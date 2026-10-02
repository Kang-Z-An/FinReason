"""Frozen v3 FinQA GRPO reward and observation-only training audit."""

import json
import os
from pathlib import Path
import time

from safetensors.torch import load_file, save_file
import torch
from transformers import TrainerCallback
from swift.callbacks import callbacks_map
from swift.rewards import ORM, orms

from finreason.bounded_sampling import count_response_tokens
from finreason.infer import append_record, atomic_json, sha
from finreason.rlvr_reward import normalize_swift_completion, score_batch
from finreason.rlvr_reward_v3 import FinQAV3Reward


ROOT = Path(os.environ.get('FINREASON_ROOT', '.')).resolve()


def adapter_state(model):
    return {key.replace('.default.', '.'): value.detach().cpu().contiguous()
            for key, value in model.named_parameters()
            if 'lora_' in key and value.requires_grad}


class FinReasonV3Execution(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args=args, **kwargs)
        self.registry = json.loads(Path(os.environ['FINREASON_REWARD_METADATA']).read_text())
        self.reward = FinQAV3Reward(
            ROOT / 'reward_support/official_evaluate.py',
            os.environ['FINREASON_REWARD_VARIANT'],
        )
        self.log = Path(os.environ['FINREASON_AUDIT_DIR']) / 'reward_calls.jsonl'

    def __call__(self, completions, id=None, trainer_state=None, **kwargs):
        if id is None:
            raise ValueError('Training ids were not passed to v3 reward')
        normalized = [normalize_swift_completion(c) for c in completions]
        programs = [program for program, _ in normalized]
        results = score_batch(self.reward, programs, id, self.registry)
        append_record(self.log, {
            'global_step': getattr(trainer_state, 'global_step', None),
            'ids': id, 'programs': programs,
            'framework_prefix_removed': [removed for _, removed in normalized],
            'rewards': [row['reward'] for row in results], 'diagnostics': results,
        })
        return [row['reward'] for row in results]


class FinReasonV3Audit(TrainerCallback):
    def __init__(self, args, trainer):
        self.trainer = trainer
        self.out = Path(os.environ['FINREASON_AUDIT_DIR'])

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.trainer.accelerator.num_processes != 1:
            raise ValueError('v3 audit expects one GPU')
        if args.generation_batch_size != args.num_generations or args.num_generations != 8:
            raise ValueError('Frozen v3 group size changed')
        source_path = Path(os.environ['FINREASON_SOURCE_ADAPTER'])
        source = load_file(str(source_path))
        initial = adapter_state(model)
        if set(initial) != set(source):
            raise ValueError('Loaded adapter tensor set differs from frozen v3 SFT')
        mismatch = [key for key in source
                    if not torch.equal(initial[key], source[key].to(initial[key].dtype))]
        if mismatch:
            raise ValueError('Frozen v3 SFT adapter was not loaded')
        save_file(initial, str(self.out / 'loaded_initial_adapter.safetensors'))
        atomic_json(self.out / 'initialization_audit.json', {
            'source_sha256': sha(source_path),
            'loaded_initial_sha256': sha(self.out / 'loaded_initial_adapter.safetensors'),
            'trainable_tensors': len(initial), 'mismatched_tensors': mismatch,
            'dtypes': sorted({str(value.dtype) for value in initial.values()}),
        })
        original_advantages = self.trainer._compute_advantages

        def observed_advantages(samples, rewards_per_func, batch_encoded_inputs):
            advantages = original_advantages(samples, rewards_per_func, batch_encoded_inputs)
            values = advantages.detach().float().cpu()
            rewards = rewards_per_func.detach().float().cpu().reshape(-1)
            groups = rewards.reshape(-1, args.num_generations)
            if not torch.isfinite(values).all() or not torch.isfinite(rewards).all():
                raise ValueError('Non-finite v3 reward or advantage')
            append_record(self.out / 'advantage_audit.jsonl', {
                'global_step': self.trainer.state.global_step,
                'rewards': rewards.tolist(), 'advantages': values.tolist(),
                'groups': len(groups),
                'mixed_groups': int((groups.amax(dim=1) > groups.amin(dim=1)).sum()),
                'nonzero_advantages': int((values != 0).sum()),
            })
            return advantages

        self.trainer._compute_advantages = observed_advantages
        lengths = json.loads(Path(os.environ['FINREASON_PROMPT_LENGTHS']).read_text())
        original_generate = self.trainer._generate_completions

        def observed_generate(samples, *positional, **named):
            started = time.monotonic()
            result = original_generate(samples, *positional, **named)
            ids = [sample.extra['id'] for sample in result]
            if len(ids) != 8 or len(set(ids)) != 1:
                raise ValueError('Expected one frozen question per eight-rollout group')
            counts = [count_response_tokens(sample.response_token_ids) for sample in result]
            if not all(count > 0 for count in counts):
                raise ValueError('Missing rollout token count')
            append_record(self.out / 'rollout_cost.jsonl', {
                'global_step': self.trainer.state.global_step, 'ids': ids,
                'completion_tokens': counts,
                'prompt_tokens_sum_unshared': sum(lengths[item_id] for item_id in ids),
                'generation_seconds': time.monotonic() - started,
                'finish_reasons': [sample.finish_reason for sample in result],
            })
            return result

        self.trainer._generate_completions = observed_generate

    def on_step_end(self, args, state, control, **kwargs):
        atomic_json(self.out / 'training_progress.json', {
            'global_step': state.global_step, 'max_steps': args.max_steps,
            'state': 'complete' if state.global_step == args.max_steps else 'running',
        })


orms['finreason_v3_execution'] = FinReasonV3Execution
callbacks_map['finreason_v3_audit'] = FinReasonV3Audit
