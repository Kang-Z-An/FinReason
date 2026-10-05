"""GPU-only, resumable inference with v3 generation settings and an explicit adapter."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from finreason.infer import encode_prompts, load_prompts, run_predictions, sha
from finreason.adapter_assets import file_sha, verify_adapter
from finreason.generation_guard import generation_policy, install_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--released-tag', help='Require the exact released adapter identity')
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; use scripts.verify_results for CPU evidence checks')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if importlib.metadata.version('transformers') != '4.57.6':
        raise RuntimeError('This audited generation resolver requires transformers==4.57.6')
    cfg = json.loads(Path('configs/inference.json').read_text())
    source = json.loads(Path('reports/tokenizer_sources.json').read_text())
    for name, record in source['files'].items():
        if sha(Path(args.model_dir) / name) != record['sha256']:
            raise ValueError('Pinned model/tokenizer source changed: ' + name)
    model_sources = json.loads(Path('configs/model_sources.json').read_text())
    for name, expected in model_sources.items():
        if file_sha(Path(args.model_dir) / name) != expected:
            raise ValueError('Pinned base weight/config changed: ' + name)
    identity = verify_adapter(args.adapter, json.loads(Path('configs/adapter_assets.json').read_text()), args.released_tag)
    if list(Path('data/processed/rlvr_v3').rglob('*.labels.jsonl')):
        raise RuntimeError('Keep evaluation labels on the local scoring machine')
    rows = load_prompts(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    encoded = encode_prompts(rows, tokenizer, cfg)
    base = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        trust_remote_code=False, dtype=torch.float16, attn_implementation='sdpa').to('cuda:0').eval()
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    policy = generation_policy(cfg, tokenizer, json.loads((Path(args.model_dir) / 'config.json').read_text())['bos_token_id'])
    generation = GenerationConfig(**{k: v for k, v in policy.items() if k != 'use_model_defaults'})
    eos_ids = policy['eos_token_id']
    resolver_calls = install_guard(base, policy)

    def generate(row, seed):
        set_seed(seed)
        inputs = torch.tensor([encoded[row['id']]], dtype=torch.long, device='cuda:0')
        torch.cuda.synchronize()
        start = time.perf_counter()
        before = len(resolver_calls)
        with torch.inference_mode():
            outputs = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                     generation_config=generation, **policy)
        torch.cuda.synchronize()
        if len(resolver_calls) != before + 1:
            raise RuntimeError('Actual generate call did not reach the audited resolver exactly once')
        tokens = outputs[0, inputs.shape[1]:].tolist()
        ended = bool(tokens) and tokens[-1] in eos_ids
        content = tokens[:-1] if ended else tokens
        return {'program': tokenizer.decode(content, skip_special_tokens=False).strip(),
            'raw_text': tokenizer.decode(tokens, skip_special_tokens=False), 'generated_token_ids': tokens,
            'prompt_tokens': inputs.shape[1], 'completion_tokens': len(tokens), 'seed': seed,
            'generation_seconds': time.perf_counter()-start, 'finish_reason': 'eos' if ended else 'length',
            'effective_generation': resolver_calls[-1], 'generation_resolver_calls': 1}

    spec = {'stage': 'public_adapter_inference_v1p1', 'config': cfg, 'adapter_identity': identity,
        'prompts_sha256': sha(args.prompts), 'adapter_sha256': identity['adapter_sha256'],
        'generation_policy': policy, 'base_files_sha256': model_sources,
        'implementation_sha256': {p: sha(p) for p in ('finreason/infer.py', 'finreason/generation_guard.py', 'finreason/adapter_assets.py')},
        'runner_sha256': sha(__file__), 'gpu': torch.cuda.get_device_name(0), 'labels_on_remote': False,
        'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft')}}
    print(json.dumps(run_predictions(rows, spec, args.output, generate, args.limit), indent=2))


if __name__ == '__main__':
    main()
