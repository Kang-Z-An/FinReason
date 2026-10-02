"""Portable v3 command plan; dry-run by default. Execute on a CUDA host only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

from finreason import training_commands as commands
from finreason.infer import sha, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['sft', 'grpo'], required=True)
    parser.add_argument('--workspace', default='.')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--adapter', help='The same SFT adapter for every GRPO variant/seed')
    parser.add_argument('--variant', choices=['binary', 'validity_0p1'], default='binary')
    parser.add_argument('--seed', type=int, choices=[42, 1234], default=42)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.stage == 'sft' and (args.adapter or args.seed != 42):
        parser.error('Frozen SFT starts from base with seed 42; adapters and other SFT seeds are forbidden')
    root = Path(args.workspace).resolve()
    cfg = json.loads((root / 'configs/v3.json').read_text())
    commands.MODEL = Path(args.model_dir).resolve()
    commands.OUT = Path(args.output).resolve()
    commands.PLUGIN = root / 'finreason/swift_plugin.py'
    data = root / cfg[args.stage]['dataset' if args.stage == 'sft' else 'prompts']
    if args.stage == 'sft':
        command = commands.build_command(cfg, data)
    else:
        if not args.adapter:
            parser.error('--adapter is required for GRPO')
        commands.ADAPTER = Path(args.adapter).resolve()
        parameters = commands.arguments(cfg, args.variant, args.seed, cfg['grpo']['steps'], data, commands.OUT)
        command = ['swift', 'rlhf']
        for key, value in parameters.items():
            command.append('--' + key)
            command.extend([str(value).lower()] if isinstance(value, bool) else
                           list(map(str, value)) if isinstance(value, list) else [str(value)])
    environment = {'FINREASON_ROOT': str(root), 'USE_TF': '0', 'USE_FLAX': '0',
        'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',
        'FINREASON_REWARD_METADATA': str(root / cfg['grpo']['reward_metadata']),
        'FINREASON_PROMPT_LENGTHS': str(root / cfg['grpo']['prompt_lengths']),
        'FINREASON_REWARD_VARIANT': args.variant,
        'FINREASON_AUDIT_DIR': str(commands.OUT.parent / (commands.OUT.name + '.audit')),
        'FINREASON_SOURCE_ADAPTER': str(commands.ADAPTER / 'adapter_model.safetensors') if args.adapter else ''}
    print(json.dumps({'execute': args.execute, 'command': command, 'environment': environment}, indent=2))
    if not args.execute:
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 22 * 1024**3:
        raise RuntimeError('Training requires a 24 GB class CUDA GPU')
    versions = {'ms-swift': '4.5.3', 'transformers': '4.57.6', 'peft': '0.20.0', 'trl': '0.29.1'}
    for name, version in versions.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError('Recorded runtime differs: ' + name)
    if args.stage == 'grpo' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GRPO requires native BF16 support')
    expected = cfg[args.stage]['dataset_sha256' if args.stage == 'sft' else 'prompts_sha256']
    if sha(data) != expected:
        raise ValueError('Training data hash changed')
    for name, expected_sha in json.loads((root / 'configs/model_sources.json').read_text()).items():
        if sha(commands.MODEL / name) != expected_sha:
            raise ValueError('Pinned model source changed: ' + name)
    # Evaluation answers never accompany remote training/inference.
    if list((root / 'data/processed/rlvr_v3').rglob('*.labels.jsonl')):
        raise RuntimeError('Remove eval label files from GPU workspace; keep them on the scoring machine')
    if commands.OUT.exists() or Path(environment['FINREASON_AUDIT_DIR']).exists():
        raise FileExistsError('Output/audit exists; inspect it instead of overwrite')
    if not shutil.which('swift'):
        raise RuntimeError('swift executable unavailable')
    from swift import get_processor, get_template
    processor = get_processor(str(commands.MODEL), model_type='qwen3')
    tokenizer = getattr(processor, 'tokenizer', processor)
    template = get_template(processor, template_type='qwen3', max_length=4096,
        loss_scale=cfg['sft']['loss_scale'], add_non_thinking_prefix=True)
    template.set_mode('train' if args.stage == 'sft' else 'transformers')
    records = [json.loads(line) for line in data.read_text().splitlines()]
    if len(records) != (3500 if args.stage == 'sft' else 1000):
        raise ValueError('Training row count differs')
    for row in records:
        messages = row['messages']
        prompts = messages[:2]
        prompt_tokens = tokenizer.apply_chat_template(prompts, tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        encoded = template.encode(row)
        if args.stage == 'sft':
            full = tokenizer.apply_chat_template(messages, tokenize=True,
                add_generation_prompt=False, enable_thinking=False)
            labels = list(encoded['labels'])
            if list(encoded['input_ids']) != full or len(full) > 4096 or \
               not any(x != -100 for x in labels) or any(x != -100 for x in labels[:len(prompt_tokens)]):
                raise ValueError('SFT template/label preflight failed')
        elif list(encoded['input_ids']) != prompt_tokens or len(prompt_tokens) + 256 > 4096:
            raise ValueError('GRPO template/length preflight failed')
    if args.stage == 'grpo':
        for field, hashfield in [('reward_metadata', 'reward_metadata_sha256'), ('prompt_lengths', 'prompt_lengths_sha256')]:
            if sha(root / cfg['grpo'][field]) != cfg['grpo'][hashfield]:
                raise ValueError('Reward support changed: ' + field)
        support = root / 'reward_support'
        support.mkdir(exist_ok=True)
        shutil.copyfile(root / 'third_party/finqa/evaluate.py', support / 'official_evaluate.py')
    Path(environment['FINREASON_AUDIT_DIR']).mkdir(parents=True)
    atomic_json(Path(environment['FINREASON_AUDIT_DIR']) / 'command.json', {'command': command,
        'portable_initial_release': True, 'config_sha256': sha(root / 'configs/v3.json')})
    subprocess.run(command, cwd=root, env={**os.environ, **environment}, check=True)


if __name__ == '__main__':
    main()
