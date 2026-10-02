"""GPU-only, resumable inference with v3 generation settings and an explicit adapter."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from finreason.infer import encode_prompts, load_prompts, run_predictions, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; use scripts.verify_results for CPU evidence checks')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    cfg = json.loads(Path('configs/inference.json').read_text())
    source = json.loads(Path('reports/tokenizer_sources.json').read_text())
    for name, record in source['files'].items():
        if sha(Path(args.model_dir) / name) != record['sha256']:
            raise ValueError('Pinned model/tokenizer source changed: ' + name)
    if list(Path('data/processed/rlvr_v3').rglob('*.labels.jsonl')):
        raise RuntimeError('Keep evaluation labels on the local scoring machine')
    rows = load_prompts(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    encoded = encode_prompts(rows, tokenizer, cfg)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        trust_remote_code=False, dtype=torch.float16, attn_implementation='sdpa').to('cuda:0').eval()
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False).eval()
    eos_ids = list(dict.fromkeys([tokenizer.eos_token_id, tokenizer.pad_token_id]))
    generation = GenerationConfig(do_sample=cfg['do_sample'], temperature=cfg['temperature'],
        top_p=cfg['top_p'], top_k=cfg['top_k'], max_new_tokens=cfg['max_new_tokens'],
        num_return_sequences=1, num_beams=1, eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        use_cache=True, repetition_penalty=1.0)

    def generate(row, seed):
        set_seed(seed)
        inputs = torch.tensor([encoded[row['id']]], dtype=torch.long, device='cuda:0')
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=generation)
        torch.cuda.synchronize()
        tokens = outputs[0, inputs.shape[1]:].tolist()
        ended = bool(tokens) and tokens[-1] in eos_ids
        content = tokens[:-1] if ended else tokens
        return {'program': tokenizer.decode(content, skip_special_tokens=False).strip(),
            'raw_text': tokenizer.decode(tokens, skip_special_tokens=False), 'generated_token_ids': tokens,
            'prompt_tokens': inputs.shape[1], 'completion_tokens': len(tokens), 'seed': seed,
            'generation_seconds': time.perf_counter()-start, 'finish_reason': 'eos' if ended else 'length'}

    spec = {'stage': 'initial_open_source_adapter_inference', 'config': cfg,
        'prompts_sha256': sha(args.prompts), 'adapter_sha256': sha(Path(args.adapter)/'adapter_model.safetensors'),
        'runner_sha256': sha(__file__), 'gpu': torch.cuda.get_device_name(0), 'labels_on_remote': False,
        'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft')}}
    print(json.dumps(run_predictions(rows, spec, args.output, generate, args.limit), indent=2))


if __name__ == '__main__':
    main()
