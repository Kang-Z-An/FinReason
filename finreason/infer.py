"""Resumable, one-answer-per-question baseline. Preflight never loads model weights."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from .data import read_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def append_record(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_prompts(path):
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if set(row) != {"id", "messages"}:
                raise ValueError("Input must contain only id and messages; do not pass labels or SFT data")
            if not isinstance(row["id"], str) or row["id"] in seen:
                raise ValueError("Invalid or duplicate input id")
            conversation = row["messages"]
            if not isinstance(conversation, list) or len(conversation) != 2:
                raise ValueError("Expected system and user messages only")
            for message, role in zip(conversation, ("system", "user")):
                if set(message) != {"role", "content"} or message["role"] != role or not isinstance(message["content"], str):
                    raise ValueError("Expected system/user text without assistant answers")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("Empty input file")
    return rows


def recover_predictions(path, allowed_ids):
    """Only an unterminated final record may be discarded; archive it first."""
    path = Path(path)
    if not path.exists():
        return {}
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        boundary = content.rfind(b"\n") + 1
        backup = path.with_name(path.name + ".interrupted-" + str(time.time_ns()))
        backup.write_bytes(content[boundary:])
        with path.open("r+b") as handle:
            handle.truncate(boundary)
            handle.flush()
            os.fsync(handle.fileno())
        content = content[:boundary]
    saved = {}
    for line in content.splitlines():
        row = json.loads(line)
        if row.get("id") not in allowed_ids or row["id"] in saved or not isinstance(row.get("program"), str):
            raise ValueError("Invalid, unknown or duplicate saved prediction; refusing to resume")
        saved[row["id"]] = row
    return saved


def item_seed(seed, item_id):
    # Each sample has its own seed so skipped samples do not shift RNG state.
    return int.from_bytes(hashlib.sha256((str(seed) + ":" + item_id).encode()).digest()[:4], "big") % (2**31)


def run_predictions(rows, spec, output_dir, generate, limit=None):
    """The callback receives prompt-only rows. Errors stop the run and remain retryable."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is writing this run directory")
        manifest = output / "run.json"
        predictions = output / "predictions.jsonl"
        if manifest.exists():
            if read_json(manifest) != spec:
                raise ValueError("Run configuration, source, or environment changed; use a new output directory")
        elif predictions.exists():
            raise ValueError("Predictions exist without a run manifest; refusing to mix runs")
        else:
            atomic_json(manifest, spec)
        saved = recover_predictions(predictions, {row["id"] for row in rows})
        selected = rows if limit is None else rows[:limit]

        def progress(state, failed_id=None):
            completed = sum(row["id"] in saved for row in selected)
            result = {"state": state, "requested": len(selected), "completed_requested": completed,
                      "completed_total": len(saved), "dataset_total": len(rows),
                      "complete_dataset": len(saved) == len(rows), "failed_id": failed_id}
            atomic_json(output / "progress.json", result)
            return result

        progress("running")
        for row in selected:
            if row["id"] in saved:
                continue
            start = time.perf_counter()
            try:
                result = generate(row, item_seed(spec["config"]["seed"], row["id"]))
                if not isinstance(result.get("program"), str) or "id" in result:
                    raise ValueError("Backend returned an invalid prediction")
                record = dict(result, id=row["id"], wall_seconds=time.perf_counter()-start)
                append_record(predictions, record)
                saved[row["id"]] = record
                progress("running")
                print("Saved", len(saved), "/", len(rows), row["id"], flush=True)
            except (Exception, KeyboardInterrupt) as exc:
                append_record(output / "errors.jsonl", {"id": row["id"], "type": type(exc).__name__,
                              "error": str(exc), "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
                progress("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", row["id"])
                raise
        return progress("complete" if len(saved) == len(rows) else "subset_complete")


def encode_prompts(rows, tokenizer, config):
    encoded = {}
    for row in rows:
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False)
        if len(ids) + config["max_new_tokens"] > config["context_length"]:
            raise ValueError("Context budget exceeded without truncation: " + row["id"])
        encoded[row["id"]] = ids
    return encoded


def make_generator(config, tokenizer, encoded, device, allow_download):
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig, set_seed
    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("Real inference currently requires an NVIDIA CUDA device; use --preflight on this Mac")
    torch.cuda.set_device(device)
    if config["dtype"] == "bfloat16" and not native_bf16_supported(torch):
        raise RuntimeError("GPU does not support bfloat16; use a separate float16 config/run")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["model_revision"],
        cache_dir="models/hf-cache", local_files_only=not allow_download, trust_remote_code=False,
        torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()
    eos_ids = list(dict.fromkeys([tokenizer.eos_token_id, tokenizer.pad_token_id]))
    generation = GenerationConfig(do_sample=config["do_sample"], temperature=config["temperature"],
        top_p=config["top_p"], top_k=config["top_k"], max_new_tokens=config["max_new_tokens"],
        num_return_sequences=1, num_beams=1, eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, use_cache=True, repetition_penalty=1.0)

    def generate(row, seed):
        set_seed(seed)
        inputs = torch.tensor([encoded[row["id"]]], dtype=torch.long, device=device)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=generation)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter()-start
        generated = outputs[0, inputs.shape[1]:].tolist()
        ended = bool(generated) and generated[-1] in eos_ids
        content_ids = generated[:-1] if ended else generated
        # Strip only the final stop token and whitespace; preserve unexpected tags/fences.
        text = tokenizer.decode(content_ids, skip_special_tokens=False)
        return {"program": text.strip(), "raw_text": tokenizer.decode(generated, skip_special_tokens=False),
                "generated_token_ids": generated, "prompt_tokens": inputs.shape[1],
                "completion_tokens": len(generated), "seed": seed, "generation_seconds": seconds,
                "finish_reason": "eos" if ended else "length"}
    return generate


def native_bf16_supported(torch):
    """T4 may allow BF16 tensor allocation without native BF16 arithmetic."""
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        # Older CUDA-only builds do not expose the keyword.
        return torch.cuda.get_device_capability()[0] >= 8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="data/processed/dev.prompts.jsonl")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--tokenizer-dir", default="models/Qwen3-1.7B-tokenizer")
    parser.add_argument("--output-dir", default="outputs/base_dev")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true", help="Tokenize/validate inputs only; never load weights")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading the pinned model weights for real inference")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = read_json(args.config)
    if config["enable_thinking"] is not False or config["dtype"] not in ("bfloat16", "float16"):
        raise ValueError("Unsupported thinking mode or dtype")
    if config["context_length"] <= config["max_new_tokens"] or config["max_new_tokens"] < 1:
        raise ValueError("Invalid context/generation budget")
    sources = read_json("reports/tokenizer_sources.json")
    if (config["model_id"], config["model_revision"]) != (sources["model_id"], sources["revision"]):
        raise ValueError("Model and audited tokenizer revisions must match")
    for filename, info in sources["files"].items():
        if sha(Path(args.tokenizer_dir)/filename) != info["sha256"]:
            raise ValueError("Tokenizer file changed: " + filename)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True, trust_remote_code=False)
    rows = load_prompts(args.prompts)
    encoded = encode_prompts(rows, tokenizer, config)
    if args.preflight:
        report = {"mode": "preflight_no_model", "examples": len(rows), "max_prompt_tokens": max(map(len, encoded.values())),
                  "context_length": config["context_length"], "max_new_tokens": config["max_new_tokens"],
                  "prompts_sha256": sha(args.prompts), "config": config, "model_loaded": False}
        atomic_json(Path(args.output_dir)/"preflight.json", report)
        print(json.dumps(report, indent=2))
        return
    import torch
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("CUDA unavailable; use --preflight to validate without loading weights")
    spec = {"config": config, "prompts_sha256": sha(args.prompts), "tokenizer": sources,
            "runner_sha256": sha(__file__), "device": args.device, "gpu": torch.cuda.get_device_name(args.device),
            "torch_cuda": torch.version.cuda,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")}}
    backend = None

    def lazy_generate(row, seed):
        nonlocal backend
        if backend is None:
            backend = make_generator(config, tokenizer, encoded, args.device, args.allow_download)
        return backend(row, seed)

    print(json.dumps(run_predictions(rows, spec, args.output_dir, lazy_generate, args.limit), indent=2))


if __name__ == "__main__":
    main()
