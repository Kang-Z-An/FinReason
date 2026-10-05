# v0.1.1 adapter assets

Five original v3 LoRA adapters are available as separate ZIP assets in the
[v0.1.1 release](https://github.com/Kang-Z-An/FinReason/releases/tag/v0.1.1).
`configs/adapter_assets.json` pins their URLs, sizes and SHA-256 hashes.

All five fixed methods are included: SFT, binary seeds 42/1234, and validity_0p1
seeds 42/1234. There is no selection of a winner using final scores. Each ZIP
contains only `adapter_model.safetensors`, `adapter_config.json`, `LICENSE`,
`NOTICE` and `README.md`. It contains no base weights, raw reports, training
data, evaluation labels, optimizer states or private SSH information.

Tensor bytes match the original checkpoint hashes in `results/v3/provenance.json`.
Only `base_model_name_or_path` in the adapter config changes from a private
filesystem path to `Qwen/Qwen3-1.7B`. Both original config hashes and the new
member hashes are recorded in `configs/adapter_assets.json`. These adapters
use Apache-2.0; the repository's original code remains MIT. The Qwen base is
obtained separately under its upstream license.

## Install one adapter

Run on the GPU machine from the repository root:

```bash
python -m scripts.download_adapter --tag sft --output models/adapters/sft
```

After downloading a ZIP separately, or for offline installation:

```bash
python -m scripts.download_adapter --tag sft --output models/adapters/sft \
  --archive /PATH/finreason_v3_sft.zip
```

The installer verifies archive size and SHA-256, the exact member names, member
sizes and every member hash before installing. Existing destinations are never
overwritten. Loading a released tag verifies the sanitized config as well as
the unchanged weights. Retrained adapters may omit `--released-tag`; their new
identity is still recorded but they are not declared released checkpoints.

## Small pipeline check

On the local scoring machine, download the pinned FinQA source and reconstruct
the original data as described in [REPRODUCE.md](REPRODUCE.md). Then:

```bash
python -m scripts.prepare_smoke
```

This selects eight SFT training IDs by a fixed SHA ordering, without inspecting
model outputs. The selection has no ID/document overlap with v3 dev/final.
Copy **only** `data/smoke_v1/train8.prompts.jsonl` to the GPU machine. Keep
`train8.labels.jsonl`, raw train JSON and selection metadata on the local scoring
machine. Download the pinned base on the GPU as in the reproduction guide.

Run a single adapter on the GPU:

```bash
python -m scripts.infer_adapter --model-dir models/base \
  --adapter models/adapters/sft --released-tag sft \
  --prompts data/smoke_v1/train8.prompts.jsonl --output outputs/smoke/sft
```

Or exercise all five adapters sequentially:

```bash
python -m scripts.smoke_adapters --model-dir models/base \
  --prompts data/smoke_v1/train8.prompts.jsonl --output outputs/smoke
```

Use `--asset-dir /PATH/ZIP_DIRECTORY` for the all-five offline route. The queue
requires prompt-only inputs; it never reconstructs data or opens answer labels.
Inference records the policy resolved inside every actual `generate` call and
requires `use_model_defaults=False`. Base weights and configs are SHA-checked,
context is not truncated, and completed predictions retain per-ID seeds when
resuming. The audited resolver requires Transformers 4.57.6.

Bring `predictions.jsonl` and `run.json` back to the local scoring machine and run:

```bash
python -m scripts.score_predictions \
  --predictions outputs/smoke/sft/predictions.jsonl \
  --run-spec outputs/smoke/sft/run.json \
  --prompts data/smoke_v1/train8.prompts.jsonl \
  --labels data/smoke_v1/train8.labels.jsonl --output outputs/smoke/sft/score.json
```

For the all-five queue, files live below `outputs/smoke/predictions/TAG/`.
The all-five offline ZIP route was run on one RTX 4090: 40/40 predictions,
40/40 actual resolver checks, 40/40 locally executable programs, with six of
eight correct for each adapter. Total GPU queue wall time was 162.5 seconds.
See [GPU_SMOKE.md](GPU_SMOKE.md) for the verification boundary.

The eight questions are seen training sources. This check proves installation,
GPU inference, resolved generation settings and local scoring execute; its
correct counts do **not** estimate generalization, RL gains or reproduce the
published 551-question metrics. Full training and full benchmark reruns of the
portable public entrypoints remain separate work. The original v0.1.0 tag and
v3 results are retained unchanged.
