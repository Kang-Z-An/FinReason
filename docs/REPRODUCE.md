# Reproduce v3 evidence and training

There are two levels: **CPU evidence reproduction**, verified for this release,
and **GPU method reproduction**, whose new portable launchers are command-plan
checked but have not been fully rerun for the public package. The original GPU
experiment is complete. Do not confuse a launcher dry-run with a training result.

## CPU evidence

Run from the repository root, using Python 3.10+ on Linux/macOS:

```bash
python -m unittest discover -s tests -v
python -m scripts.verify_results
python -m scripts.check_release
```

No third-party Python packages or GPU are needed. Optional original data replay:

```bash
python -m scripts.prepare_data
python -m scripts.verify_results --source-training-file data/raw/finqa/train.json
```

The preparer downloads only the pinned FinQA train source, validates SHA-256,
reconstructs the original 3,500 SFT examples, 1,000 GRPO prompts/reward metadata,
407 dev prompts/local labels and 551 final prompts/local labels. Every generated
file must match its original hash. `--source-training-file PATH` avoids network
when the exact file is already available. Existing outputs are not overwritten.

Outputs live under `data/reconstructed_v3/`. This reconstruction contains
evaluation labels **for local scoring only**. Do not upload those label files or
the raw train JSON to the GPU host. The source train also contains questions
reserved for dev/final, so uploading the entire source defeats that boundary.

## GPU workspace

Clone this code on a CUDA host, then transfer only the reconstructed training
files and prompt-only evaluation files into its `data/processed/rlvr_v3/` tree.
Use your own SSH alias; for example, from the scoring machine:

```bash
rsync -av --exclude='*.labels.jsonl' \
  data/reconstructed_v3/data/processed/rlvr_v3/ \
  YOUR_GPU_ALIAS:/PATH/FinReason/data/processed/rlvr_v3/
```

Create the remote destination first. Gold answers in **training reward metadata**
are expected; they are never model inputs. Only dev/final labels remain local.

Recorded hardware: one RTX 4090, 24 GB. Recorded core runtime: Python 3.10,
PyTorch 2.6.0+cu124, ms-swift 4.5.3, Transformers 4.57.6, PEFT 0.20.0,
TRL 0.29.1 and datasets 4.8.4. These are the measured versions, not a complete
dependency solver lock. Install CUDA/PyTorch only on the GPU host:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-runtime.txt
hf download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir models/base
```

Check the model card and license. The repository does not redistribute weights.
The historical model-file hashes are available in `configs/model_sources.json`.

## SFT and GRPO

Print the exact command plan without loading a model:

```bash
python -m scripts.train --stage sft --model-dir models/base --output outputs/v3/sft
python -m scripts.train --stage grpo --model-dir models/base \
  --adapter outputs/v3/sft/RUN_DIRECTORY/checkpoint-875 \
  --variant binary --seed 42 --output outputs/v3/binary_seed42
```

Replace `RUN_DIRECTORY` with the actual Swift output directory. Add `--execute`
only on the GPU host after reviewing the plan. SFT uses FP32, not FP16/BF16;
GRPO uses BF16. All four GRPO runs must start from the **same SFT adapter**;
they are not chained. Run both `binary` and `validity_0p1` with seeds 42 and 1234.
The plugin logs rewards, advantages, generation costs, and checks the loaded
adapter tensors. Fresh output directories are required.

The launchers check frozen data, model sources, runtime versions, sequence/template
alignment and label boundaries before execution. The original experiment also
performed longest-sample smoke trials and final optimizer/weight-update audits;
the portable first-release runner does not replace all of those resource/audit
steps. Inspect finite gradients, nonzero LoRA updates and final optimizer steps
before reporting any new training as successful. Training loss alone is not a
task metric. Exact checkpoint bytes across hardware/software are not promised.

## Adapter inference and scoring

Use the frozen generation configuration: non-thinking, FP16 inference,
temperature 0.7, top-p 0.8, top-k 20, 256 new tokens, context 4,096 and a
SHA-derived per-question seed from 42. Inference can resume with the same inputs:

```bash
python -m scripts.infer_adapter \
  --prompts data/processed/rlvr_v3/split_v1/dev.prompts.jsonl \
  --model-dir models/base --adapter YOUR_CHECKPOINT \
  --output outputs/eval/YOUR_RUN
```

Bring predictions back to the local scoring machine. A generic scoring command
for **new** predictions is documented in `python -m scripts.score_predictions --help`;
it is separate from the immutable published-result verifier. Evaluate dev first;
only open a new final set after methods, checkpoints and seeds are fixed. The
published v3 final set is already open and cannot confirm a newly tuned method.
