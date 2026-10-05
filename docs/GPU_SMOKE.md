# Public adapter path: small GPU check

On 2026-10-05, all five v3 ZIP assets were installed with member SHA-256 checks
and run through `scripts.infer_adapter` on one RTX 4090. Each adapter used the
same eight SFT training-source questions selected by a fixed SHA ordering,
without conditioning on outcomes. There was no new training or final-set
model inference. Only prompt-only inputs were uploaded; labels stayed local.

| Adapter | Predictions | Executable | Correct |
| --- | ---: | ---: | ---: |
| SFT | 8 | 8 | 6 |
| binary seed 42 | 8 | 8 | 6 |
| binary seed 1234 | 8 | 8 | 6 |
| validity_0p1 seed 42 | 8 | 8 | 6 |
| validity_0p1 seed 1234 | 8 | 8 | 6 |

The queue completed in 162.5 seconds. All 40 generation calls reached the
guarded resolver exactly once with model-default fallback disabled and the
specified sampling settings. Base model shards, adapters and sanitized configs
were hash-checked. Returned predictions, run manifests and queue/log artifacts
were checked against remote hashes before local official execution scoring.

Runtime: PyTorch 2.6.0, Transformers 4.57.6, PEFT 0.20.0. Generation: FP16 SDPA,
non-thinking, temperature 0.7, top-p 0.8, top-k 20, maximum 256 new tokens,
4,096-token budget without truncation and SHA-derived per-ID seeds from 42.
Each saved output includes token IDs and the actual resolved generation policy.
See `results/reproduction_smoke_v1/` for the compact summary and all 40 outputs.

**This is an installation/inference/scoring check on seen training questions.**
It does not estimate accuracy, measure a new RL gain or reproduce the published
551-question benchmark. The all-five GPU check used ZIP files uploaded directly
to the remote machine; public URL download is a separate installation check.
Full public training and full benchmark reruns are still not claimed. The
historical v3/v4 results, final-method choices and the original v0.1.0 tag were
not modified by this check.
