# Attribution and release scope

FinReason's original code is MIT licensed, copyright 2026 Kang-Z-An.

`third_party/finqa/evaluate.py` is the unmodified official FinQA evaluator from
czyssrs/FinQA commit `0f16e2867befa6840783e58be38c9efb9229d742`. Its MIT notice,
copyright 2021 Zhiyu Chen, is preserved in `third_party/finqa/LICENSE`.
Source: https://github.com/czyssrs/FinQA.

FinQA data is downloaded from the official source by the user; raw financial
report text and training/evaluation label files are not included in this release.
Dataset citation: Chen et al., *FinQA: A Dataset of Numerical Reasoning over
Financial Data*, EMNLP 2021, https://aclanthology.org/2021.emnlp-main.300/.

The Qwen3-1.7B base model is not redistributed. Its model card and Apache-2.0
license remain with the upstream model: https://huggingface.co/Qwen/Qwen3-1.7B.
ms-swift and PEFT are external dependencies, not claimed as original algorithms.

Released result programs and benchmark IDs are derived from FinQA. Checkpoint
hashes identify archived local artifacts; they are not download links. Release
v0.1.0 distributes code, configurations, ordered split IDs, predictions, measured
results and attribution, but no model weights. The portable launchers adapt path
handling and have CPU command-plan checks; they have not been rerun through full
GPU training as part of this publication.

The v0.1.1 release adds all five frozen v3 adapters as separate Apache-2.0
ZIP assets. Every archive includes the Apache license and
upstream/model attribution. Tensor bytes are unchanged; only the private base
path in adapter_config.json is replaced by a public model ID. Base weights and
training/evaluation data are not redistributed. FinReason's original source
code remains MIT. Installation and small GPU inference checks are distinct from
rerunning full training or the full benchmark; see docs/ADAPTERS.md.
