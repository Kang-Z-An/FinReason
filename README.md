# FinReason

**Financial numerical reasoning with LoRA SFT, GRPO and verifiable execution rewards.**

[中文说明](README.zh-CN.md) · [Reproduction](docs/REPRODUCE.md) · [Results and limitations](docs/RESULTS.md) · [MIT](LICENSE)

FinReason trains Qwen3-1.7B to turn a financial table, surrounding text and a
question into a short executable FinQA program. A restricted DSL computes the
answer, and the official FinQA evaluator supplies task metrics. The experiment
compares SFT with two GRPO reward designs at two fixed RL seeds.

The **v0.1.1 release** adds all five frozen [adapter assets](docs/ADAPTERS.md)
and a short verified GPU inference/scoring path for the completed v3 experiment.
FinReason is an
experimental portfolio project, not an end-to-end financial assistant or a new
RL algorithm. There is no RAG, Agent tool loop or generated natural-language CoT.

## Measured results

The final set contains **551 questions in 91 company/year groups**, held out
prospectively within the original FinQA **training** data. It is **not the
official FinQA test set**, and it was not historically untouched during all
earlier project work. All five checkpoints were fixed before this final run.

| Model | Correct / 551 | Accuracy | Executable / 551 | Change from SFT |
| --- | ---: | ---: | ---: | ---: |
| SFT | 388 | 70.42% | 550 | — |
| Binary GRPO, seed 42 | 397 | 72.05% | 550 | +1.63 pp |
| Binary GRPO, seed 1234 | 398 | 72.23% | 550 | +1.81 pp |
| Validity bonus 0.1, seed 42 | 399 | 72.41% | 550 | +2.00 pp |
| Validity bonus 0.1, seed 1234 | 397 | 72.05% | 550 | +1.63 pp |

Both binary runs improved the correct count without reducing executability.
Paired document-cluster 95% intervals for their gains are **[0.00, 3.30]** and
**[0.52, 3.25]** percentage points. Seed 42's interval touches zero: these results
do not establish statistical significance for both seeds. Both share **one SFT
starting run**. The validity bonus changes the paired correct count by +2 / −1
relative to binary at the corresponding seed, with no consistent advantage.

On the separate 407-question development set, the fresh base and SFT obtained
38/407 and 262/407 correct, respectively. Do not compare that development base
score directly with the final-set RL score.

## What is implemented

- Prompt-only input contracts and a non-thinking Qwen chat-template match.
- 3,500-row FP32 LoRA SFT: rank 16, alpha 32, all linear layers, 875 updates.
- 1,000-question GRPO pool, eight rollouts per question, 1,000 updates per run;
  binary correctness reward versus a 0.1 reward for valid-but-wrong programs.
- BF16 GRPO, group reward scaling, **KL coefficient 0**, no dynamic sampling and
  no vLLM in v3. These are explicit experimental settings, not general advice.
- Restricted DSL validation, official execution scoring, resume-safe generation,
  finite-update audits, paired regressions and 10,000 document bootstrap draws.
- Public ordered split IDs, per-question programs and correctness flags, exact
  dataset reconstruction hashes, original checkpoint hashes and source notices.

## Quick start without a GPU

Python 3.10+ on Linux/macOS. These commands require only the standard library:

```bash
git clone https://github.com/Kang-Z-An/FinReason.git
cd FinReason
python -m unittest discover -s tests -v
python -m scripts.verify_results
python -m scripts.check_release
```

`verify_results` recomputes all published counts, paired changes and bootstrap
intervals from the released per-question records. For an independent official
rescore, download the pinned upstream train file and reconstruct the frozen data:

```bash
python -m scripts.prepare_data
python -m scripts.verify_results --source-training-file data/raw/finqa/train.json
```

The second verification executes all **2,755 saved programs** against the source
tables and answers. It uses no model weights or GPU. Downloading data requires
network access; the offline quick start does not.

## GPU reproduction

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for data transfer boundaries, recorded
runtime versions, dry-run training plans, model revision and adapter inference.
The measured runs used one RTX 4090. The release's portable launchers have CPU
checks, but a full GPU rerun of the public package has **not** been performed.

**Weights were not included in v0.1.0.** The v0.1.1 release adds verified ZIP
installation, pinned adapter identities, actual generation-policy checks, and
an eight-training-question pipeline check. Download the LoRA assets using
[ADAPTERS.md](docs/ADAPTERS.md). Base weights are obtained separately. Exact
bitwise reproduction across machines is not promised. Full public training and
benchmark reruns have not been performed.

## Repository

```text
finreason/             DSL, rewards, resumable inference, statistics, Swift plugin
scripts/               data reconstruction, result checks, portable GPU launchers
configs/               frozen v3 method, generation settings, ordered data IDs
results/v3/            all five final results, per-question programs, provenance
prompts/               frozen financial program-generation prompt
tests/                 CPU contract, reward, execution and result tests
third_party/finqa/     official evaluator and its original MIT notice
docs/                  experimental scope and reproduction instructions
```

## Limitations and attribution

Correct execution alone does not prove the operands came from the right evidence;
answer shortcuts remain a known risk. The lexical decontamination policy cannot
guarantee semantic independence. Each evaluation question has one frozen
stochastic generation; there is no inference-seed sweep. Intervals are pointwise
and not adjusted for multiple comparisons. The final set is now open and must
not be reused as an independent confirmation set for future methods.

FinReason builds on [FinQA](https://github.com/czyssrs/FinQA),
[Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B),
[ms-swift](https://github.com/modelscope/ms-swift) and
[PEFT](https://github.com/huggingface/peft). See [NOTICE.md](NOTICE.md) and
[CITATION.cff](CITATION.cff). Future data expansion is outside this release.
