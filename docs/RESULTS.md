# v3 result interpretation

The final evaluation finished on 2026-10-01. Initial publication is dated
2026-10-02. Five fixed checkpoints produced 551 predictions each with no length
stops. All are reported; neither a reward nor a seed was chosen from final scores.

`results/v3/summary.json` stores the full counts, six paired comparisons and
document-bootstrap intervals. `per_question.jsonl` contains the 551 ordered
questions with all five predicted programs and the original execution flags.
`provenance.json` records checkpoint hashes, training-update checks and inference
versions. No financial text or gold answer field is redistributed.

Binary GRPO gains 9 and 10 net correct questions over the same SFT checkpoint.
The corresponding new-correct/lost-correct counts are 15/6 and 14/4. The small
validity reward has no consistent paired benefit. A larger final correct count
for one reward/seed is not a reason to claim it is the best method.

Bootstrap samples 91 company/year groups with replacement, keeping every
question in each drawn group. The rate is question-weighted, not an unweighted
average of document accuracies. Every comparison uses the same 10,000 draws
(seed 20261001), with pointwise percentile 95% intervals. No multiplicity
adjustment or power claim is made.

The splits are within original FinQA train: 5,266 training candidates, 407 dev
questions and 551 final questions. The final split is prospective for v3, not
historically untouched across the entire earlier project. It is not the official
FinQA test or leaderboard evaluation. The public ID manifest preserves the
exact assignments; release-time reconstruction does not select new splits.

Two RL seeds conditional on one SFT checkpoint do not establish end-to-end
multi-seed stability. One stochastic generation per item does not establish
inference-seed stability. Exact/near lexical filtering cannot prove absence of
all semantic duplicates. Execution scoring can reward answer shortcuts and
does not fully validate operand provenance. The opened final set must be treated
as regression data in later experiments, not as a new independent test.
