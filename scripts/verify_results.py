"""Verify released counts and CIs; optionally rescore programs against FinQA train."""
import argparse
from collections import Counter
import json
from pathlib import Path

from finreason.analyze_run import official_functions
from finreason.infer import sha
from finreason.official_score import score_one
from finreason.statistics import cluster_intervals, paired_counts


def verify(source_training_file=None, with_intervals=True):
    root = Path('results/v3')
    summary = json.loads((root / 'summary.json').read_text())
    rows = [json.loads(l) for l in (root / 'per_question.jsonl').read_text().splitlines()]
    contract = json.loads(Path('configs/v3_data.json').read_text())
    ids = contract['splits']['final']['ids']
    if [r['id'] for r in rows] != ids or len(set(ids)) != 551:
        raise ValueError('Final IDs or order changed')
    tags = list(summary['summary'])
    for tag, totals in summary['summary'].items():
        for field, metric in [('official_correct', 'correct'), ('official_valid', 'valid')]:
            if sum(r[tag][metric] for r in rows) != totals[field]:
                raise ValueError('Aggregate differs: ' + tag)
    comparisons = {t + '_vs_sft': (t, 'sft') for t in tags if t != 'sft'}
    comparisons.update({f'validity_vs_binary_seed{s}': (f'full1000_validity_0p1_seed{s}',
                         f'full1000_binary_seed{s}') for s in (42, 1234)})
    for name, pair in comparisons.items():
        if paired_counts(rows, *pair) != summary['paired'][name]:
            raise ValueError('Paired counts differ: ' + name)
    if with_intervals:
        prior = summary['uncertainty']
        actual = cluster_intervals(rows, tags, comparisons, prior['replicates'], prior['seed'])
        if actual != prior:
            raise ValueError('Document bootstrap differs')
    rescored = 0
    if source_training_file:
        if sha(source_training_file) != contract['source_train_sha256']:
            raise ValueError('FinQA source hash changed')
        source = {r['id']: r for r in json.loads(Path(source_training_file).read_text())}
        official = official_functions('third_party/finqa/evaluate.py')
        for row in rows:
            gold = source[row['id']]
            for tag in tags:
                actual = score_one(row[tag]['program'], gold['table'], gold['qa']['exe_ans'], official)
                if any(actual[k] != row[tag][k] for k in ('correct', 'valid', 'error')):
                    raise ValueError('Official rescore differs: ' + row['id'] + '/' + tag)
                rescored += 1
    return {'state': 'passed', 'examples': len(rows), 'models': len(tags),
            'paired_comparisons': len(comparisons), 'bootstrap_checked': with_intervals,
            'official_predictions_rescored': rescored, 'GPU_used': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-training-file')
    parser.add_argument('--skip-bootstrap', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.source_training_file, not args.skip_bootstrap), indent=2))


if __name__ == '__main__':
    main()
