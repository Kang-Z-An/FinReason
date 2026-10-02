"""Unmodified v3 paired-count and document bootstrap functions."""
from collections import defaultdict
import random

def doc_key(item_id):
    parts = item_id.split('/')
    if len(parts) != 3 or not parts[1].isdigit():
        raise ValueError('Unexpected FinQA ID: ' + item_id)
    return '/'.join(parts[:2])

def paired_counts(details, left, right):
    return {metric: {
        'left_only': sum(r[left][metric] and not r[right][metric] for r in details),
        'right_only': sum(r[right][metric] and not r[left][metric] for r in details),
        'net': sum(int(r[left][metric])-int(r[right][metric]) for r in details),
        'delta_percentage_points': 100*sum(
            int(r[left][metric])-int(r[right][metric]) for r in details)/len(details),
    } for metric in ('correct', 'valid')}


def percentile(values, q):
    values = sorted(values)
    x = (len(values)-1)*q
    lo = int(x)
    hi = min(lo+1, len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(x-lo)


def cluster_intervals(details, models, comparisons, replicates, seed):
    """Sample documents, retaining all questions; use the same draws for every metric."""
    grouped = defaultdict(list)
    for row in details:
        grouped[doc_key(row['id'])].append(row)
    groups = [grouped[k] for k in sorted(grouped)]
    keys = [(tag, metric) for tag in models for metric in ('correct', 'valid')]
    totals = [[sum(int(r[tag][metric]) for r in group) for tag, metric in keys]
              for group in groups]
    sizes = [len(group) for group in groups]
    index = {key: i for i, key in enumerate(keys)}
    distributions = {key: [] for key in keys}
    diffs = {(name, metric): [] for name in comparisons for metric in ('correct', 'valid')}
    rng = random.Random(seed)
    for _ in range(replicates):
        selected = [rng.randrange(len(groups)) for _ in groups]
        denominator = sum(sizes[i] for i in selected)
        values = [100*sum(totals[i][j] for i in selected)/denominator for j in range(len(keys))]
        for key in keys:
            distributions[key].append(values[index[key]])
        for name, (left, right) in comparisons.items():
            for metric in ('correct', 'valid'):
                diffs[name, metric].append(values[index[left, metric]]-values[index[right, metric]])
    def interval(values):
        return [percentile(values, .025), percentile(values, .975)]
    return {'documents': len(groups), 'replicates': replicates, 'seed': seed,
            'interpretation': 'pointwise_95_percentile_CI_not_multiplicity_adjusted',
            'rates_percent': {tag: {m: interval(distributions[tag, m]) for m in ('correct', 'valid')}
                              for tag in models},
            'paired_delta_percentage_points': {
                name: {m: interval(diffs[name, m]) for m in ('correct', 'valid')}
                for name in comparisons}}
