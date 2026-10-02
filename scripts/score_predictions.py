"""Score a NEW saved prediction file locally; never overwrites published results."""
import argparse
from collections import Counter
import json
from pathlib import Path

from finreason.analyze_run import official_functions
from finreason.data import write_json
from finreason.executor import execute
from finreason.infer import load_prompts, sha
from finreason.official_score import score_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--labels', required=True, help='Local scoring only; never place on the inference host')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Score output exists; refusing to overwrite')
    prompts = load_prompts(args.prompts)
    predictions = [json.loads(l) for l in Path(args.predictions).read_text().splitlines()]
    labels = [json.loads(l) for l in Path(args.labels).read_text().splitlines()]
    ids = [p['id'] for p in prompts]
    if [p['id'] for p in predictions] != ids or [p['id'] for p in labels] != ids:
        raise ValueError('Prediction/label IDs, count or order differ from prompt file')
    official_path = 'third_party/finqa/evaluate.py'
    if sha(official_path) != json.loads(Path('configs/v3_data.json').read_text())['official_evaluator_sha256']:
        raise ValueError('Official evaluator changed')
    official = official_functions(official_path)
    rows = []
    for prompt, pred, label in zip(prompts, predictions, labels):
        table = json.loads(prompt['messages'][1]['content'])['table']
        safe = execute(pred['program'], table)
        scored = score_one(pred['program'], table, label['gold_answer'], official) if safe.valid else \
            {'valid': False, 'correct': False, 'error': safe.error}
        rows.append({'id': pred['id'], **scored})
    write_json(args.output, {'scope': 'new_run_local_scoring_with_bounded_DSL_precheck',
        'examples': len(rows), 'correct': sum(r['correct'] for r in rows),
        'valid': sum(r['valid'] for r in rows), 'details': rows,
        'sources_sha256': {p: sha(p) for p in (args.predictions, args.prompts, args.labels, official_path)}})
    print(json.dumps({'examples': len(rows), 'correct': sum(r['correct'] for r in rows),
                      'valid': sum(r['valid'] for r in rows)}))


if __name__ == '__main__':
    main()
