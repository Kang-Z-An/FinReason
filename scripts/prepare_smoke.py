"""Make eight unconditioned training-source prompts; labels stay on this machine."""
import argparse
import hashlib
import json
from pathlib import Path

from finreason.data import write_json, write_jsonl
from finreason.executor import canonical_program
from finreason.infer import sha
from finreason.statistics import doc_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-training-file', default='data/raw/finqa/train.json')
    parser.add_argument('--output', default='data/smoke_v1')
    args = parser.parse_args()
    contract = json.loads(Path('configs/v3_data.json').read_text())
    if sha(args.source_training_file) != contract['source_train_sha256']:
        raise ValueError('Source hash mismatch')
    system_path = Path('prompts/finqa_v2.txt')
    if sha(system_path) != contract['prompt_sha256']:
        raise ValueError('System prompt hash mismatch')
    ids = sorted(contract['sft_ids'], key=lambda i: hashlib.sha256(('public-smoke-v1:' + i).encode()).hexdigest())[:8]
    held = contract['splits']['dev']['ids'] + contract['splits']['final']['ids']
    if set(ids) & set(held) or {doc_key(i) for i in ids} & {doc_key(i) for i in held}:
        raise ValueError('Smoke IDs overlap held-out IDs or documents')
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Smoke output exists; use a fresh directory')
    by_id = {r['id']: r for r in json.loads(Path(args.source_training_file).read_text())}
    prompts, labels = [], []
    for i in ids:
        r = by_id[i]
        context = {k: r[k] for k in ('pre_text', 'table', 'post_text')}
        context['question'] = r['qa']['question']
        prompts.append({'id': i, 'messages': [{'role': 'system', 'content': system_path.read_text().strip()},
            {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}]})
        labels.append({'id': i, 'gold_program': canonical_program(r['qa']['program']), 'gold_answer': r['qa']['exe_ans']})
    write_jsonl(destination / 'train8.prompts.jsonl', prompts)
    write_jsonl(destination / 'train8.labels.jsonl', labels)
    write_json(destination / 'selection.json', dict(scope='training_source_pipeline_smoke_only', ids=ids,
        selection='first_8_by_sha256_public_smoke_v1_id_no_outcome_filter',
        source_sha256=sha(args.source_training_file), prompts_sha256=sha(destination / 'train8.prompts.jsonl'),
        labels_local_only=True, heldout_document_overlap=0, quality_estimate=False))
    print(json.dumps({'state': 'passed', 'examples': 8, 'quality_estimate': False}))


if __name__ == '__main__':
    main()
