"""Reconstruct exact frozen v3 files from upstream train and public ordered IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from finreason.data import write_json, write_jsonl
from finreason.executor import canonical_program
from finreason.infer import sha
from finreason.statistics import doc_key


def reconstruct(source, destination):
    contract = json.loads(Path('configs/v3_data.json').read_text())
    if sha(source) != contract['source_train_sha256']:
        raise ValueError('Pinned FinQA train source changed')
    if sha('prompts/finqa_v2.txt') != contract['prompt_sha256']:
        raise ValueError('Frozen system prompt changed')
    raw = json.loads(Path(source).read_text())
    by_id = {r['id']: r for r in raw}
    if len(by_id) != len(raw) or len(raw) != 6251:
        raise ValueError('Source count or IDs changed')
    split = contract['splits']
    for a, b in [('train', 'dev'), ('train', 'final'), ('dev', 'final')]:
        if set(split[a]['ids']) & set(split[b]['ids']) or \
           {doc_key(i) for i in split[a]['ids']} & {doc_key(i) for i in split[b]['ids']}:
            raise ValueError('Split overlaps')
    system = Path('prompts/finqa_v2.txt').read_text().strip()

    def prompt(item_id):
        row = by_id[item_id]
        context = {k: row[k] for k in ('pre_text', 'table', 'post_text')}
        context['question'] = row['qa']['question']
        return {'id': item_id, 'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}]}

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Destination exists; verify existing files instead of overwriting')
    destination.mkdir(parents=True)
    for part in ('dev', 'final'):
        prefix = destination / 'data/processed/rlvr_v3/split_v1'
        ids = split[part]['ids']
        write_jsonl(prefix / (part + '.prompts.jsonl'), [prompt(i) for i in ids])
        write_jsonl(prefix / (part + '.labels.jsonl'), [{'id': i,
            'gold_program': canonical_program(by_id[i]['qa']['program']),
            'gold_answer': by_id[i]['qa']['exe_ans']} for i in ids])
    prefix = destination / 'data/processed/rlvr_v3/method_v1'
    training_ids = set(split['train']['ids'])
    if not set(contract['sft_ids'] + contract['grpo_ids']) <= training_ids:
        raise ValueError('Training IDs escape the training pool')
    write_jsonl(prefix / 'train.3500.sft.jsonl', [{'messages': prompt(i)['messages'] + [
        {'role': 'assistant', 'content': canonical_program(by_id[i]['qa']['program'])}]}
        for i in contract['sft_ids']])
    write_jsonl(prefix / 'grpo.main1000.prompts.jsonl', [prompt(i) for i in contract['grpo_ids']])
    write_json(prefix / 'grpo.main1000.reward_metadata.json', {i: {'table': by_id[i]['table'],
        'gold_answer': by_id[i]['qa']['exe_ans']} for i in contract['grpo_ids']})
    shutil.copyfile('configs/v3_grpo_prompt_lengths.json', prefix / 'grpo.main1000.prompt_lengths.json')
    for relative, expected in contract['expected_outputs'].items():
        if sha(destination / relative) != expected:
            raise ValueError('Reconstructed file hash differs: ' + relative)
    write_json(destination / 'verification.json', {'state': 'passed',
        'files_verified': len(contract['expected_outputs']), 'source_sha256': sha(source),
        'evaluation_labels_local_only': True, 'models_loaded': False})
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-training-file', help='Use a local train.json; no download')
    parser.add_argument('--output', default='data/reconstructed_v3')
    args = parser.parse_args()
    contract = json.loads(Path('configs/v3_data.json').read_text())
    source = Path(args.source_training_file or 'data/raw/finqa/train.json')
    if not args.source_training_file and not source.exists():
        url = f'https://raw.githubusercontent.com/czyssrs/FinQA/{contract["finqa_revision"]}/dataset/train.json'
        source.parent.mkdir(parents=True, exist_ok=True)
        temporary = source.with_suffix('.json.part')
        urllib.request.urlretrieve(url, temporary)
        if sha(temporary) != contract['source_train_sha256']:
            raise ValueError('Downloaded source hash mismatch')
        temporary.replace(source)
    result = reconstruct(source, args.output)
    print(json.dumps({'state': 'passed', 'output': str(result), 'GPU_used': False}))


if __name__ == '__main__':
    main()
