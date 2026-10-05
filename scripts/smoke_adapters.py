"""Run all five pinned v3 adapters on prompt-only smoke inputs on a GPU host."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from finreason.adapter_assets import install_adapter, verify_adapter
from finreason.infer import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--prompts', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--asset-dir', help='Offline ZIP directory; otherwise download published assets')
    args = parser.parse_args()
    manifest = json.loads(Path('configs/adapter_assets.json').read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for tag, record in manifest['adapters'].items():
        adapter = root / 'adapters' / tag
        if adapter.exists():
            verify_adapter(adapter, manifest, tag)
        else:
            archive = Path(args.asset_dir) / Path(record['url']).name if args.asset_dir else None
            install_adapter(record, adapter, archive)
        cmd = [sys.executable, '-m', 'scripts.infer_adapter', '--model-dir', args.model_dir,
               '--adapter', str(adapter), '--released-tag', tag, '--prompts', args.prompts,
               '--output', str(root / 'predictions' / tag)]
        print('Starting', tag, flush=True)
        subprocess.run(cmd, check=True)
        if not json.loads((root / 'predictions' / tag / 'progress.json').read_text())['complete_dataset']:
            raise RuntimeError('Smoke predictions incomplete: ' + tag)
    atomic_json(root / 'queue.json', dict(state='complete', models=list(manifest['adapters']),
        wall_seconds=time.perf_counter() - start, labels_accessed=False,
        quality_estimate=False, public_download_used=args.asset_dir is None))
    print('All adapter smoke runs complete', flush=True)


if __name__ == '__main__':
    main()
