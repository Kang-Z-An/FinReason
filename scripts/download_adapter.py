"""Download one pinned v3 LoRA asset, or verify/install an already downloaded ZIP."""
import argparse
import json
from pathlib import Path

from finreason.adapter_assets import install_adapter


def main():
    manifest = json.loads(Path('configs/adapter_assets.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True, choices=list(manifest['adapters']))
    parser.add_argument('--output', required=True)
    parser.add_argument('--archive', help='Existing asset ZIP; avoids network, but retains all hash checks')
    args = parser.parse_args()
    result = install_adapter(manifest['adapters'][args.tag], args.output, args.archive)
    print(json.dumps(dict(result, tag=args.tag, model_loaded=False)))


if __name__ == '__main__':
    main()
