"""Hash-checked adapter assets; no framework imports or model loading."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def install_adapter(record, destination, archive=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Destination exists; use a fresh directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as work:
        work = Path(work)
        source = Path(archive) if archive else work / 'asset.zip'
        if archive is None:
            with urllib.request.urlopen(record['url'], timeout=60) as response, source.open('wb') as out:
                shutil.copyfileobj(response, out)
        if source.stat().st_size != record['bytes'] or file_sha(source) != record['sha256']:
            raise ValueError('Adapter archive size or SHA-256 mismatch')
        output = work / 'adapter'
        output.mkdir()
        with zipfile.ZipFile(source) as z:
            members = z.infolist()
            if len(members) != len(record['files']) or {m.filename for m in members} != set(record['files']):
                raise ValueError('Unexpected or duplicate archive member')
            for member in members:
                if Path(member.filename).name != member.filename or member.file_size != record['files'][member.filename]['bytes']:
                    raise ValueError('Unsafe archive member or changed member size')
                # Never extract paths or symlinks; create a regular file ourselves.
                with z.open(member) as src, (output / member.filename).open('wb') as out:
                    shutil.copyfileobj(src, out)
        for name, info in record['files'].items():
            if file_sha(output / name) != info['sha256']:
                raise ValueError('Adapter member SHA-256 mismatch: ' + name)
        output.rename(destination)
    return {'state': 'passed', 'files_verified': len(record['files']),
            'adapter_sha256': record['files']['adapter_model.safetensors']['sha256']}


def verify_adapter(folder, manifest, tag=None):
    folder = Path(folder)
    weight_sha = file_sha(folder / 'adapter_model.safetensors')
    config_sha = file_sha(folder / 'adapter_config.json')
    tags = [tag] if tag else [key for key, value in manifest['adapters'].items()
                             if value['files']['adapter_model.safetensors']['sha256'] == weight_sha]
    if tag or tags:
        if len(tags) != 1 or tags[0] not in manifest['adapters']:
            raise ValueError('Unknown or ambiguous released adapter tag')
        record = manifest['adapters'][tags[0]]
        for name, info in record['files'].items():
            if file_sha(folder / name) != info['sha256']:
                raise ValueError('Released adapter changed: ' + name)
        tag = tags[0]
    config = json.loads((folder / 'adapter_config.json').read_text())
    if config.get('peft_type') != 'LORA' or config.get('task_type') != 'CAUSAL_LM':
        raise ValueError('Expected causal-language-model LoRA adapter')
    return {'tag': tag, 'adapter_sha256': weight_sha, 'adapter_config_sha256': config_sha,
            'released_asset_verified': tag is not None}
