"""Verify release file hashes and exclude credentials, private paths and weights."""
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    manifest = json.loads(Path('release_manifest.json').read_text())
    tracked = subprocess.run(['git', 'ls-files'], text=True, capture_output=True, check=True).stdout.splitlines()
    # Before git initialization, the author verifies manifest entries directly.
    paths = tracked or list(manifest['files']) + ['release_manifest.json']
    for name, expected in manifest['files'].items():
        if sha(name) != expected:
            raise ValueError('Release file changed: ' + name)
    findings = []
    patterns = [r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',
                r'\bgh[pousr]_[A-Za-z0-9]{30,}\b', r'\bgithub_pat_[A-Za-z0-9_]{30,}\b',
                r'\bhf_[A-Za-z0-9]{25,}\b', r'/Users/[^\s"\']+',
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b']
    for name in paths:
        path = Path(name)
        if path.suffix in ('.safetensors', '.pt', '.bin', '.pem', '.key') or \
           name.startswith(('data/', 'models/', 'outputs/')) or '.env' in path.name:
            findings.append(name + ': forbidden artifact')
            continue
        text = path.read_text()
        # Checker source necessarily contains its own detection expressions.
        if name == 'scripts/check_release.py':
            continue
        for pattern in patterns:
            if re.search(pattern, text):
                findings.append(name + ': private identifier/credential pattern')
    if findings:
        raise ValueError('Release scan failed: ' + '; '.join(findings))
    return {'state': 'passed', 'hashes_verified': len(manifest['files']), 'scanned_files': len(paths)}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
