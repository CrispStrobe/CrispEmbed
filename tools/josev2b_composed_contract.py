#!/usr/bin/env python3
"""Stdlib-only composition closure. No model, tokenizer or framework access."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def verify():
    cfg = json.loads((ROOT / 'docs/josev2b-composed-source.json').read_text())
    if cfg['default_enabled'] or cfg['full_model_qualified']:
        raise ValueError('unqualified default/promotion')
    old = set(git('ls-tree', '-r', '--name-only', cfg['base'], 'src').decode().splitlines())
    if set(cfg['protected']) != old - set(cfg['changed_existing']):
        raise ValueError('old source closure omitted')
    current = set(git('ls-tree', '-r', '--name-only', 'HEAD', 'src').decode().splitlines())
    if current != old | set(cfg['new_runtime']):
        raise ValueError('unexpected native source delta')
    for name, source in cfg['changed_existing'].items():
        if git('show', 'HEAD:' + name) != git('show', source + ':' + name):
            raise ValueError('shared decoder differs: ' + name)
    for name, value in cfg['protected'].items():
        if git('rev-parse', 'HEAD:' + name).decode().strip() != value:
            raise ValueError('old source changed: ' + name)
    for name in ('CMakeLists.txt', 'ggml'):
        if git('rev-parse', 'HEAD:' + name).decode().strip() != cfg['protected_roots'][name]:
            raise ValueError('root build/backend changed: ' + name)
    for name, digest in cfg['owned_sha256'].items():
        actual = (ROOT / name).read_bytes()
        if hashlib.sha256(actual).hexdigest() != digest or git('show', 'HEAD:' + name) != actual:
            raise ValueError('owned source differs: ' + name)
    paths = list(cfg['protected']) + list(cfg['owned_sha256']) + ['docs/josev2b-composed-source.json']
    if git('status', '--porcelain', '--', *paths):
        raise ValueError('dirty protected source')
    return cfg

if __name__ == '__main__':
    cfg = verify()
    print(json.dumps({'completed': True, 'old_sources': len(cfg['protected']),
                      'owned_sources': len(cfg['owned_sha256']), 'model_payloads_accessed': False}))
