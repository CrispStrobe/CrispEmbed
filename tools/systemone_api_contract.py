#!/usr/bin/env python3
"""Closed source profile for the initial, model-free SystemOne public surface."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'docs/systemone-api-profile.json'
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT)
def verify():
    cfg = json.loads(PATH.read_text())
    if cfg['schema'] != 'crispembed-systemone-public-api-v1' or cfg['default_enabled'] or cfg['model_qualified']:
        raise ValueError('initial public API policy differs')
    old = set(git('ls-tree', '-r', '--name-only', cfg['base'], 'src').decode().splitlines())
    current = set(git('ls-tree', '-r', '--name-only', 'HEAD', 'src').decode().splitlines())
    if current != old | {'src/systemone_api.cpp'} or set(cfg['protected_runtime']) != old-{'src/crispembed.h'}:
        raise ValueError('incomplete runtime source closure')
    for name, value in cfg['protected_runtime'].items():
        if git('rev-parse', 'HEAD:'+name).decode().strip() != value or git('rev-parse', cfg['base']+':'+name).decode().strip() != value:
            raise ValueError('frozen inference source differs: '+name)
    for name, value in cfg['owned_sha256'].items():
        data = (ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != value or git('show', 'HEAD:'+name) != data:
            raise ValueError('public source differs: '+name)
    if git('ls-tree', 'HEAD', 'ggml').decode().split()[2] != cfg['ggml']:
        raise ValueError('ggml gitlink differs')
    if git('status', '--porcelain', '--', *cfg['owned_sha256'], *cfg['protected_runtime'], str(PATH.relative_to(ROOT))):
        raise ValueError('dirty public source')
    return cfg
if __name__ == '__main__':
    cfg = verify()
    print(json.dumps({'completed':True, 'protected_runtime_files':len(cfg['protected_runtime']),
                      'owned_files':len(cfg['owned_sha256']), 'model_qualified':False}))
