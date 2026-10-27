"""Versioned closure for root-wired MoEdern public text on top of the compiled draft.

The draft guard (public_contract.py) stays unchanged and correctly rejects this
source. This guard pins every changed file by SHA256 and every other src/,
ggml and draft object to the compiled draft commit.
"""
from pathlib import Path
import hashlib
import json
import subprocess
ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT/'docs/moedern-public-root.json'
def require(ok, why):
    if not ok: raise ValueError(why)
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT)
def source():
    cfg = json.loads(PATH.read_text())
    require(cfg['schema'] == 'crispembed-moedern-public-root-v1' and cfg['default_enabled'] is False
            and cfg['public_model_proof'] is False, 'root wiring policy differs')
    base = cfg['base']
    changed = set(git('diff', '--name-only', base, 'HEAD').decode().split())
    require(changed <= set(cfg['owned_sha256']) | {str(PATH.relative_to(ROOT))}, 'unpinned change: '+str(sorted(changed-set(cfg['owned_sha256']))))
    for name, value in cfg['owned_sha256'].items():
        data = (ROOT/name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == value and git('show', 'HEAD:'+name) == data, 'wired source differs: '+name)
    for name in ('src', 'ggml'):
        require(git('rev-parse', 'HEAD:'+name) == git('rev-parse', base+':'+name), 'protected tree differs: '+name)
    require(not git('status', '--porcelain', '--', 'src', 'ggml', 'CMakeLists.txt', *cfg['owned_sha256']), 'dirty wired source')
    return cfg
if __name__ == '__main__':
    cfg = source()
    print(json.dumps({'source_closed': True, 'base': cfg['base'], 'owned': len(cfg['owned_sha256']), 'public_model_proof': False}))
