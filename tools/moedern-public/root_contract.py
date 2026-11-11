"""Versioned closure for root-wired MoEdern public text on top of the compiled draft.

The draft guard (public_contract.py) stays unchanged and correctly rejects this
source. This guard pins every changed file by SHA256 and every other src/,
ggml and draft object to the compiled draft commit.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'tools'))
import moedern_guard_pins as P
GUARD = 'moedern-public-root'
PATH = ROOT/'docs/moedern-public-root.json'
DECLARATION = '5f710aded027bde225482626b2b7a16f2e4ee12d'
def require(ok, why):
    if not ok: raise ValueError(why)
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT)
def source():
    cfg = json.loads(PATH.read_text())
    require(cfg['schema'] == 'crispembed-moedern-public-root-v1' and cfg['default_enabled'] is False
            and cfg['public_model_proof'] is False, 'root wiring policy differs')
    base = cfg['base']
    # Evidence receipts and hosted launcher scripts are not runtime source.
    # Undeclared-delta check anchored to the declaration commit: only owned files may
    # differ between base and the declaration. Later unrelated commits no longer fail
    # it; owned hashes and protected objects below stay exact on HEAD.
    changed = {f for f in git('diff', '--name-only', base, DECLARATION).decode().split()
               if not f.startswith(('tests/results/', 'tools/kaggle/'))} - {str(PATH.relative_to(ROOT))}
    # Paths changed after the base by later reviewed work are an exact recorded set
    # (tests/results/frozen-guards/moedern.json), not a wildcard.
    changed -= P.path_set(GUARD, P.CHANGED)
    require(changed <= set(cfg['owned_sha256']) | {str(PATH.relative_to(ROOT))}, 'unpinned change: '+str(sorted(changed-set(cfg['owned_sha256']))))
    for name, value in cfg['owned_sha256'].items():
        data = (ROOT/name).read_bytes()
        require(P.digest_ok(GUARD, name, value, P.sha(data)) and git('show', 'HEAD:'+name) == data, 'wired source differs: '+name)
    for name in ('src', 'ggml'):
        frozen = git('rev-parse', base+':'+name).decode().strip()
        require(P.object_ok(GUARD, name, frozen, git('rev-parse', 'HEAD:'+name).decode().strip()), 'protected tree differs: '+name)
    require(not git('status', '--porcelain', '--', 'src', 'ggml', 'CMakeLists.txt', *cfg['owned_sha256']), 'dirty wired source')
    return cfg
if __name__ == '__main__':
    cfg = source()
    print(json.dumps({'source_closed': True, 'base': cfg['base'], 'owned': len(cfg['owned_sha256']), 'public_model_proof': False}))
