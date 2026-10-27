"""Separately versioned closure for MoEdern 27M/115M public raw-text dispatch.

The 1.5M root closure (root_contract.py + docs/moedern-public-root.json) is not
edited. This guard pins every file changed since the family base by SHA256,
keeps the 1.5M engine, family engine and ggml byte-identical to the base,
requires every 1.5M root pin to still hold except the explicitly superseded
files, and checks that the 1.5M packer constants are unchanged.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT/'docs/moedern-family-public.json'
def require(ok, why):
    if not ok: raise ValueError(why)
def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT)
def sha(data): return hashlib.sha256(data).hexdigest()
def packer_constants():
    spec = importlib.util.spec_from_file_location('moedern_family_packer_guard', ROOT/'tools/pack_moedern_text_f32.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return dict(MODEL=m.MODEL, TOKENIZER=m.TOKENIZER, COMPOSED=m.COMPOSED, BOUNDS=m.BOUNDS)
def source():
    cfg = json.loads(PATH.read_text())
    require(cfg['schema'] == 'crispembed-moedern-family-public-v1' and cfg['default_enabled'] is False
            and cfg['gate'] == 'CRISPEMBED_MOEDERN_FAMILY' and cfg['public_model_proof'] is False, 'family policy differs')
    base = cfg['base']
    changed = set(git('diff', '--name-only', base, 'HEAD').decode().split())
    require(changed <= set(cfg['owned_sha256']) | {str(PATH.relative_to(ROOT))}, 'unpinned change: '+str(sorted(changed-set(cfg['owned_sha256']))))
    for name, value in cfg['owned_sha256'].items():
        data = (ROOT/name).read_bytes()
        require(sha(data) == value and git('show', 'HEAD:'+name) == data, 'family source differs: '+name)
    for name in cfg['unchanged_from_base']:
        require(git('rev-parse', 'HEAD:'+name) == git('rev-parse', base+':'+name), 'protected object differs: '+name)
    root = json.loads((ROOT/'docs/moedern-public-root.json').read_text())
    require(sha((ROOT/'docs/moedern-public-root.json').read_bytes()) == cfg['root_closure_sha256'], '1.5M root closure edited')
    for name, value in root['owned_sha256'].items():
        if name in cfg['supersedes_root_pins'] or name in cfg['root_drift_before_base']: continue
        require(sha((ROOT/name).read_bytes()) == value, '1.5M root pin differs: '+name)
    require(set(cfg['supersedes_root_pins']) <= set(cfg['owned_sha256']), 'superseded pin not re-pinned')
    require(packer_constants() == cfg['packer_one_five_constants'], '1.5M packer constants differ')
    require(not git('status', '--porcelain', '--', 'src', 'ggml', 'CMakeLists.txt', *cfg['owned_sha256']), 'dirty family source')
    return cfg
if __name__ == '__main__':
    cfg = source()
    print(json.dumps({'source_closed': True, 'base': cfg['base'], 'owned': len(cfg['owned_sha256']), 'public_model_proof': False}))
