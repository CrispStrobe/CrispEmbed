"""Separately versioned Q4_K closure. Historical source guards are unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT/'docs/moedern-q4k-padded.json'
RUNTIME_OWNED = {'src/moedern_q4k_profile.h', 'src/moedern_family_engine.cpp',
                 'src/moedern_family_engine.h', 'src/moedern_text.cpp',
                 'src/moedern_text_frontend.cpp', 'src/moedern_text_frontend.h'}
UNCHANGED_REQUIRED = {
    'CMakeLists.txt','.gitmodules','tools/quantize.cpp','tools/pack_moedern_text_f32.py',
    'tools/moedern-colbert/contract.py','tools/moedern-colbert/quantize_gguf.py',
    'tools/moedern-native-v2/replay.py','tools/moedern-native-v2/contract_v2.py',
    'tools/moedern-native-v2/diff_adapter.py','tools/ci-heavy/moedern_public_consumers.py',
    'tests/moedern-text/diff.cpp','tests/moedern-public/consumer.c',
    'docs/moedern-colbert-source-contract.json','docs/moedern-colbert-native-contract-v2.json',
    'tests/fixtures/moedern-colbert/panel-v1.json',
    'tests/results/moedern-colbert/publication-1p5m.json',
    'tests/results/moedern-colbert/publication-27m.json',
    'tests/results/moedern-colbert/publication-115m.json',
    'tests/results/moedern-colbert/original-27m.json',
    'tests/results/moedern-colbert/original-115m.json',
}


def require(ok, why):
    if not ok: raise ValueError(why)


def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT)
def sha(data): return hashlib.sha256(data).hexdigest()


def validate(cfg):
    require(cfg['schema'] == 'crispembed-moedern-q4k-padded-source-v1' and
            cfg['default_enabled'] is False and cfg['publication_qualified'] is False and
            cfg['retrieval_quality']['reviewed'] is False and
            cfg['gates'] == {'max_absolute': 0.002, 'min_row_cosine': 0.99999, 'maxsim_absolute': 0.002},
            'Q4_K source policy differs')
    base_runtime = set(git('ls-tree', '-r', '--name-only', cfg['base'], 'src').decode().splitlines())
    require(set(cfg['runtime_owned']) == RUNTIME_OWNED and
            base_runtime - RUNTIME_OWNED <= set(cfg['unchanged_sha256']) and
            UNCHANGED_REQUIRED <= set(cfg['unchanged_sha256']) and
            RUNTIME_OWNED <= set(cfg['owned_sha256']), 'complete production closure required')
    require(not (set(cfg['owned_sha256']) & set(cfg['unchanged_sha256'])), 'overlapping closure declarations')
    return cfg


def source():
    cfg = validate(json.loads(PATH.read_text()))
    require(git('show', 'HEAD:'+str(PATH.relative_to(ROOT))) == PATH.read_bytes(), 'dirty source contract')
    owned = set(cfg['owned_sha256']) | {str(PATH.relative_to(ROOT))}
    changed = set(git('diff', '--name-only', cfg['base'], 'HEAD').decode().splitlines())
    require(changed <= owned, 'undeclared Q4_K source delta: '+str(sorted(changed-owned)))
    for group in ('owned_sha256','unchanged_sha256'):
        for name, pin in cfg[group].items():
            path = ROOT/name
            require(path.is_file() and not path.is_symlink() and sha(path.read_bytes()) == pin and
                    git('show', 'HEAD:'+name) == path.read_bytes(), 'source differs: '+name)
            if group == 'unchanged_sha256':
                require(git('show', cfg['base']+':'+name) == path.read_bytes(), 'protected base differs: '+name)
    require(git('rev-parse','HEAD:ggml').decode().strip() == cfg['ggml'], 'ggml source differs')
    require(not git('status','--porcelain','--untracked-files=no','--','src','ggml','CMakeLists.txt','tools/quantize.cpp'),
            'dirty production source')
    return cfg


if __name__ == '__main__': print(json.dumps(source(), indent=2))
