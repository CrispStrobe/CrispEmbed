"""New closed four-case storage experiment; no mutation of old A/B/C/gold."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'docs/josev2b-protected-recipes-v1.json'
from josev2b_q4k_contract import require, REFERENCE_SHA, FREEZE_SHA
from quantize_josev2b_protected import COUNTS, SOURCE, recipe, schema


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(4 * 2**20):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def plans():
    rows = schema()
    return {name: dict(counts={str(t): c for t, c in counts.items()},
                       inventory_sha256=hashlib.sha256(json.dumps(
                           recipe(rows, name), sort_keys=True, separators=(',', ':')).encode()).hexdigest())
            for name, counts in COUNTS.items()}


def validate(cfg):
    require(cfg['schema'] == 'crispembed-josev-protected-recipes-v1' and
            all(cfg[k] is False for k in ('default_enabled', 'publication_qualified',
                                          'holdout_access', 'recipe_selected')), 'development scope differs')
    require(cfg['runtime_source'] == '6781befb66b5f65afd47c1b921ac9cd1cb8e568a' and
            git('rev-parse', cfg['base'] + ':src') == git('rev-parse', cfg['runtime_source'] + ':src'),
            'reviewed Q8 runtime differs')
    require(cfg['source_model'] == SOURCE and cfg['recipes'] == plans(), 'immutable role recipes differ')
    require(cfg['gates'] == dict(max_absolute=.002, min_row_cosine=.99999, logits_absolute=.002,
                                 probabilities_absolute=.0001, ordinal_expectation_absolute=.001),
            'historical budgets changed')
    require(cfg['original_reference_sha256'] == REFERENCE_SHA and cfg['original_freeze_sha256'] == FREEZE_SHA,
            'original identities differ')
    previous = json.loads((ROOT / 'docs/josev2b-mixed-recipes-v1.json').read_text())
    protected = set(previous['protected_sha256']) | set(previous['owned_sha256']) | {
        'docs/josev2b-mixed-recipes-v1.json', 'tools/gguf_file_type_canonicalize.py',
        'tools/quantize.cpp', 'docs/portfolio-q8-profile-controls-v1.json'}
    protected |= set(git('ls-tree', '-r', '--name-only', cfg['base'], 'src', 'tests/josev-q4k',
                         'tests/josev-packed').decode().splitlines())
    require(protected <= set(cfg['protected_sha256']), 'complete previous/runtime/reference closure required')
    require(not any(p.startswith('src/') for p in cfg['owned_sha256']), 'no runtime edits in storage caller')
    require(cfg['original_artifact'] == previous['original_artifact'], 'original archive differs')
    return cfg


def verify():
    cfg = validate(json.loads(PATH.read_text()))
    rel = str(PATH.relative_to(ROOT))
    require(git('show', 'HEAD:' + rel) == PATH.read_bytes(), 'dirty new contract')
    require(set(git('diff', '--name-only', cfg['base'], 'HEAD').decode().splitlines()) <=
            set(cfg['owned_sha256']) | {rel}, 'undeclared source delta')
    require(git('ls-tree', '-r', 'HEAD', 'src') == git('ls-tree', '-r', cfg['base'], 'src'),
            'runtime inventory differs')
    for group in ('protected_sha256', 'owned_sha256'):
        for name, pin in cfg[group].items():
            p = ROOT / name
            require(p.is_file() and not p.is_symlink() and sha(p) == pin and
                    git('show', 'HEAD:' + name) == p.read_bytes(), 'source differs: ' + name)
            if group == 'protected_sha256':
                require(git('show', cfg['base'] + ':' + name) == p.read_bytes(),
                        'protected source changed: ' + name)
    require(git('rev-parse', 'HEAD:ggml').decode().strip() == cfg['ggml'], 'backend gitlink differs')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'dirty tracked source')
    return cfg


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
