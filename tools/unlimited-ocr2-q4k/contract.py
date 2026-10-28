"""Separate true-Q4_K experiment; historical precision/reference workers stay frozen."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = '83eac856cc684808e417ebfed7db7c2f677dc09f'
CASES = ['paragraphs', 'table', 'receipt', 'german']
TRIAL_SHA = '9818063ba01dcf830fcdc2acffdbbab295b50397c1c90f71f1ac009b49636996'
F16_SHA = '487b887eddaf73e070cd98c3e10d112d21749036ca06b5e7fc103c3fb4def517'
F16_BYTES = 6692677696
F16_URL = ('https://huggingface.co/cstr/Unlimited-OCR-2-GGUF/resolve/'
           'e3af0c0c4bfa241c161db64ca36f3c63d3ca8378/unlimited-ocr-2-f16.gguf')
SELECTED = re.compile(r'l\.blk\.(\d+)\.ffn_(gate|up)_exps\.weight')
RECIPES = {'last1': [11], 'last4': list(range(8, 12)), 'all11': list(range(1, 12))}
PREFIXES = ['src', 'ggml', '.gitmodules', 'VERSION', 'CMakeLists.txt', 'cmake', 'examples/cli',
            'models/convert-unlimited-ocr-to-gguf.py', 'tools/quantize.cpp',
            'tools/minicpm-ocr/graph_adapter.py', 'tools/josev2b_composed_panel.py',
            'tools/josev2b_composed_diff.py',
            'tools/kaggle/unlimited-ocr2-trial/trial_worker.py',
            'tools/kaggle/unlimited-ocr2-precision/precision_worker.py',
            'tests/results/unlimited-ocr2/trial.json']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args]).decode().strip()


def guard():
    expected = git('ls-tree', '-r', BASE, '--', *PREFIXES)
    require(git('ls-tree', '-r', 'HEAD', '--', *PREFIXES) == expected,
            'Declared production/blueprint/build closure differs from BASE')
    require(not git('diff', '--name-only', 'HEAD', '--', *PREFIXES), 'Dirty protected files')
    require(git('-C', 'ggml', 'rev-parse', 'HEAD') == git('rev-parse', BASE + ':ggml'), 'ggml gitlink differs')
    owned = ['tools/unlimited-ocr2-q4k', 'tests/unlimited-q4k',
             'tools/kaggle/unlimited-ocr2-q4k', 'docs/unlimited-ocr2-q4k-v1.json',
             'docs/unlimited-ocr2-q4k-v1.md', 'tests/test_unlimited_ocr2_q4k.py',
             'tools/unlimited-ocr2/replay_real_decode.py', 'tools/unlimited-ocr2/gen_text_fixture.py',
             'tests/unlimited-real-decode/probe.cpp', 'tests/test_unlimited_real_decode.py',
             'tools/kaggle/ocr-real-scans/receipt.py', 'docs/ocr-real-scans-v1.json',
             'tests/results/ocr-real-scans', 'tests/results/unlimited-ocr2/publication.json']
    require(not git('diff', '--name-only', 'HEAD', '--', *owned), 'Dirty audit files')
    require(not git('ls-files', '--others', '--exclude-standard', '--', *owned), 'Untracked audit source')
    files = [line.split('\t', 1)[1] for line in expected.splitlines() if line.startswith('100')]
    files += git('ls-files', '--', *owned).splitlines()
    require(bool(files), 'Missing source inventory')
    return {p: digest(ROOT / p) for p in sorted(set(files))}


def references():
    path = ROOT / 'tests/results/unlimited-ocr2/trial.json'
    require(digest(path) == TRIAL_SHA, 'Frozen four-page original differs')
    trial = json.loads(path.read_text())
    require(trial['reference']['completed'] and trial['reference']['exit'] == 0,
            'Original ID/text capture incomplete')
    records = {x['id']: x for x in trial['reference']['cases']}
    require(set(records) == set(CASES), 'Exact four original cases required')
    for row in records.values():
        row['ids'] = json.loads(row['ids']) if isinstance(row['ids'], str) else row['ids']
        require(row['ids'] and all(type(x) is int for x in row['ids']), 'Original IDs malformed')
    images = {x['id']: x['sha256'] for x in trial['pages'] if x['id'] in CASES}
    require(set(images) == set(CASES), 'Four original image hashes required')
    return records, images


def inventory(path):
    # Imported only on hosted CPU after source/reference authentication.
    from gguf import GGUFReader
    records = {}
    for tensor in GGUFReader(str(path)).tensors:
        require(tensor.name not in records, 'Duplicate tensor')
        h = hashlib.sha256()
        view = memoryview(tensor.data).cast('B')
        for at in range(0, len(view), 8 << 20):
            h.update(view[at:at + (8 << 20)])
        records[tensor.name] = dict(type=int(tensor.tensor_type), shape=[int(x) for x in tensor.shape],
                                    bytes=len(view), sha256=h.hexdigest())
    require(records, 'Empty tensor inventory')
    return records


def selected(names, recipe):
    require(recipe in RECIPES, 'Unknown predeclared recipe')
    targets = {f'l.blk.{layer}.ffn_{part}_exps.weight'
               for layer in RECIPES[recipe] for part in ['gate', 'up']}
    require(targets <= set(names), 'Missing selected stacked expert tensor')
    return targets


def overrides(records, recipe):
    targets = selected(records, recipe)
    require(all(r['type'] in (0, 1) for r in records.values()), 'Floating source only')
    require(not any(len(r['shape']) == 4 and r['type'] == 0 for r in records.values()),
            'F32 four-axis source would be physically flattened by the generic quantizer')
    # Exact names, no broad rule can accidentally cover a selected expert.
    return [name + '=' + ('f32' if row['type'] == 0 else 'f16')
            for name, row in sorted(records.items()) if name not in targets]


def protected(control, candidate, recipe):
    targets = selected(control, recipe)
    require(set(control) == set(candidate), 'Candidate tensor names differ')
    for name, row in control.items():
        other = candidate[name]
        if name in targets:
            require(row['type'] in (0, 1) and row['shape'] == [1280, 896, 64], 'Expert blueprint shape differs')
            require(other['type'] == 12 and other['shape'] == row['shape'], 'Selected tensor must be actual Q4_K')
            require(other['bytes'] == 1280 * 896 * 64 // 256 * 144, 'Q4_K packed extent differs')
        else:
            require(row == other, 'Protected source type/shape/payload differs: ' + name)
    require(sum(row['type'] == 12 for row in candidate.values()) == len(targets), 'Unexpected Q4_K coverage')
    require(not any(row['type'] == 2 for row in candidate.values()), 'Q4_0 fallback forbidden')
    return dict(actual_q4_k_stacked_tensors=len(targets), selected=sorted(targets),
                protected_tensors=len(control) - len(targets), all_protected_byte_exact=True,
                critical_precision='exact published floating source types, including F16 norms and vision')
