"""Standard-library source admission and immutable authored router inputs."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'docs/moedern-router-topk-operator-v1.json'
BASE = '2f3cd74e035a27d46582e153f07f764ccacd2d66'
GGML = 'a06fc0bb46b73cc10011fc90ae4054aee8438aeb'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def validate(p):
    required = {'src', 'ggml', 'cmake', 'tests/moedern-native-v2/diff.cpp',
                'tools/moedern-native-v2/diff_adapter.py', 'tools/ci_process.py',
                'tools/moedern-colbert/scifact_original.py', 'docs/moedern-colbert-source-contract.json',
                'tests/results/moedern-colbert/scifact-original-27m.json',
                'tests/results/moedern-colbert/scifact-original-115m.json'}
    require(p['schema'] == 'crispembed-moedern-router-topk-operator-v1' and p['base'] == BASE
            and p['ggml'] == GGML and p['model_access'] is False and p['production_qualified'] is False,
            'Model-free source scope differs')
    require(set(p['protected_git_objects']) == required and p['experts'] == 8 and p['top_k'] == 2
            and p['largest'] is True and p['sorted'] is True and p['denominator_epsilon'] == 1e-9
            and p['absolute_budget'] == .002 and p['row_cosine_budget'] == .99999,
            'Source/operator/fixed gate contract differs')
    require(p['threads'] == [1, 2] and p['repeats'] == 3 and p['layouts'] == ['contiguous', 'column_stride2', 'row_stride2', 'transpose'],
            'Operator coverage differs')
    return p


def verify(source):
    require(re.fullmatch('[a-f0-9]{40}', source or '') and git('rev-parse', 'HEAD') == source,
            'Exact reviewed caller required')
    p = validate(read(PROFILE))
    paths = [*p['protected_git_objects'], *p['owned_sha256'], str(PROFILE.relative_to(ROOT))]
    require(not git('status', '--porcelain', '--', *paths), 'Dirty audited source')
    for path, obj in p['protected_git_objects'].items():
        require(git('rev-parse', BASE + ':' + path) == obj and git('rev-parse', 'HEAD:' + path) == obj,
                'Protected source differs: ' + path)
        if (ROOT / path).is_file():
            require(git('hash-object', '--', path) == obj, 'Protected worktree differs: ' + path)
    for path, digest in p['owned_sha256'].items():
        require(sha(ROOT / path) == digest, 'Owned source differs: ' + path)
    require(git('-C', str(ROOT / 'ggml'), 'rev-parse', 'HEAD') == GGML
            and not git('-C', str(ROOT / 'ggml'), 'status', '--porcelain', '--untracked-files=no'),
            'Actual ggml checkout differs')
    return p


def rows(profile):
    result = {}
    for mode, values in profile['inputs'].items():
        require(mode in ('probabilities', 'logits') and 1 <= len(values) <= 64, 'Bounded input mode required')
        names = set()
        for row in values:
            require(re.fullmatch('[a-z0-9_]+', row['name']) and row['name'] not in names, 'Unique safe row ID required')
            names.add(row['name'])
            require(len(row['bits']) == 8 and all(type(b) is int and 0 <= b < 2**32 for b in row['bits']),
                    'Exactly eight F32 bit patterns required')
            floats = struct.unpack('<8f', struct.pack('<8I', *row['bits']))
            require(all(math.isfinite(x) for x in floats), 'Finite authored inputs required')
        result[mode] = values
    require(set(result) == {'probabilities', 'logits'}, 'Both selection and softmax arms required')
    return result


def packet(path, values):
    require(1 <= len(values) <= 64 and all(len(v) == 8 for v in values), 'Packet axes differ')
    require(all(math.isfinite(x) for row in values for x in row), 'Nonfinite packet input')
    data = b'MRTOPK1\0' + struct.pack('<I', len(values)) + b''.join(struct.pack('<8f', *v) for v in values)
    Path(path).write_bytes(data)
    return identity(path)
