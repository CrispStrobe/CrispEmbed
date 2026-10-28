"""Actual root/overlay compiler and backend agreement before any model access."""
import json
from pathlib import Path
import shlex

KEYS = ('GGML_NATIVE', 'CRISPEMBED_NATIVE', 'GGML_LLAMAFILE', 'GGML_CUDA', 'GGML_BLAS',
        'GGML_OPENMP', 'GGML_BACKEND_DL', 'GGML_CPU_REPACK', 'GGML_CPU_ALL_VARIANTS',
        'GGML_AVX', 'GGML_AVX2', 'GGML_AVX512', 'GGML_FMA', 'GGML_F16C')


def cache(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith(('#', '//')) or ':' not in line or '=' not in line:
            continue
        name, rest = line.split(':', 1)
        _, rows[name] = rest.split('=', 1)
    return rows


def require(ok, why):
    if not ok:
        raise ValueError(why)


def one(rows, name, target=None):
    found = []
    for row in rows:
        argv = row.get('arguments') or shlex.split(row['command'])
        if Path(row['file']).name == name and (target is None or f'CMakeFiles/{target}.dir/' in ' '.join(argv)):
            found.append(argv)
    require(len(found) == 1, 'Unique actual compile row required: ' + name)
    return found[0]


def flags(argv):
    return sorted(x for x in argv if x.startswith(('-D', '-m', '-O', '-f')))


def verify(plain, observed, plain_source, observed_source):
    a, b = cache(plain / 'CMakeCache.txt'), cache(observed / 'CMakeCache.txt')
    require(all(k in a and k in b and a[k] == b[k] for k in KEYS), 'Actual configured profiles differ')
    require(a['CRISPEMBED_NATIVE'] == a['GGML_NATIVE'] == a['GGML_CUDA'] == 'OFF' and
            a['GGML_LLAMAFILE'] == 'ON', 'Portable CPU/root LLAMAFILE profile required')
    aa = json.loads((plain / 'compile_commands.json').read_text())
    bb = json.loads((observed / 'compile_commands.json').read_text())
    rows = []
    for name, target in [('unlimited_ocr.cpp', 'crispembed'), ('ggml-cpu.c', None), ('ggml-cpu.cpp', None)]:
        x, y = one(aa, name, target), one(bb, name, target)
        require(flags(x) == flags(y), 'Actual implementation/backend flags differ: ' + name)
        require(not any(f.startswith(('-ffast-math', '-Ofast', '-march=native')) for f in x + y), 'Unreviewed arithmetic flags')
        rows.append(dict(file=name, plain=x, observed=y, exact_arithmetic_flags=flags(x)))
    return dict(passed=True, cache={k: a[k] for k in KEYS}, compile_rows=rows,
                plain_source=str(plain_source), observed_source=str(observed_source))
