"""Pure admission/metric/decision logic for the layer3 FFN mul_mat replay (stdlib only).

Nothing here runs a build or a model. The hosted caller feeds it saved compile
inventories and the probe's output bytes; tests feed it synthetic data.
"""
import hashlib
import math
import shlex
import struct
from pathlib import Path

ABSOLUTE, COSINE = 0.002, 0.99999  # unchanged diagnostic budgets
TOKENS = 12
KERNEL_FILES = {
    'native': ('ggml-cpu/ggml-cpu.c', 'ggml-cpu/ggml-cpu.cpp', 'ggml-cpu/quants.c', 'ggml-cpu/arch/x86/quants.c',
               'ggml-cpu/ops.cpp', 'ggml-cpu/vec.cpp', 'ggml-cpu/repack.cpp', 'ggml-cpu/llamafile/sgemm.cpp',
               'ggml-quants.c', 'ggml.c'),
    'wasm': ('ggml-cpu/ggml-cpu.c', 'ggml-cpu/ggml-cpu.cpp', 'ggml-cpu/quants.c', 'ggml-cpu/arch/wasm/quants.c',
             'ggml-cpu/ops.cpp', 'ggml-cpu/vec.cpp', 'ggml-cpu/repack.cpp', 'ggml-quants.c', 'ggml.c'),
}
ABSENT = {'native': (), 'wasm': ('ggml-cpu/llamafile/sgemm.cpp',)}
LINKAGE_ONLY = {'-DGGML_SHARED', '-DGGML_BACKEND_SHARED', '-DGGML_BACKEND_BUILD', '-DGGML_BUILD'}
FORBIDDEN = ('-ffast-math', '-Ofast', '-funsafe-math-optimizations', '-fassociative-math', '-freciprocal-math',
             '-ffinite-math-only', '-fno-signed-zeros', '-ffp-model=fast', '-fapprox-func')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def profile(command):
    keep = []
    for x in shlex.split(command)[1:]:
        if x.startswith(('-m', '-O', '-ffast', '-fno-fast', '-ffp-', '-funsafe', '-std=')) or x == '-pthread':
            keep.append(x)
        elif x.startswith('-D') and x not in LINKAGE_ONLY and not x.endswith('_EXPORTS'):
            keep.append(x)
    return sorted(keep)


def by_ggml_file(commands):
    out = {}
    for r in commands:
        f = r['file'].replace('\\', '/')
        if '/ggml/src/' in f:
            rel = f.split('/ggml/src/', 1)[1]
            require(rel not in out, 'Duplicate GGML compile entry: ' + rel)
            out[rel] = r
    return out


def admit(kind, actual, production, probe_source):
    """Kernel-file arithmetic profiles must equal the production full build of the same kind."""
    require(kind in KERNEL_FILES, 'Build kind')
    a, p = by_ggml_file(actual), by_ggml_file(production)
    rows = []
    for rel in KERNEL_FILES[kind]:
        require(rel in a and rel in p, 'Kernel file absent: ' + rel)
        pa, pp = profile(a[rel]['command']), profile(p[rel]['command'])
        require(pa == pp, 'Kernel arithmetic profile differs from production: %s %r != %r' % (rel, pa, pp))
        rows.append(dict(file=rel, profile=pa))
    for rel in ABSENT[kind]:
        require(rel not in a and rel not in p, 'Kernel file must be absent for this kind: ' + rel)
    probe = [r for r in actual if r['file'].replace('\\', '/').endswith('/' + probe_source)]
    require(len(probe) == 1, 'Exactly one probe compile entry')
    t = shlex.split(probe[0]['command'])
    contract = [x for x in t if x.startswith('-ffp-contract')]
    require(contract == ['-ffp-contract=off'] and not any(x.startswith(FORBIDDEN) for x in t), 'Probe rounding flags differ')
    return dict(kind=kind, kernel_files=rows, probe_fp_contract='off', relaxed=False)


def floats(data):
    require(len(data) % 4 == 0, 'F32 byte length')
    return [x[0] for x in struct.iter_unpack('<f', data)]


def metrics(ref, new, width):
    """Full-row metrics (rows of `width`), same arithmetic as the live-norm independent()."""
    a, b = floats(ref), floats(new)
    require(len(a) == len(b) == width * TOKENS and all(math.isfinite(v) for v in a + b), 'Exact finite F32 comparison required')
    d = [y - x for x, y in zip(a, b)]
    cos = []
    for at in range(0, len(a), width):
        x, y = a[at:at + width], b[at:at + width]
        aa = sum(v * v for v in x); bb = sum(v * v for v in y)
        cos.append(sum(v * w for v, w in zip(x, y)) / math.sqrt(aa * bb) if aa > 1e-18 and bb > 1e-18 else 1. if aa <= 1e-18 and bb <= 1e-18 else 0.)
    row = dict(elements=len(a), bit_exact=ref == new, differing_elements=sum(x != y for x, y in zip(a, b)),
               max_absolute=max(abs(x) for x in d), rms=math.sqrt(sum(x * x for x in d) / len(d)), min_row_cosine=min(cos))
    row['passed'] = row['max_absolute'] <= ABSOLUTE and row['min_row_cosine'] >= COSINE
    return row


def block_diff(a, b, block_bytes):
    require(len(a) == len(b) and len(a) % block_bytes == 0, 'Activation byte geometry')
    blocks = len(a) // block_bytes
    return dict(bytes_equal=a == b, blocks=blocks, differing_blocks=sum(a[i:i + block_bytes] != b[i:i + block_bytes] for i in range(0, len(a), block_bytes)))


BLOCK_BYTES = {'q8_0': 34, 'q8_K': 292}


def selftest(meta, files):
    """files[(side, label, kind)] -> bytes; meta[side] -> probe JSON."""
    checks = []
    for side in ['native', 'wasm']:
        cases = {c['label']: c for c in meta[side]['cases']}
        require(set(cases) == {'authored_iq4_nl', 'authored_iq4_xs'}, 'Selftest cases')
        for label, wt, at in [('authored_iq4_nl', 'iq4_nl', 'q8_0'), ('authored_iq4_xs', 'iq4_xs', 'q8_K')]:
            c = cases[label]
            checks.append(dict(name=side + '/' + label + '/types', passed=c['weight_type'] == wt and c['activation_type'] == at))
            checks.append(dict(name=side + '/' + label + '/plain-cpu-buffer', passed=c['weight_buffer'] == 'CPU'))
            prod = files[(side, label, 'production')]
            own = 'native_order' if side == 'native' else 'generic_order'
            other = 'generic_order' if side == 'native' else 'native_order'
            checks.append(dict(name=side + '/' + label + '/production==' + own, passed=prod == files[(side, label, own)]))
            checks.append(dict(name=side + '/' + label + '/sensitivity:production!=' + other, passed=prod != files[(side, label, other)]))
    return dict(passed=all(c['passed'] for c in checks), checks=checks)


def replay(meta, files, captured):
    """Own controls, four-way crosswise metrics, activation bytes and candidate effect."""
    controls = []
    for side in ['native', 'wasm']:
        m = meta[side]
        controls.append(dict(name=side + '/fc1-type', passed=m['fc1']['type'] == 'iq4_nl' and m['fc1']['ne'] == [384, 1536]))
        controls.append(dict(name=side + '/fc2-type', passed=m['fc2']['type'] == 'iq4_xs' and m['fc2']['ne'] == [1536, 384]))
        for c in m['cases']:
            controls.append(dict(name=side + '/' + c['label'] + '/plain-cpu-buffer', passed=c['weight_buffer'] == 'CPU'))
            own = 'native_order' if side == 'native' else 'generic_order'
            controls.append(dict(name=side + '/' + c['label'] + '/production==' + own,
                                 passed=files[(side, c['label'], 'production')] == files[(side, c['label'], own)]))
    for side, label, arm, op in [('native', 'fc1_native', 'native', 'fc1_raw'), ('native', 'fc2_native', 'native', 'fc2_raw'),
                                 ('wasm', 'fc1_wasm0', 'wasm0', 'fc1_raw'), ('wasm', 'fc1_wasm1', 'wasm1', 'fc1_raw'),
                                 ('wasm', 'fc2_wasm0', 'wasm0', 'fc2_raw')]:
        controls.append(dict(name='own-reproduction/' + side + '/' + label, passed=files[(side, label, 'production')] == captured[(arm, op)]))
    ops = {}
    for op, width, act in [('fc1', 1536, 'q8_0'), ('fc2', 384, 'q8_K')]:
        N, W = op + '_native', op + '_wasm0'
        NKN = files[('native', N, 'production')]
        rows = {
            'native_kernel(native_input)': dict(bytes_sha256=sha(NKN), reference=True),
            'wasm_kernel(wasm_input)': metrics(NKN, files[('wasm', W, 'production')], width),
            'native_kernel(wasm_input)': metrics(NKN, files[('native', W, 'production')], width),
            'wasm_kernel(native_input)': metrics(NKN, files[('wasm', N, 'production')], width),
        }
        decomposition = {
            'kernel_effect_on_native_input': metrics(NKN, files[('wasm', N, 'production')], width),
            'kernel_effect_on_wasm_input': metrics(files[('native', W, 'production')], files[('wasm', W, 'production')], width),
            'input_effect_native_kernel': metrics(NKN, files[('native', W, 'production')], width),
            'input_effect_wasm_kernel': metrics(files[('wasm', N, 'production')], files[('wasm', W, 'production')], width),
        }
        activation = {lab: block_diff(files[('native', lab, 'activation')], files[('wasm', lab, 'activation')], BLOCK_BYTES[act])
                      for lab in [N, W]}
        candidate = {
            # native-order emulation executed by the WASM build on its own production activation bytes
            'candidate(native_input) vs native_kernel(native_input)': metrics(NKN, files[('wasm', N, 'native_order')], width),
            'candidate(wasm_input) vs native_kernel(wasm_input)': metrics(files[('native', W, 'production')], files[('wasm', W, 'native_order')], width),
            'candidate(wasm_input) vs native_kernel(native_input)': metrics(NKN, files[('wasm', W, 'native_order')], width),
        }
        ops[op] = dict(activation_type=act, four_way=rows, decomposition=decomposition, activation_bytes=activation, candidate_expected_effect=candidate)
    if ('fc1_wasm1') in {c['label'] for c in meta['wasm']['cases']}:
        ops['fc1']['wasm1_input'] = dict(
            native_kernel=metrics(files[('native', 'fc1_native', 'production')], files[('native', 'fc1_wasm1', 'production')], 1536),
            wasm_kernel=metrics(files[('native', 'fc1_native', 'production')], files[('wasm', 'fc1_wasm1', 'production')], 1536),
            activation_bytes=block_diff(files[('native', 'fc1_wasm1', 'activation')], files[('wasm', 'fc1_wasm1', 'activation')], 34))
    numeric = [r for op in ops.values() for k, r in op['four_way'].items() if not r.get('reference')]
    return dict(controls_passed=all(c['passed'] for c in controls), controls=controls, ops=ops,
                four_way_within_budget=all(r['passed'] for r in numeric))
