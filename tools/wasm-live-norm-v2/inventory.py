"""Closed component compile-inventory admission for live norm v2.

v1 (ff3b8a01) assumed exactly one wasm_native_order_norm.cpp compile entry.
The live caller adds tests/wasm-live-norm on top of the frozen controls, so the
helper is compiled once per declared target. This module admits only the
declared (target, source) set, requires the frozen arithmetic profile for each
entry and the frozen GGML backend inventory, and never widens a flag rule.
Callers must save the full actual commands before calling check().
"""
import shlex
from pathlib import Path

HELPER = 'src/core/wasm_native_order_norm.cpp'
# (target, source, directory relative to the build root). Exactly these five.
DECLARED = (
    ('norm-candidate-controls', 'tests/wasm-native-order-norm/controls.cpp', 'frozen-controls'),
    ('norm-candidate-controls', HELPER, 'frozen-controls'),
    ('norm-candidate-ref', 'tests/moedern-native-v2/diff.cpp', 'frozen-controls'),
    ('live-norm-replay', 'tests/wasm-live-norm/replay.cpp', '.'),
    ('live-norm-replay', HELPER, '.'),
)
FROZEN_TARGETS = {'norm-candidate-controls', 'norm-candidate-ref'}
# Any of these, in any position, rejects. Prefix match covers =value forms.
FORBIDDEN = ('-ffast-math', '-Ofast', '-funsafe-math-optimizations', '-fassociative-math',
             '-freciprocal-math', '-ffinite-math-only', '-fno-signed-zeros', '-fapprox-func',
             '-ffp-model=fast', '-ffp-model=aggressive', '-fcx-limited-range', '-msimd128')


def require(value, message):
    if not value:
        raise ValueError(message)


def tokens(command):
    return shlex.split(command)


def target_of(output):
    parts = Path(output).parts
    require(len(parts) >= 3 and parts[0] == 'CMakeFiles' and parts[1].endswith('.dir'), 'Unrecognised compile output: ' + output)
    return parts[1][:-4]


def profile(command):
    """Arithmetic/backend profile: optimisation, -m, fp/fast-math, defines, std, pthread."""
    t = tokens(command)
    keep = sorted(x for x in t[1:] if x.startswith(('-m', '-O', '-ffast', '-fno-fast', '-ffp-', '-funsafe', '-fassociative',
                                                    '-freciprocal', '-ffinite', '-fno-signed', '-fapprox', '-D', '-std=')))
    return dict(flags=keep, pthread='-pthread' in t)


def flag_violations(command):
    t = tokens(command)
    bad = [x for x in t if x.startswith(FORBIDDEN)]
    contract = [x for x in t if x.startswith('-ffp-contract')]
    if not contract or any(x != '-ffp-contract=off' for x in contract):
        bad.append('ffp-contract:' + ','.join(contract or ['absent']))
    return bad


def split(commands, root):
    """Classify every entry as declared component, GGML backend, or unexpected."""
    root = Path(root)
    own, backend, unexpected = [], {}, []
    for r in commands:
        f = Path(r['file'])
        try:
            rel = f.relative_to(root).as_posix()
        except ValueError:
            unexpected.append(dict(file=r['file'], reason='outside source root'))
            continue
        if rel.startswith('ggml/'):
            require(rel not in backend, 'Duplicate GGML backend entry: ' + rel)
            backend[rel] = r
        else:
            own.append((rel, r))
    return own, backend, unexpected


def check(kind, commands, frozen, root, build, frozen_root):
    """Return an inventory report or raise. `frozen` is the run-40852835039
    component inventory for the same kind (frozen-controls built at top level)."""
    require(kind in ('native', 'wasm'), 'Backend kind')
    build = Path(build)
    own, backend, unexpected = split(commands, root)
    require(not unexpected, 'Unexpected compile entries: ' + repr(unexpected))
    frozen_own, frozen_backend, frozen_unexpected = split(frozen, frozen_root)
    require(not frozen_unexpected, 'Frozen inventory entries outside root')

    seen, rows = {}, []
    for rel, r in own:
        t = tokens(r['command'])
        require('-o' in t, 'Compile output absent: ' + rel)
        target = target_of(t[t.index('-o') + 1])
        key = (target, rel)
        require(key not in seen, 'Duplicate declared compile entry: ' + repr(key))
        seen[key] = r
    declared = {(t, s): d for t, s, d in DECLARED}
    require(set(seen) == set(declared), 'Component target inventory differs: actual ' + repr(sorted(seen)) + ' declared ' + repr(sorted(declared)))

    frozen_profiles = {}
    for rel, r in frozen_own:
        ft = tokens(r['command']);frozen_profiles[(target_of(ft[ft.index('-o') + 1]), rel)] = profile(r['command'])
    require(set(frozen_profiles) == {k for k in declared if k[0] in FROZEN_TARGETS}, 'Frozen component inventory differs')

    for key, r in sorted(seen.items()):
        target, rel = key
        directory = build if declared[key] == '.' else build / declared[key]
        require(Path(r['directory']) == directory, 'Compile directory differs: ' + repr(key))
        t = tokens(r['command'])
        # CMake records `output` relative to the top-level build, `-o` relative to `directory`.
        require(Path(r['output']) == Path(declared[key]) / t[t.index('-o') + 1], 'Compile output path differs: ' + repr(key))
        require('-c' in t and Path(t[t.index('-c') + 1]) == Path(root) / rel and Path(r['file']) == Path(root) / rel, 'Compiled source path differs: ' + repr(key))
        compiler = Path(t[0]).name
        require(compiler in (('em++',) if kind == 'wasm' else ('c++', 'g++', 'clang++')), 'Compiler differs: ' + repr(key) + ' ' + compiler)
        bad = flag_violations(r['command'])
        require(not bad, 'Actual component rounding flags differ: ' + repr(key) + ' ' + repr(bad))
        p = profile(r['command'])
        require(p['pthread'] == (kind == 'wasm'), 'Thread profile differs: ' + repr(key))
        reference = frozen_profiles[key] if target in FROZEN_TARGETS else frozen_profiles[('norm-candidate-controls', rel if rel == HELPER else 'tests/wasm-native-order-norm/controls.cpp')]
        require(p == reference, 'Component backend profile differs from frozen: ' + repr(key))
        rows.append(dict(target=target, source=rel, directory=declared[key], profile=p, frozen_profile=reference, output=r['output']))

    helper = [profile(seen[(t, HELPER)]['command']) for t in ('norm-candidate-controls', 'live-norm-replay')]
    require(helper[0] == helper[1], 'Helper profiles differ between targets')

    require(set(backend) == set(frozen_backend), 'GGML backend inventory differs: +' + repr(sorted(set(backend) - set(frozen_backend))) + ' -' + repr(sorted(set(frozen_backend) - set(backend))))
    for rel in sorted(backend):
        require(profile(backend[rel]['command']) == profile(frozen_backend[rel]['command']), 'GGML backend profile differs: ' + rel)

    return dict(kind=kind, declared_entries=len(DECLARED), helper_entries=2, entries=rows, ggml_backend_entries=len(backend),
                ggml_backend_profile_equals_frozen=True, total_entries=len(commands), relaxed_flags=False)
