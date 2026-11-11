#!/usr/bin/env python3
"""Successor-asset protocol v4: v3 plus one measured native cross-platform desktop tolerance (stdlib only).

Owner decision (3 Nov 2026): accept the measured macOS arm64 desktop drift without
attribution. Measured evidence narrows the scope: Linux arm64 vs Linux x86_64 passes
the strict gate (1.2e-4), while macOS arm64 differs from *both* Linux x86_64 (0.00255)
and Linux arm64 (0.00260). The bound therefore applies only to the comparison the
review makes for platform macos-arm64 against the linux-x86_64 reference. Every other
cross-platform comparison and every same-platform comparison (CLI/Python/C/offline vs
C ABI, bindings, repeats) stays strict (0.002/0.99999). v1, v2 and v3 files are read-only.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import frozen_guard_amendments as _amend  # noqa: E402  (recorded re-pins, exact chain semantics)
PROTOCOL = ROOT / 'docs/successor-assets-v4.json'
SCHEMA = 'crispembed-successor-assets-v4'
_spec = importlib.util.spec_from_file_location('successor_assets_v3_frozen', ROOT / 'tools/successor_assets_v3.py')
V3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V3)
require, ProtocolError, git_blob = V3.require, V3.ProtocolError, V3.git_blob
STRICT_ABSOLUTE, STRICT_COSINE = V3.STRICT_ABSOLUTE, V3.STRICT_COSINE
V4_OWN = ('.github/workflows/successor-assets-v4.yml', '.github/workflows/successor-assets-consumers-v4.yml',
          '.github/workflows/successor-bindings-v4.yml', 'docs/successor-assets-v4.md', 'docs/successor-assets-v4.json',
          'tools/successor_assets_v4.py', 'tools/ci-heavy/successor_review_v4.py', 'tools/ci-heavy/successor_bindings_v4.py',
          'tests/test_successor_assets_v4.py')


def load(path=PROTOCOL):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def desktop_gate(platform, measured, protocol=None):
    """Called by the derived review for each desktop vs the linux-x86_64 reference. Strict unless the declared pair."""
    t = (protocol or load())['desktop_tolerance']
    strict = measured.get('max_abs', math.inf) <= STRICT_ABSOLUTE and math.isfinite(measured.get('cosine', math.nan)) \
        and measured['cosine'] >= STRICT_COSINE
    if strict:
        return {'passed': True, 'basis': 'strict', 'platform': platform}
    scoped = platform == t['platform'] and t['reference_platform'] == 'linux-x86_64'
    within = scoped and measured['max_abs'] <= t['max_abs'] and measured['cosine'] >= t['min_cosine']
    return {'passed': bool(within), 'basis': 'desktop-cross-platform-tolerance-v4' if within else 'rejected',
            'platform': platform, 'bound': {'max_abs': t['max_abs'], 'min_cosine': t['min_cosine']} if scoped else None}


def check_tolerance(protocol):
    t = protocol['desktop_tolerance']
    require(t['platform'] == 'macos-arm64' and t['reference_platform'] == 'linux-x86_64'
            and t['output'] == 'final pooled 384-d vector', 'Desktop tolerance scope must be macos-arm64 vs linux-x86_64 final vector')
    require(t['model_sha256'] == '77087a5dd455f5eeae6d6d098bac82793610438518a91071c72c253940db18a2', 'Desktop tolerance model differs')
    m = t['measured']
    require(len(m) >= 2 and len({(r['max_abs'], r['cosine']) for r in m}) == 1, 'Measured desktop drift must be deterministic')
    require(t['margin'] == 0.2, 'Stated margin differs')
    bound = V3.bound_from_measured(m[0]['max_abs'], m[0]['cosine'], t['margin'])
    require((t['max_abs'], t['min_cosine']) == bound, 'Desktop bound must equal the stated rule applied to the measured drift')
    require(t['attribution'] == 'none (accepted without attribution)', 'Desktop tolerance must state it is unattributed')
    for r in t['other_pairs_measured_strict']:
        require(r['max_abs'] <= STRICT_ABSOLUTE and r['cosine'] >= STRICT_COSINE, 'A pair listed as strict-passing exceeds the strict gate')
    return bound


_V3_REVIEW = dict(V3.DERIVATIONS['tools/ci-heavy/successor_review_v3.py'][1])
DERIVATIONS = {
    '.github/workflows/successor-assets-consumers-v4.yml': ('.github/workflows/successor-assets.yml', [
        ('name: Immutable successor packaged consumers\n',
         'name: Successor packaged consumers v4 (derived from v1; WASM and macOS-arm64 desktop tolerances)\n'),
        ('node tests/test_successor_wasm.cjs "$RUNNER_TEMP/browser" "$RUNNER_TEMP/browser/browser.json"',
         'node tests/test_successor_wasm_v3.cjs "$RUNNER_TEMP/browser" "$RUNNER_TEMP/browser/browser.json"'
         ' && python tools/successor_assets_v3.py browser-gate --report "$RUNNER_TEMP/browser/browser.json"'),
        ('python tools/ci-heavy/successor_review.py --root', 'python tools/ci-heavy/successor_review_v4.py --root'),
        ('    uses: ./.github/workflows/successor-bindings.yml\n', '    uses: ./.github/workflows/successor-bindings-v4.yml\n'),
    ]),
    '.github/workflows/successor-bindings-v4.yml': ('.github/workflows/successor-bindings.yml', [
        ('name: Successor Linux prebuilt Rust and Dart consumers\n', 'name: Successor Linux prebuilt Rust and Dart consumers v4 (derived; caller allowlist only)\n'),
        ('python tools/ci-heavy/successor_bindings.py --self-test', 'python tools/ci-heavy/successor_bindings_v4.py --self-test'),
        ('python tools/ci-heavy/successor_bindings.py --native-run', 'python tools/ci-heavy/successor_bindings_v4.py --native-run'),
    ]),
    'tools/ci-heavy/successor_bindings_v4.py': ('tools/ci-heavy/successor_bindings.py', [
        ("                                '.github/workflows/successor-assets-v2.yml')",
         "                                '.github/workflows/successor-assets-v2.yml', '.github/workflows/successor-assets-v4.yml')"),
    ]),
    'tools/ci-heavy/successor_review_v4.py': ('tools/ci-heavy/successor_review.py', [
        ('"""Collect completed packaged-consumer evidence; device/model limits remain explicit."""\n',
         '"""Derived from successor_review.py by tools/successor_assets_v4.py; only browser and macOS-arm64 desktop parity differ."""\n'),
        ('from successor_assets import MODEL_SHA as MINILM_MODEL_SHA, TEXT as MINILM_TEXT\n',
         'from successor_assets import MODEL_SHA as MINILM_MODEL_SHA, TEXT as MINILM_TEXT\nfrom successor_assets_v3 import wasm_gate\nfrom successor_assets_v4 import desktop_gate\n'),
        ("or len(d.get('embed',{}).get('vector',[]))!=384 or not parity(dict(d,passed=True))):",
         _V3_REVIEW["or len(d.get('embed',{}).get('vector',[]))!=384 or not parity(dict(d,passed=True))):"]),
        ("  if not measured['passed']:\n   raise ValueError('Cross-platform desktop vector parity failed: '+platform+' '+repr(measured))",
         "  measured['gate']=desktop_gate(platform,measured)\n  if not measured['gate']['passed']:\n   raise ValueError('Cross-platform desktop vector parity failed: '+platform+' '+repr(measured))"),
        ("return {'schema':'crispembed-successor-bounded-consumer-review-v1',",
         "return {'schema':'crispembed-successor-bounded-consumer-review-v4','browser_parity':wasm_gate(load('successor-wasm-consumer','browser.json')),"),
    ]),
}


def derive(target, root=Path(ROOT), source_text=None):
    source, replacements = DERIVATIONS[target]
    text = source_text if source_text is not None else (root / source).read_text(encoding='utf-8')
    for old, new in replacements:
        require(text.count(old) == 1, 'Derivation anchor must occur exactly once in ' + source + ': ' + old[:60])
        text = text.replace(old, new)
    return text


def check_derived(root=ROOT):
    for target in DERIVATIONS:
        require((root / target).read_text(encoding='utf-8') == derive(target, root), 'Derived file differs from v1 + declared replacements: ' + target)
    return len(DERIVATIONS)


def check_pins(protocol, root=ROOT):
    pinned = protocol['read_only_blobs']
    missing = [p for p in pinned if not (root / p).is_file()]
    require(not missing, 'Pinned read-only files removed: ' + ', '.join(missing))
    changed = [p for p, b in pinned.items() if git_blob(root / p) != _amend.expected('successor-assets-v4', p, b)]
    require(not changed, 'Pinned read-only files changed: ' + ', '.join(changed))
    v3 = V3.load()
    require(set(v3['read_only_blobs']) <= set(pinned) and set(V3.V3_OWN) <= set(pinned), 'v4 must pin every v3 pin and every v3 file')
    return len(pinned)


def check_declared(protocol):
    v3 = V3.load()
    require(protocol.get('schema') == SCHEMA and protocol.get('protocol_version') == 4, 'not a v4 successor-asset protocol')
    for k in ('final_base', 'ggml', 'producers', 'strict_gates', 'wasm_tolerance'):
        require(protocol[k] == v3[k], 'v4 keeps v3 ' + k + ' unchanged')
    require(protocol.get('qualification') is False and protocol.get('release_alias_allowed') is False
            and protocol.get('app_upgrade_allowed') is False, 'v4 claims no qualification, alias or app upgrade')
    require(protocol['expectations'] == v3['expectations'], 'Surface expectations unchanged from v3')
    return protocol['final_base']


def controls(protocol=None, root=ROOT):
    protocol = protocol or load()
    V3.check_tolerance(V3.load())
    return {'final_base': check_declared(protocol), 'pinned_read_only': check_pins(protocol, root),
            'derived_files': check_derived(root), 'desktop_bound': check_tolerance(protocol),
            'consumer_jobs': len(__import__('re').findall(r'^  ([a-z][\w-]*):\s*$', (root / '.github/workflows/successor-assets-consumers-v4.yml').read_text(), 8))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['controls', 'producer', 'derive'])
    p.add_argument('--kind'); p.add_argument('--run-json', type=Path)
    a = p.parse_args(argv)
    if a.command == 'controls':
        print(json.dumps(controls(), indent=2))
    elif a.command == 'producer':
        run = json.loads(a.run_json.read_text(encoding='utf-8'))
        print(a.kind, 'producer at runtime-equivalent final source:', V3.check_producer_run(a.kind, run, V3.git_closure, load()))
    else:
        for target in DERIVATIONS:
            (ROOT / target).write_text(derive(target), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
