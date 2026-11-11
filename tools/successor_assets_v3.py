#!/usr/bin/env python3
"""Successor-asset protocol v3: v2 plus an explicit WASM-only platform tolerance (stdlib only).

Owner decision (2 Nov 2026): the browser/WASM build of the published quantized
MiniLM (all-MiniLM-L6-v2-iq4_xs, sha 77087a5d...) may differ from native by a
bounded, attributed amount. Nothing else changes:

* v1 and v2 files, guards and receipts are read-only; their blobs are pinned here.
* Every native/desktop/binding/mobile/Apple gate stays the strict v1 gate. The v3
  consumer workflow, browser consumer and review are *derived* from the frozen v1
  files with exactly the replacements in DERIVATIONS, and the controls reject any
  other difference.
* The WASM tolerance is applied only to the packaged browser consumer's final pooled
  vector for that model and fixture text, with no page errors and the classical OCR
  cleanup still exact. Drift beyond the bound fails.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/successor-assets-v3.json'
SCHEMA = 'crispembed-successor-assets-v3'
_spec = importlib.util.spec_from_file_location('successor_assets_v2_frozen', ROOT / 'tools/successor_assets_v2.py')
V2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V2)
git_blob, RUNTIME_PATHS, PRODUCER_PATHS, git_closure = V2.git_blob, V2.RUNTIME_PATHS, V2.PRODUCER_PATHS, V2.git_closure
STRICT_ABSOLUTE, STRICT_COSINE = 0.002, 0.99999  # unchanged native/desktop/binding gates
V3_OWN = ('.github/workflows/successor-assets-v3.yml', '.github/workflows/successor-assets-consumers-v3.yml',
          'docs/successor-assets-v3.md', 'docs/successor-assets-v3.json', 'tools/successor_assets_v3.py',
          'tools/ci-heavy/successor_review_v3.py', 'tests/test_successor_wasm_v3.cjs', 'tests/test_successor_assets_v3.py',
          '.github/workflows/successor-bindings-v3.yml', 'tools/ci-heavy/successor_bindings_v3.py')


sys.path.insert(0, str(ROOT / 'tools'))
import frozen_guard_amendments as _amend  # noqa: E402  (recorded re-pins, exact chain semantics)


class ProtocolError(ValueError):
    pass


def require(value, message):
    if not value:
        raise ProtocolError(message)


def load(path=PROTOCOL):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ---------------------------------------------------------------- WASM tolerance

def bound_from_measured(max_abs, cosine, margin):
    """Stated rule: margin over the measured deterministic drift, rounded outward to 3 significant digits."""
    def up(x):
        e = math.floor(math.log10(x)) - 2
        return math.ceil(x / 10 ** e) * 10 ** e
    deficit = 1.0 - cosine
    return round(up(max_abs * (1 + margin)), 12), round(1.0 - up(deficit * (1 + margin)), 12)


def wasm_gate(report, protocol=None):
    """Decide the packaged browser consumer under the WASM-only tolerance. Never raises on numbers."""
    p = (protocol or load())['wasm_tolerance']
    vector, reference = report.get('embed', {}).get('vector'), report.get('native_vector')
    structural = (report.get('schema') == 'crispembed-successor-wasm-consumer-v3'
                  and report.get('model_sha256') == p['model_sha256'] and report.get('text') == p['text']
                  and report.get('page_errors') == []
                  and report.get('ocr', {}).get('authored_black_blob_cleared_rgb_exact') is True
                  and isinstance(vector, list) and isinstance(reference, list) and len(vector) == len(reference) == 384
                  and all(type(v) in (float, int) and math.isfinite(v) for v in vector + reference))
    if not structural:
        return {'passed': False, 'structural': False, 'reason': 'browser consumer structure/model/text/OCR/page-error contract differs'}
    errors = [abs(a - b) for a, b in zip(vector, reference)]
    na, nb = math.sqrt(math.fsum(v * v for v in vector)), math.sqrt(math.fsum(v * v for v in reference))
    if not (na > 0 and nb > 0):
        return {'passed': False, 'structural': False, 'reason': 'zero vector'}
    max_abs, cosine = max(errors), math.fsum(a * b for a, b in zip(vector, reference)) / (na * nb)
    agree = abs(max_abs - report.get('max_abs', math.inf)) <= 1e-12 and abs(cosine - report.get('cosine', math.inf)) <= 1e-12
    strict = max_abs <= STRICT_ABSOLUTE and cosine >= STRICT_COSINE
    within = max_abs <= p['max_abs'] and cosine >= p['min_cosine']
    return {'passed': bool(agree and within), 'structural': True, 'max_abs': max_abs, 'cosine': cosine,
            'reported_metrics_agree': agree, 'strict_native_gate_passed': strict,
            'bound': {'max_abs': p['max_abs'], 'min_cosine': p['min_cosine']},
            'basis': 'wasm-platform-tolerance-v3' if not strict else 'strict'}


def check_tolerance(protocol):
    t = protocol['wasm_tolerance']
    require(t['applies_to'] == {'consumer': 'packaged browser crispembed-embed-wasm', 'output': 'final pooled 384-d vector',
                                'platform': 'wasm'}, 'Tolerance scope must be the packaged browser final vector only')
    require(t['model_sha256'] == '77087a5dd455f5eeae6d6d098bac82793610438518a91071c72c253940db18a2', 'Tolerance model differs')
    m = t['measured']
    require(len(m) >= 2 and len({(r['max_abs'], r['cosine']) for r in m}) == 1, 'Measured drift must be deterministic across producer runs')
    require(t['margin'] == 0.2, 'Stated margin differs')
    bound = bound_from_measured(m[0]['max_abs'], m[0]['cosine'], t['margin'])
    require((t['max_abs'], t['min_cosine']) == bound, 'Bound must equal the stated rule applied to the measured drift')
    require(all(r['max_abs'] <= t['max_abs'] and r['cosine'] >= t['min_cosine'] for r in m), 'Measured evidence outside bound')
    require(t['max_abs'] > STRICT_ABSOLUTE and t['min_cosine'] < STRICT_COSINE, 'Tolerance must be explicit, not the strict gate')
    for path in t['attribution_receipts']:
        require((ROOT / path).is_file(), 'Attribution receipt missing: ' + path)
    return bound


# ------------------------------------------------- derived consumer artifacts (v1 + exact replacements)

DERIVATIONS = {
    '.github/workflows/successor-assets-consumers-v3.yml': ('.github/workflows/successor-assets.yml', [
        ('name: Immutable successor packaged consumers\n',
         'name: Successor packaged consumers v3 (derived from v1; WASM-only tolerance)\n'),
        ('node tests/test_successor_wasm.cjs "$RUNNER_TEMP/browser" "$RUNNER_TEMP/browser/browser.json"',
         'node tests/test_successor_wasm_v3.cjs "$RUNNER_TEMP/browser" "$RUNNER_TEMP/browser/browser.json"'
         ' && python tools/successor_assets_v3.py browser-gate --report "$RUNNER_TEMP/browser/browser.json"'),
        ('python tools/ci-heavy/successor_review.py --root', 'python tools/ci-heavy/successor_review_v3.py --root'),
        ('    uses: ./.github/workflows/successor-bindings.yml\n', '    uses: ./.github/workflows/successor-bindings-v3.yml\n'),
    ]),
    # Identity allowlist only: the frozen bindings consumer accepts fixtures from named caller workflows.
    '.github/workflows/successor-bindings-v3.yml': ('.github/workflows/successor-bindings.yml', [
        ('name: Successor Linux prebuilt Rust and Dart consumers\n', 'name: Successor Linux prebuilt Rust and Dart consumers v3 (derived; caller allowlist only)\n'),
        ('python tools/ci-heavy/successor_bindings.py --self-test', 'python tools/ci-heavy/successor_bindings_v3.py --self-test'),
        ('python tools/ci-heavy/successor_bindings.py --native-run', 'python tools/ci-heavy/successor_bindings_v3.py --native-run'),
    ]),
    'tools/ci-heavy/successor_bindings_v3.py': ('tools/ci-heavy/successor_bindings.py', [
        ("                                '.github/workflows/successor-assets-v2.yml')",
         "                                '.github/workflows/successor-assets-v2.yml', '.github/workflows/successor-assets-v3.yml')"),
    ]),
    'tests/test_successor_wasm_v3.cjs': ('tests/test_successor_wasm.cjs', [
        ("schema:'crispembed-successor-wasm-consumer-v1',", "schema:'crispembed-successor-wasm-consumer-v3',text:fixture.text,native_vector:b,"),
        ("check(errors.length===0 && maxAbs<=.002 && cosine>=.99999,'Packaged browser/native parity failed');",
         "check(errors.length===0,'Browser page errors'); // numeric decision: tools/successor_assets_v3.py browser-gate"),
    ]),
    'tools/ci-heavy/successor_review_v3.py': ('tools/ci-heavy/successor_review.py', [
        ('"""Collect completed packaged-consumer evidence; device/model limits remain explicit."""\n',
         '"""Derived from successor_review.py by tools/successor_assets_v3.py; only the browser parity check differs."""\n'),
        ('from successor_assets import MODEL_SHA as MINILM_MODEL_SHA, TEXT as MINILM_TEXT\n',
         'from successor_assets import MODEL_SHA as MINILM_MODEL_SHA, TEXT as MINILM_TEXT\nfrom successor_assets_v3 import wasm_gate\n'),
        ("or len(d.get('embed',{}).get('vector',[]))!=384 or not parity(dict(d,passed=True))):",
         "or len(d.get('embed',{}).get('vector',[]))!=384 or not wasm_gate(d)['passed']):"),
        ("return {'schema':'crispembed-successor-bounded-consumer-review-v1',",
         "return {'schema':'crispembed-successor-bounded-consumer-review-v3','browser_parity':wasm_gate(load('successor-wasm-consumer','browser.json')),"),
    ]),
}


def derive(target, root=ROOT, source_text=None):
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


# ----------------------------------------------------------------- pins, matrix, producers

def frozen_files(root=ROOT):
    found = set()
    for pattern in V2.V1_GLOBS:
        found.update(str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file())
    return sorted(found - set(V3_OWN))


def check_pins(protocol, root=ROOT):
    """Every v1/v2 file pinned at v3 creation must be unchanged; later-added files are allowed."""
    pinned = protocol['read_only_blobs']
    missing = [p for p in pinned if not (root / p).is_file()]
    require(not missing, 'Pinned read-only files removed: ' + ', '.join(missing))
    changed = [p for p, blob in pinned.items() if git_blob(root / p) != _amend.expected('successor-assets-v3', p, blob)]
    require(not changed, 'Pinned read-only files changed: ' + ', '.join(changed))
    for p in ('tools/successor_assets_v2.py', 'docs/successor-assets-v2.json', '.github/workflows/successor-assets.yml',
              'tools/ci-heavy/successor_review.py', 'tests/test_successor_wasm.cjs', 'tools/ci-heavy/successor_assets.py'):
        require(p in pinned, 'Required frozen file not pinned: ' + p)
    return len(pinned)


def check_declared(protocol):
    require(protocol.get('schema') == SCHEMA and protocol.get('protocol_version') == 3, 'not a v3 successor-asset protocol')
    v2 = V2.load()
    require(protocol['final_base'] == v2['final_base'] and protocol['ggml'] == v2['ggml'], 'v3 keeps the v2 final base and ggml')
    require(protocol['producers'] == v2['producers'], 'Producer set/workflows identical to v2')
    require(protocol.get('qualification') is False and protocol.get('release_alias_allowed') is False
            and protocol.get('app_upgrade_allowed') is False, 'v3 claims no qualification, alias or app upgrade')
    require(protocol['strict_gates'] == {'max_abs': STRICT_ABSOLUTE, 'min_cosine': STRICT_COSINE,
                                         'applies_to': 'every non-WASM consumer and cross-platform comparison, unchanged from v1'},
            'Strict native gates must be unchanged')
    exp = protocol['expectations']
    require(set(exp) == set(v2['expectations']), 'Every v2 surface keeps an explicit expectation')
    require(exp['browser'] == {'expected': 'pass-within-wasm-tolerance', 'may_claim': 'wasm-tolerance-only'},
            'Browser expectation must name the WASM tolerance')
    require({k: v for k, v in exp.items() if k != 'browser'} == {k: v for k, v in v2['expectations'].items() if k != 'browser'},
            'Non-browser expectations unchanged from v2')
    return protocol['final_base']


def check_matrix(root=ROOT):
    jobs = set(re.findall(r'^  ([a-z][\w-]*):\s*$', (root / '.github/workflows/successor-assets-consumers-v3.yml').read_text(encoding='utf-8'), re.M))
    v1 = set(re.findall(r'^  ([a-z][\w-]*):\s*$', (root / '.github/workflows/successor-assets.yml').read_text(encoding='utf-8'), re.M))
    require(jobs == v1, 'v3 consumer jobs differ from v1')
    return len(jobs)


def check_producer_run(kind, run, closure, protocol=None):
    protocol = protocol or load()
    require(run.get('status') == 'completed' and run.get('conclusion') == 'success' and run.get('event') == 'workflow_dispatch'
            and run.get('path') == PRODUCER_PATHS[kind], kind + ' producer run is not a successful dispatch of ' + PRODUCER_PATHS[kind])
    require(closure(run['head_sha']) == closure(protocol['final_base']), kind + ' producer runtime closure differs from final base')
    return run['head_sha']


def controls(protocol=None, root=ROOT):
    protocol = protocol or load()
    return {'final_base': check_declared(protocol), 'pinned_read_only': check_pins(protocol, root),
            'derived_files': check_derived(root), 'consumer_jobs': check_matrix(root), 'wasm_bound': check_tolerance(protocol)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['controls', 'producer', 'browser-gate', 'derive'])
    p.add_argument('--kind', choices=sorted(PRODUCER_PATHS)); p.add_argument('--run-json', type=Path); p.add_argument('--report', type=Path)
    a = p.parse_args(argv)
    if a.command == 'controls':
        print(json.dumps(controls(), indent=2))
    elif a.command == 'producer':
        run = json.loads(a.run_json.read_text(encoding='utf-8'))
        print(a.kind, 'producer at runtime-equivalent final source:', check_producer_run(a.kind, run, git_closure))
    elif a.command == 'derive':
        for target in DERIVATIONS:
            (ROOT / target).write_text(derive(target), encoding='utf-8')
    else:
        report = json.loads(a.report.read_text(encoding='utf-8'))
        decision = wasm_gate(report)
        a.report.with_name('browser-gate.json').write_text(json.dumps(decision, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(decision, indent=2))
        if not decision['passed']:
            raise SystemExit('Packaged browser drift outside the v3 WASM tolerance (or structure differs)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
