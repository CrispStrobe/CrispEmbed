#!/usr/bin/env python3
"""Successor-asset protocol v2 for the declared final source (stdlib only).

v1 files, guards and receipts are read-only evidence: this module pins their
Git blob identities and refuses to run if any changed. It never downloads
archives or models; the producer gate only reads run metadata and Git trees.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'docs/successor-assets-v2.json'
FINAL_DECLARATION = ROOT / 'docs/ovisocr2-final-integrated-v2.json'
V1_WORKFLOW = '.github/workflows/successor-assets.yml'
SCHEMA = 'crispembed-successor-assets-v2'
PENDING = 'PENDING'
PRODUCER_PATHS = {'native': '.github/workflows/release.yml',
                  'wasm': '.github/workflows/release-wasm.yml'}
# Same closure as v1 (tools/ci-heavy/successor_assets.py RUNTIME_PATHS); copied,
# not imported, so v1 stays untouched and v2 cannot silently widen it.
RUNTIME_PATHS = ('src', 'ggml', 'cmake', 'CMakeLists.txt', 'VERSION', 'Cargo.lock', 'examples',
                 'python', 'flutter', 'crispembed', 'crispembed-sys', 'wasm', 'tools/quantize.cpp',
                 'build-wasm.sh', 'build-embed-wasm.sh', 'scripts/check-cpu-baseline.py',
                 'scripts/check-bundled-deps.py', 'tools/successor_asset_manifest.py', '.gitattributes',
                 '.github/workflows/release.yml', '.github/workflows/release-wasm.yml', 'README.md', 'LICENSE')
V1_GLOBS = ('docs/successor-*.md', 'tools/successor_asset_manifest.py', 'tools/successor_wasm_trace_contract.py',
            'tools/ci-heavy/successor_*.py', '.github/workflows/successor-*.yml',
            'tests/test_successor_*', 'tests/successor_*', 'tests/results/packaging/*.json')
V2_OWN = {'.github/workflows/successor-assets-v2.yml', 'docs/successor-assets-v2.md', 'tests/test_successor_assets_v2.py'}


class ProtocolError(ValueError):
    pass


def git_blob(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def v1_files(root=ROOT):
    found = set()
    for pattern in V1_GLOBS:
        found.update(str(p.relative_to(root)) for p in root.glob(pattern) if p.is_file())
    return sorted(found - V2_OWN)


def load(path=PROTOCOL):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def is_pending(value):
    return not isinstance(value, str) or not re.fullmatch('[0-9a-f]{40}', value)


def check_declared(protocol, final=None):
    """Base/ggml must be declared, match the final-v2 declaration, and qualification stays false."""
    final = final if final is not None else load(FINAL_DECLARATION)
    if protocol.get('schema') != SCHEMA or protocol.get('protocol_version') != 2:
        raise ProtocolError('not a v2 successor-asset protocol')
    if is_pending(protocol.get('final_base')) or is_pending(protocol.get('ggml')):
        raise ProtocolError('final base/ggml pending: producers and consumers blocked')
    if protocol['final_base'] != final.get('runtime_base') or protocol['ggml'] != final.get('ggml'):
        raise ProtocolError('v2 pins differ from the final-integrated-v2 declaration')
    if final.get('protected_git_objects', {}).get('ggml') != protocol['ggml']:
        raise ProtocolError('ggml pin differs from protected final ggml object')
    if protocol.get('qualification') is not False or protocol.get('release_alias_allowed') is not False \
            or protocol.get('app_upgrade_allowed') is not False:
        raise ProtocolError('v2 preparation must not claim qualification, aliases or app upgrades')
    validation = protocol.get('final_source_validation')
    if not isinstance(validation, dict) or validation.get('status') not in ('pending', 'positive'):
        raise ProtocolError('final source validation status must be pending or positive')
    if validation['status'] == 'pending' and validation.get('run') is not None:
        raise ProtocolError('pending validation cannot name a run')
    return validation['status']


def check_v1_unchanged(protocol, root=ROOT):
    pinned = protocol.get('v1_read_only_blobs', {})
    current = v1_files(root)
    if sorted(pinned) != current:
        raise ProtocolError('v1 file set changed: ' + ', '.join(sorted(set(pinned) ^ set(current))))
    changed = [p for p in current if git_blob(root / p) != pinned[p]]
    if changed:
        raise ProtocolError('v1 files changed: ' + ', '.join(changed))
    return len(current)


def v1_matrix(root=ROOT):
    text = (root / V1_WORKFLOW).read_text(encoding='utf-8')
    jobs = set(re.findall(r'^  ([a-z][\w-]*):\s*$', text, re.M))
    artifacts = set(re.findall(r'^\s+artifact: ([\w.-]+)\s*$', text, re.M))
    return jobs, artifacts


def check_matrix(protocol, root=ROOT):
    sys.path.insert(0, str(root / 'tools'))
    import successor_asset_manifest as manifest
    producers = protocol['producers']
    native = {a['archive'] for a in producers['native']['archives']}
    wasm = {a['archive'] for a in producers['wasm']['archives']}
    if native != set(manifest.NATIVE_REQUIRED) or len(native) != 7:
        raise ProtocolError('native producer set must be exactly the seven v1 archives')
    if wasm != set(manifest.WASM_REQUIRED) or len(wasm) != 2:
        raise ProtocolError('WASM producer set must be exactly the two v1 bundles')
    for kind, path in PRODUCER_PATHS.items():
        if producers[kind]['workflow'] != path:
            raise ProtocolError('producer workflow changed for ' + kind)
    jobs, artifacts = v1_matrix(root)
    consumers = protocol['consumers']
    if set(consumers['v1_jobs']) != jobs - {'producer-identity'}:
        raise ProtocolError('consumer jobs differ from v1 matrix')
    if set(consumers['v1_artifacts']) != artifacts:
        raise ProtocolError('consumer artifacts differ from v1 matrix')
    stems = {a.split('crispembed-', 1)[1].rsplit('.tar.gz', 1)[0].rsplit('.zip', 1)[0] for a in native}
    if not stems <= artifacts:
        raise ProtocolError('native archive without a v1 consumer: ' + ', '.join(sorted(stems - artifacts)))
    required = {'c', 'python', 'cli', 'cache', 'rust', 'dart', 'browser', 'windows-utf8-path',
                'apple-png-mac50', 'mobile-cross-link'}
    surfaces = set(consumers['surfaces'])
    if surfaces != required:
        raise ProtocolError('consumer surfaces incomplete: ' + ', '.join(sorted(required ^ surfaces)))
    expected = protocol['expectations']
    if set(expected) != required:
        raise ProtocolError('every consumer surface needs an explicit expectation')
    if expected['browser']['expected'] != 'fail' or expected['browser'].get('may_claim') is not False:
        raise ProtocolError('WASM/browser remains negative and must not be claimed')
    return len(native), len(wasm), len(surfaces)


def check_evidence(protocol, root=ROOT):
    evidence = protocol['carried_evidence']
    receipts = {str(p.relative_to(root)) for p in (root / 'tests/results/packaging').glob('*.json')}
    if set(evidence) != receipts:
        raise ProtocolError('carried evidence must list every packaging receipt')
    for path, item in evidence.items():
        if item.get('read_only') is not True or item.get('verdict') not in ('positive', 'negative', 'context'):
            raise ProtocolError('bad carried evidence entry ' + path)
        if item['blob'] != protocol['v1_read_only_blobs'].get(path):
            raise ProtocolError('carried evidence blob mismatch ' + path)
    return len(evidence)


def check_producer_run(kind, run, closure):
    """run: GitHub run JSON; closure(sha) -> ls-tree lines of RUNTIME_PATHS."""
    if run.get('status') != 'completed' or run.get('conclusion') != 'success' \
            or run.get('event') != 'workflow_dispatch' or run.get('path') != PRODUCER_PATHS[kind]:
        raise ProtocolError(kind + ' producer run is not a successful dispatch of ' + PRODUCER_PATHS[kind])
    protocol = load()
    if closure(run['head_sha']) != closure(protocol['final_base']):
        raise ProtocolError(kind + ' producer runtime closure differs from final base')
    return run['head_sha']


def git_closure(sha):
    subprocess.run(['git', 'cat-file', '-e', sha + '^{commit}'], cwd=ROOT, check=True)
    return subprocess.check_output(['git', 'ls-tree', sha, '--', *RUNTIME_PATHS], cwd=ROOT,
                                   text=True, encoding='utf-8').splitlines()


def controls(protocol=None):
    protocol = protocol or load()
    status = check_declared(protocol)
    pinned = check_v1_unchanged(protocol)
    native, wasm, surfaces = check_matrix(protocol)
    evidence = check_evidence(protocol)
    return {'final_source_validation': status, 'v1_files_pinned': pinned, 'native_archives': native,
            'wasm_archives': wasm, 'surfaces': surfaces, 'carried_evidence': evidence}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('controls', 'producer', 'pin-v1'))
    p.add_argument('--kind', choices=tuple(PRODUCER_PATHS))
    p.add_argument('--run-json', type=Path)
    a = p.parse_args(argv)
    if a.command == 'pin-v1':
        print(json.dumps({f: git_blob(ROOT / f) for f in v1_files()}, indent=2))
    elif a.command == 'controls':
        print(json.dumps(controls(), indent=2))
    else:
        controls()
        run = json.loads(a.run_json.read_text(encoding='utf-8'))
        print(a.kind, 'producer at runtime-equivalent final source:', check_producer_run(a.kind, run, git_closure))


if __name__ == '__main__':
    sys.exit(main())
