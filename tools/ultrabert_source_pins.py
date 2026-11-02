#!/usr/bin/env python3
"""Recorded re-pins for the UltraBERT SciFact driver's frozen source hashes.

tools/ci-heavy/ultrabert_scifact.py PINS freezes historical receipts and the scoped
UltraBERT runtime sources its evidence ran on. Those digests are never rewritten.
When a pinned source later changes legitimately (a new default-off path), the change
is declared in tests/results/ultrabert/source-pin-amendments.json: the frozen hash,
the commit the evidence ran at, and every re-pin (previous -> new hash, the commits
that changed it and why). Same chain rules as tools/qwen35_source_pins.py.

  tools/ultrabert_source_pins.py --check
  tools/ultrabert_source_pins.py --amend PATH --reason TEXT   # after reviewing the diff
  tools/ultrabert_source_pins.py --verify-git
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMENDMENTS = 'tests/results/ultrabert/source-pin-amendments.json'
SCHEMA = 'ultrabert-source-pin-amendments/v1'
DRIVER = 'tools/ci-heavy/ultrabert_scifact.py'

sys.path.insert(0, str(ROOT / 'tools'))
from qwen35_source_pins import accepted, sha  # noqa: E402  (same chain semantics)


def frozen_pins(repo=ROOT):
    spec = importlib.util.spec_from_file_location('ultrabert_scifact_pins', Path(repo) / DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.PINS)


def load(repo=ROOT):
    path = Path(repo) / AMENDMENTS
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA:
        raise ValueError('Unknown UltraBERT source-pin amendment schema')
    return {(a['path'], a['frozen_sha256']): a for a in data['amendments']}


def matches(path, digest, repo=ROOT, amendments=None):
    """True when the file is the frozen digest or the newest recorded re-pin of it."""
    now = sha(Path(repo) / path)
    if now == digest:
        return True
    amendment = (load(repo) if amendments is None else amendments).get((path, digest))
    return amendment is not None and accepted(amendment)[-1] == now


def drift(repo=ROOT, pins=None):
    amendments = load(repo)
    return sorted((p, d) for p, d in (pins or frozen_pins(repo)).items() if not matches(p, d, repo, amendments))


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def verify_git(repo=ROOT):
    checked = 0
    for (path, frozen), amendment in load(repo).items():
        commit = amendment['frozen_commit']
        blob = subprocess.check_output(['git', '-C', str(repo), 'cat-file', 'blob', '%s:%s' % (commit, path)])
        if hashlib.sha256(blob).hexdigest() != frozen:
            raise ValueError('Frozen digest is not the blob at %s for %s' % (commit[:12], path))
        if git(repo, 'hash-object', str(Path(repo) / path)) != amendment['repins'][-1]['to_blob']:
            raise ValueError('Recorded current blob differs: ' + path)
        checked += 1
    return checked


def amend(path, reason, frozen_commit, repo=ROOT):
    if not reason.strip():
        raise ValueError('A re-pin needs a reason')
    targets = [(p, d) for p, d in drift(repo) if p == path]
    if not targets:
        raise ValueError('No unrecorded drift for ' + path)
    file = Path(repo) / AMENDMENTS
    data = json.loads(file.read_text()) if file.exists() else dict(schema=SCHEMA, amendments=[])
    now, blob, head = sha(Path(repo) / path), git(repo, 'hash-object', str(Path(repo) / path)), git(repo, 'rev-parse', 'HEAD')
    for _, frozen in targets:
        entry = next((a for a in data['amendments'] if a['path'] == path and a['frozen_sha256'] == frozen), None)
        if entry is None:
            entry = dict(path=path, frozen_sha256=frozen, frozen_commit=frozen_commit, repins=[])
            data['amendments'].append(entry)
        base = entry['frozen_commit'] if not entry['repins'] else entry['repins'][-1]['recorded_at_commit']
        changed_by = git(repo, 'log', '--format=%H %s', base + '..HEAD', '--', path).splitlines()
        entry['repins'].append(dict(from_sha256=accepted(entry)[-1], to_sha256=now, to_blob=blob,
                                    recorded_at_commit=head, recorded=datetime.date.today().isoformat(),
                                    changed_by=changed_by, reason=reason.strip()))
    file.write_text(json.dumps(data, indent=2) + '\n')
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--amend')
    parser.add_argument('--reason', default='')
    parser.add_argument('--frozen-commit', help='commit whose blob is the frozen digest (first re-pin only)')
    parser.add_argument('--verify-git', action='store_true')
    args = parser.parse_args()
    if args.amend:
        for path, frozen in amend(args.amend, args.reason, args.frozen_commit or ''):
            print('re-pinned %s (frozen %s)' % (path, frozen))
    if args.verify_git:
        print('verified %d amended UltraBERT pins against git' % verify_git())
    remaining = drift()
    for path, frozen in remaining:
        print('UNRECORDED DRIFT %s frozen %s now %s' % (path, frozen, sha(ROOT / path)))
    return 1 if remaining else 0


if __name__ == '__main__':
    sys.exit(main())
