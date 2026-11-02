#!/usr/bin/env python3
"""Recorded re-pins for source hashes frozen inside retained Qwen3.5-OCR evidence.

The quality/quantization receipts under tests/results/qwen35-ocr pin the exact source
bytes their inference ran on. Those receipts are frozen and are never rewritten. When a
pinned file later changes legitimately (a new default-off path, an unrelated quantizer
family), the change is declared here instead: the frozen hash, the commit the evidence
ran at, every re-pin (previous hash -> new hash, the commits that changed it and why).

The auditor accepts a drifted file only when the frozen digest and the current bytes are
both covered by a recorded chain. An undeclared change still fails with
"Source bytes changed". The chain's starting point is checkable against git: the frozen
digest must be the blob at each receipt's own source_commit.

  tools/qwen35_source_pins.py --check
  tools/qwen35_source_pins.py --amend PATH --reason TEXT   # after reviewing the diff
  tools/qwen35_source_pins.py --verify-git                 # needs the receipt commits
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMENDMENTS = 'tests/results/qwen35-ocr/source-pin-amendments.json'
SCHEMA = 'qwen35-ocr-source-pin-amendments/v1'
# Receipts whose runtime/diagnostic pins the OCR quality contract verifies byte for byte
# (tools/audit_qwen35_quality_quant.py). quality.json's runtime pins are the same values;
# its own diagnostic pins were superseded by these receipts and are not byte-checked.
RECEIPTS = tuple('tests/results/qwen35-ocr/quality-quant-%s-inventory-table.json' % p
                 for p in ('q8', 'q8-q6', 'q4'))
GROUPS = ('runtime_sources', 'diagnostic_sources')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(repo=ROOT):
    path = Path(repo) / AMENDMENTS
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA:
        raise ValueError('Unknown source-pin amendment schema')
    return {(a['path'], a['frozen_sha256']): a for a in data['amendments']}


def accepted(amendment):
    """Every digest a recorded chain admits, in order (frozen first)."""
    chain = [amendment['frozen_sha256']]
    for repin in amendment['repins']:
        if repin['from_sha256'] != chain[-1] or repin['to_sha256'] in chain:
            raise ValueError('Broken source-pin chain: ' + amendment['path'])
        chain.append(repin['to_sha256'])
    return chain


def source_matches(repo, path, digest, amendments):
    now = sha(Path(repo) / path)
    if now == digest:
        return True
    amendment = amendments.get((path, digest))
    # Only the newest re-pin is current; older chain links stay as history.
    return amendment is not None and accepted(amendment)[-1] == now


def pins(repo=ROOT):
    """(receipt, group, path, frozen digest, receipt source commit) for every pin."""
    for receipt in RECEIPTS:
        report = json.loads((Path(repo) / receipt).read_text())
        for group in GROUPS:
            for path, digest in report.get(group, {}).items():
                yield receipt, group, path, digest, report['source_commit']


def drift(repo=ROOT):
    amendments = load(repo)
    return sorted({(path, digest) for _, _, path, digest, _ in pins(repo)
                   if not source_matches(repo, path, digest, amendments)})


def git(repo, *args, binary=False):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=not binary)


def verify_git(repo=ROOT):
    """Each amended frozen digest is the blob at every receipt commit that pinned it."""
    amendments = load(repo)
    checked = 0
    for receipt, _, path, digest, commit in pins(repo):
        if (path, digest) not in amendments:
            continue
        amendment = amendments[(path, digest)]
        if sha_bytes(git(repo, 'cat-file', 'blob', '%s:%s' % (commit, path), binary=True)) != digest:
            raise ValueError('Frozen digest is not the blob at %s for %s' % (commit[:12], path))
        if commit not in amendment['receipt_commits']:
            raise ValueError('Receipt commit not recorded for %s: %s' % (path, receipt))
        checked += 1
    for amendment in amendments.values():
        last = amendment['repins'][-1]
        if git(repo, 'hash-object', str(Path(repo) / amendment['path'])).strip() != last['to_blob']:
            raise ValueError('Recorded current blob differs: ' + amendment['path'])
    return checked


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def amend(path, reason, repo=ROOT):
    """Append a re-pin for one drifted path; frozen receipts are not touched."""
    if not reason.strip():
        raise ValueError('A re-pin needs a reason')
    targets = sorted({(p, d) for p, d in drift(repo) if p == path})
    if not targets:
        raise ValueError('No unrecorded drift for ' + path)
    file = Path(repo) / AMENDMENTS
    data = json.loads(file.read_text()) if file.exists() else dict(schema=SCHEMA, amendments=[])
    now, blob = sha(Path(repo) / path), git(repo, 'hash-object', str(Path(repo) / path)).strip()
    head = git(repo, 'rev-parse', 'HEAD').strip()
    for _, frozen in targets:
        commits = sorted({c for _, _, p, d, c in pins(repo) if p == path and d == frozen})
        entry = next((a for a in data['amendments'] if a['path'] == path and a['frozen_sha256'] == frozen), None)
        if entry is None:
            entry = dict(path=path, frozen_sha256=frozen, receipt_commits=commits, repins=[])
            data['amendments'].append(entry)
        previous = accepted(entry)[-1]
        base = commits[-1] if not entry['repins'] else entry['repins'][-1]['recorded_at_commit']
        changed_by = git(repo, 'log', '--format=%H %s', base + '..HEAD', '--', path).strip().splitlines()
        entry['repins'].append(dict(from_sha256=previous, to_sha256=now, to_blob=blob, recorded_at_commit=head,
                                    recorded=datetime.date.today().isoformat(), changed_by=changed_by,
                                    reason=reason.strip()))
    file.write_text(json.dumps(data, indent=2) + '\n')
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true', help='exit 1 when any pin drifted without a record')
    parser.add_argument('--amend', metavar='PATH')
    parser.add_argument('--reason', default='')
    parser.add_argument('--verify-git', action='store_true')
    args = parser.parse_args()
    try:
        if args.amend:
            for path, frozen in amend(args.amend, args.reason):
                print('re-pinned %s (frozen %s)' % (path, frozen))
        if args.verify_git:
            print('verified %d amended receipt pins against git' % verify_git())
        stale = drift()
        for path, frozen in stale:
            print('UNRECORDED DRIFT %s frozen %s now %s' % (path, frozen, sha(ROOT / path)))
        if args.check and stale:
            sys.exit(1)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, 'Source pin check failed: %s\n' % error)


if __name__ == '__main__':
    main()
