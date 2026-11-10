#!/usr/bin/env python3
"""Recorded re-pins for frozen guards that pin file hashes, git objects or source text.

Some model-free guards freeze the exact bytes (sha256), git objects (blob/tree/gitlink
ids) or literal source text their evidence ran on. Those frozen values are never
rewritten. When a pinned value later changes for a reviewed, behaviour-neutral reason
(formatting, comments, refactors, includes, additive default-off gates), the change is
declared in one of tests/results/frozen-guards/*.json (one file per guard family): the guard, the path, the
frozen value and every re-pin (previous -> new value, the commits that changed it and
why). Same chain rules as tools/qwen35_source_pins.py: a guard accepts only the frozen
value or the newest recorded re-pin, so exact-match semantics are unchanged.

A change to guarded arithmetic or contract on the default path is never recorded here;
it needs a fresh replay instead.

  tools/frozen_guard_amendments.py --check        # every chain is well formed
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AMENDMENTS = 'tests/results/frozen-guards'
SCHEMA = 'crispembed-frozen-guard-amendments/v1'
KINDS = {'sha256', 'git-object', 'text'}


def load(repo=ROOT):
    out = {}
    for path in sorted((Path(repo) / AMENDMENTS).glob('*.json')):
        data = json.loads(path.read_text())
        if data.get('schema') != SCHEMA:
            raise ValueError('Unknown frozen-guard amendment schema: ' + path.name)
        for a in data['amendments']:
            if a['kind'] not in KINDS:
                raise ValueError('Unknown amendment kind: %r' % a['kind'])
            key = (a['guard'], a['path'], a['frozen'])
            if key in out:
                raise ValueError('Duplicate amendment: %r' % (key,))
            out[key] = a
    return out


def chain(amendment):
    """Every value a recorded chain admits, in order (frozen first)."""
    values = [amendment['frozen']]
    for repin in amendment['repins']:
        if repin['from'] != values[-1] or repin['to'] in values:
            raise ValueError('Broken frozen-guard chain: %s %s' % (amendment['guard'], amendment['path']))
        if repin.get('classification') != 'behaviour-neutral' or not repin.get('changed_by') or not repin.get('reason'):
            raise ValueError('Re-pin must be reviewed behaviour-neutral with commits and reason: ' + amendment['path'])
        values.append(repin['to'])
    return values


_CACHE = {}


def expected(guard, path, frozen, repo=ROOT):
    """The single value a guard must now match exactly: frozen, or the newest re-pin."""
    key = str(repo)
    if key not in _CACHE:
        _CACHE[key] = load(repo)
    amendment = _CACHE[key].get((guard, path, frozen))
    return frozen if amendment is None else chain(amendment)[-1]


def check(repo=ROOT):
    amendments = load(repo)
    for amendment in amendments.values():
        chain(amendment)
    return len(amendments)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.check:
        print('%d frozen-guard amendments, all chains valid' % check())
    return 0


if __name__ == '__main__':
    sys.exit(main())
