"""Exact-match helpers for the frozen MoEdern guards' recorded re-pins.

Each helper keeps the guard's frozen value as the key and compares the current
checkout against exactly one accepted value: the frozen one, or the newest
reviewed behaviour-neutral re-pin recorded in tests/results/frozen-guards/moedern.json
(see tools/frozen_guard_amendments.py). Nothing here widens a comparison.
"""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('frozen_guard_amendments', ROOT / 'tools/frozen_guard_amendments.py')
_amend = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_amend)

ADDED = '<added-src-files>'
CHANGED = '<changed-paths>'


def sha(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def expected(guard, path, frozen):
    return _amend.expected(guard, path, frozen)


def object_ok(guard, path, frozen_object, current_object):
    """A git blob/tree/gitlink id equals the frozen id or its newest recorded re-pin."""
    return current_object == expected(guard, path, frozen_object)


def bytes_ok(guard, path, frozen_bytes, current_bytes):
    """Current bytes hash to the frozen bytes' sha256 or its newest recorded re-pin."""
    return sha(current_bytes) == expected(guard, path, sha(frozen_bytes))


def digest_ok(guard, path, frozen_digest, current_digest):
    return current_digest == expected(guard, path, frozen_digest)


def path_set(guard, key, frozen=()):
    """A recorded exact path set (newline separated, sorted); frozen default is empty."""
    value = expected(guard, key, '\n'.join(sorted(frozen)))
    return set(value.split('\n')) - {''}
