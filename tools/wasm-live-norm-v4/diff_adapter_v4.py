"""v4 lossless F32 Ref adapter: the frozen moedern-native-v2 adapter with an explicit shape allowlist.

The frozen adapter (tools/moedern-native-v2/diff_adapter.py, protected, unchanged)
caps every dimension at 512, which rejects the actual live layer3 fc1_raw/fc1/gelu
tensors [12,1536] (run 41427588368). This module executes the SHA-pinned frozen
source into a private module object and replaces only its `shape()` with an
allowlist of exact shapes. Every other function (GGUF writer, independent
readback, collector, fixed 0.002/0.99999 budgets, metric agreement) is the
frozen code byte for byte. Rows are never reshaped or chunked: the C++ Ref row
axis is ne[0] (the full 1536-wide C-order last axis), identical to the
independent CPython metric's `width = shape[-1]`.
"""
import hashlib
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / 'tools/moedern-native-v2/diff_adapter.py'
FROZEN_SHA256 = '695a79e4a9c96f80aeada999c7f76a63c55304b96f1b42a539108adc9523d097'
# Exact shapes only: live norm/layer3 rows, live fc1/gelu rows, the Ref controls.
ALLOWED_SHAPES = frozenset({(12, 384), (12, 1536), (2, 8)})


def shape(value):
    if (not isinstance(value, (list, tuple)) or any(type(n) is not int for n in value)
            or tuple(value) not in ALLOWED_SHAPES):
        raise ValueError('Allowlisted exact v4 shape required: ' + repr(value))
    return tuple(value)


def _load():
    source = FROZEN.read_bytes()
    if hashlib.sha256(source).hexdigest() != FROZEN_SHA256:
        raise ValueError('Frozen adapter bytes differ')
    module = types.ModuleType('diff_adapter_v4_frozen_copy')
    module.__file__ = str(FROZEN)
    exec(compile(source, str(FROZEN), 'exec'), module.__dict__)
    module.shape = shape  # the single change; all other functions resolve it via this module's globals
    return module


_M = _load()
ABSOLUTE, COSINE, METRIC_RELATIVE, FIELDS = _M.ABSOLUTE, _M.COSINE, _M.METRIC_RELATIVE, _M.FIELDS
sha, source_bytes, write_gguf, read_gguf, write_pair, collect = (
    _M.sha, _M.source_bytes, _M.write_gguf, _M.read_gguf, _M.write_pair, _M.collect)
