# Adapted from CrispEmbed6fccd863 tools/moedern-sdpa/diff_adapter.py.
"""Lossless test-only F32 GGUF adapters and independent comparator collection.

Standard library only. This is not a model converter or production dependency.
Reported gates remain the conjunction of independent metrics and actual Ref.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

ABSOLUTE = 0.002
COSINE = 0.99999
METRIC_RELATIVE = 2e-7  # Reporting consistency, never a model acceptance tolerance.
FIELDS = ('max_absolute', 'mean_absolute', 'mean_error', 'rms', 'min_row_cosine', 'original_l2', 'native_l2',
          'original_min', 'original_max', 'original_abs_min', 'original_abs_max',
          'native_min', 'native_max', 'native_abs_min', 'native_abs_max')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def shape(value):
    if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 4 or
            any(type(n) is not int or not 1 <= n <= 2048 for n in value)):
        raise ValueError('Bounded exact original/native shape required')
    return tuple(value)


def source_bytes(spec):
    dims = shape(spec['shape'])
    path = Path(spec['path'])
    if path.is_symlink() or not path.is_file() or path.stat().st_size != 4 * math.prod(dims):
        raise ValueError('F32 source file shape/bytes differ')
    data = path.read_bytes()
    if sha(data) != spec['sha256']:
        raise ValueError('F32 source SHA256 differs')
    if not all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', data)):
        raise ValueError('Nonfinite F32 adapter input')
    return data, dims


def _string(value):
    encoded = value.encode('ascii')
    return struct.pack('<Q', len(encoded)) + encoded


def write_gguf(path, tensors):
    """Keep every input float bit; GGUF ne dimensions reverse C-order axes."""
    if not 1 <= len(tensors) <= 128:
        raise ValueError('Bounded nonempty F32 adapter required')
    names, rows, offset = set(), [], 0
    for item in tensors:
        name = item['name']
        if not re.fullmatch('[A-Za-z0-9_.]+', name) or name in names:
            raise ValueError('Unsafe/duplicate adapter tensor name')
        names.add(name)
        data, dims = source_bytes(item)
        rows.append((name, data, dims, offset, item.get('label', Path(item['path']).name)))
        offset = (offset + len(data) + 31) // 32 * 32
    if offset > 256 * 1024**2:
        raise ValueError('Adapter payload bound exceeded')
    header = b'GGUF' + struct.pack('<IQQ', 3, len(rows), 0)
    for name, data, dims, start, _ in rows:
        header += (_string(name) + struct.pack('<I', len(dims)) +
                   struct.pack('<' + 'Q' * len(dims), *reversed(dims)) + struct.pack('<IQ', 0, start))
    data_start = (len(header) + 31) // 32 * 32
    path = Path(path)
    with path.open('xb') as stream:
        stream.write(header)
        stream.write(b'\0' * (data_start - len(header)))
        for _, data, _, start, _ in rows:
            stream.write(b'\0' * (data_start + start - stream.tell()))
            stream.write(data)
    actual = read_gguf(path)
    expected = {name: {'shape': list(dims), 'bytes': len(data), 'sha256': sha(data)} for name, data, dims, _, _ in rows}
    if actual != expected:
        raise ValueError('Lossless adapter readback differs')
    return {'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes()), 'tensors': {
        name: {**expected[name], 'source': label} for name, _, _, _, label in rows}}


def read_gguf(path):
    """Bounded independent readback of this metadata-free F32 test format."""
    data = Path(path).read_bytes()
    if len(data) < 24 or data[:4] != b'GGUF':
        raise ValueError('Invalid test adapter header')
    version, count, metadata = struct.unpack_from('<IQQ', data, 4)
    if version != 3 or not 1 <= count <= 128 or metadata:
        raise ValueError('Not the declared test F32 adapter')
    at, records = 24, []
    for _ in range(count):
        length = struct.unpack_from('<Q', data, at)[0]; at += 8
        if not 1 <= length <= 256:
            raise ValueError('Invalid test tensor name')
        name = data[at:at+length].decode('ascii'); at += length
        rank = struct.unpack_from('<I', data, at)[0]; at += 4
        if not 1 <= rank <= 4:
            raise ValueError('Invalid test tensor rank')
        ne = struct.unpack_from('<' + 'Q' * rank, data, at); at += 8 * rank
        kind, offset = struct.unpack_from('<IQ', data, at); at += 12
        dims = shape(tuple(reversed(ne)))
        if kind != 0 or offset % 32:
            raise ValueError('Only aligned lossless F32 accepted')
        records.append((name, dims, offset))
    start, result = (at + 31) // 32 * 32, {}
    ranges = []
    for name, dims, offset in records:
        end = start + offset + math.prod(dims) * 4
        if name in result or end > len(data):
            raise ValueError('Duplicate/truncated F32 adapter')
        ranges.append((start+offset,end))
        payload = data[start+offset:end]
        result[name] = {'shape': list(dims), 'bytes': len(payload), 'sha256': sha(payload)}
    ordered = sorted(ranges)
    if any(b[0] < a[1] for a,b in zip(ordered, ordered[1:])) or ordered[-1][1] != len(data):
        raise ValueError('Overlapping/extra test adapter bytes')
    return result


def write_pair(original, native, pairs):
    if any(shape(p['original']['shape']) != shape(p['native']['shape']) for p in pairs):
        raise ValueError('Original/native shapes differ; no reshape adapter allowed')
    return {'original': write_gguf(original, [{'name':p['name'], **p['original']} for p in pairs]),
            'native': write_gguf(native, [{'name':p['name'], **p['native']} for p in pairs])}


def collect(returncode, path, independent):
    """Return numeric failures as records. Structural/coverage errors reject."""
    if returncode not in (0, 1):
        raise ValueError('crispembed-diff structural/process rejection: ' + str(returncode))
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if (report.get('schema') != 'crispembed-josev-backbone-ref-v1' or
            report.get('max_absolute_budget') != ABSOLUTE or report.get('minimum_row_cosine') != COSINE):
        raise ValueError('Comparator schema/fixed budgets differ')
    if report.get('status') != ('passed' if returncode == 0 else 'numeric_rejection'):
        raise ValueError('Comparator exit/status mismatch')
    rows = report.get('stages', [])
    if len(rows) != len(independent) or {r['name'] for r in rows} != set(independent):
        raise ValueError('Comparator stage coverage differs')
    numeric_failure = False
    for row in rows:
        name = row['name']; measured = independent[name]
        if (row.get('elements') != measured['elements'] or row.get('row_axis') != 0 or
                list(reversed(row.get('ggml_shape', []))) != measured['shape']):
            raise ValueError('Actual Ref shape/row axis/element count differs')
        if row.get('found') is not True or any(not math.isfinite(row[k]) for k in FIELDS):
            raise ValueError('Missing/nonfinite actual Ref metric')
        gate = row['max_absolute'] <= ABSOLUTE and row['min_row_cosine'] >= COSINE
        if row.get('passed') is not gate:
            raise ValueError('Actual Ref budget decision differs from fixed gates')
        agreement = {}
        for key in FIELDS:
            distance = abs(row[key] - measured[key])
            limit = METRIC_RELATIVE * max(1., abs(row[key]), abs(measured[key]))
            agreement[key] = {'absolute_difference': distance, 'limit': limit, 'passed': distance <= limit}
        agree = all(v['passed'] for v in agreement.values())
        measured.update(independent_passed=measured['passed'], cpp_ref=row, metric_agreement=agreement,
                        passed=measured['passed'] and gate and agree)
        numeric_failure |= not gate
    if numeric_failure != (returncode == 1):
        raise ValueError('Comparator numeric exit code differs from its records')
    return {'exit_code':returncode,'status':report['status'],'stages':len(rows),
            'all_metrics_agree':all(all(v['passed'] for v in independent[r['name']]['metric_agreement'].values()) for r in rows)}
