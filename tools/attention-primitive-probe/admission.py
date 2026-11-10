"""Closed source-fed attention input admission; no model or network execution."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate(p):
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    require(p['schema'] == 'lighton4b-attention-source-fed-primitives/v1', 'schema')
    require(p['physical_source'] == 'ac34f72ca89823a177c538ad598dfa00fc69f124', 'physical source')
    require(p['physical_report_sha256'] == '712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2', 'report')
    require(p['upstream_commit'] == '6c01da6278ab6c69ed67d9124806b50d2b299ba7', 'upstream')
    require(p['native_ggml_commit'] == '3a27b6c665b191cb230be367db5c52d81dd3e2ac', 'native ggml')
    require(p['candidate_sha256'] == '843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43', 'candidate')
    require(p['layer'] == 3 and p['tokens'] == [4, 16], 'probe scope')
    require((p['query_heads'], p['kv_heads'], p['head_dim']) == (16, 4, 256), 'geometry')
    require((p['context'], p['visible_kv_padded'], p['threads']) == (2048, 256, 4), 'execution profile')
    require(p['flash_attention'] is False and p['model_forward_allowed'] is False, 'model prohibition')
    require(p['cache_k_type'] == p['cache_v_type'] == 'f16', 'cache precision')
    require(p['total_timeout_seconds'] == 1800, 'deadline')
    require(p['minimum_memory_bytes'] == 2 << 30 and p['minimum_disk_bytes'] == 4 << 30, 'resource guards')
    rows = p['tensors']
    require(len(rows) == 11 and len({r['name'] for r in rows}) == 11, 'tensor census')
    spans = []
    for r in rows:
        require(r['name'].startswith('l.layers.3.') and r['physical_name'].startswith('blk.3.'), 'layer boundary')
        shape = r['stored_shape']
        require(len(shape) in (1, 2) and all(isinstance(x, int) and x > 0 for x in shape), 'shape')
        kind = r['type']
        require(kind in (0, 12, 14), 'weight precision')
        width = shape[0]
        count = shape[1] if len(shape) == 2 else 1
        require(kind == 0 or width % 256 == 0, 'block alignment')
        size = width * count * 4 if kind == 0 else width // 256 * count * {12: 144, 14: 210}[kind]
        require(r['bytes'] == size and r['source_offset'] >= 0 and r['source_offset'] + size <= p['source_bytes'], 'tensor extent')
        require(len(r['source_packed_sha256']) == 64 and all(c in '0123456789abcdef' for c in r['source_packed_sha256']), 'tensor hash')
        spans.append((r['source_offset'], r['source_offset'] + size))
    spans.sort()
    require(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), 'overlapping tensors')
    captures = p['actual_captures']
    require(set(captures) == {'upstream_features/prefill.layer.2.f32', 'upstream_features/prefill.layer.3.f32', 'upstream_features/prefill.positions.f32'}, 'capture scope')
    require(sorted(x['bytes'] for x in captures.values()) == [1848, 1576960, 1576960], 'capture extent')
    return p


def admit_capture_files(protocol, case_root):
    """Authenticate whole captures before slicing; reject path escape or drift."""
    validate(protocol)
    root = Path(case_root).resolve()
    for name, expected in protocol['actual_captures'].items():
        path = (root / name).resolve()
        if root not in path.parents or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('actual capture drift: ' + name)
    return True


def load(path):
    return validate(json.loads(Path(path).read_text()))
