"""Frozen single-sentence F32-boundary attribution. No model qualification."""
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
BASE = '580e3ae3be3671ddce5e945b0725d0583ec34110'
GGML = 'a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
TRACE_RUN = 40424064413
TRACE_SOURCE = 'dfd7bb4f35d3460f1ea67f6f5ce35cc408e2e6f6'
TRACE_REPORT = 'da8767490c953a6dbf544fc3b1a22b8aabb058be3f46dd7527c5cd3ddde1782d'
REVIEW = dict(id=12702046203, bytes=1031136, sha256='4847ab1b9f2f10680e9647bc19fc462bbbea212379756fffd896bba9cd122e05')
RAW = dict(id=12702610607, bytes=24900090, sha256='1444edaf1e3751ee7f0c6d5a05faef0af16f476b2c8fce4bd7785ec950e52721')
MODEL = '77087a5dd455f5eeae6d6d098bac82793610438518a91071c72c253940db18a2'
AXES = ('native', 'wasm0', 'wasm1')
STAGES = {'native': 'native/1/capture/stages.gguf', 'wasm0': 'browser/arrays/wasm0-stages.gguf',
          'wasm1': 'browser/arrays/wasm1-stages.gguf'}
# op, raw input stage, optional second/third stage or exact F32 bias/scale, output stage.
# Prefix w: identifies frozen model data, never a newly estimated coefficient.
OPERATORS = {
    'q_bias': (2, 'q_raw', 'w:attn_q.bias', None, 'q'),
    'k_bias': (2, 'k_raw', 'w:attn_k.bias', None, 'k'),
    'v_bias': (2, 'v_raw', 'w:attn_v.bias', None, 'v'),
    'flash': (4, 'q', 'k', 'v', 'flash'),
    'o_bias': (2, 'o_raw', 'w:attn_output.bias', None, 'o'),
    'attention_residual': (2, 'input', 'o', None, 'attention_residual'),
    'attention_norm': (0, 'attention_residual', None, None, 'attention_norm'),
    'attention_mul': (1, 'attention_norm', 'w:attn_output_norm.weight', None, 'attention_scaled'),
    'attention_bias': (2, 'attention_scaled', 'w:attn_output_norm.bias', None, 'attention_output'),
    'fc1_bias': (2, 'fc1_raw', 'w:ffn_up.bias', None, 'fc1'),
    'gelu_erf': (3, 'fc1', None, None, 'gelu'),
    'fc2_bias': (2, 'fc2_raw', 'w:ffn_down.bias', None, 'fc2'),
    'ffn_residual': (2, 'attention_output', 'fc2', None, 'ffn_residual'),
    'ffn_norm': (0, 'ffn_residual', None, None, 'ffn_norm'),
    'ffn_mul': (1, 'ffn_norm', 'w:layer_output_norm.weight', None, 'ffn_scaled'),
    'ffn_bias': (2, 'ffn_scaled', 'w:layer_output_norm.bias', None, 'output'),
}
WEIGHTS = {x[2:] for row in OPERATORS.values() for x in row[1:4] if x and x.startswith('w:')}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_zip(path, maximum):
    import zipfile
    # ZIP directories are harmless but are never extracted. Every actual file
    # must be authenticated by the frozen producer manifest below.
    with zipfile.ZipFile(path) as archive:
        records = archive.infolist()
        require(len(records) <= 600 and sum(row.file_size for row in records) <= maximum, 'ZIP budget exceeded')
        result = {}
        for row in records:
            if row.is_dir():
                continue
            name = row.filename
            require(not row.flag_bits & 1 and ((row.external_attr >> 16) & 0o170000) != 0o120000,
                    'Encrypted/symlink ZIP member')
            require(not name.startswith('/') and '\\' not in name and all(x not in ('', '.', '..') for x in name.split('/')),
                    'Unsafe ZIP name')
            require(name not in result, 'Duplicate ZIP member')
            result[name] = archive.read(row)
        return result


def authenticate_files(files, declared, binaries):
    expected = {**declared, **binaries}
    # upload-artifact follows these two SONAME symlinks if present. Their bytes
    # must equal the one already authenticated library, not an extra binary.
    aliases = {'native-build/libcrispembed.so.0', 'native-build/libcrispembed.so.0.17.14'}
    extra = set(files) - set(expected)
    require(extra <= aliases, 'Unexpected old raw ZIP coverage')
    for name in extra:
        expected[name] = binaries['native-build/libcrispembed.so']
    require(set(files) == set(expected), 'Missing old raw ZIP member')
    for name, row in expected.items():
        require({'bytes': len(files[name]), 'sha256': sha(files[name])} == row, 'Old raw member hash differs: ' + name)
    return dict(members=len(expected), aliases=sorted(extra), all_sha256_exact=True)


def own_anchors(packets, outputs):
    require(set(outputs) == {'native1', 'native2', 'wasm'}, 'Actual backend/thread coverage differs')
    for backend, rows in outputs.items():
        require(set(rows) == set(packets), 'Missing/extra replay output: ' + backend)
        for key, packet in packets.items():
            require(rows[key]['input_hashes'] == {k: r['sha256'] for k, r in packet['inputs'].items()} and
                    rows[key]['shape'] == [12, packet['width']] and rows[key]['bytes'] == 48 * packet['width'] and
                    rows[key]['repeat_bit_exact'] is True, 'Actual input/shape/repeat differs: ' + backend + '/' + key)
    checked = []
    for backend, axes in [('native1', ['native']), ('native2', ['native']), ('wasm', ['wasm0', 'wasm1'])]:
        for axis in axes:
            for name in OPERATORS:
                key = axis + '/' + name
                expected = packets[key]['expected_sha256']
                require(key in outputs[backend] and outputs[backend][key]['sha256'] == expected,
                        'Own-input/output bit anchor failed before causal comparison: ' + backend + '/' + key)
                checked.append(backend + '/' + key)
    return dict(passed=True, exact_own_input_output_anchors=len(checked), keys=checked,
                cross_backend_interpretation_authorized=True, full_model_qualified=False)


def epsilon(meta):
    # Exact source580 ordered opt_f32 keys for architecture bert. Metadata does
    # not get to introduce a different epsilon without a new protocol.
    keys = ['bert.layer_norm_eps', 'bert.attention.layer_norm_epsilon', 'xlmr.attention.layer_norm_epsilon']
    present = next((key for key in keys if key in meta), None)
    value = meta[present] if present else 1e-12
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            struct.pack('<f', value) == struct.pack('<f', 1e-12), 'Actual model norm epsilon differs')
    return dict(value=struct.unpack('<f', struct.pack('<f', value))[0], key=present,
                source='model metadata' if present else 'source580 opt_f32 default1e-12f')
