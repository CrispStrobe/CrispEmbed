#!/usr/bin/env python3
"""Hosted-only MIXED padded-Q4_K / Q8_0 / F32 MoEdern packages (portfolio Q4_K lane).

A recipe starts from the Q8_0 baseline (every engine-quantizable matmul weight
Q8_0, embeddings/router/norms/conv kernels/dense head F32) and lowers named
tensor groups to real Q4_K with explicit zero-padded storage columns, exactly as
the plain padded profile (quantize_q4k_padded.py) does. Groups may also be raised
to F32. Only numeric block encoding is delegated to the production C quantizer;
every output type, shape, byte extent and F32 payload is read back.

Loading needs CRISPEMBED_MOEDERN_Q4_K_MIXED=1 plus the padded Q4_K gates
(Q4_K_PADDED, QUANT, FAST, FAMILY, ...). Recipes are named policies below or an
ad-hoc grammar `q8|q4` followed by `+group` (Q4_K), `~group` (Q8_0) or `-group`
(F32) terms applied left to right, used by the sensitivity sweep. `q4` lowers all.

  quantize_q4k_mixed.py --input F32.gguf --output OUT.gguf --variant 115M \
      --recipe moedern115m_q4k_v1 --quantizer crispembed-quantize --report r.json
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_gguf import quantizable  # noqa: E402
from quantize_q4k_padded import PROFILES, publication, sha_file, BLOCK, TYPE_BYTES  # noqa: E402

RECIPE = 'q4_k_padded_mixed_v1'
FORMAT = 'crispembed-moedern-q4k-padded-v1'
SUFFIX_GROUPS = {
    'attn_q': ('attn_q.weight',), 'attn_k': ('attn_k.weight',), 'attn_v': ('attn_v.weight',),
    'attn_o': ('attn_output.weight',),
    'attn': ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight'),
    'exps_up': ('ffn_up_exps.weight',), 'exps_gate': ('ffn_gate_exps.weight',),
    'exps_down': ('ffn_down_exps.weight',),
    'exps': ('ffn_up_exps.weight', 'ffn_gate_exps.weight', 'ffn_down_exps.weight'),
    'shexp': ('ffn_up_shexp.weight', 'ffn_gate_shexp.weight', 'ffn_down_shexp.weight'),
    'ffn_down': ('ffn_down_exps.weight', 'ffn_down_shexp.weight'),
    'conv': ('conv_in.weight', 'conv_out.weight'),
    'all': ('',),
}
# Named opt-in policies = measured winners (docs/moedern-q4k-mixed.md). Each is a
# recipe string in the sweep grammar, frozen per variant; never a default.
POLICIES = {
    # 115M: every engine-quantizable matmul Q4_K (padded), embeddings/router/norms/
    # conv kernels/dense head F32. Same tensor payloads as the plain padded profile;
    # measured final cos min 0.9793 (hosted CPU, portable) vs Q8_0 0.9670.
    'moedern115m_q4k_v1': {'variant': '115M', 'recipe': 'q4'},
    # 27M: Q8_0 base, routed-expert up/down projections and attn_q Q4_K (padded
    # 640->768); gate experts, shared experts and attn_k/v/o Q8_0; criticals F32.
    # Final cos min 0.9685 (x86) / 0.9688 (ARM) where plain Q4_K gives 0.9250 / 0.9219.
    'moedern27m_q4k_v1': {'variant': '27M', 'recipe': 'q8+exps_up+exps_down+attn_q'},
}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def group_members(group, name):
    m = re.fullmatch(r'L(\d+)(?:-(\d+))?', group)
    if m:
        lo, hi = int(m[1]), int(m[2] or m[1])
        layer = re.match(r'blk\.(\d+)\.', name)
        return bool(layer) and lo <= int(layer[1]) <= hi
    require(group in SUFFIX_GROUPS, 'unknown recipe group: ' + group)
    return any(name.endswith('.' + s) if s else True for s in SUFFIX_GROUPS[group])


def resolve(recipe, variant):
    if recipe in POLICIES:
        require(POLICIES[recipe]['variant'] == variant, 'policy belongs to another variant')
        return recipe, POLICIES[recipe]['recipe']
    return recipe, recipe


def assign(recipe, names, variant):
    """Return {name: 'Q4_K'|'Q8_0'|'F32'} for the engine-quantizable weights."""
    _, spec = resolve(recipe, variant)
    require(re.fullmatch(r'(q8|q4)([+~-][A-Za-z0-9_]+(-\d+)?)*', spec) is not None, 'invalid recipe: ' + spec)
    base = 'Q4_K' if spec.startswith('q4') else 'Q8_0'
    types = {n: base for n in names if quantizable(n)}
    for sign, group in re.findall(r'([+~-])([A-Za-z0-9_]+(?:-\d+)?)', spec[2:]):
        hit = [n for n in types if group_members(group, n)]
        require(hit, 'recipe group selects nothing: ' + group)
        for n in hit:
            types[n] = {'+': 'Q4_K', '~': 'Q8_0', '-': 'F32'}[sign]
    require(any(t == 'Q4_K' for t in types.values()), 'mixed recipe needs at least one Q4_K tensor')
    return types


def quantize(source, output, variant, quantizer, recipe):
    kaggle = (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir()
              and os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Batch', 'Interactive'))
    require(os.environ.get('GITHUB_ACTIONS') == 'true' or kaggle, 'hosted GHA/Kaggle execution required')
    require(variant in PROFILES, 'explicit supported variant required')
    source, output, quantizer = Path(source), Path(output), Path(quantizer)
    cfg = PROFILES[variant]
    pin = publication(variant)
    require(source.is_file() and source.stat().st_size == pin['bytes'] and sha_file(source) == pin['sha256'],
            'immutable published F32 input differs')
    require(quantizer.is_file() and not output.exists(), 'existing quantizer/fresh output required')
    import numpy as np
    import gguf
    reader = gguf.GGUFReader(source)
    field = lambda n: reader.fields[n].contents()  # noqa: E731
    require(field('general.architecture') == 'modernbert_hash_moe_v10'
            and field('modernbert_hash_moe_v10.source_revision') == cfg['revision']
            and field('moedern.packed.format') == cfg['format'], 'source profile differs')
    require(len(reader.tensors) == cfg['tensors']
            and all(t.tensor_type == gguf.GGMLQuantizationType.F32 for t in reader.tensors), 'exact F32 inventory required')
    policy, spec = resolve(recipe, variant)
    types = assign(recipe, [t.name for t in reader.tensors], variant)
    rows = []
    for t in reader.tensors:
        shape = list(map(int, t.shape))
        kind = types.get(t.name, 'F32')
        logical = shape[0]
        stored = (logical + 255) // 256 * 256 if kind == 'Q4_K' else logical
        require(kind != 'Q4_K' or stored // BLOCK * TYPE_BYTES < logical * 4, 'Q4_K must compress: ' + t.name)
        require(kind != 'Q8_0' or logical % 32 == 0, 'Q8_0 needs 32-aligned rows: ' + t.name)
        n_rows = math.prod(shape[1:])
        nbytes = {'F32': n_rows * logical * 4, 'Q8_0': n_rows * logical // 32 * 34,
                  'Q4_K': n_rows * stored // BLOCK * TYPE_BYTES}[kind]
        rows.append(dict(name=t.name, logical_shape=shape, storage_shape=[stored, *shape[1:]], type=kind,
                         bytes=nbytes, original_bytes=math.prod(shape) * 4))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.padded-f32.gguf')
    require(not temporary.exists(), 'fresh padded scratch required')
    writer = gguf.GGUFWriter(str(temporary), arch='modernbert_hash_moe_v10', use_temp_file=True)
    for f in reader.fields.values():
        if f.name.startswith('GGUF.') or f.name in ('general.architecture', 'moedern.packed.format',
                                                    'modernbert_hash_moe_v10.quant_recipe'):
            continue
        kind = f.types[0]
        sub = f.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(f.name, f.contents(), kind, sub_type=sub)
    writer.add_string('moedern.packed.format', FORMAT)
    writer.add_string('modernbert_hash_moe_v10.quant_recipe', RECIPE)
    writer.add_string('moedern.q4k.variant', variant)
    writer.add_string('moedern.q4k.format', 'q4_k-padded-columns-v1')
    writer.add_string('moedern.q4k.source_sha256', pin['sha256'])
    writer.add_string('moedern.q4k.source_format', cfg['format'])
    writer.add_string('moedern.q4k.policy', policy[:64])
    writer.add_string('moedern.q4k.recipe_spec', spec)
    selected = [r for r in rows if r['type'] == 'Q4_K']
    writer.add_array('moedern.q4k.names', [r['name'] for r in selected])
    for key, index in [('logical_columns', 'logical_shape'), ('storage_columns', 'storage_shape')]:
        writer.add_key_value('moedern.q4k.' + key, [r[index][0] for r in selected], gguf.GGUFValueType.ARRAY,
                             sub_type=gguf.GGUFValueType.INT32)
    for t, row in zip(reader.tensors, rows):
        a = np.asarray(t.data, dtype='<f4').reshape(tuple(reversed(row['logical_shape'])))
        row['original_f32_sha256'] = hashlib.sha256(a.tobytes()).hexdigest()
        pad = row['storage_shape'][0] - row['logical_shape'][0]
        if pad:
            a = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, pad)], mode='constant', constant_values=0)
        writer.add_tensor(t.name, a, raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    args = [str(quantizer), str(temporary), str(output), 'q4_k']
    for row in rows:
        if row['type'] != 'Q4_K':
            args += ['--tensor-type', row['name'] + '=' + row['type'].lower()]
    log = output.with_suffix(output.suffix + '.quantize.log')
    with log.open('w') as stream:
        subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900,
                       env={k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')})
    require('(fallback ' not in log.read_text(), 'generic fallback occurred; cannot claim Q4_K')
    result = gguf.GGUFReader(output)
    actual = {t.name: t for t in result.tensors}
    require(set(actual) == {r['name'] for r in rows}, 'output names differ')
    code = {'F32': gguf.GGMLQuantizationType.F32, 'Q8_0': gguf.GGMLQuantizationType.Q8_0,
            'Q4_K': gguf.GGMLQuantizationType.Q4_K}
    source_by_name = {t.name: t for t in reader.tensors}
    for row in rows:
        t = actual[row['name']]
        require(t.tensor_type == code[row['type']] and list(map(int, t.shape)) == row['storage_shape']
                and t.data.nbytes == row['bytes'], 'actual output type/shape/bytes differ: ' + row['name'])
        payload = hashlib.sha256(t.data.tobytes()).hexdigest()
        if row['type'] == 'F32':
            require(payload == row['original_f32_sha256'], 'critical F32 payload changed: ' + row['name'])
            continue
        decoded = gguf.quants.dequantize(t.data, t.tensor_type).reshape(tuple(reversed(row['storage_shape'])))
        a = np.asarray(source_by_name[row['name']].data, dtype='<f4').reshape(tuple(reversed(row['logical_shape'])))
        require(np.isfinite(decoded).all(), 'nonfinite dequantized weight')
        row['max_absolute_weight_error'] = float(np.max(np.abs(decoded[..., :row['logical_shape'][0]] - a)))
    counts = {k: sum(r['type'] == k for r in rows) for k in ('F32', 'Q8_0', 'Q4_K')}
    return dict(schema='crispembed-moedern-q4k-mixed-v1', variant=variant, policy=policy, recipe_spec=spec,
                completed=True, source_sha256=pin['sha256'], output_sha256=sha_file(output),
                output_bytes=output.stat().st_size, counts=counts,
                bytes_by_type={k: sum(r['bytes'] for r in rows if r['type'] == k) for k in counts},
                tensors=rows, quantizer_sha256=sha_file(quantizer), quantizer_log_sha256=sha_file(log),
                semantic_widths_unchanged=True, quantized_publication_qualified=False)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('input', 'output', 'quantizer', 'report', 'recipe'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--variant', choices=list(PROFILES), required=True)
    a = p.parse_args()
    Path(a.report).write_text(json.dumps(quantize(a.input, a.output, a.variant, a.quantizer, a.recipe), indent=2) + '\n')
