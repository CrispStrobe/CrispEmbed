#!/usr/bin/env python3
"""Compare retained fused-ON prefill and same-prefix logits; never emits text/IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

SOURCE_SHA = '4a9776508758866dbdcf02f41dfec94844230c7e215f72ef1061142834992dc4'
NATIVE_SHA = 'c723dd349d5e3bdce449d2c0fa5bf8daffb34ab1be24eb2a58a6eaa9a21250f1'
STAGES = [('attn_norm-0', 'input_norm', 2560, None),
          ('linear_attn_qkv_mixed-0', 'qkv', 8192, 'qkv'),
          ('z-0', 'z', 4096, 'z'), ('alpha-0', 'a', 32, 'a'), ('beta-0', 'b', 32, 'b'),
          ('linear_attn_out-0', 'attention', 2560, None),
          ('attn_residual-0', 'attention_residual', 2560, None),
          ('attn_post_norm-0', 'post_norm', 2560, None),
          ('ffn_gate-0', 'gate', 9216, None), ('ffn_up-0', 'up', 9216, None),
          ('ffn_swiglu-0', 'activation', 9216, None), ('ffn_out-0', 'down', 2560, None)]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def owned(root, name, manifest):
    root = Path(root).resolve()
    p = (root / name).resolve()
    require(p.is_relative_to(root) and name in manifest, 'unowned artifact')
    seal = manifest[name]
    require(p.is_file() and p.stat().st_size == seal['bytes'] and digest(p) == seal['sha256'], 'artifact seal mismatch')
    return p


def delta(candidate, reference):
    import numpy as np
    require(candidate.shape == reference.shape and candidate.size > 0 and np.isfinite(candidate).all() and np.isfinite(reference).all(), 'array contract')
    x, y = candidate.astype('float64'), reference.astype('float64')
    d = x-y
    return dict(elements=int(x.size), bit_exact=candidate.tobytes() == reference.tobytes(),
                error_rms=float(np.sqrt(np.mean(d*d))), error_max_abs=float(np.abs(d).max()),
                native_rms=float(np.sqrt(np.mean(x*x))), source_rms=float(np.sqrt(np.mean(y*y))),
                native_max_abs=float(np.abs(x).max()), source_max_abs=float(np.abs(y).max()))


def analyze(source_path, native_path):
    import numpy as np
    require(digest(source_path) == SOURCE_SHA and digest(native_path) == NATIVE_SHA, 'exact frozen reports')
    s, n = [json.loads(Path(p).read_text()) for p in [source_path, native_path]]
    require(s['source_sha'] == 'e3164d09de4578370082e42c5954a77a6cd643e3' and s['all654_scores_observer_neutral'], 'source neutrality')
    require(n['status'] == 'failed' and n['phase'] == 'inventory-table-preflight' and n['error'] == 'per-arm resources/seal', 'immutable partial negative')
    require(n['candidate_source'] == '519c3d9765c234cd93e9069d5efdc239e3cad3f2', 'fixed native source')
    sr, nr = Path(source_path).parent, Path(native_path).parent
    def source(g, name):
        rows = [x for x in s['graph_indexes'][g]['records'] if x['name'] == name]
        require(len(rows) == 1, 'unique graph record')
        r = rows[0]
        if r['status'] == 'zero-sized':
            return None
        require(r['status'] == 'captured' and r['sample_count'] == '0' and r['type'] in ['0', '26'], 'source record contract')
        shape = [int(r['ne%d' % i]) for i in range(4)]
        stride = 4
        for axis in range(4):
            require(shape[axis] <= 1 or int(r['nb%d'%axis]) == stride, 'source contiguous layout')
            stride *= shape[axis]
        p = owned(sr, 'on/'+r['file'], s['artifacts'])
        require(p.stat().st_size == stride == int(r['bytes']), 'source dimensions')
        return np.fromfile(p, dtype='<f4' if r['type'] == '0' else '<i4')
    def native(suffix, shape=None):
        p = owned(nr, 'two-column-page/on/'+suffix, n['artifacts'])
        a = np.fromfile(p, dtype='<f4')
        return a.reshape(shape) if shape else a
    graphs = [1, 2, 3, 4]
    counts = [int(s['graph_indexes'][g]['records'][0]['sample_count']) for g in graphs]
    require(counts == [0]*4, 'prefill only')
    embeds = np.concatenate([source(g, 'model.input_embed').reshape(-1, 2560) for g in graphs])
    require(embeds.shape == (266, 2560), 'exact prompt geometry')
    positions = np.concatenate([source(g, 'inp_pos').reshape(4, -1)[:3] for g in graphs], axis=1).astype('<f4')
    require(embeds.tobytes() == native('prefill.input.f32', (266, 2560)).tobytes(), 'same actual embeddings')
    require(positions.tobytes() == native('prefill.positions.f32', (3, 266)).tobytes(), 'same actual positions')
    case = n['cases']['two-column-page'];arm = case['arms']['on']
    require(arm['summary']['prompt_tokens'] == 266 and arm['summary']['image_tokens'] == 252, 'sealed prompt dimensions')
    comparisons = []
    for name, suffix, width, role in STAGES:
        ref = source(1, name).reshape(4, width)
        head = lambda h: 16*(h % 2) + h//2
        if role == 'qkv':
            index = list(range(4096))+[4096+128*head(i//128)+i%128 for i in range(4096)]
        elif role == 'z':
            index = [128*head(i//128)+i%128 for i in range(width)]
        elif role in ['a', 'b']:
            index = [head(i) for i in range(width)]
        else:
            index = list(range(width))
        comparisons.append(dict(scope='first4_layer0_operators', stage=name, **delta(native('prefill.first_chunk.layer0.'+suffix+'.f32', (4, width)), ref[:, index])))
    for layer in range(32):
        candidate = native('prefill.layer.%d.f32'%layer, (266, 2560));offset=0
        for g, count in zip(graphs, [4, 252, 6, 4]):
            ref = source(g, 'l_out-%d'%layer)
            if ref is not None:
                ref = ref.reshape(-1, 2560)
                indices = np.arange(count) if ref.shape[0] == count else source(g, 'out_ids')
                require(indices.size == ref.shape[0] and np.all((0 <= indices)&(indices < count)), 'output crop contract')
                comparisons.append(dict(scope='prefill_layers', layer=layer, graph=g, **delta(candidate[offset+indices], ref)))
            offset += count
    sid = s['arms']['on']['generated_ids'];nid = arm['ids']
    first = next((i for i, (a,b) in enumerate(zip(sid,nid)) if a != b), min(len(sid),len(nid)))
    require(first == 85, 'frozen first difference')
    archive = owned(nr, 'two-column-page/on/greedy-raw-rows.zip', n['artifacts'])
    logits=[]
    with zipfile.ZipFile(archive) as z:
        for pos in [0, 84, 85]:
            row = arm['raw_rows']['rows'][pos];name='greedy.%d.logits.f32'%pos
            require(z.getinfo(name).file_size == row['bytes'] == 248320*4, 'row dimensions')
            raw=z.read(name);require(hashlib.sha256(raw).hexdigest() == row['sha256'], 'row SHA')
            c=np.frombuffer(raw,dtype='<f4')
            refpath=owned(sr,'on/source.logits.%d.f32'%pos,s['artifacts']);r=np.fromfile(refpath,dtype='<f4')
            ci,ri=int(c.argmax()),int(r.argmax())
            require(ci == nid[pos] and ri == sid[pos], 'actual greedy argmax closure')
            # Export margins only; private vocabulary IDs never enter the report.
            logits.append(dict(position=pos, same_input_prefix=True, argmax_same=ci==ri,
                source_margin=float(r[ri]-r[ci]), candidate_margin=float(c[ci]-c[ri]), **delta(c,r)))
            if pos == 0:
                require(raw == native('prefill.logits.f32').tobytes() == native('teacher.0.logits.f32').tobytes(), 'firstscore triple closure')
    return dict(schema='lighton4b-post-fused-source-trace/v1', status='measured-artifact-only',
        source_receipt_sha256=SOURCE_SHA, native_receipt_sha256=NATIVE_SHA,
        inference_executed=False, qualified=False, text_or_ids_emitted=False,
        same_prefill_embeddings_bits=True, same_positions_bits=True, prompt_tokens=266,
        prefill_graph_rows=[4,252,6,4], head_permutation='physical new-to-old inversion',
        comparisons=comparisons,
        first_recorded_operator_difference=next((x for x in comparisons if x['scope']=='first4_layer0_operators' and not x['bit_exact']),None),
        first_recorded_layer_difference=next((x for x in comparisons if x['scope']=='prefill_layers' and not x['bit_exact']),None),
        same_prefix_logits=logits, first_different_generated_position=first,
        native_generated_tokens=len(nid), native_eos=arm['summary']['eos'], source_generated_tokens=len(sid),source_eos=s['arms']['on']['eos'],
        limits=['Source image-layer0 operator and recurrent-cache captures unavailable; first differing hidden boundary is not unique operator attribution.',
                'No native intermediate capture at decode85; same-prefix logits establish scoring divergence without localizing that decode operator.',
                'First4 exact fused SwiGLU does not prove all later layers/chunks or decoded quality.',
                'Partial producer failure and capped negative book output remain unchanged.'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--native',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=analyze(a.source,a.native);a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
