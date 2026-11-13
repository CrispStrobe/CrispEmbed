#!/usr/bin/env python3
"""Fail-closed retained source/native405 comparator; no IDs or text in output."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import zipfile

SOURCE_REPORT = '4a9776508758866dbdcf02f41dfec94844230c7e215f72ef1061142834992dc4'
HISTORICAL = 'd2ea232376c2fa4697613d615d33b8b10a4dc0d5ab987fdac7d1ed3d99561760'
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


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def owned(root, name, manifest):
    root = Path(root).resolve()
    path = (root / name).resolve()
    require(path.is_relative_to(root) and name in manifest, 'unowned/escaping artifact')
    record = manifest[name]
    require(path.is_file() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], 'artifact hash/size mismatch')
    return path


def delta(candidate, reference):
    import numpy as np
    require(candidate.shape == reference.shape and candidate.size > 0 and np.isfinite(candidate).all() and np.isfinite(reference).all(), 'incompatible/nonfinite arrays')
    x, y = candidate.astype('float64'), reference.astype('float64')
    difference = x - y
    return dict(elements=int(x.size), bit_exact=candidate.tobytes() == reference.tobytes(),
                error_rms=float(np.sqrt(np.mean(difference*difference))), error_max_abs=float(np.abs(difference).max()),
                native_rms=float(np.sqrt(np.mean(x*x))), source_rms=float(np.sqrt(np.mean(y*y))),
                native_max_abs=float(np.abs(x).max()), source_max_abs=float(np.abs(y).max()))


def permutation(role, width):
    # Existing authenticated physical new-to-old head contract, inverted to HF/grouped order.
    def head(h):
        return 16*(h % 2) + h//2
    if role == 'qkv':
        return list(range(4096)) + [4096 + 128*head(i//128) + i % 128 for i in range(4096)]
    if role == 'z':
        return [128*head(i//128) + i % 128 for i in range(width)]
    if role in ['a', 'b']:
        return [head(i) for i in range(width)]
    return list(range(width))


def read_graphs(root, receipt):
    import numpy as np
    graphs, total = [], 0
    for number, index in enumerate(receipt['graph_indexes']):
        require(index['file'] == 'graph-%d.tsv' % number, 'contiguous graph inventory')
        path = owned(root, 'on/' + index['file'], receipt['artifacts'])
        records = list(csv.DictReader(path.open(), delimiter='\t'))
        require(records == index['records'] and len(records) == 55 and len({r['name'] for r in records}) == 55, 'sealed complete graph index')
        arrays, metadata = {}, {}
        for row in records:
            name = row['name']
            require(int(row['sample_count']) in [0, 405], 'unexpected phase')
            ne = [int(row['ne%d' % i]) for i in range(4)]
            require(all(x >= 0 for x in ne), 'negative shape')
            metadata[name] = ne
            if row['status'] == 'zero-sized':
                require(math.prod(ne) == 0 and int(row['bytes']) == 0, 'invalid zero-sized record')
                continue
            require(row['status'] == 'captured' and int(row['type']) in [0, 26], 'unsupported source layout/type/status')
            count = math.prod(ne)
            require(0 < count*4 == int(row['bytes']) <= 16 << 20, 'bounded complete tensor')
            stride = 4
            for axis in range(4):
                require(ne[axis] <= 1 or int(row['nb%d' % axis]) == stride, 'noncontiguous captured tensor')
                stride *= ne[axis]
            expected = 'graph-%d.%s.bin' % (number, name)
            require(row['file'] == expected, 'exact graph tensor filename')
            file = owned(root, 'on/' + expected, receipt['artifacts'])
            require(file.stat().st_size == count*4, 'shape/storage mismatch')
            array = np.fromfile(file, dtype='<f4' if int(row['type']) == 0 else '<i4')
            require(int(row['type']) != 0 or np.isfinite(array).all(), 'nonfinite tensor')
            arrays[name] = array
            total += count*4
            require(total <= 256 << 20, 'bounded complete source capture')
        require(len({int(r['sample_count']) for r in records}) == 1, 'mixed graph phases')
        graphs.append(dict(number=number, phase=int(records[0]['sample_count']), arrays=arrays, ne=metadata))
    require(len(graphs) == 6, 'exact captured graph count')
    probe = graphs[0]
    require(probe['phase'] == 0 and probe['ne']['model.input_embed'] == [2560, 2, 1, 1] and probe['arrays']['inp_tokens'].tolist() == [0, 0] and probe['arrays']['inp_pos'].tolist() == [0, 1]*3 + [0, 0], 'unrecognized auxiliary graph; do not exclude')
    prefill, step = graphs[1:5], graphs[5]
    require([g['ne']['model.input_embed'] for g in prefill] == [[2560, n, 1, 1] for n in [4, 252, 6, 4]], 'actual book prefill geometry')
    require(all(g['phase'] == 0 for g in prefill) and step['phase'] == 405 and step['ne']['model.input_embed'] == [2560, 1, 1, 1], 'actual trace405 phase')
    sampling = owned(root, 'on/sampling-steps.tsv', receipt['artifacts'])
    require(list(map(int, sampling.read_text().split())) == list(range(654)), 'complete sampling progression')
    return prefill, step


def source_rows(graph, name, width, expected_tokens, role=None):
    ne = graph['ne'][name]
    if role == 'b':
        require(ne in [[1, width, expected_tokens, 1], [width, expected_tokens, 1, 1]], 'beta layout contract')
    else:
        require(ne == [width, expected_tokens, 1, 1], 'source row layout contract')
    values = graph['arrays'][name].reshape(expected_tokens, width)
    return values[:, permutation(role, width)] if role else values


def analyze(source_root, source, native_root, native):
    import numpy as np
    require(source['source_sha'] == 'e3164d09de4578370082e42c5954a77a6cd643e3' and source['status'] == 'measured_neutral_book_text_trace' and source['all654_scores_observer_neutral'] and source['historical_reference_sha256'] == '9b88195d5a8058dea4fab87472009d3826bc1fa3b5d0575aa2e33698316d5722', 'source neutrality admission')
    require(native['status'] == 'measured_neutral_native405_book_trace' and native['all1536_rows_ids_text_observer_neutral'] and native['trace_position'] == 405 and native['native_trace_layers'] == 32 and native['neutrality_reference_sha256'] == HISTORICAL and native['strict_1054_prefix_and_stages_reproduced'], 'native neutrality admission')
    for i in range(654):
        require(source['artifacts']['off/source.logits.%d.f32' % i] == source['artifacts']['on/source.logits.%d.f32' % i], 'source score neutrality manifest')
    source_ids, native_ids = source['arms']['on']['generated_ids'], native['arm']['ids']
    require(len(source_ids) == 654 and len(native_ids) == 1536 and source_ids[:405] == native_ids[:405] and source_ids[405] != native_ids[405], 'actual same consumed-prefix405')
    prefill, step = read_graphs(source_root, source)
    def native_vector(name, tokens, width):
        p = owned(native_root, 'planner-on/' + name + '.f32', native['artifacts'])
        require(p.stat().st_size == tokens*width*4, 'native stage dimensions')
        array = np.fromfile(p, dtype='<f4').reshape(tokens, width)
        require(np.isfinite(array).all(), 'native finite values')
        return array
    def native_file(name):
        return owned(native_root, 'planner-on/' + name, native['artifacts'])
    sealpath = owned(native_root, 'pre-forward-inputs.json', native['artifacts'])
    seal = json.loads(sealpath.read_text())
    require(sha(sealpath) == 'cbdc21be0e226c58cd3cb44341cc640616f4aa29ae42ffde16d664e20a6a4ed1', 'exact original book input seal')
    require(list(map(int, native_file('prompt.ids').read_text().split())) == seal['prompt_ids'] and len(seal['prompt_ids']) == 266, 'exact sealed prompt')
    source_input = np.concatenate([source_rows(g, 'model.input_embed', 2560, g['ne']['model.input_embed'][1]) for g in prefill])
    source_positions = np.concatenate([g['arrays']['inp_pos'].reshape(4, -1)[:3] for g in prefill], axis=1).astype('<f4')
    require(source_input.tobytes() == native_vector('prefill.input', 266, 2560).tobytes() and source_positions.tobytes() == native_vector('prefill.positions', 3, 266).tobytes(), 'prefill embedding/position input mismatch')
    require(step['arrays']['inp_tokens'].tolist() == [source_ids[404]] and source_rows(step, 'model.input_embed', 2560, 1).tobytes() == native_vector('greedy.405.input', 1, 2560).tobytes(), 'step405 embedding/consumed token mismatch')
    require(step['arrays']['inp_pos'].reshape(4, 1)[:3].astype('<f4').tobytes() == native_vector('greedy.405.positions', 3, 1).tobytes(), 'step405 positions mismatch')
    comparisons = []
    for phase, graphs in [('prefill', prefill), ('greedy.405', [step])]:
        offset = 0
        for graph in graphs:
            n = graph['ne']['model.input_embed'][1]
            for name, filename, width, role in STAGES:
                # Native prefill duplicate is only the first source-aligned4-token chunk.
                if phase == 'prefill' and offset:
                    continue
                take = 4 if phase == 'prefill' else 1
                original = source_rows(graph, name, width, n, role)[:take]
                candidate = native_vector(('prefill.first_chunk.layer0.' if phase == 'prefill' else 'greedy.405.layer0.') + filename, take, width)
                comparisons.append(dict(phase=phase, graph=graph['number'], stage=name, **delta(candidate, original)))
            for layer in range(32):
                name = 'l_out-%d' % layer
                if name not in graph['arrays']:
                    continue  # Explicit zero-sized last-layer crop; never synthesize rows.
                ne = graph['ne'][name]
                require(ne[0] == 2560 and ne[2:] == [1, 1], 'hidden tensor shape')
                count = ne[1]
                if count == n:
                    indices = np.arange(n)
                else:
                    require('out_ids' in graph['arrays'], 'missing exact output crop mapping')
                    indices = graph['arrays']['out_ids']
                    require(indices.size == count and len(set(map(int, indices))) == count and np.all((0 <= indices) & (indices < n)), 'invalid output crop mapping')
                full = native_vector(('prefill.layer.' if phase == 'prefill' else 'greedy.405.layer.') + str(layer), 266 if phase == 'prefill' else 1, 2560)
                comparisons.append(dict(phase=phase, graph=graph['number'], stage=name, **delta(full[indices + offset], graph['arrays'][name].reshape(count, 2560))))
            offset += n
    for phase, graph in [('prefill', prefill[-1]), ('greedy.405', step)]:
        source_score = source_rows(graph, 'result_output', 248320, 1)
        src_file = owned(source_root, 'on/source.logits.%d.f32' % (0 if phase == 'prefill' else 405), source['artifacts'])
        require(source_score.tobytes() == src_file.read_bytes(), 'live score/sampler closure')
        norm = source_rows(graph, 'result_norm', 2560, 1)
        candidate_norm = native_vector('prefill.norm' if phase == 'prefill' else 'greedy.405.norm', 266 if phase == 'prefill' else 1, 2560)[-1:]
        comparisons.append(dict(phase=phase, graph=graph['number'], stage='result_norm', **delta(candidate_norm, norm)))
        # Raw score rows are archived by native producer; prefill remains an owned plain row.
        if phase == 'prefill':
            comparisons.append(dict(phase=phase, graph=graph['number'], stage='result_output', **delta(native_vector('prefill.logits', 1, 248320), source_score)))
        else:
            raw = native['arm']['raw_rows']
            archive = owned(native_root, 'planner-on/' + raw['archive']['file'], native['artifacts'])
            require(sha(archive) == raw['archive']['sha256'], 'native archived-score closure')
            with zipfile.ZipFile(archive) as sealed:
                value = sealed.read('greedy.405.logits.f32')
            require(len(value) == 248320*4 and hashlib.sha256(value).hexdigest() == raw['rows'][405]['sha256'], 'native405 raw score identity')
            comparisons.append(dict(phase=phase, graph=graph['number'], stage='result_output', **delta(np.frombuffer(value, dtype='<f4').reshape(1, 248320), source_score)))
    earliest = {}
    for phase in ['prefill', 'greedy.405']:
        earliest[phase] = next((dict(stage=row['stage'], graph=row['graph']) for row in comparisons if row['phase'] == phase and not row['bit_exact']), None)
    return dict(comparisons=comparisons, earliest_recorded_differing_stage=earliest,
                same_prefix_tokens=405, same_prefill_input=True, same_step405_input=True,
                exclusions=['Exact two-zero-token auxiliary compatibility graph excluded before prompt concatenation.',
                            'Zero-sized last-layer outputs omitted using actual out_ids crop mapping.',
                            'Source convolution/recurrent-state tensors have no matching native capture; not compared.',
                            'Shared context does not imply equal cache state; operator causality requires separate same-input replay.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-sha256', required=True)
    args = parser.parse_args()
    require(re.fullmatch('[0-9a-f]{64}', args.native_sha256), 'immutable native receipt digest required')
    require(Path('/kaggle/input').is_dir(), 'Kaggle artifact-only worker required')
    signal.alarm(600)
    def locate(name, digest):
        matches = [p for p in Path('/kaggle/input').rglob(name) if p.is_file() and p.stat().st_size < 4 << 20 and sha(p) == digest]
        require(len(matches) == 1, 'one authenticated producer report required')
        return matches[0]
    result = dict(schema='lighton4b-book-stage-compare/v1', status='failed', inference_executed=False,
                  source_receipt_sha256=SOURCE_REPORT, native_receipt_sha256=args.native_sha256,
                  text_or_ids_emitted=False, qualified=False)
    try:
        sp, np = locate('book-text-trace.json', SOURCE_REPORT), locate('planner-book-extended.json', args.native_sha256)
        result.update(analyze(sp.parent, json.loads(sp.read_text()), np.parent, json.loads(np.read_text())))
        result['status'] = 'complete-aligned-artifact-comparison'
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        Path('/kaggle/working/book-stage-compare.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
