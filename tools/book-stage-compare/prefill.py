#!/usr/bin/env python3
"""Artifact-only prefill comparison against the preserved incomplete book replay."""
import json
from pathlib import Path
import signal
import audit

TIMEOUT = 'f231ab48c9bff411a1f9c96cfb79eda084f2f0b7cee46f9ea747ccc734d5f8e6'
SEAL = 'cbdc21be0e226c58cd3cb44341cc640616f4aa29ae42ffde16d664e20a6a4ed1'


def analyze(source_root, source, native_root, native):
    import numpy as np
    audit.require(source['source_sha'] == 'e3164d09de4578370082e42c5954a77a6cd643e3' and source['status'] == 'measured_neutral_book_text_trace' and source['all654_scores_observer_neutral'], 'neutral source405 producer')
    audit.require(native['status'] == 'failed' and native['error_type'] == 'TimeoutExpired' and native['phase'] == 'corrected-composed' and native['source_sha'] == 'a6164205620c71427a39c97ab386591406c4ca73' and native['old_baseline_all_rows_stages_exact'], 'preserved incomplete native producer')
    audit.require(native['native_gates']['on']['CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL'] == '1' and native['pre_forward_input_seal_sha256'] == SEAL, 'exact native planner policy/input seal')
    for i in range(654):
        audit.require(source['artifacts']['off/source.logits.%d.f32' % i] == source['artifacts']['on/source.logits.%d.f32' % i], 'producer source neutrality manifests')
    graphs, unused_step = audit.read_graphs(source_root, source)
    def file(name):
        return audit.owned(native_root, 'corrected-composed/' + name, native['artifacts'])
    def vector(name, tokens, width):
        p = file(name + '.f32')
        audit.require(p.stat().st_size == tokens*width*4, 'native prefill dimensions')
        a = np.fromfile(p, dtype='<f4').reshape(tokens, width)
        audit.require(np.isfinite(a).all(), 'native finite prefill')
        return a
    seal = audit.owned(native_root, 'pre-forward-inputs.json', native['artifacts'])
    audit.require(audit.sha(seal) == SEAL, 'authenticated original seal')
    sealed = json.loads(seal.read_text())
    audit.require(list(map(int, file('prompt.ids').read_text().split())) == sealed['prompt_ids'] and len(sealed['prompt_ids']) == 266 and list(map(int, file('modalities.ids').read_text().split())) == sealed['modalities'], 'sealed native prompt and modalities')
    embeddings = np.concatenate([audit.source_rows(g, 'model.input_embed', 2560, g['ne']['model.input_embed'][1]) for g in graphs])
    positions = np.concatenate([g['arrays']['inp_pos'].reshape(4, -1)[:3] for g in graphs], axis=1).astype('<f4')
    audit.require(embeddings.tobytes() == vector('prefill.input', 266, 2560).tobytes(), 'actual same prefill embeddings')
    audit.require(positions.tobytes() == vector('prefill.positions', 3, 266).tobytes(), 'actual same prefill positions')
    comparisons, omitted = [], []
    for name, suffix, width, role in audit.STAGES:
        reference = audit.source_rows(graphs[0], name, width, 4, role)
        comparisons.append(dict(scope='first4_layer0_operators', graph=graphs[0]['number'], stage=name,
            **audit.delta(vector('prefill.first_chunk.layer0.' + suffix, 4, width), reference)))
    for layer in range(32):
        full = vector('prefill.layer.' + str(layer), 266, 2560)
        offset = 0
        for g in graphs:
            n = g['ne']['model.input_embed'][1]
            name = 'l_out-' + str(layer)
            if name not in g['arrays']:
                audit.require(g['ne'][name][1] == 0, 'only zero-sized hidden outputs may be absent')
                omitted.append(dict(graph=g['number'], stage=name, reason='source zero-sized cropped output'))
                offset += n
                continue
            ne = g['ne'][name]
            audit.require(ne[0] == 2560 and ne[2:] == [1, 1], 'source hidden dimensions')
            count = ne[1]
            if count == n:
                indices = np.arange(n)
            else:
                indices = g['arrays']['out_ids']
                audit.require(indices.size == count and len(set(map(int, indices))) == count and np.all((0 <= indices) & (indices < n)), 'actual source output crop')
            comparisons.append(dict(scope='prefill_layer_outputs', graph=g['number'], stage=name,
                source_input_rows=n, compared_rows=count, **audit.delta(full[offset + indices], g['arrays'][name].reshape(count, 2560))))
            offset += n
    reference_norm = audit.source_rows(graphs[-1], 'result_norm', 2560, 1)
    comparisons.append(dict(scope='final_last_prompt_row', graph=4, stage='result_norm', **audit.delta(vector('prefill.norm', 266, 2560)[-1:], reference_norm)))
    reference_score = audit.source_rows(graphs[-1], 'result_output', 248320, 1)
    original_score = audit.owned(source_root, 'on/source.logits.0.f32', source['artifacts'])
    audit.require(reference_score.tobytes() == original_score.read_bytes(), 'source live firstscore closure')
    native_score = vector('prefill.logits', 1, 248320)
    audit.require(native_score.tobytes() == file('teacher.0.logits.f32').read_bytes() == file('greedy.0.logits.f32').read_bytes(), 'native actual firstscore triple closure')
    comparisons.append(dict(scope='final_last_prompt_row', graph=4, stage='result_output', **audit.delta(native_score, reference_score)))
    return dict(same_prefill_embeddings_bits=True, same_positions_bits=True, actual_prefill_graph_rows=[4, 252, 6, 4],
        comparisons=comparisons, omitted=omitted,
        earliest_recorded_first4_operator_difference=next((x for x in comparisons if x['scope'] == 'first4_layer0_operators' and not x['bit_exact']), None),
        earliest_recorded_layer_output_difference=next((x for x in comparisons if x['scope'] == 'prefill_layer_outputs' and not x['bit_exact']), None),
        limits=['Native timeout remains incomplete/negative; only authenticated prefill arrays are compared.',
                'Producer source654 raw neutrality is inherited from frozen receipt; this reader verifies manifests and selected actual arrays.',
                'No matched intermediate native image-layer0 operator or recurrent-cache capture exists; not inferred.',
                'Observed accumulated hidden differences do not establish unique operator causality; no decoded quality upgrade.'])


def main():
    audit.require(Path('/kaggle/input').is_dir(), 'authorized Kaggle artifact-only worker')
    signal.alarm(600)
    def locate(name, digest):
        rows = [p for p in Path('/kaggle/input').rglob(name) if p.is_file() and p.stat().st_size < 4 << 20 and audit.sha(p) == digest]
        audit.require(len(rows) == 1, 'one exact receipt required')
        return rows[0]
    result = dict(schema='lighton4b-book-prefill-stage-compare/v1', status='failed', inference_executed=False,
        source_receipt_sha256=audit.SOURCE_REPORT, native_receipt_sha256=TIMEOUT, text_or_ids_emitted=False, qualified=False)
    try:
        s, n = locate('book-text-trace.json', audit.SOURCE_REPORT), locate('planner-heldout.json', TIMEOUT)
        result.update(analyze(s.parent, json.loads(s.read_text()), n.parent, json.loads(n.read_text())))
        result['status'] = 'complete-aligned-prefill-artifact-comparison'
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        Path('/kaggle/working/book-prefill-stage-compare.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
