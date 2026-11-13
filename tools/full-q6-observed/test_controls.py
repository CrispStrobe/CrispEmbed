#!/usr/bin/env python3
"""Small model-free controller controls; run on the hosted worker before weights."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full_q6_worker', HERE/'worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class Controls(unittest.TestCase):
    def test_protocol_hash_and_fixed_scope(self):
        raw = (HERE/'protocol.json').read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        cfg = worker.admit_protocol(HERE/'protocol.json', digest)
        self.assertEqual([cfg['old_q4_cap'], cfg['full_q6_cap'], cfg['threads']], [512, 256, 4])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'protocol.json'
            path.write_bytes(raw+b' ')
            worker.admit_protocol(path, digest)  # Harmless formatting is not an admission barrier.
            changed = json.loads(raw)
            changed['heldout_allowed'] = True
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                worker.admit_protocol(path, worker.sha(path))

    def test_old_and_full_q6_gate_profiles(self):
        cfg = json.loads((worker.ROOT/'tests/results/lighton4b/receipt-layer0-operators/aligned-observed.json').read_text())
        original = cfg['cases']['simple_form']['arms']['on']['gates']
        before = copy.deepcopy(original)
        old, new = worker.gates(original)
        self.assertEqual(original, before)
        self.assertEqual({k for k in old if old[k] != new[k]}, {worker.GATE, worker.REPACK})
        self.assertEqual((old[worker.GATE], old[worker.REPACK], new[worker.GATE], new[worker.REPACK]), ('0', '1', '1', '0'))
        for key in ['CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE', 'CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL_ALIGNED']:
            bad = dict(original)
            bad[key] = '0'
            with self.assertRaises(ValueError):
                worker.gates(bad)

    def test_complete_and_capped_stops(self):
        summary = dict(generated_tokens=3, cap=256, eos=True, upstream_physical=True,
                       source_fed_vision=False, teacher_positions=1)
        worker.stop_state(summary, [27, 17251, 248046], 256)
        for ids in [[248046, 17251, 248046], [27, 17251, -1], [27, 17251, 248320], [27, 17251, 29]]:
            with self.assertRaises(ValueError):
                worker.stop_state(summary, ids, 256)
        capped = dict(summary, generated_tokens=256, eos=False)
        worker.stop_state(capped, [27]*256, 256)
        with self.assertRaises(ValueError):
            worker.stop_state(dict(capped, cap=512), [27]*256, 256)
        with self.assertRaises(ValueError):
            worker.stop_state(dict(summary, source_fed_vision=True), [27, 17251, 248046], 256)

    def test_edit_distance_unicode_and_reference_denominators(self):
        self.assertEqual(worker.edits('kitten', 'sitting'), 3)
        self.assertEqual(worker.edits('é', 'e\u0301'), 2)
        self.assertEqual(worker.edits([], ['a']), 1)
        metric = worker.metric([27, 12, 248046], 'un chat', [27, 15, 248046], 'un grand chat')
        self.assertEqual((metric['matching_prefix_ids'], metric['character_edits'], metric['word_edits']), (1, 6, 1))
        self.assertEqual((metric['reference_codepoints'], metric['reference_words']), (13, 3))

    def test_frozen_q4_schema_derived_from_publisher(self):
        cfg = json.loads((HERE/'protocol.json').read_text())
        census = json.loads((worker.ROOT/'tests/fixtures/lightonocr3-4b-source/publisher-q4km-type-census.json').read_text())['tensors']
        mapping = worker.load('physical_mapping_test', worker.ROOT/'tools/ci-heavy/lighton4b_upstream_physical_audit.py')
        rows = [line.split() for line in (HERE/'q4-control-schema.tsv').read_text().splitlines()]
        self.assertEqual(len(rows), 426)
        self.assertEqual(len({r[0] for r in rows}), 426)
        for row in rows:
            self.assertEqual(int(row[1]), census[mapping.physical_name(row[0])]['type'])
        self.assertEqual(sum(int(row[1]) == 12 for row in rows), 216)
        self.assertEqual(sum(int(row[1]) == 14 for row in rows), 33)

    def test_artifact_content_corruption_and_size_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'owned.bin'
            path.write_bytes(b'abc')
            row = dict(bytes=3, sha256=worker.sha(path))
            self.assertEqual(worker.verified(path, row), path)
            path.write_bytes(b'abd')
            with self.assertRaises(ValueError):
                worker.verified(path, row)
            path.write_bytes(b'abcd')
            with self.assertRaises(ValueError):
                worker.verified(path, row)

    def test_real_frozen_bf16_and_q6_denominators(self):
        cfg = json.loads((HERE/'protocol.json').read_text())
        source = json.loads((worker.ROOT/cfg['source_report_path']).read_text())
        baseline = json.loads((worker.ROOT/'docs/lighton4b-independent-llama-baseline-v1.json').read_text())
        expected = {'simple_form': (182, 154, 31), 'simple_table': (222, 0, 0), 'arabic_printed_line': (6, 0, 0)}
        for case, values in expected.items():
            row = source['arms']['Q6_K'][case]
            ref_cfg = next(c for c in baseline['cases'] if c['id'] == case)
            path = (worker.ROOT/cfg['source_report_path']).parent/'simple_form-original.json' if case == 'simple_form' else worker.ROOT/ref_cfg['reference_path']
            original = json.loads(path.read_text())
            metric = worker.metric(row['ids'], row['decoded'], original['generated_ids'], original['decoded'])
            self.assertEqual((len(row['ids']), metric['character_edits'], metric['word_edits']), values)

    def test_actual_full_vocabulary_row_validation(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            values = np.zeros(worker.VOCAB, dtype='<f4')
            values[27] = 1
            for name in ['prefill.logits.f32', 'teacher.0.logits.f32', 'greedy.0.logits.f32']:
                values.tofile(folder/name)
            self.assertEqual(len(worker.rows(folder, [27], np)), 1)
            with self.assertRaises(ValueError):
                worker.rows(folder, [28], np)
            values[7] = np.nan
            values.tofile(folder/'greedy.0.logits.f32')
            with self.assertRaises(ValueError):
                worker.rows(folder, [27], np)
            (folder/'greedy.0.logits.f32').write_bytes(b'\0'*4)
            with self.assertRaises(ValueError):
                worker.rows(folder, [27], np)

    def test_numerical_error_retains_magnitudes(self):
        import numpy as np
        candidate = np.asarray([2, -1], dtype='<f4')
        reference = np.asarray([1, -1], dtype='<f4')
        result = worker.numerical(candidate, reference, np)
        self.assertFalse(result['bit_exact'])
        self.assertEqual(result['max_abs'], 1)
        self.assertAlmostEqual(result['rms'], 0.5**0.5)
        self.assertEqual((result['candidate_abs_max'], result['reference_abs_max']), (2, 1))
        with self.assertRaises(ValueError):
            worker.numerical(candidate, reference[:1], np)

    def test_actual_source_response_admission(self):
        cfg = json.loads((HERE/'protocol.json').read_text())
        path = worker.ROOT/cfg['source_report_path']
        report = json.loads(path.read_text())
        source = report['arms']['Q6_K']['simple_form']
        response = json.loads((path.parent/'Q6_K-simple_form/response.json').read_text())
        worker.admit_source_response(response, source, 154)
        for name, value in [('max_tokens', 512), ('n_predict', 512), ('ignore_eos', True), ('temperature', 1), ('top_k', 40)]:
            bad = copy.deepcopy(response)
            bad['generation_settings'][name] = value
            with self.assertRaises(ValueError):
                worker.admit_source_response(bad, source, 154)
        for name, value in [('tokens_evaluated', 153), ('content', 'changed'), ('stop_type', 'limit')]:
            bad = copy.deepcopy(response)
            bad[name] = value
            with self.assertRaises(ValueError):
                worker.admit_source_response(bad, source, 154)
        with self.assertRaises(ValueError):
            worker.admit_source_response(response, dict(source, prompt_ids_exact_original=False), 154)


if __name__ == '__main__':
    unittest.main()
