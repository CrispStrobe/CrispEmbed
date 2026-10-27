#!/usr/bin/env python3
"""Bounded stdlib source/typed-protocol controls. No original gold or CPP run here."""
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import josev2b_answer_serializer as lane


def driver():
    spec = importlib.util.spec_from_file_location('josev_answer_driver', ROOT / 'tools/ci-heavy/josev2b_answer_serializer.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def valid():
    freeze = json.loads((ROOT / 'tests/results/josev2b/original-input-freeze.json').read_text())['records'][0]
    reference = json.loads((ROOT / 'tests/results/josev2b/original-reference.json').read_text())['cases'][0]
    return {'id': freeze['id'], 'request': lane.canonical(freeze['request']),
            'encoded': lane.lexical(dict(freeze['encoded'], attention_mask=freeze['attention_mask'])),
            'probabilities_u32': [[lane.f32_bits(x) for x in f['probabilities']] for f in reference['fields']]}


class Controls(unittest.TestCase):
    def test_frozen_originals_and_seven_f32_fields(self):
        for name, expected in [('original-reference.json', lane.REFERENCE_SHA), ('original-input-freeze.json', lane.FREEZE_SHA)]:
            self.assertEqual(lane.digest((ROOT / 'tests/results/josev2b' / name).read_bytes()), expected)
        cases = json.loads((ROOT / 'tests/results/josev2b/original-reference.json').read_text())['cases']
        self.assertEqual(len(cases), 4)
        self.assertEqual(sum(len(c['fields']) for c in cases), 7)
        for case in cases:
            for field in case['fields']:
                self.assertEqual([lane.f32_value(lane.f32_bits(p)) for p in field['probabilities']], field['probabilities'])

    def test_f32_transport_preserves_negative_zero_and_nan_without_json_nan(self):
        self.assertEqual(lane.f32_bits('-0.0'), 0x80000000)
        self.assertEqual(lane.f32_bits('0.0'), 0)
        self.assertTrue(math.isnan(lane.f32_value(0x7fc00000)))
        self.assertEqual(lane.canonical({'bits': [0x7fc00000]}), '{"bits":[2143289344]}')
        self.assertEqual(struct.pack('<f', lane.f32_value(0x80000000)), b'\0\0\0\x80')

    def test_owned_recipe_bounds_and_declared_adversarial_cases(self):
        fixture = json.loads((ROOT / 'tests/fixtures/josev2b-answer-serializer-v1.json').read_text())
        cases = fixture['cases']; self.assertEqual(len(cases), 13)
        self.assertEqual(len({c['id'] for c in cases}), 13)
        counter = next(c for c in cases if c['id'] == 'score-compensated-counter')
        self.assertEqual(len(counter['request']['questions']['q']['criteria']), 64)
        self.assertEqual(counter['probability_recipe'], 'compensated-positive-64')
        maximum = next(c for c in cases if c['id'] == 'score-max-128-unrenormalized')
        self.assertEqual(len(maximum['probabilities']['q']), 128)
        self.assertEqual(sum(range(128)), 8128)
        self.assertLess(((1 << 53) - 1) * 625, 1 << 63)
        ties = next(c for c in cases if c['id'] == 'choice-negative-zero-tie')
        self.assertEqual(list(ties['request']['questions']['q']['criteria']), ['z', 'a'])
        self.assertEqual(ties['probabilities']['q'], {'a': '0.0', 'z': '-0.0'})
        counter_usage = next(c for c in cases if c['id'] == 'counter-512-instruction-fallback')
        self.assertEqual(counter_usage['tokens'], 512)

    def test_binary64_rounding_recipe_includes_neighbors_ties_subnormal_signed_zero(self):
        rows = lane.rounding_inputs(); self.assertEqual(len(rows), 66)
        bits = {int(r['binary64_hex'], 16) for r in rows}
        for value in (0.0, -0.0, 0.03125, 0.09375):
            self.assertIn(struct.unpack('<Q', struct.pack('<d', value))[0], bits)
        self.assertIn(1, bits)
        for row in rows:
            value = struct.unpack('<d', struct.pack('<Q', int(row['binary64_hex'], 16)))[0]
            self.assertTrue(math.isfinite(value) and 0 <= value <= 8128)

    def test_manual_wrapper_calls_original_and_preserves_f32_order(self):
        calls = []
        class Pure:
            def systemone_answer(self, question, probabilities):
                calls.append((deepcopy(question), dict(probabilities)))
                return {'original-call': len(calls)}
        case = valid()
        result = lane.original_answers(Pure(), [case])
        self.assertEqual(len(calls), 3)
        self.assertEqual(list(calls[1][1]), ['a_bezahlt', 'z_offen'])
        self.assertEqual(result[0]['answer']['usage'], {'input_tokens': 319, 'output_tokens': 0})
        self.assertIsNone(result[0]['official_response_equal'])
        for call, bits in zip(calls, case['probabilities_u32']):
            self.assertEqual(list(call[1].values()), [lane.f32_value(b) for b in bits])
        self.assertNotIn('torch', sys.modules)

    def test_invalid_fixtures_are_owned_copies_and_typed(self):
        original = valid(); before = deepcopy(original); negative = lane.native_negatives(original)
        self.assertEqual(original, before)
        self.assertEqual(len({r['id'] for r in negative}), len(negative))
        self.assertEqual(len(negative), 24)
        for row in negative:
            json.loads(lane.canonical(row))
            self.assertNotEqual(row, original)
        nan = next(r for r in negative if r['id'] == 'nan-probability')
        self.assertEqual(nan['probabilities_u32'][0][0], 0x7fc00000)
        no_model = next(r for r in negative if r['id'] == 'missing-model')
        self.assertNotIn('model', json.loads(no_model['request']))

    def test_all_old_frontend_runtime_and_schema_bytes_unchanged(self):
        paths = sorted(lane.OLD_PATHS | lane.old_src())
        for name in paths:
            obj = subprocess.check_output(['git', 'rev-parse', lane.BASE + ':' + name], cwd=ROOT, text=True).strip()
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD:' + name], cwd=ROOT, text=True).strip()
            self.assertEqual(obj, head, name)
        for name in ('src/josev2b_text_frontend.cpp', 'src/josev2b_text_frontend.h', 'tools/josev2b_tokenizer_schema.py'):
            self.assertEqual((ROOT / name).read_bytes(), subprocess.check_output(['git', 'show', lane.BASE + ':' + name], cwd=ROOT))

    def test_cpp_gate_and_exact_binary_arithmetic_contract(self):
        source = (ROOT / 'src/josev2b_answer_serializer.cpp').read_text()
        self.assertIn('core_env::on("CRISPEMBED_JOSEV_ANSWER_SERIALIZER")', source)
        self.assertEqual(source.count('core_env::on('), 1)
        self.assertRegex(source, r'output.clear\(\);\s*error.clear\(\);')
        self.assertIn('std::memcpy(&bits, &value, sizeof(bits))', source)
        self.assertIn('mantissa * 625', source)
        self.assertIn('remainder == midpoint && (rounded & 1)', source)
        self.assertNotIn('std::round(', source)
        self.assertNotIn('nearbyint', source)
        self.assertIn('compensation += (result - t) + x', source)
        self.assertNotIn('core_decision::state_text', source)
        self.assertNotIn('core_decision::question(', source)
        header = (ROOT / 'src/josev2b_answer_serializer.h').read_text()
        self.assertIn('#ifdef CRISPEMBED_JOSEV_ANSWER_TESTING', header)
        self.assertIn('core_util::clean_exit(rc)', (ROOT / 'tests/test_josev2b_answer.cpp').read_text())

    def test_no_frontend_backend_dependency_or_historical_guard_call(self):
        cmake = (ROOT / 'tests/josev-answer/CMakeLists.txt').read_text()
        self.assertNotIn('frontend.cpp', cmake)
        self.assertNotIn('add_subdirectory', cmake)
        self.assertIn('josev-answer-plain', cmake)
        self.assertIn('josev-answer-tested', cmake)
        self.assertIn('-ffp-contract=off', cmake)
        tree = ast.parse((ROOT / 'tools/ci-heavy/josev2b_answer_serializer.py').read_text())
        imports = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'josev2b_tokenizer_schema']
        self.assertEqual([[a.name for a in n.names] for n in imports], [['blueprint']])
        self.assertNotIn('torch', sys.modules)

    def test_original_first_workflow_and_no_package_payloads(self):
        workflow = (ROOT / '.github/workflows/josev2b-answer-serializer.yml').read_text()
        self.assertLess(workflow.index('--phase preflight'), workflow.index('--phase original'))
        self.assertLess(workflow.index('--phase original'), workflow.index('cmake -S'))
        self.assertIn("python-version: '3.13.15'", workflow)
        self.assertIn('timeout-minutes: 15', workflow)
        self.assertIn('-j2', workflow)
        self.assertNotIn('pip ', workflow)
        self.assertNotIn('submodule update', workflow)
        host = driver()
        with mock.patch.dict('os.environ', {}, clear=True), mock.patch.object(host, 'source_guard') as guard:
            with self.assertRaisesRegex(ValueError, 'hosted'):
                host.hosted()
            guard.assert_not_called()

    def test_metadata_only_new_phase_and_job(self):
        old = subprocess.check_output(['git', 'show', lane.BASE + ':.github/workflows/ovisocr2-metadata.yml'], cwd=ROOT, text=True)
        expected = old.replace('options: [josev-tokenizer-schema,', 'options: [josev-answer-serializer, josev-tokenizer-schema,', 1)
        expected += "\n  josev-answer-serializer:\n    if: github.event_name == 'workflow_dispatch' && inputs.phase == 'josev-answer-serializer'\n    uses: ./.github/workflows/josev2b-answer-serializer.yml\n"
        self.assertEqual((ROOT / '.github/workflows/ovisocr2-metadata.yml').read_text(), expected)

    def test_contract_closes_sources_and_omission_rejects_before_payloads(self):
        cfg = json.loads(lane.CONTRACT.read_text())
        self.assertEqual(set(cfg['source_sha256']), lane.LANE_PATHS)
        self.assertEqual(set(cfg['protected_objects']), lane.OLD_PATHS | lane.old_src())
        for name, value in cfg['source_sha256'].items():
            self.assertEqual(lane.digest((ROOT / name).read_bytes()), value, name)
        bad = deepcopy(cfg); del bad['protected_objects']['src/josev2b_text_frontend.cpp']
        with mock.patch.object(lane.json, 'loads', return_value=bad):
            with self.assertRaisesRegex(ValueError, 'closure omitted'):
                lane.source_guard()


if __name__ == '__main__':
    unittest.main()
