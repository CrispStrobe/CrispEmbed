#!/usr/bin/env python3
"""Small stdlib-only controls. Native/Rust execution belongs to hosted CI."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import josev2b_tokenizer_schema as lane


def load_driver():
    spec = importlib.util.spec_from_file_location('josev_tokenizer_driver', ROOT / 'tools/ci-heavy/josev2b_tokenizer_schema.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class Controls(unittest.TestCase):
    def test_original_four_freeze_identity(self):
        for path, expected in [('original-reference.json', lane.REFERENCE_SHA), ('original-input-freeze.json', lane.FREEZE_SHA)]:
            self.assertEqual(lane.digest((ROOT / 'tests/results/josev2b' / path).read_bytes()), expected)
        rows = lane.panel_inputs()
        self.assertEqual([r['id'] for r in rows[:4]], [r['id'] for r in json.loads((ROOT / 'tests/results/josev2b/original-input-freeze.json').read_text())['records']])

    def test_old_sources_unchanged_and_only_new_frontend(self):
        before = subprocess.check_output(['git', 'ls-tree', '-r', lane.BASE, '--', 'src'], cwd=ROOT, text=True)
        for entry in before.splitlines():
            _, path = entry.split('\t')
            self.assertEqual((ROOT / path).read_bytes(), subprocess.check_output(['git', 'show', lane.BASE + ':' + path], cwd=ROOT), path)

    def test_pure_ast_contract_is_the_original_audited_set(self):
        tree = ast.parse((ROOT / 'tools/josev2b_contract.py').read_text())
        inspect = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inspect_serialization')
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in inspect.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('declarations', 'constants')}
        self.assertEqual(lane.PURE_DECLARATIONS, values['declarations'])
        self.assertEqual(lane.PURE_CONSTANTS, values['constants'])
        with self.assertRaisesRegex(ValueError, 'source differs'):
            lane.blueprint(ROOT / 'tools/josev2b_tokenizer_schema.py')
        self.assertNotIn('torch', sys.modules)

    def test_original_first_and_no_payload_allowlist_bypass(self):
        driver = load_driver()
        with mock.patch.dict('os.environ', {}, clear=True), mock.patch.object(driver, 'source_guard') as source:
            with self.assertRaisesRegex(ValueError, 'hosted'):
                driver.hosted()
            source.assert_not_called()
        with mock.patch.object(driver.urllib.request, 'urlopen') as network:
            with self.assertRaisesRegex(ValueError, 'not permitted'):
                driver.fetch('model.safetensors', {'bytes': 1}, Path('unused'))
            network.assert_not_called()
        workflow = (ROOT / '.github/workflows/josev2b-tokenizer-schema.yml').read_text()
        self.assertLess(workflow.index('--phase original'), workflow.index('cmake -S'))
        self.assertLess(workflow.index('--phase preflight'), workflow.index('pip --python'))
        self.assertIn('--no-deps --only-binary=:all: --require-hashes', workflow)
        self.assertIn('timeout-minutes: 25', workflow)
        self.assertIn('-j2', workflow)
        cmake = (ROOT / 'tests/josev-text/CMakeLists.txt').read_text()
        self.assertIn('set(GGML_CPU OFF', cmake)
        self.assertIn('PUBLIC ggml-base', cmake)
        self.assertIn('PRIVATE "${CRISPEMBED_ROOT}/src/core/third_party"', cmake)
        self.assertTrue((ROOT / 'src/core/third_party/ryu/ryu.h').is_file())

    def test_metadata_only_writer_and_no_tensors(self):
        payload = lane.gguf_bytes({'text': 'é', 'strings': ['a', 'b'], 'flags': [0, 16]})
        self.assertEqual(payload[:4], b'GGUF')
        self.assertEqual(struct.unpack_from('<IQQ', payload, 4), (3, 0, 3))
        self.assertEqual(len(payload) % 32, 0)
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            lane.gguf_bytes({'empty': []})

    def test_profile_converter_rejects_before_vocab_on_bad_normalizer(self):
        with self.assertRaisesRegex(ValueError, 'normalizer'):
            lane.component_metadata({'normalizer': None}, {})
        with self.assertRaisesRegex(ValueError, 'pretokenizer'):
            lane.component_metadata({'normalizer': {'type': 'NFC'}, 'pre_tokenizer': {}}, {})

    def test_authenticated_policy_accepts_only_observed_bytelevel_descriptor(self):
        # Actual small report40444002658, artifact12708375999: no vocab/model payload.
        path = ROOT / 'tests/fixtures/josev2b-tokenizer-policy-v1.json'
        self.assertEqual(lane.digest(path.read_bytes()), '1e3ff110f014cfb3baf35f07dcea52df21e7fc3454bc2e34dc07634e86ced3ec')
        policy = json.loads(path.read_text())
        self.assertEqual(policy['tokenizer_sha256'], lane.TOKENIZER_SHA)
        self.assertEqual(policy['config_sha256'], lane.CONFIG_SHA)
        config = dict(pretokenize_regex=policy['pretokenize_regex'], split_special_tokens=False,
                      tokenizer_class='Qwen2Tokenizer')
        lane.validate_tokenizer_policy(policy, config)
        self.assertIs(policy['pre_tokenizer']['pretokenizers'][1]['trim_offsets'], False)
        changes = [lambda x:x['pre_tokenizer']['pretokenizers'][1].update(trim_offsets=True),
                   lambda x:x['pre_tokenizer']['pretokenizers'][1].update(add_prefix_space=True),
                   lambda x:x['pre_tokenizer']['pretokenizers'][1].update(use_regex=True),
                   lambda x:x['pre_tokenizer']['pretokenizers'][0].update(invert=True),
                   lambda x:x['pre_tokenizer']['pretokenizers'][0].update(behavior='Removed'),
                   lambda x:x['pre_tokenizer']['pretokenizers'][0]['pattern'].update(Regex='unobserved'),
                   lambda x:x.update(normalizer={'type':'NFD'})]
        for change in changes:
            altered = deepcopy(policy); change(altered)
            with self.assertRaises(ValueError): lane.validate_tokenizer_policy(altered, config)
        for key, value in [('pretokenize_regex','unobserved'), ('split_special_tokens',True),
                           ('tokenizer_class','unobserved')]:
            with self.assertRaises(ValueError):
                lane.validate_tokenizer_policy(policy, dict(config, **{key:value}))

    def test_frozen_comparator_detects_one_id_span_mask_and_lexical_change(self):
        frozen = json.loads((ROOT / 'tests/results/josev2b/original-input-freeze.json').read_text())
        rows = []
        for record in frozen['records']:
            encoded = deepcopy(record['encoded']); encoded['attention_mask'] = record['attention_mask']
            for field in encoded['questions']:
                field['lexical_ids'] = [encoded['input_ids'][a:b] for a, b in field['option_spans']]
            rows.append({'id': record['id'], 'accepted': True, 'encoded': encoded})
        lane.frozen_match(deepcopy(rows), deepcopy(frozen))
        for key in ('input_ids', 'attention_mask', 'question_span', 'lexical_ids'):
            changed = deepcopy(rows)
            value = changed[0]['encoded']
            if key in value:
                value[key][0] += 1
            elif key == 'question_span':
                value['questions'][0][key][0] += 1
            else:
                value['questions'][0][key][0][0] += 1
            with self.assertRaisesRegex(ValueError, 'original tokenizer-only reproduction differs'):
                lane.frozen_match(changed, deepcopy(frozen))

    def test_authored_cases_order_and_raw_duplicates_are_preserved(self):
        cases = lane.panel_inputs()
        self.assertEqual(len(cases), 18)
        duplicate = next(c for c in cases if c['id'] == 'json-duplicate-order')
        value = json.loads(duplicate['request'])
        self.assertEqual(list(value['questions']), ['z', 'a'])
        self.assertEqual(value['state'], 'last')
        self.assertIs(value['questions']['z']['instructions'], False)
        float_case = next(c for c in cases if c['id'] == 'float-roundtrip')
        self.assertIn('5e-324', float_case['request'])
        self.assertEqual(sum(c['expect'] == 'reject' for c in cases), 4)
        self.assertTrue(all(c['max_length'] <= 512 for c in cases))

    def test_original_records_all_expectations_and_rejection_provenance(self):
        class Original:
            def encode_record(self, tokenizer, record, max_length):
                raise ValueError('schema requires 999 tokens before state; maximum is 1')
        cases = [{'id': 'schema-overflow', 'request': '{}', 'max_length': 1, 'expect': 'reject'},
                 {'id': 'should-have-accepted', 'request': '{}', 'max_length': 1, 'expect': 'accept'}]
        rows = lane.capture_original(Original(), None, cases)
        self.assertTrue(rows[0]['expectation_matched'])
        self.assertFalse(rows[1]['expectation_matched'])
        self.assertEqual(len(rows), 2)
        class WrongReason:
            def encode_record(self, *args, **kwargs):
                raise ValueError('unrelated failure')
        rows = lane.capture_original(WrongReason(), None, cases[:1])
        self.assertFalse(rows[0]['expectation_matched'])

    def test_cpp_context_gate_and_no_gemma_renderer(self):
        source = (ROOT / 'src/josev2b_text_frontend.cpp').read_text()
        self.assertIn('core_env::on("CRISPEMBED_JOSEV_TEXT_FRONTEND")', source)
        self.assertEqual(source.count('core_env::on('), 1)
        self.assertIn('bytelevel_pretokenize(core_nfc::normalize(', source)
        self.assertIn('1, true)', source)
        self.assertIn('core_decision::python_str(v)', source)
        self.assertNotIn('core_decision::state_text', source)
        self.assertNotIn('core_decision::question(', source)
        self.assertRegex(source, r'output = \{\};\s*error.clear\(\);')
        self.assertIn('gguf_get_n_tensors(g.get()) == 0', source)
        self.assertIn('invalid JSON escape', source)
        self.assertNotIn('ggml_backend', source)
        self.assertIn('core_util::clean_exit(rc)', (ROOT / 'tests/test_josev2b_text.cpp').read_text())

    def test_metadata_registration_only_adds_phase_and_job(self):
        old = subprocess.check_output(['git', 'show', lane.BASE + ':.github/workflows/ovisocr2-metadata.yml'], cwd=ROOT, text=True)
        new = (ROOT / '.github/workflows/ovisocr2-metadata.yml').read_text()
        expected = old.replace('options: [josev-wheel-metadata,', 'options: [josev-tokenizer-schema, josev-wheel-metadata,', 1)
        expected += "\n  josev-tokenizer-schema:\n    if: github.event_name == 'workflow_dispatch' && inputs.phase == 'josev-tokenizer-schema'\n    uses: ./.github/workflows/josev2b-tokenizer-schema.yml\n"
        self.assertEqual(new, expected)

    def test_lock_pins_exact_original_rust_wheel(self):
        lock = json.loads((ROOT / 'docs/josev2b-environment-lock.json').read_text())
        row, = [p for p in lock['packages'] if p['name'] == 'tokenizers']
        self.assertEqual(row['version'], '0.22.2')
        self.assertEqual(row['sha256'], '369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67')
        self.assertEqual(row['bytes'], 3274982)

    def test_omitted_old_source_closure_rejects_before_lane_or_payload(self):
        cfg = json.loads(lane.CONTRACT_PATH.read_text())
        lines = subprocess.check_output(['git', 'ls-tree', '-r', lane.BASE, '--', 'src'], cwd=ROOT, text=True)
        for path in ('src/crispembed.cpp', 'src/qwen35_tokenizer.cpp', 'src/core/bpe.h'):
            altered = deepcopy(cfg); del altered['protected_objects'][path]
            with mock.patch.object(lane.json, 'loads', return_value=altered), mock.patch.object(lane, 'git', return_value=lines):
                with self.assertRaisesRegex(ValueError, 'closure omitted'):
                    lane.source_guard()

    def test_contract_exact_protected_and_source_sets(self):
        cfg = json.loads(lane.CONTRACT_PATH.read_text())
        src = {l.split('\t')[1] for l in subprocess.check_output(['git', 'ls-tree', '-r', lane.BASE, '--', 'src'], cwd=ROOT, text=True).splitlines()}
        self.assertEqual(set(cfg['protected_objects']), lane.PROTECTED_ROOTS | src)
        self.assertEqual(set(cfg['source_sha256']), lane.LANE_SOURCES)
        self.assertFalse(cfg['default_enabled'])
        self.assertFalse(cfg['model_weights'])
        for path, expected in cfg['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected, path)


if __name__ == '__main__':
    unittest.main()
