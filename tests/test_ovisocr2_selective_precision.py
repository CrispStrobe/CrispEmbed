"""Pure gates for the development-only 72-FFN selective precision recipe."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load('ovis_selective_frozen_contract',
                'tools/ci-heavy/ovisocr2_quality_contract.py')
selective = load('ovis_selective_recipe',
                 'tools/ci-heavy/ovisocr2_selective_precision.py')
driver = load('ovis_selective_driver', 'tools/ci-heavy/ovisocr2_quality_run.py')


class SelectivePrecisionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = contract.source_schema()
        cls.targets = selective.target_names(cls.schema, contract)

    def before(self):
        return {name: {'shape': list(reversed(shape)),
                       'type': 'F32' if name.startswith('v.') and len(shape) == 2 else 'F16',
                       'sha256': 'a' * 64}
                for name, shape in self.schema.items()}

    def after(self, before):
        result = copy.deepcopy(before)
        for name in self.targets:
            result[name]['type'] = 'Q4_K'
            result[name]['sha256'] = 'b' * 64
        return result

    def test_exact_72_ffn_and_401_explicit_old_type_overrides(self):
        before = self.before()
        command = selective.quant_command('quantizer', 'f16.gguf', 'candidate.gguf',
                                          before, contract, self.schema)
        self.assertEqual(command[:4], ['quantizer', 'f16.gguf', 'candidate.gguf', 'q4_k'])
        overrides = dict(item.split('=', 1) for item in command[5::2])
        self.assertEqual(len(self.targets), 72)
        self.assertEqual(len(overrides), 401)
        self.assertEqual(set(overrides), set(self.schema) - self.targets)
        self.assertEqual(command[4::2], ['--tensor-type'] * 401)
        self.assertEqual(overrides['v.blocks.0.attn.proj.weight'], 'f32')
        self.assertEqual(overrides['l.layers.0.linear_attn.in_proj_qkv.weight'], 'f16')
        result = selective.validate_quant_tensors(before, self.after(before),
                                                   contract, self.schema)
        self.assertEqual(result['q4_k_ffn_count'], 72)
        self.assertEqual(result['unchanged_tensor_count'], 401)
        self.assertEqual(result['unchanged_attention_matrix_count'], 78)
        self.assertTrue(result['non_ffn_byte_exact'])

    def test_attention_vision_norm_or_one_ffn_drift_is_rejected(self):
        before = self.before()
        after = self.after(before)
        for name in ('l.layers.0.linear_attn.in_proj_qkv.weight',
                     'l.layers.3.self_attn.q_proj.weight',
                     'v.blocks.0.attn.proj.weight', 'l.layers.0.input_layernorm.weight'):
            changed = copy.deepcopy(after)
            changed[name]['sha256'] = 'c' * 64
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'non-FFN'):
                selective.validate_quant_tensors(before, changed, contract, self.schema)
        target = 'l.layers.0.mlp.down_proj.weight'
        for field, value in (('type', 'Q4_0'), ('sha256', before[target]['sha256']),
                             ('shape', [1, 1])):
            changed = copy.deepcopy(after)
            changed[target][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'FFN Q4_K'):
                selective.validate_quant_tensors(before, changed, contract, self.schema)
        changed = copy.deepcopy(after)
        del changed[target]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            selective.validate_quant_tensors(before, changed, contract, self.schema)

    def test_baseline_type_schema_and_alignment_are_fail_closed(self):
        before = self.before()
        target = 'l.layers.0.mlp.gate_proj.weight'
        changed = copy.deepcopy(before)
        changed[target]['type'] = 'F32'
        with self.assertRaisesRegex(ValueError, 'not F16'):
            selective.quant_command('q', 'f', 'out', changed, contract, self.schema)
        changed = copy.deepcopy(before)
        changed['v.blocks.0.attn.proj.weight']['type'] = 'Q8_0'
        with self.assertRaisesRegex(ValueError, 'record differs'):
            selective.quant_command('q', 'f', 'out', changed, contract, self.schema)
        shape_changed = copy.deepcopy(self.schema)
        shape_changed[target][1] += 1
        with self.assertRaisesRegex(ValueError, 'ineligible'):
            selective.target_names(shape_changed, contract)

    def test_full_f16_fallback_requires_all_eleven_exact_ids_text_and_scores(self):
        cases = [{'id': str(i), 'rgb_sha256': 'a' * 64, 'png_sha256': 'b' * 64,
                  'cap': 12} for i in range(11)]
        frozen = {case['id']: {'token_ids': [42, 248046], 'raw_text': 'hello',
                               'processed_text': 'hello', 'stop_id': 248046,
                               'finish_reason': 'stop'} for case in cases}
        actual = {name: {**row, 'input_rgb_sha256': 'a' * 64,
                         'input_png_sha256': 'b' * 64, 'cap': 12,
                         'confidences': [0.5, 0.5], 'mean_confidence': 0.5}
                  for name, row in frozen.items()}
        self.assertTrue(driver.validate_full_f16_host_control(actual, frozen, cases)
                        ['frozen_ids_text_stop_exact'])
        changed = copy.deepcopy(actual)
        changed['10']['token_ids'] = [43, 248046]
        with self.assertRaisesRegex(ValueError, 'IDs/text'):
            driver.validate_full_f16_host_control(changed, frozen, cases)
        changed = copy.deepcopy(actual)
        changed['10']['mean_confidence'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'confidence'):
            driver.validate_full_f16_host_control(changed, frozen, cases)

    def test_frozen_recipe_file_and_runtime_ast_remain_exact(self):
        frozen = ROOT / 'tools/ci-heavy/ovisocr2_quality_contract.py'
        self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(),
                         'c1e77fd2a20bb2aeae3e09f030df13389891d188957cb7db78488c9d4460b8b3')
        hashes = driver.f16_runtime_ast_contracts(
            (ROOT / 'tools/ci-heavy/ovisocr2_quality_run.py').read_text())
        self.assertEqual(hashes, {
            'DIFF_FLOAT_ZERO': 'a2c1f24c858a3ddc4e5a77fa61b9755a7a50fa35b8e6ffa8c11db0a41921934a',
            'ENV_GATES': '0b8415fd77c2022b3d99fc65481db354adedd32714b53bd45430c0b2e96c1832',
            'cmake-configure': '6a2e50167228f0ec9617c0037f0410c162e38e1de595dc0b5125d04a974fb452',
            'native_environment': '3f6da0564c3c5053dbb4a8ca5c09424c43875cdde548a3abb63de919e8ad6342',
            'numerical_environment': 'ebaab4bd474e40ed7a27b87b4bbb5c629dc8a23c88d19e92a0059afaf491b917',
            'vision_contract': '6d318a1668db09cc8d0d102e7736093b2dd874c5520dbd19cf08b549f8e5e7c5',
        })
        self.assertIn(selective.PHASE, driver.QUANT_PHASES)
        self.assertIn(selective.PHASE, driver.Q4_PHASES)
        workflow = (ROOT / '.github/workflows/ovisocr2-quality.yml').read_text()
        self.assertIn('q4-attention-f16]', workflow)
        self.assertIn('ovisocr2-q4-attention-f16.gguf', workflow)

    def test_new_phase_requires_reviewed_q8_before_candidate_work(self):
        missing = ROOT / 'tests/results/qwen35-ocr/no-such-q8-review.json'
        with patch.object(driver, 'Q8_REVIEW', missing):
            with self.assertRaisesRegex(ValueError, 'Reviewed Q8'):
                driver.require_baseline(selective.PHASE)


if __name__ == '__main__':
    unittest.main()
