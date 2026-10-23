"""Model-free checks of the frozen OvisOCR2 quality and quantization gates."""

import importlib.util
import json
import copy
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'ovis_quality_contract', ROOT / 'tools/ci-heavy/ovisocr2_quality_contract.py')
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


class OvisQualityContract(unittest.TestCase):
    def test_real_proof_and_predeclared_disjoint_panel(self):
        panel = contract.load_panel()
        schema = contract.source_schema()
        self.assertEqual(len(schema), 473)
        self.assertEqual(len(panel['cases']), 8)
        self.assertEqual(len({case['gold'] for case in panel['cases']}), 8)
        self.assertEqual({case['split'] for case in panel['cases'][:4]}, {'development'})
        self.assertEqual({case['split'] for case in panel['cases'][4:]}, {'heldout'})
        self.assertEqual(panel['license'], 'CC0-1.0')

    def test_actual_quantizer_rules_match_qualified_ovis_schema(self):
        schema = contract.source_schema()
        for recipe in contract.RECIPES:
            command = contract.quant_command('crispembed-quantize', 'f16.gguf',
                                             'candidate.gguf', recipe, schema)
            self.assertEqual(command[:3], ['crispembed-quantize', 'f16.gguf', 'candidate.gguf'])
            self.assertIn('v.*=f16', command)
            self.assertIn('l.embed_tokens.weight=f16', command)
            self.assertIn('l.layers.*.linear_attn.in_proj_a.weight=f16', command)
            self.assertIn('l.layers.*.linear_attn.in_proj_b.weight=f16', command)

    def test_heldout_quality_screen_retains_raw_completion_and_format_failures(self):
        panel = contract.load_panel()
        rendered = {'panel_canonical_sha256': contract.PANEL_SHA,
                    'cases': [{'id': row['id'], 'rgb_sha256': str(i) * 64,
                               'png_sha256': str(i + 1) * 64}
                              for i, row in enumerate(panel['cases'])]}
        outputs = {row['id']: {'input_rgb_sha256': rendered['cases'][i]['rgb_sha256'],
                               'input_png_sha256': rendered['cases'][i]['png_sha256'],
                               'raw_text': row['gold'], 'processed_text': row['gold'],
                               'token_ids': [12, 248046], 'finish_reason': 'stop',
                               'stop_reason': 248046}
                   for i, row in enumerate(panel['cases'])}
        reference = contract.score_panel(rendered, outputs)
        candidate = contract.score_panel(rendered, outputs)
        self.assertTrue(contract.paired_quality_screen(reference, candidate)['screen_passed'])
        changed = json.loads(json.dumps(outputs))
        changed['hold-batch']['processed_text'] = ''
        changed['hold-batch']['finish_reason'] = 'length'
        failed = contract.score_panel(rendered, changed)
        self.assertFalse(contract.paired_quality_screen(reference, failed)['screen_passed'])
        self.assertEqual(failed['summary']['heldout']['cap_hits'], 1)
        self.assertEqual(failed['cases'][4]['raw_text'], panel['cases'][4]['gold'])

    def test_no_model_framework_import(self):
        import sys
        for name in ('torch', 'transformers', 'gguf', 'PIL', 'vllm'):
            self.assertNotIn(name, sys.modules)

    def test_paired_screen_rejects_missing_or_relabelled_cases(self):
        panel = contract.load_panel()
        rows = []
        for row in panel['cases']:
            rows.append({'id': row['id'], 'split': row['split'], 'gold': row['gold'],
                         'input_rgb_sha256': 'a' * 64, 'raw_text': row['gold'],
                         'processed_text': row['gold'], 'generated_tokens': 2,
                         'cap_hit': False, 'completed': True, 'format_issues': [],
                         'cer': 0.0, 'wer': 0.0, 'finish_reason': 'stop',
                         'raw_exact': True, 'processed_exact': True, 'stop_id': 248046})
        reference = {'panel_canonical_sha256': contract.PANEL_SHA, 'cases': rows}
        self.assertTrue(contract.paired_quality_screen(reference, reference)['screen_passed'])
        for mutation in ('missing', 'split', 'nan', 'empty'):
            candidate = copy.deepcopy(reference)
            if mutation == 'missing':
                candidate['cases'].pop()
            elif mutation == 'split':
                candidate['cases'][4]['split'] = 'development'
            elif mutation == 'nan':
                candidate['cases'][4]['cer'] = float('nan')
            else:
                candidate['cases'][4]['generated_tokens'] = 0
            with self.assertRaises(ValueError):
                contract.paired_quality_screen(reference, candidate)

    def test_quant_audit_rejects_shape_drift_and_q4_fallback(self):
        schema = {'l.example.weight': [256, 512]}
        before = {'l.example.weight': {'shape': [512, 256], 'type': 'F16', 'sha256': 'a' * 64}}
        after = {'l.example.weight': {'shape': [512, 256], 'type': 'Q4_K', 'sha256': 'b' * 64}}
        self.assertEqual(contract.validate_quant_tensors(before, after, 'q4-attention-q8', schema)
                         ['changed_matrix_count'], 1)
        changed = copy.deepcopy(after)
        changed['l.example.weight']['shape'] = [256, 512]
        with self.assertRaisesRegex(ValueError, 'shape'):
            contract.validate_quant_tensors(before, changed, 'q4-attention-q8', schema)
        changed = copy.deepcopy(after)
        changed['l.example.weight']['type'] = 'Q4_0'
        with self.assertRaisesRegex(ValueError, 'precision'):
            contract.validate_quant_tensors(before, changed, 'q4-attention-q8', schema)

    def test_realistic_document_screen_checks_completion_and_15_cells(self):
        rows = []
        for name, license in [('historical-receipt', 'CC0-1.0'),
                              ('two-column-page', 'CC0-1.0'), ('inventory-table', 'MIT')]:
            rows.append({'id': name, 'rgb_sha256': 'a' * 64, 'license': license,
                         'completed': True, 'cap_hit': False, 'format_issues': [],
                         'structure': {'complete': True, 'parse_ok': True,
                                       'raw': {'cer': 0.1, 'wer': 0.2},
                                       'content': {'cer': 0.1, 'wer': 0.2},
                                       'table': {'expected_cells': 15, 'cell_errors': 0}
                                       if name == 'inventory-table' else None}})
        self.assertTrue(contract.paired_document_screen(rows, copy.deepcopy(rows))['screen_passed'])
        worse = copy.deepcopy(rows)
        worse[0]['structure']['content']['cer'] = 0.11
        self.assertFalse(contract.paired_document_screen(rows, worse)['screen_passed'])
        worse = copy.deepcopy(rows)
        worse[2]['structure']['table']['cell_errors'] = 1
        self.assertFalse(contract.paired_document_screen(rows, worse)['screen_passed'])
        worse = copy.deepcopy(rows)
        worse[1]['cap_hit'] = True
        self.assertFalse(contract.paired_document_screen(rows, worse)['screen_passed'])


if __name__ == '__main__':
    unittest.main()
