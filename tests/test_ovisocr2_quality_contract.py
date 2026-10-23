"""Model-free checks of the frozen OvisOCR2 quality and quantization gates."""

import importlib.util
import json
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


if __name__ == '__main__':
    unittest.main()
