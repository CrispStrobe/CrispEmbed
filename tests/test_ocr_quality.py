"""Independent positive/negative controls for content, table and completion scores."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ci-heavy')]
from ocr_quality import parse_document, score
from ocr_quality_prepare import validate_panel, verify_prepared


class Quality(unittest.TestCase):
    gold = dict(text='Stock\nPart Count\nBolt 2\nNut 7\nEnd', tables=[[['Part', 'Count'], ['Bolt', '2'], ['Nut', '7']]])
    html = '# Stock\n<table><thead><tr><th>Part</th><th>Count</th></tr></thead><tbody><tr><td>Bolt</td><td>2</td></tr><tr><td>Nut</td><td>7</td></tr></tbody></table>\nEnd'
    markdown = 'Stock\n| Part | Count |\n| --- | ---: |\n| Bolt | 2 |\n| Nut | 7 |\nEnd'

    def test_serialization_keeps_raw_errors(self):
        for text in (self.html, self.markdown):
            r = score(self.gold, text, complete=True)
            self.assertEqual(r['raw_text'], text)
            self.assertGreater(r['raw']['cer'], 0)
            self.assertEqual(r['content']['cer'], 0)
            self.assertTrue(r['table']['exact'])
            self.assertTrue(r['exact_quality'])

    def test_plain_text_does_not_prove_table_structure(self):
        r = score(self.gold, self.gold['text'], complete=True)
        self.assertEqual(r['raw']['cer'], 0)
        self.assertEqual(r['content']['wer'], 0)
        self.assertFalse(r['table']['exact'])
        self.assertEqual(r['table']['cell_errors'], 6)
        self.assertFalse(r['exact_quality'])

    def test_unknown_table_gold_is_not_zero_tables(self):
        r = score(dict(text=self.gold['text'], tables=None), self.html, complete=True)
        self.assertEqual(r['content']['wer'], 0)
        self.assertTrue(r['content_exact'])
        self.assertFalse(r['table_gold_available'])
        self.assertIsNone(r['table'])
        self.assertIsNone(r['exact_quality'])

    def test_order_not_a_bag_of_words(self):
        text = self.html.replace('<td>Bolt</td><td>2</td>', '<td>Nut</td><td>7</td>', 1)
        r = score(self.gold, text, complete=True)
        self.assertGreater(r['content']['wer'], 0)
        self.assertEqual(r['table']['cell_errors'], 2)
        self.assertFalse(r['exact_quality'])
        plain = dict(text='Left first\nLeft second\nRight first\nRight second', tables=[])
        wrong = 'Left first\nRight first\nLeft second\nRight second'
        self.assertGreater(score(plain, wrong, complete=True)['content']['wer'], 0)

    def test_unicode_case_punctuation_and_hallucination_preserved(self):
        gold = dict(text='Café A-7 & < 9', tables=[])
        self.assertTrue(score(gold, '<p>Café A-7 &amp; &lt; 9</p>', complete=True)['exact_quality'])
        for text in ('Cafe A-7 & < 9', 'café A-7 & < 9', 'Café A7 & < 9', 'Café A-7 & < 9 Extra'):
            self.assertFalse(score(gold, text, complete=True)['exact_quality'])

    def test_whitespace_view_does_not_change_raw(self):
        gold = dict(text='Alpha beta\nGamma', tables=[])
        r = score(gold, 'Alpha\n beta   Gamma', complete=True)
        self.assertTrue(r['exact_quality'])
        self.assertGreater(r['raw']['cer'], 0)

    def test_incomplete_identical_output_is_not_complete_quality(self):
        r = score(dict(text='Alpha', tables=[]), 'Alpha', complete=False)
        self.assertEqual(r['content']['cer'], 0)
        self.assertFalse(r['exact_quality'])
        with self.assertRaises(ValueError): score(self.gold, self.html, complete=1)

    def test_missing_extra_transposed_cells_and_tables(self):
        for text in (self.html.replace('<td>7</td>', '<td></td>'), self.html + self.html,
                     self.html.replace('<th>Part</th><th>Count</th>', '<th>Count</th><th>Part</th>'),
                     self.html.replace('<td>7</td>', '<td>8</td>')):
            r = score(self.gold, text, complete=True)
            self.assertFalse(r['exact_quality'])
            self.assertGreater(r['table']['cell_errors'], 0)

    def test_unsupported_or_malformed_structure_is_explicit(self):
        for text in ('<script>hidden</script>Stock', '<!-- hidden -->Stock', '<table><tr><td>x</table>',
                     '<table><tr><td colspan="2">x</td></tr></table>', '<p style="display:none">x</p>',
                     '<table><tr>extra<td>x</td></tr></table>', '| A | B |\n| -- | --- |',
                     '| A | B |\n| --- | --- |\n| x |', '```html\n' + self.html + '\n```',
                     '<!DOCTYPE html>text', '<p>unclosed'):
            r = score(self.gold, text, complete=True)
            self.assertFalse(r['parse_ok'], text)
            self.assertIsNone(r['content'])
            self.assertFalse(r['exact_quality'])
            self.assertIn('cer', r['raw'])

    def test_actual_development_receipt_keeps_frozen_raw_metrics(self):
        previous = json.loads((ROOT / 'tests/results/qwen35-ocr/receipt.json').read_text())
        gold = dict(text=previous['gold']['text'], tables=None)
        for c in previous['cases']:
            actual = c['decode']
            r = score(gold, actual['native_text'], complete=actual['complete_document'])
            self.assertEqual(r['raw'], actual['native_vs_gold'])
            self.assertEqual(r['characters'], actual['native_characters'])
            self.assertEqual(r['lines'], actual['native_lines'])
            self.assertTrue(r['parse_ok'])
            self.assertEqual(len(r['parsed_tables']), 1)
            self.assertIsNone(r['table'])
            self.assertIsNone(r['exact_quality'])

    def test_prepared_pixel_and_annotation_drift_rejected(self):
        # Control reports are independent of renderer/model dependencies.
        report = dict(policy='crispembed-ocr-quality-v1', panel_sha256='panel',
            font=dict(name='font', sha256='fontsha'), split_evidence_sha256={'reference':'sha'},
            cases=[dict(id=str(i), source_sha256='source', gold_sha256='gold', original_size=[100,200],
                size=[32,64], png_sha256='png', rgb_sha256='rgb', file=str(i)+'.png', nominal_patches=8)
                for i in range(3)])
        self.assertTrue(verify_prepared(report, copy.deepcopy(report)))
        for field in ('png_sha256', 'rgb_sha256', 'gold_sha256', 'source_sha256'):
            changed = copy.deepcopy(report); changed['cases'][1][field] = 'changed'
            with self.assertRaises(ValueError): verify_prepared(report, changed)
        changed = copy.deepcopy(report); changed['cases'].pop()
        with self.assertRaises(ValueError): verify_prepared(report, changed)
        changed = copy.deepcopy(report); changed['cases'][1]['id'] = '0'
        with self.assertRaises(ValueError): verify_prepared(report, changed)
        changed = copy.deepcopy(report); changed['font']['sha256'] = 'changed'
        with self.assertRaises(ValueError): verify_prepared(report, changed)

    def test_panel_frozen_and_disjoint(self):
        self.assertEqual(len(validate_panel(ROOT)['cases']), 3)

    def test_panel_mutations_rejected(self):
        panel = validate_panel(ROOT)
        for key, value in [('source_sha256', '0'*64), ('gold_sha256', '0'*64), ('resize', [1024,1024]),
                           ('source', '../outside'), ('prompt', 'answer from gold'), ('license', 'restricted')]:
            bad = copy.deepcopy(panel); bad['cases'][0][key] = value
            with self.assertRaises(ValueError): validate_panel(ROOT, bad)
        bad = copy.deepcopy(panel); bad['cases'][1] = copy.deepcopy(bad['cases'][0])
        with self.assertRaises(ValueError): validate_panel(ROOT, bad)
        development = json.loads((ROOT / panel['development_reports'][0]).read_text())['cases'][0]
        bad = copy.deepcopy(panel)
        bad['cases'][0]['source'] = development['source']['local_file']
        bad['cases'][0]['source_sha256'] = development['source']['sha256']
        with self.assertRaises(ValueError): validate_panel(ROOT, bad)


if __name__ == '__main__':
    unittest.main()
