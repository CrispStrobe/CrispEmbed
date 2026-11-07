"""Unit tests for the OCR gold corpus tooling (no network, no inference).

python3 -m unittest discover -s tests -p test_ocr_gold_corpus.py
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'ocr-gold-corpus'))
import align_boxes as A  # noqa: E402
import build_corpus as B  # noqa: E402
import layout_score as L  # noqa: E402
import wikisource_layout as W  # noqa: E402


def blk(i, t, text, bbox=None, **kw):
    return dict(id=f'b{i}', order=i, type=t, text=text, bbox=bbox, **kw)


GOLD = [
    blk(0, 'header', '17 TABLEAU DE LA FRANCE', [100, 50, 900, 90]),
    blk(1, 'text', 'Capets, et les ducs de Bourgogne, de Bretagne', [100, 120, 900, 600]),
    blk(2, 'table', 'a b c d', [100, 650, 900, 900],
        table=dict(rows=2, cols=2, cells=[dict(row=0, col=0, text='a'), dict(row=0, col=1, text='b'),
                                          dict(row=1, col=0, text='c'), dict(row=1, col=1, text='d')])),
    blk(3, 'footnote', 'App. 8.', [100, 950, 400, 990]),
]


class KendallTau(unittest.TestCase):
    def test_identity_and_reverse(self):
        self.assertEqual(L.kendall_tau_b([0, 1, 2, 3], [0, 1, 2, 3]), 1.0)
        self.assertEqual(L.kendall_tau_b([0, 1, 2, 3], [3, 2, 1, 0]), -1.0)

    def test_one_swap(self):
        # 6 pairs, one discordant -> (5 - 1) / 6
        self.assertAlmostEqual(L.kendall_tau_b([0, 1, 2, 3], [1, 0, 2, 3]), 4 / 6)

    def test_degenerate(self):
        self.assertIsNone(L.kendall_tau_b([0], [0]))
        self.assertIsNone(L.kendall_tau_b([], []))


class IoU(unittest.TestCase):
    def test_values(self):
        self.assertEqual(L.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(L.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertAlmostEqual(L.iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)


class TableF1(unittest.TestCase):
    def test_html_and_markdown_parse(self):
        h = L.cells_from_html('<table><tr><td colspan="2">x</td></tr><tr><td>a</td><td>b</td></tr></table>')
        self.assertEqual([(c['row'], c['col'], c['text']) for c in h], [(0, 0, 'x'), (1, 0, 'a'), (1, 1, 'b')])
        md = L.cells_from_markdown('| a | b |\n|---|---|\n| c | d |')
        self.assertEqual([(c['row'], c['col'], c['text']) for c in md], [(0, 0, 'a'), (0, 1, 'b'), (1, 0, 'c'), (1, 1, 'd')])

    def test_rowspan_grid(self):
        h = L.cells_from_html('<table><tr><td rowspan="2">r</td><td>a</td></tr><tr><td>b</td></tr></table>')
        self.assertEqual([(c['row'], c['col']) for c in h], [(0, 0), (0, 1), (1, 1)])

    def test_f1(self):
        g = GOLD[2]['table']['cells']
        self.assertEqual(L.table_f1(g, g), (1.0, 1.0))
        shifted = [dict(c, col=c['col'] + 1) for c in g]  # right content, wrong columns
        cell, content = L.table_f1(g, shifted)
        self.assertEqual(cell, 0.0)
        self.assertEqual(content, 1.0)
        half = g[:2]
        cell, _ = L.table_f1(g, half)
        self.assertAlmostEqual(cell, 2 * 1.0 * 0.5 / 1.5)


class ScorePage(unittest.TestCase):
    def test_perfect(self):
        r = L.score_page(GOLD, [dict(b) for b in GOLD])
        self.assertEqual(r['block_f1'], 1.0)
        self.assertEqual(r['mean_iou'], 1.0)
        self.assertEqual(r['type_accuracy'], 1.0)
        self.assertEqual(r['reading_order_tau'], 1.0)
        self.assertEqual(r['table_cell_f1'], 1.0)

    def test_reversed_order_and_aliases(self):
        pred = [dict(b) for b in reversed(GOLD)]
        pred[-1]['type'] = 'page_header'  # alias of header
        r = L.score_page(GOLD, pred)
        self.assertEqual(r['block_f1'], 1.0)
        self.assertEqual(r['type_accuracy'], 1.0)
        self.assertEqual(r['reading_order_tau'], -1.0)

    def test_text_mode_without_boxes(self):
        pred = [dict(type=b['type'], text=b['text']) for b in GOLD]
        pred[2] = dict(type='table', text='<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>')
        r = L.score_page(GOLD, pred)
        self.assertEqual(r['matched_by_text'], 4)  # markup is stripped before text similarity
        self.assertIsNone(r['mean_iou'])
        self.assertEqual(r['table_cell_f1'], 1.0)  # predicted HTML table parsed into cells

    def test_low_iou_not_matched(self):
        pred = [dict(GOLD[1], bbox=[0, 2000, 10, 2010])]
        r = L.score_page(GOLD, pred)
        self.assertEqual(r['matched'], 0)
        self.assertEqual(r['block_recall'], 0.0)
        self.assertEqual(r['table_cell_f1'], 0.0)  # unmatched gold table scores 0

    def test_corpus_summary(self):
        corpus = dict(pages=[dict(id='p', blocks=GOLD), dict(id='q', blocks=GOLD)])
        res = L.score_corpus(corpus, {'p': GOLD})
        self.assertEqual(res['summary']['pages'], 1)
        self.assertEqual(res['summary']['missing_pages'], ['q'])


class WikisourceLayout(unittest.TestCase):
    WT = ('<noinclude><pagequality level="4" user="X" />{{rh|17|TITLE|}}</noinclude>'
          'Body one<ref>Note.</ref>\n\nBody two\n<noinclude><references/></noinclude>')

    def test_sections(self):
        h, b, f = W.split_sections(self.WT)
        self.assertEqual(h, '{{rh|17|TITLE|}}')
        self.assertTrue(b.startswith('Body one'))
        self.assertEqual(f, '<references/>')

    def test_blocks_from_rendered_html(self):
        html = ('<div class="mw-parser-output"><div class="prp-page-qualityheader quality4">validated</div>'
                '<p>17 TITLE</p><p>' + W.SENTINEL + '</p>'
                '<p>Body one<sup class="reference"><a>[1]</a></sup></p><p>Body two</p>'
                '<h2>Chapter</h2><table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>'
                '<ul><li>x</li><li>y</li></ul><div style="text-align:center">CENTRED</div>'
                '<p>' + W.SENTINEL + '</p><ol class="references"><li id="cite_note-1">'
                '<span class="mw-cite-backlink">↑</span> Note.</li></ol></div>')
        bl = W.blocks_from_html(html, 'en')
        self.assertEqual([b['type'] for b in bl],
                         ['header', 'text', 'text', 'title', 'table', 'list', 'title', 'footnote'])
        self.assertEqual(bl[1]['text'], 'Body one')
        self.assertEqual(bl[1]['markers'], ['1'])
        self.assertEqual(bl[4]['table']['rows'], 2)
        self.assertEqual(bl[5]['items'], ['x', 'y'])
        self.assertEqual(bl[7]['text'], 'Note.')
        self.assertEqual([b['order'] for b in bl], list(range(8)))

    def test_catchword(self):
        html = ('<div class="mw-parser-output"><p>' + W.SENTINEL + '</p><p>Long text here.</p><p>the</p>'
                '<p>' + W.SENTINEL + '</p></div>')
        bl = W.blocks_from_html(html, 'en')
        self.assertEqual(bl[-1]['type'], 'footer')
        self.assertIn('catchword', bl[-1]['type_provenance'])


class Licence(unittest.TestCase):
    def test_policy(self):
        for ok in ('Public domain', 'PD-old', 'CC0', 'CC BY 4.0', 'CC BY-SA 3.0'):
            self.assertTrue(B.licence_ok(ok)[0], ok)
        for bad in ('CC BY-NC 4.0', 'CC BY-ND 2.0', 'CC BY-NC-SA 4.0', '', 'All rights reserved'):
            self.assertFalse(B.licence_ok(bad)[0], bad)


class Align(unittest.TestCase):
    def test_boxes(self):
        tsv = 'level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
        words = [('Hello', 10, 10), ('world', 60, 10), ('Second', 10, 50), ('blok', 80, 50), ('here', 140, 50)]
        for w, x, y in words:
            tsv += f'5\t1\t1\t1\t1\t1\t{x}\t{y}\t40\t20\t90\t{w}\n'
        res = A.align([dict(text='Hello world'), dict(text='Second block here'), dict(text='missing words')],
                      A.read_tsv(tsv))
        self.assertEqual(res[0][0], [10, 10, 100, 30])
        self.assertEqual(res[1][0], [10, 50, 180, 70])  # 'blok' fuzzy-matched
        self.assertIsNone(res[2][0])


class CommittedCorpus(unittest.TestCase):
    def test_schema(self):
        path = ROOT / 'docs' / 'ocr-gold-corpus-v1.json'
        m = json.loads(path.read_text())
        self.assertEqual(m['schema'], 'crispembed-ocr-gold-corpus-v1')
        ids = [p['id'] for p in m['pages']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 40)
        for p in m['pages']:
            self.assertTrue(B.licence_ok(p['image_license']['license'])[0], p['id'])
            self.assertRegex(p['image'].get('sha256') or '', r'^[0-9a-f]{64}$', p['id'])
            self.assertEqual([b['order'] for b in p['blocks']], list(range(len(p['blocks']))), p['id'])
            for b in p['blocks']:
                self.assertIn(b['type'], L.TYPES, p['id'])
                self.assertRegex(b['type_provenance'], r'^(derived|manual):', p['id'])
                if b['bbox'] is not None:
                    self.assertRegex(b['bbox_provenance'], r'^(derived|manual):', p['id'])
                    x0, y0, x1, y1 = b['bbox']
                    self.assertTrue(0 <= x0 < x1 <= p['image']['width'] and 0 <= y0 < y1 <= p['image']['height'],
                                    (p['id'], b['id'], b['bbox']))
            if p['source'] == 'wikisource':
                self.assertIn('revision_id', p['provenance'])
        # v2 manifest stays untouched and every v2 page is reused
        v2 = json.loads((ROOT / 'docs' / 'ocr-real-scans-v2.json').read_text())
        self.assertEqual(v2['schema'], 'crispembed-ocr-real-scans-v2')
        self.assertEqual({p['id'] for p in v2['pages']}, {p['v2_id'] for p in m['pages'] if p.get('v2_id')})


if __name__ == '__main__':
    unittest.main()
