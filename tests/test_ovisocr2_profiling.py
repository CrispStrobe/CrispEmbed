"""Fail-closed controls for actual profile identity and measurement pairing."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ci-heavy'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ovisocr2_profiling import require_answer, require_neutral, attribution
from qwen35_profiling import parse_benchmark
from test_qwen35_profiling import fixture

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=json.loads((ROOT/'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']['native_cases']['f16']['dev-hello']

class Profiling(unittest.TestCase):
    def row(self):
        return dict(ids=REFERENCE['token_ids'],text=REFERENCE['raw_text'],stop=1,
                    confidences=REFERENCE['confidences'],mean_confidence=REFERENCE['mean_confidence'])
    def test_frozen_ids_text_and_eos(self):
        self.assertEqual(require_answer(self.row(),REFERENCE)['selected_probability_max_abs'],0)
        for key,value in [('ids',[1]*9),('text','wrong'),('stop',2)]:
            row=self.row();row[key]=value
            with self.assertRaises(ValueError):require_answer(row,REFERENCE)
    def pairs(self):
        rows=[]
        for pair in range(3):
            for on in ((0,1) if pair%2==0 else (1,0)):
                row=parse_benchmark(fixture(bool(on)));row.update(pair=pair,resources={'max_rss_kib':1024})
                rows.append(row)
        return rows
    def test_complete_neutral_pairs(self):
        self.assertEqual(require_neutral(self.pairs())['median_paired_on_over_off'],1)
    def test_different_scores_reject(self):
        rows=self.pairs();rows[1]['confidences'][0]=0.7
        with self.assertRaises(ValueError):require_neutral(rows)
    def test_missing_pair_rejects(self):
        with self.assertRaises(ValueError):require_neutral(self.pairs()[:-1])
    def test_changed_pair_order_rejects(self):
        rows=self.pairs();rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaises(ValueError):require_neutral(rows)
    def test_attribution_keeps_nested_phases_separate(self):
        summary=attribution(self.pairs()[1])
        self.assertEqual(set(summary),{'vision','prefill','decode'})
        self.assertEqual(summary['decode']['f32_bytes_materialized'],1017118720)

if __name__=='__main__':unittest.main()
