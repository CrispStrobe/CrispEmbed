"""Small protocol/negative controls; no checkpoint, NumPy, Torch or native build."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('original_retrieval',ROOT/'tools/moedern-colbert/scifact_original.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
    def test_local_host_rejected_before_framework_imports(self):
        with patch.dict(m.os.environ,{},clear=True),patch.object(m.Path,'is_dir',return_value=False):
            with self.assertRaisesRegex(ValueError,'hosted'):m.hosted()
    def test_exact_full_panel_and_immutable_source_declarations(self):
        cfg=json.loads((ROOT/'docs/moedern-scifact-parity-v1.json').read_text())
        self.assertTrue(cfg['original_first']);self.assertEqual(cfg['queries'],300);self.assertEqual(cfg['documents'],5183)
        self.assertEqual(cfg['variants'],['1.5M','27M','115M'])
        self.assertEqual(cfg['gates'],dict(max_absolute=.002,min_row_cosine=.99999,max_metric_drop=.001,min_top1_agreement=.99))
        for name,pin in cfg['source_sha256'].items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),pin,name)
    def test_full_retrieval_rejects_regression_and_partial_coverage(self):
        original=[dict(query_id=str(i),top1='doc',ndcg_at_10=.7,mrr_at_10=.6,recall_at_100=.9)for i in range(300)]
        gates=dict(max_metric_drop=.001,min_top1_agreement=.99)
        self.assertTrue(m.compare_rankings(original,copy.deepcopy(original),gates)['passed'])
        bad=copy.deepcopy(original);bad[0]['ndcg_at_10']=0
        self.assertFalse(m.compare_rankings(original,bad,gates)['passed'])
        bad=copy.deepcopy(original)
        for i in range(4):bad[i]['top1']='changed'
        self.assertFalse(m.compare_rankings(original,bad,gates)['passed'])
        with self.assertRaisesRegex(ValueError,'300'):m.compare_rankings(original,bad[:-1],gates)
        bad=copy.deepcopy(original);bad[0]['query_id']='changed'
        with self.assertRaisesRegex(ValueError,'order'):m.compare_rankings(original,bad,gates)
if __name__=='__main__':unittest.main()
