"""Meaningful scorer controls; synthetic vectors are never model evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ultrabert_retrieval import FIXTURE_SHA256, QUERY, evaluate, fixture

class ScorerControls(unittest.TestCase):
    def setUp(self):
        self.data=fixture(Path(__file__).resolve().parent/'regression/ultrabert/retrieval-smoke-v1.json')
        rows=[]
        for i,q in enumerate(self.data['queries']):
            v=[0.]*768;v[i]=1.;rows.append(dict(id=q['id'],raw_cls=v,normalized_cls=v.copy()))
        for i,d in enumerate(self.data['documents']):
            v=[0.]*768;v[i//3 if i%3==0 else 8+i]=1.;rows.append(dict(id=d['id'],raw_cls=v,normalized_cls=v.copy()))
        self.record=dict(pooling='cls',query_prompt=QUERY,document_prompt='',fixture_sha256=FIXTURE_SHA256,vectors=rows)
    def test_perfect_control_and_non_equivalent_vector_rejected(self):
        good=evaluate(self.data,self.record,self.record)
        self.assertTrue(good['passed']);self.assertEqual(len(good['vectors']),64)
        bad=deepcopy(self.record);bad['vectors'][0]['raw_cls'][0]=-1;bad['vectors'][0]['normalized_cls'][0]=-1
        result=evaluate(self.data,self.record,bad)
        self.assertFalse(result['vector_pass']);self.assertFalse(result['passed'])
    def test_exact_numerical_agreement_does_not_hide_bad_retrieval(self):
        bad=deepcopy(self.record)
        for row in bad['vectors'][:8]:
            row['raw_cls']=[1.]+[0.]*767;row['normalized_cls']=row['raw_cls'].copy()
        result=evaluate(self.data,bad,bad)
        self.assertTrue(result['vector_pass']);self.assertFalse(result['quality_pass']);self.assertFalse(result['passed'])
    def test_wrong_pooling_prefix_order_or_missing_vector_rejected(self):
        for key,value in [('pooling','mean'),('query_prompt',''),('document_prompt',QUERY)]:
            bad=deepcopy(self.record);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):evaluate(self.data,self.record,bad)
        for rows in [self.record['vectors'][:-1],list(reversed(self.record['vectors']))]:
            bad=deepcopy(self.record);bad['vectors']=rows
            with self.assertRaises(ValueError):evaluate(self.data,self.record,bad)
    def test_nonfinite_zero_and_wrong_dimension_rejected(self):
        for v in [[float('nan')]+[0.]*767,[0.]*768,[1.]*767]:
            bad=deepcopy(self.record);bad['vectors'][0]['raw_cls']=v
            with self.assertRaises(ValueError):evaluate(self.data,self.record,bad)

if __name__=='__main__':unittest.main()
