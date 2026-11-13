#!/usr/bin/env python3
import ast,hashlib,importlib.util,json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def module(name):
 spec=importlib.util.spec_from_file_location('attention252_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

worker=module('worker');patch=module('source_patch')

class Controls(unittest.TestCase):
 def test_closed_scope(self):
  cfg=json.loads((HERE/'protocol.json').read_text())
  self.assertFalse(any(cfg[k]for k in['full_f32_model_allowed','native_full_model_forward_allowed','full_decode_allowed']))
  self.assertEqual(len(cfg['tensors']),11);self.assertEqual({r['type']for r in cfg['tensors']},{0,12,14})
  tree=ast.parse((HERE/'source_worker.py').read_text());predict=[v.value for n in ast.walk(tree)if isinstance(n,ast.Dict)for k,v in zip(n.keys,n.values)if isinstance(k,ast.Constant)and k.value=='n_predict'and isinstance(v,ast.Constant)]
  self.assertEqual(predict,[1])
 def test_observer_reads_aliases_without_mutating_tensors(self):
  m=patch.configured();self.assertIn('ggml_tensor operand=*t->src[i]',m.HELPER);self.assertIn('ggml_set_name(&operand',m.HELPER)
  self.assertNotIn('unsupported-strides',m.HELPER);self.assertIn('GGML_TYPE_F16',m.HELPER)
  self.assertIn('n==0)',m.HELPER);self.assertNotIn('n==31',m.HELPER)
  self.assertIn('kq_soft_max-3',m.STAGES);self.assertIn('Qcur_normed-3',m.STAGES)
 def test_duplicate_source_labels_only_disambiguate_raw_callbacks(self):
  before='    cb(Qcur_full, "Qcur_full", il);\n    cb(Kcur, "Kcur", il);\n    cb(Vcur, "Vcur", il);\nCALL_COMPUTATION();\n    cb(Kcur, "Kcur", il);\n    cb(Vcur, "Vcur", il);'
  after=patch.disambiguate(before)
  self.assertEqual(after.replace('"Kcur_raw"','"Kcur"').replace('"Vcur_raw"','"Vcur"'),before)
  self.assertEqual(after.count('cb(Kcur, "Kcur", il)'),1);self.assertEqual(after.count('cb(Vcur, "Vcur", il)'),1)
  with self.assertRaises(ValueError):patch.disambiguate(before+'\ncb(Kcur, "Kcur", il)')
 def test_stride_aware_f16_reader(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'stage.bin').write_bytes(struct.pack('<6e',1,2,99,3,4,99))
   r={'name':'x','status':'captured','type':'1','file':'stage.bin','bytes':'12','ne0':'2','ne1':'2','ne2':'1','ne3':'1','nb0':'2','nb1':'6','nb2':'12','nb3':'12'}
   x=worker.source_array(p,{'records':[r]},'x');self.assertEqual(x.reshape(-1).tolist(),[1,2,3,4])
   r['bytes']='13'
   with self.assertRaises(ValueError):worker.source_array(p,{'records':[r]},'x')
 def test_seals_and_finite_metrics(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);(p/'a').write_bytes(b'abcd');seal={'a':{'bytes':4,'sha256':hashlib.sha256(b'abcd').hexdigest()}}
   self.assertEqual(worker.owned(p,'a',seal),p/'a');(p/'a').write_bytes(b'efgh')
   with self.assertRaises(ValueError):worker.owned(p,'a',seal)
   with self.assertRaises(ValueError):worker.owned(p,'../escape',seal)
  m=worker.delta(np.array([3.,4.],dtype='<f4'),np.zeros(2,dtype='<f4'));self.assertEqual(m['error_max_abs'],4);self.assertAlmostEqual(m['error_rms'],(12.5)**.5)
  with self.assertRaises(ValueError):worker.delta(np.array([np.nan]),np.array([0.]))

if __name__=='__main__':unittest.main()
