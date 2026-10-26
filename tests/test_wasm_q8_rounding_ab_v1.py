#!/usr/bin/env python3
"""Small source/protocol/parser controls. No weights, compilers or inference."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import wasm_q8_rounding_ab_v1 as c
module=importlib.util.spec_from_file_location('ab_driver_v1',ROOT/'tools/ci-heavy/wasm_q8_rounding_ab_v1.py')
driver=importlib.util.module_from_spec(module);module.loader.exec_module(driver)

class Contract(unittest.TestCase):
 def setUp(self):self.spec=json.loads((ROOT/'docs/wasm-q8-rounding-ab-v1.json').read_text())
 def test_actual_operator_receipts_precede_model_access(self):
  declaration={'candidate':'fixed'}
  source={'schema':'crispembed-wasm-rounding-operator-source-v1','source':'a'*40,'ggml':c.CANDIDATE_GGML,'declaration':declaration,'model_access':False,'qualification':False}
  rows=[{'control':'converter','passed':True}]+[{'control':'backend','type':t,'indirect':i,'legacy':158,'nearest':189,'passed':True} for t in (8,20) for i in (False,True)]
  result={'schema':'crispembed-wasm-rounding-operator-result-v1','passed':True,'qualification':False,'controls':rows}
  c.operator_receipt(source,result,'a'*40,declaration)
  for key,value in [('source','b'*40),('model_access',True),('ggml',c.OLD_GGML)]:
   bad=copy.deepcopy(source);bad[key]=value
   with self.assertRaises(ValueError):c.operator_receipt(bad,result,'a'*40,declaration)
  for edit in ('missing','wrong-value','duplicate','failure'):
   bad=copy.deepcopy(result)
   if edit=='missing':bad['controls'].pop()
   elif edit=='wrong-value':bad['controls'][1]['nearest']=158
   elif edit=='duplicate':bad['controls'][1]=copy.deepcopy(bad['controls'][2])
   else:bad['passed']=False
   with self.assertRaises(ValueError):c.operator_receipt(source,bad,'a'*40,declaration)
 def test_review_pins_fail_closed(self):
  pending=copy.deepcopy(self.spec);pending['trace']['reviewed']=False
  with self.assertRaisesRegex(ValueError,'independently reviewed'):c.reviewed_spec(pending)
  trial=copy.deepcopy(self.spec);trial['trace']['reviewed']=True
  for key in ('review_artifact','raw_artifact'):trial['trace'][key].update(id=None,bytes=None,sha256=None)
  with self.assertRaisesRegex(ValueError,'artifact identity'):c.reviewed_spec(trial)
  for key in ('review_artifact','raw_artifact'):trial['trace'][key].update(id=1,bytes=100,sha256='a'*64)
  trial['trace']['review_artifact']['report_sha256']='b'*64
  self.assertIs(c.reviewed_spec(trial),trial)
  for field,bad in [('id',True),('bytes',0),('sha256','')]:
   broken=copy.deepcopy(trial);broken['trace']['raw_artifact'][field]=bad
   with self.assertRaises(ValueError):c.reviewed_spec(broken)
 def test_frozen_owned_panel(self):
  cases=c.panel(self.spec)
  self.assertEqual(c.digest(c.canonical(cases)),self.spec['panel_sha256'])
  self.assertEqual(cases[1]['text'],'');self.assertIn('\t',cases[5]['text']);self.assertIn('😀',cases[3]['text'])
  self.assertEqual(len(cases[-2]['text'].split()),420);self.assertEqual(len(cases[-1]['text'].split()),800)
  changed=copy.deepcopy(self.spec);changed['panel'][0]['literal']+=' '
  with self.assertRaisesRegex(ValueError,'content differs'):c.panel(changed)
  changed=copy.deepcopy(self.spec);changed['panel'][-1]['repeat']['count']=10**9
  with self.assertRaisesRegex(ValueError,'budget'):c.panel(changed)
 def test_trace_requires_all_neutrality_and_all_stages(self):
  report={'schema':'crispembed-successor-wasm-source-trace-v1','state':'attribution-complete','source':c.TRACE_SOURCE,
   'producer_source':c.PRODUCER,'model_sha256':c.MODEL_SHA,'token_ids':[101,102],
   'neutrality':{k:{'bit_exact':True,'sha256':'a'*64,'expected_sha256':'a'*64} for k in c.NEUTRALITY},
   'stages':{name:[{'stage':stage,'shape':[384,2],'native_sha256':'a'*64,'wasm_sha256':'b'*64} for stage in c.STAGES] for name in ('packaged-1-dump','packaged-2-dump')}}
  self.assertIs(c.trusted_report(report),report)
  bad=copy.deepcopy(report);bad['neutrality']={}
  with self.assertRaises(ValueError):c.trusted_report(bad)
  bad=copy.deepcopy(report);bad['neutrality'][c.NEUTRALITY[0]]['bit_exact']=False
  with self.assertRaises(ValueError):c.trusted_report(bad)
  bad=copy.deepcopy(report);bad['neutrality'][c.NEUTRALITY[0]].update(sha256=None,expected_sha256=None)
  with self.assertRaises(ValueError):c.trusted_report(bad)
  bad=copy.deepcopy(report);bad['stages']['packaged-2-dump'].pop()
  with self.assertRaises(ValueError):c.trusted_report(bad)
  bad=copy.deepcopy(report);bad['stages']['packaged-1-dump'][0]['shape']=[2,384]
  with self.assertRaises(ValueError):c.trusted_report(bad)
 def test_zip_path_duplicate_and_budget_controls(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'test.zip'
   with zipfile.ZipFile(p,'w') as z:z.writestr('safe/a',b'abc')
   self.assertEqual(driver.zip_files(p),{'safe/a':b'abc'})
   with self.assertRaises(ValueError):driver.zip_files(p,2)
   with zipfile.ZipFile(p,'w') as z:z.writestr('../escape',b'x')
   with self.assertRaises(ValueError):driver.zip_files(p)
   with zipfile.ZipFile(p,'w') as z:z.writestr('/absolute',b'x')
   with self.assertRaises(ValueError):driver.zip_files(p)
 def test_minimal_test_overlay_has_no_trait_quantizer_probe(self):
  paths=('src/crispembed.cpp','wasm/embed_wrapper.c','build-embed-wasm.sh');originals={p:(ROOT/p).read_bytes() for p in paths}
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for name,b in originals.items():
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
   changes=c.overlay_ab(root);self.assertEqual(set(changes),set(paths))
   self.assertNotIn('wasm_trace_q8',(root/'wasm/embed_wrapper.c').read_text())
   self.assertIn('wasm_trace_setenv',(root/'wasm/embed_wrapper.c').read_text())
   self.assertEqual((root/'src/crispembed.cpp').read_text().count('Test-only A/B capture'),2)
   for name in paths:self.assertEqual(changes[name]['before'],c.digest(originals[name]))
   with self.assertRaises(ValueError):c.overlay_ab(root)
  for name,b in originals.items():self.assertEqual((ROOT/name).read_bytes(),b)
 def test_exact_candidate_source_guard_and_dirty_rejection(self):
  proof=c.source_guard(ROOT,driver.RUNTIME_PATHS)
  self.assertEqual(proof['ggml'],c.CANDIDATE_GGML)
  self.assertEqual(proof['encoder_sha256'],c.ENCODER_SHA)
  temporary=ROOT/'src/.wasm-q8-ab-source-guard-negative'
  if temporary.exists():self.fail('Task-owned source guard negative already exists')
  try:
   temporary.write_text('untracked production drift must reject\n')
   with self.assertRaisesRegex(ValueError,'Dirty production'):c.source_guard(ROOT,driver.RUNTIME_PATHS)
  finally:temporary.unlink(missing_ok=True)
 def test_actual_norm_parser_is_strict(self):
  self.assertEqual(driver.raw_norm('[crispembed-rawnorm] 0.123456\n')['six_decimal_log'],.123456)
  for log in ('','[crispembed-rawnorm] 0\n','[crispembed-rawnorm] 1\n[crispembed-rawnorm] 1\n'):
   with self.assertRaises(ValueError):driver.raw_norm(log)
if __name__=='__main__':unittest.main()
