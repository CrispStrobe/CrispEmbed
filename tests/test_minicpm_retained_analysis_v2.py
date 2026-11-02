#!/usr/bin/env python3
"""Stdlib-only shape/coverage/source controls; no optional package imports."""
import copy,hashlib,importlib.util,json,pathlib,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/minicpm-retained-analysis-v2'))
import analysis_contract as c
import prior
spec=importlib.util.spec_from_file_location('scope_v2',ROOT/'tools/minicpm-retained-analysis-v2/analyze.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class Controls(unittest.TestCase):
 def test_complete_new_source_and_runtime83(self):
  d=c.source_guard(c.inherited.git('rev-parse','HEAD').decode().strip());self.assertFalse(d['runtime_math_changes']);self.assertEqual(d['inherited_guard']['old_source'],c.inherited.OLD_SOURCE)
 def test_line_full_overview_only(self):
  p=a.view_plan([235,1024],[235,1024],[940],[235],1024,True);self.assertEqual(p['native_rows'],[0,235]);self.assertEqual(p['unmapped'],[])
 def test_paragraph_and_tall_only_first_list_tensor(self):
  for patches,tokens in [([1008]*5,[252]*5),([1008]+[1040]*8,[252]+[260]*8)]:
   p=a.view_plan([252,1024],[sum(tokens),1024],patches,tokens,1024,True)
   self.assertEqual(p['native_rows'],[0,252]);self.assertEqual(sum(x['shape'][0]for x in p['unmapped'])+252,sum(tokens));self.assertEqual(p['unmapped'][-1]['native_rows'][1],sum(tokens));self.assertTrue(all(not x['reference_parity_claimed']for x in p['unmapped']))
 def test_no_padding_replication_full_reference_or_wrong_width(self):
  for original,native in [([1260,1024],[1260,1024]),([252,1024],[1259,1024]),([1,252,1024],[1260,1024]),([252,1152],[1260,1024])]:
   with self.assertRaises(ValueError):a.view_plan(original,native,[1008]*5,[252]*5,1024,True)
 def test_all_five_tower_observations_keep_full_patch_rows(self):
  self.assertEqual(a.view_plan([1,5040,1152],[5040,1152],[1008]*5,[252]*5,1152)['shape'],[5040,1152])
  with self.assertRaises(ValueError):a.view_plan([1,1008,1152],[5040,1152],[1008]*5,[252]*5,1152)
 def test_incomplete_receipt_exists_before_nonzero(self):
  with tempfile.TemporaryDirectory()as tmp:
   p=pathlib.Path(tmp)/'analysis.json'
   with self.assertRaisesRegex(ValueError,'incomplete'):a.finish({'completed':False,'cases':[]},p)
   self.assertFalse(json.loads(p.read_text())['completed'])
   catalog=json.loads(c.inherited.CATALOG.read_text());rows=[]
   for pin in catalog['cases']:
    rows.append(dict(**{k:pin[k]for k in ['id','dtype','graph']},completed=True,pixels_bit_exact=True,carried={}if pin['id']=='line'else False,vision={'stages':{'vision_'+str(i):{}for i in range(6)}}))
   a.finish({'cases':rows},p);self.assertTrue(json.loads(p.read_text())['completed'])
   with self.assertRaises(ValueError):a.finish({'completed':True,'cases':rows[:-1]},p)
   self.assertFalse(json.loads(p.read_text())['completed'])
 def test_reference_hook_first_list_policy_frozen(self):
  s=(ROOT/'tools/minicpm-ocr/reference.py').read_text();self.assertIn('if name in captured: return',s);self.assertIn('t = o[0] if isinstance(o, (tuple, list)) else o',s)
  initial=json.loads(c.INITIAL.read_text());self.assertEqual(initial['blueprint']['sha256'],'45d60914094cdc51fab67dc8edcf85a36a4db61a770eaca891cc820675fcf352')
 def test_no_second_canonical_compaction_or_forward(self):
  s=(ROOT/'tools/minicpm-retained-analysis-v2/analyze.py').read_text();self.assertNotIn('retention.compact_old(',s);self.assertNotIn('run_case(',s);self.assertIn("new_canonical_copy=False",s);self.assertIn("model_accessed=False",s)
  self.assertIn("if old_row['completed']:",s);self.assertIn("new_comparisons=0",s);self.assertIn("new_comparisons=6*result['new_arms']",s)
 def test_workflow_zero_model_and_failure_evidence(self):
  s=(ROOT/'.github/workflows/minicpm-graph-retained-analysis-v2.yml').read_text();self.assertIn('if: always()',s);self.assertIn('launch.py --timeout 10800',s);self.assertIn('build.py analysis',s);self.assertIn('fetch.py analysis',s);self.assertNotIn('build.py models',s);self.assertNotIn('continue.py',s)
 def test_prior_carried_six36_proof_and_negative_flags(self):
  cfg=c.declaration();cases=[];members={}
  arms=[('f32',0),('f32',1),('f16',0),('f16',1),('q8_0',1),('q4_k',1)]
  for i,(dtype,graph)in enumerate(arms):
   stages={};cpp=[]
   for j in range(6):
    name='vision_'+str(j);passed=j!=3;row={'name':name,'passed':passed};cpp.append(row)
    stages[name]=dict(cpp_ref=copy.deepcopy(row),passed=passed,independent_passed=passed,metric_agreement={'absolute':{'passed':True}})
   actual={'stages':6,'all_metrics_agree':True};cases.append(dict(id='line',dtype=dtype,graph=graph,completed=True,pixels_bit_exact=True,vision={'actual_ref':actual,'stages':stages}))
   members[f'analysis/{i}/diff.json']=json.dumps({'stages':cpp}).encode();members[f'analysis/{i}/comparison.json']=json.dumps({'actual_ref':actual}).encode()
  cases += [dict(id='para-en',dtype='authored'+str(i),graph=0,completed=False,error='Only known batch1 shape')for i in range(17)]
  report=dict(completed=False,comparisons=36,cases=cases,all_pixels_bit_exact=True,model_accessed=False,native_forwards=0,original_rerun=False,source={'source':c.SOURCE},original={'reference_sha256':cfg['original_reference_sha256']})
  members['analysis/analysis.json']=json.dumps(report).encode();self.assertEqual(len(prior.verify(members,cfg)['cases']),23)
  cases[0]['vision']['stages']['vision_3']['passed']=True;members['analysis/analysis.json']=json.dumps(report).encode()
  with self.assertRaises(ValueError):prior.verify(members,cfg)
 def test_no_optional_framework_imported_by_controls(self):
  self.assertNotIn('numpy',sys.modules);self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)
if __name__=='__main__':unittest.main()
