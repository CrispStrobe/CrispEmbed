#!/usr/bin/env python3
"""Tiny stdlib source/coverage/prior-negative controls; no arrays or models."""
import ast,copy,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/minicpm-seven-v2'))
import seven_contract as c
s=importlib.util.spec_from_file_location('seven_owner',ROOT/'tools/minicpm-seven-v2/run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Controls(unittest.TestCase):
 def fixture(self):
  cases=[]
  for row in json.loads(c.old.CATALOG.read_text())['cases']:
   stages={}
   for i in range(6):
    name='vision_'+str(i);flag=i!=3
    stages[name]=dict(independent_passed=flag,passed=flag,cpp_ref=dict(name=name,passed=flag),metric_agreement={'authored':{'passed':True}})
   cases.append(dict(id=row['id'],dtype=row['dtype'],graph=row['graph'],completed=True,pixels_bit_exact=True,vision=dict(stages=stages,actual_ref=dict(stages=6,all_metrics_agree=True))))
  return dict(completed=True,qualified=False,native_forwards=0,model_accessed=False,original_rerun=False,comparisons=138,carried_comparisons=36,new_comparisons=102,carried_arms=6,new_arms=17,source={'source':'cec1faebd39582a6103774dd9598b3e26d6533f1'},cases=cases,all_pixels_bit_exact=True,historical_numeric_failures_preserved=True)
 def test_complete_negative_prior_admitted(self):self.assertEqual(len(c.verify_analysis(self.fixture())['cases']),23)
 def test_incomplete_prior_extra_rows_pixels_or_waived_flags_reject(self):
  original=self.fixture();bad=[]
  x=copy.deepcopy(original);x['completed']=False;bad.append(x)
  x=copy.deepcopy(original);x['cases']=x['cases'][:-1];bad.append(x)
  x=copy.deepcopy(original);x['cases'][0]['pixels_bit_exact']=False;bad.append(x)
  x=copy.deepcopy(original);x['cases'][0]['vision']['stages']['vision_3']['passed']=True;bad.append(x)
  x=copy.deepcopy(original);x['cases'][0]['vision']['stages']['vision_0']['metric_agreement']['authored']['passed']=False;bad.append(x)
  for x in bad:
   with self.assertRaises(ValueError):c.verify_analysis(x)
 def test_exact23_seven_partition(self):
  catalog=json.loads(c.old.CATALOG.read_text());missing=c.old.remaining(catalog)
  self.assertEqual(missing,[('tall','q4_k',1)]+[('tiny',d,g)for d,g in [('f32',0),('f32',1),('f16',0),('f16',1),('q8_0',1),('q4_k',1)]])
  self.assertFalse(set(missing)&{(r['id'],r['dtype'],r['graph'])for r in catalog['cases']})
 def test_atomic_incomplete_and_complete42_180_classification(self):
  with tempfile.TemporaryDirectory()as temp:
   path=Path(temp)/'result.json'
   with self.assertRaises(ValueError):m.finish({'old_seven_completed':False,'new_vision':[]},path)
   self.assertFalse(json.loads(path.read_text())['completed'])
   rows=[dict(id=v[0],dtype=v[1],graph=v[2],completed=True,pixels_bit_exact=True,vision=dict(stages={'vision_'+str(i):{}for i in range(6)},actual_ref=dict(stages=6,all_metrics_agree=True)))for v in c.declaration()['remaining_arms']]
   m.finish(dict(old_seven_completed=True,new_vision=rows),path);x=json.loads(path.read_text());self.assertTrue(x['completed']);self.assertFalse(x['qualified']);self.assertEqual((x['new_vision_comparisons'],x['combined_vision_comparisons']),(42,180))
   rows[0]['vision']['actual_ref']['all_metrics_agree']=False
   with self.assertRaises(ValueError):m.finish(dict(old_seven_completed=True,new_vision=rows),path)
   self.assertFalse(json.loads(path.read_text())['completed'])
 def test_actual_old_runtime_helpers_guards_byte_exact(self):
  for p in ['tools/minicpm-retained-v3/common.py','tools/minicpm-retained-v3/continue.py','tools/minicpm-retained-v3/build.py','tools/minicpm-retained-v3/fetch.py','docs/minicpm-graph-seven-arm-v1.json']:
   self.assertEqual((ROOT/p).read_bytes(),subprocess.check_output(['git','show','19c3f7bdf6d452614721946adefb95f6d3cf0180:'+p],cwd=ROOT))
  self.assertEqual(subprocess.check_output(['git','rev-parse','HEAD:src'],cwd=ROOT),subprocess.check_output(['git','rev-parse',c.old.OLD_SOURCE+':src'],cwd=ROOT))
 def test_no_guard_monkeypatch_or_second_forward(self):
  code=(ROOT/'tools/minicpm-seven-v2/run.py').read_text();vision=(ROOT/'tools/minicpm-seven-v2/vision.py').read_text()
  self.assertNotIn('run_case(',code+vision);self.assertNotIn('source_guard =',code+vision);self.assertNotIn('source_guard=',code+vision)
  self.assertIn('frozen.view_plan(',vision);self.assertIn('retention.compare(',vision);self.assertIn('original/native packed pixels must be bit exact'.lower(),vision.lower())
  self.assertIn('raw_span_sha256',vision);self.assertIn('first_strict_failure',vision)
 def test_dependency_before_models_and_only_frozen_body_runs(self):
  workflow=(ROOT/'.github/workflows/minicpm-graph-seven-arm-v2.yml').read_text()
  self.assertLess(workflow.index('run.py prepare'),workflow.index('pip install'))
  self.assertLess(workflow.index('run.py verify'),workflow.index('fetch.py continue'))
  self.assertIn('tools/minicpm-retained-v3/continue.py',workflow);self.assertIn('launch.py --timeout 14400',workflow);self.assertIn('timeout-minutes: 360',workflow)
  self.assertIn('--threads 4',workflow);self.assertIn('secrets.KAGGLE_TOKEN',workflow);self.assertIn('actions: read',workflow)
  self.assertEqual(workflow.count('if: always()'),4);self.assertEqual(workflow.count('retention-days: 90'),3)
 def test_signed_storage_receives_no_api_token(self):
  code=(ROOT/'tools/minicpm-seven-v2/run.py').read_text();self.assertIn("urllib.request.urlopen(e.headers['Location'],timeout=180)",code)
  self.assertIn("'Signed HTTPS redirect required'",code);self.assertIn('len(z.infolist())==134',code);self.assertIn("set(members)==set(expected['small_members'])",code)
 def test_newcomparator_and_all23_case_proof_checked(self):
  code=(ROOT/'tools/minicpm-seven-v2/run.py').read_text()
  for key in ["c.sha(diff)==build['binary_sha256']['diff']","old['carried_rows']==json.loads(panel.read_text())['cases']","all_omitted_adapter_bytes_reconstructed_and_sha_verified","files_manifest_sha256","case_report_sha256"]:self.assertIn(key,code)
 def test_closed_source_and_exact_actual_dependency(self):
  value=c.source_guard(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());self.assertFalse(value['runtime_math_changes'])
  cfg=c.declaration();self.assertEqual(cfg['completed_analysis_service']['id'],40831080193);self.assertEqual(cfg['completed_analysis_report']['sha256'],'9f84356c5c31b0aec4f3182b1b84acb780ef5a9ddfaabffcd233917b97df3442')
 def test_real_isolated_help_and_frozen_view_import_need_no_packages(self):
  result=subprocess.run([sys.executable,'-I',str(ROOT/'tools/minicpm-seven-v2/run.py'),'--help'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
  self.assertEqual(result.returncode,0,result.stderr);self.assertIn('finalize',result.stdout)
  import vision
  value=vision.frozen.view_plan([252,1024],[1260,1024],[1008]*5,[252]*5,1024,True)
  self.assertEqual(value['native_rows'],[0,252]);self.assertEqual(len(value['unmapped']),4)
 def test_optional_packages_never_imported_by_source_controls(self):
  for name in ['numpy','gguf','torch','transformers']:self.assertNotIn(name,sys.modules)
 def test_python_ast_only(self):
  for p in(ROOT/'tools/minicpm-seven-v2').glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
