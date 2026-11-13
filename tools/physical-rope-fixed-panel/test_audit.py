"""Small independent raw-audit corruption and privacy controls; no model data."""
import hashlib,importlib.util,json,math,random,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
import audit
class AuditControls(unittest.TestCase):
 def build(self,root,values,duplicate=False):
  p=root/'greedy-raw-rows.zip';b=values.astype('<f4').tobytes();row=dict(position=0,bytes=len(b),argmax=7,sha256=hashlib.sha256(b).hexdigest())
  with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_STORED)as z:
   z.writestr('greedy.0.logits.f32',b)
   if duplicate:z.writestr('extra.logits.f32',b)
  record=dict(rows=[row],archive=dict(file=p.name,bytes=p.stat().st_size,sha256=audit.sha(p)));return record,{p.resolve():audit.sha(p)}
 def test_actual_finite_argmax_and_corrupt_rows(self):
  for mode in ['valid','wrong-argmax','nonfinite','extra-member']:
   with self.subTest(mode=mode),tempfile.TemporaryDirectory()as folder:
    root=Path(folder);x=np.zeros(audit.VOCAB,dtype='<f4');x[7]=1
    if mode=='wrong-argmax':x[8]=2
    if mode=='nonfinite':x[8]=np.nan
    r,known=self.build(root,x,mode=='extra-member')
    if mode=='valid':self.assertEqual(audit.raw_archive(root,r,[7],known)['rows'],1)
    else:
     with self.assertRaises(ValueError):audit.raw_archive(root,r,[7],known)
 def test_closed_paths_and_no_book_text(self):
  with tempfile.TemporaryDirectory()as folder:
   with self.assertRaises(ValueError):audit.owned(Path(folder),'../other')
  got=audit.safe_score(dict(raw_text='PROTECTED BOOK SENTENCE',parsed_text='PROTECTED BOOK SENTENCE',parsed_tables=[['PROTECTED']],parse_error='PROTECTED',complete=False,raw={'cer':2.,'wer':3.},content={'cer':1.,'wer':1.}))
  self.assertNotIn('PROTECTED',json.dumps(got));self.assertFalse(got['complete']);self.assertEqual(got['raw']['cer'],2.)
 def test_independent_edit_distance(self):
  self.assertEqual(audit.edit_count('kitten','sitting'),3);self.assertEqual(audit.edit_count('',[1,2]),2);self.assertEqual(audit.edit_count(['one','two'],['one','three','two']),1)
  score=dict(raw={'cer':1/3,'wer':1.},parsed_text='abd',content={'cer':1/3,'wer':1.});v=audit.independent_edits({'text':'abc'},'abd',score);self.assertEqual(v['raw']['character_edits'],1)
 def test_bit_parallel_matches_separate_frozen_dp(self):
  path=Path(__file__).resolve().parents[1]/'ci-heavy/qwen35_quant_calibration.py';spec=importlib.util.spec_from_file_location('frozen_distance',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);rng=random.Random(0)
  for _ in range(100):
   left=''.join(rng.choices('abcأعé',k=rng.randrange(65)));right=''.join(rng.choices('abcأعé',k=rng.randrange(65)));self.assertEqual(audit.edit_count(left,right),module.distance(left,right))
 def test_only_derived_rms_roundoff_is_bounded(self):
  expected={'rms':1.,'max_abs':2.,'source_rms_magnitude':1.};actual=dict(expected);actual['rms']+=8*math.ulp(1.);self.assertEqual(audit.compare_derived(actual,expected),8.)
  actual['rms']+=math.ulp(1.)
  with self.assertRaises(ValueError):audit.compare_derived(actual,expected)
  actual=dict(expected);actual['max_abs']+=math.ulp(2.)
  with self.assertRaises(ValueError):audit.compare_derived(actual,expected)
 def test_only_exact_outer_bootstrap_finalization_collision(self):
  final={'status':'complete','candidate_source':'a8a22749ab0b757ae1f317b6fa5b84ffd148b939','diagnostic_source':'2ffa681dafa89cc5091715b74d15d6f937eda08f','model_forward_executed':False,'source_capture_repeated':False,'full_f32_model_instantiated':False,'qualified':False,'phase':'worker','elapsed_seconds':169.47456825699993}
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);p=root/'physical-rope-bootstrap.json';p.write_text(json.dumps(final,indent=2)+'\n');other=root/'retained.log';other.write_bytes(b'retained')
   rows={'physical-rope-bootstrap.json':{'bytes':312,'sha256':'db1fa9c9a743752a988b2f9b5f8aaa7f79c8f6caac15232698570bc6bb8d0829'},'retained.log':{'bytes':8,'sha256':audit.sha(other)}}
   known,journal=audit.control_manifest(root,rows,final['candidate_source']);self.assertEqual(len(known),1);self.assertTrue(journal['retained_final_independently_rehashed']);self.assertFalse(journal['recorded_intermediate_independently_rehashed'])
   other.write_bytes(b'corrupt!')
   with self.assertRaises(ValueError):audit.control_manifest(root,rows,final['candidate_source'])
   other.write_bytes(b'retained');rows['physical-rope-bootstrap.json']['bytes']=313
   with self.assertRaises(ValueError):audit.control_manifest(root,rows,final['candidate_source'])
   rows['physical-rope-bootstrap.json']['bytes']=312;p.write_text('{}')
   with self.assertRaises(ValueError):audit.control_manifest(root,rows,final['candidate_source'])
 def test_complete_admission_rejects_missing_or_failed_arms(self):
  cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())
  def fixture():
   return dict(schema=cfg['schema'],status='measured_physical_rope_fixed_panel',candidate_source=cfg['candidate_source'],controller_source=cfg['controller_source'],qualified=False,references_expert_adjudicated=False,reference_forward_executed=False,source_feature_forward_executed=False,heldout_forward_executed=True,observed_receipt_sha256=audit.OBSERVED_SHA,observed_controls_reused_without_forward=True,all_three_historical_inputs_baselines_audits_authenticated_before_models=True,cases={name:dict(arms={'off':{},'on':{}},off_all_historical_rows_stages_stop_exact=True)for name in cfg['case_order']},commands={name+'-'+arm:dict(status='completed',returncode=0)for name in cfg['case_order']for arm in ['off','on']},admissions={name+'-'+arm:dict(accepted=True)for name in cfg['case_order']for arm in ['off','on']})
  audit.admit_complete(fixture(),cfg)
  for change in [lambda r:r.update(status='failed'),lambda r:r['cases']['two-column-page']['arms'].pop('on'),lambda r:r['commands']['two-column-page-on'].update(returncode=1),lambda r:r['admissions']['inventory-table-on'].update(accepted=False),lambda r:r['cases']['historical-receipt'].update(off_all_historical_rows_stages_stop_exact=False),lambda r:r.update(reference_forward_executed=True)]:
   r=fixture();change(r)
   with self.assertRaises(ValueError):audit.admit_complete(r,cfg)
 def test_no_ignored_earlier_eos(self):
  s=dict(generated_tokens=2,cap=1536,eos=True,upstream_physical=True,source_fed_vision=False,teacher_positions=1,physical_source_prefill_chunks=True,physical_q6_tiled=True,physical_q4_repack=True,physical_repack_entries=216,physical_repack_bytes=1662861312,converted_f32_weight_entries=0,converted_f32_weight_bytes=0,prompt_tokens=86,image_tokens=72)
  audit.actual_summary(s,[7,248046],1536)
  s['generated_tokens']=3
  with self.assertRaises(ValueError):audit.actual_summary(s,[248046,7,248046],1536)
if __name__=='__main__':unittest.main()
