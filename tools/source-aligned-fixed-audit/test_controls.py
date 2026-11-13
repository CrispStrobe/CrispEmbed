#!/usr/bin/env python3
import copy,hashlib,importlib.util,json,math,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('aligned_audit',HERE/'audit.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
class Controls(unittest.TestCase):
 def test_frozen_sources_and_scope(self):
  self.assertEqual(a.sha(a.ROOT/'tools/receipt-aligned-decoded/fixed-protocol.json'),a.CFG['fixed_protocol_sha256']);self.assertEqual(a.sha(a.ROOT/'tools/physical-rope-fixed-panel/audit.py'),a.CFG['legacy_audit_helper_sha256']);self.assertFalse(any(a.CFG[k]for k in ['model_forward_allowed','source_forward_allowed','network_allowed','raw_local_download_allowed']));self.assertEqual((a.CFG['expected_observed_rows'],a.CFG['expected_baseline_rows']),(916,1370))
 def test_complete_order_and_negative_results_remain_measured(self):
  cfg=json.loads((a.ROOT/'tools/receipt-aligned-decoded/fixed-protocol.json').read_text());names=cfg['case_order'];r=dict(schema=cfg['schema'],status='measured_source_aligned_fixed_panel',candidate_source=cfg['candidate_source'],controller_source=cfg['candidate_source'],qualified=False,references_expert_adjudicated=False,reference_forward_executed=False,source_feature_forward_executed=False,model_acquisition=False,heldout_forward_executed=True,retained_model_reused=True,observed_receipt_sha256=a.CFG['observed_sha256'],controls_sha256=a.CFG['controls_sha256'],cases={n:dict(off_all_historical_rows_stages_stop_exact=True,arms={'off':{},'on':{}})for n in names},commands={n+'-'+arm:dict(returncode=0,status='completed')for n in names for arm in ['off','on']},admissions={n+'-'+arm:dict(accepted=True)for n in names for arm in ['off','on']},all_on_complete=False,on_source_floors_passed=0);a.admit_complete(r,cfg)
  for mutation in ['partial','reordered','reference']:
   x=copy.deepcopy(r)
   if mutation=='partial':x['cases'][names[0]]['arms'].pop('on')
   elif mutation=='reordered':x['cases']=dict(reversed(list(x['cases'].items())))
   else:x['reference_forward_executed']=True
   with self.assertRaises(ValueError):a.admit_complete(x,cfg)
 def test_raw_row_finite_argmax_and_closed_zip(self):
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);path=root/'greedy-raw-rows.zip';x=np.zeros(248320,dtype='<f4');x[7]=1
   def record(data,extra=False):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED)as z:
     z.writestr('greedy.0.logits.f32',data)
     if extra:z.writestr('../escape',b'x')
    digest=hashlib.sha256(path.read_bytes()).hexdigest();return dict(archive=dict(file=path.name,bytes=path.stat().st_size,sha256=digest),rows=[dict(position=0,bytes=248320*4,argmax=7,sha256=hashlib.sha256(data).hexdigest())]),{path.resolve():digest}
   row,known=record(x.tobytes());self.assertEqual(a.raw_archive(root,row,[7],known)['rows'],1)
   row,known=record(x.tobytes(),True)
   with self.assertRaises(ValueError):a.raw_archive(root,row,[7],known)
   x[0]=np.nan;row,known=record(x.tobytes())
   with self.assertRaises(ValueError):a.raw_archive(root,row,[7],known)
 def test_stats_preserve_signed_zero_identity(self):
  r=a.stats(np.array([0.,1.],dtype='<f4'),np.array([-0.,1.],dtype='<f4'));self.assertFalse(r['bit_exact']);self.assertEqual((r['rms'],r['max_abs']),(0.,0.))
  with self.assertRaises(ValueError):a.stats(np.array([np.nan]),np.array([0.]))
 def test_derived_bound_does_not_relax_maxima(self):
  value=1.
  for _ in range(8):value=math.nextafter(value,2.)
  self.assertEqual(a.compare_derived(dict(rms=1.,max_abs=2.),dict(rms=value,max_abs=2.)),8.)
  with self.assertRaises(ValueError):a.compare_derived(dict(rms=1.,max_abs=2.),dict(rms=math.nextafter(value,2.),max_abs=2.))
  with self.assertRaises(ValueError):a.compare_derived(dict(rms=1.,max_abs=2.),dict(rms=1.,max_abs=math.nextafter(2.,3.)))
 def test_safe_score_drops_private_payload(self):
  text='PRIVATE_BOOK_SENTINEL';score=dict(policy='frozen',raw_text=text,parsed_text=text,parsed_tables=[[text]],raw=dict(cer=0.,wer=0.),content=dict(cer=0.,wer=0.));self.assertNotIn(text,json.dumps(a.safe_score(score)))
 def test_safe_projection_altered_floor_hash_plan_rejected(self):
  av=dict(summary=dict(eos=True),chunk_sizes=[4],firstscore=dict(rms=0.),raw_rows=dict(archive=dict(sha256='a'*64)));actual=dict(tokens=1,score={},independent_edits={},source_output_ids_equal=True,source_output_text_equal=True,source_floor=True,raw=dict(ordered_row_sha256='b'*64));p=dict(summary=av['summary'],chunk_sizes=[4],score={},independent_edits={},source_output_ids_equal=True,source_output_text_equal=True,meets_provisional_source_floor=True,firstscore=av['firstscore'],generated_ids_sha256='c'*64,decoded_utf8_sha256='d'*64,raw_rows=dict(count=1,archive=av['raw_rows']['archive'],ordered_row_sha256='b'*64));a.check_projected_arm(p,av,actual,'c'*64,'d'*64)
  for key,value in [('meets_provisional_source_floor',False),('generated_ids_sha256','e'*64),('chunk_sizes',[3]),('firstscore',dict(rms=1.))]:
   changed=copy.deepcopy(p);changed[key]=value
   with self.assertRaises(ValueError):a.check_projected_arm(changed,av,actual,'c'*64,'d'*64)
 def test_closed_owned_path(self):
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);(root/'ok').write_text('x');self.assertEqual(a.owned(root,'ok'),root/'ok')
   with self.assertRaises(ValueError):a.owned(root,'../escape')
if __name__=='__main__':unittest.main()
