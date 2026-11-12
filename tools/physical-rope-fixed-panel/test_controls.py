#!/usr/bin/env python3
import copy,hashlib,importlib.util,json,tempfile,unittest,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixed',HERE/'worker.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)

class Controls(unittest.TestCase):
 def test_predeclared_order_caps_and_negative_baseline(self):
  cfg,selection=worker.admit(HERE/'protocol.json');self.assertEqual(cfg['arm_order'],['two-column-page/off','two-column-page/on','historical-receipt/off','historical-receipt/on','inventory-table/off','inventory-table/on']);self.assertEqual([(x['baseline_tokens'],x['baseline_eos'])for x in cfg['cases']],[(1536,False),(465,True),(248,True)]);self.assertEqual((cfg['cap'],cfg['context'],cfg['threads']),(1536,2048,4));self.assertTrue(cfg['continue_after_decoded_negative']);self.assertFalse(cfg['qualified']);self.assertFalse(cfg['tuning_allowed'])
 def test_only_rope_changes_fused_stays_on(self):
  before={'CRISPEMBED_LIGHTON4B_'+k:'1'for k in ['PHYSICAL_SOURCE_PREFILL','PHYSICAL_GDN_SOURCE_OPS','PHYSICAL_Q4_REPACK','PHYSICAL_Q6_TILED','PHYSICAL_PILLOW_RESIZE','PHYSICAL_F16_VISION','PHYSICAL_F16_ATTENTION','PHYSICAL_SWIGLU']};before.update({'CRISPEMBED_LIGHTON4B_'+k:'0'for k in ['HEAD_BF16','BF16_LINEAR','BF16_QUANT_WEIGHTS']});off,on=worker.gates(before);self.assertEqual({k for k in off if off[k]!=on[k]},{worker.GATE});self.assertEqual(off['CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU'],'1')
  before['CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU']='0'
  with self.assertRaises(RuntimeError):worker.gates(before)
 def test_stop_and_observed_admission_fail_closed(self):
  summary=dict(generated_tokens=1536,eos=False,cap=1536,upstream_physical=True,prompt_tokens=266,image_tokens=252,physical_source_prefill_chunks=True,physical_q6_tiled=True,physical_q4_repack=True,physical_repack_entries=216,physical_repack_bytes=1,converted_f32_weight_entries=0,converted_f32_weight_bytes=0,teacher_positions=1,source_fed_vision=False);worker.stop_state(summary,[7]*1536,1536,False)
  with self.assertRaises(RuntimeError):worker.stop_state(summary,[7]*1535,1536,False)
  with self.assertRaises(RuntimeError):worker.stop_state(summary,[248044]+[7]*1535,1536,False)
  cfg=json.loads((HERE/'protocol.json').read_text());good=dict(status='measured_physical_rope_observed_ab',runtime_producer_source=cfg['candidate_source'],candidate_controls_sha256='x',runtime_bundle={},cases={name:dict(off_all_rows_stages_exact=True,arms={'off':{},'on':dict(summary={'eos':True},meets_observed_upstream_floor=True)})for name in ['simple_table','simple_form','arabic_printed_line']});worker.admit_observed(good,cfg,'x',{})
  bad=copy.deepcopy(good);bad['cases']['simple_table']['arms']['on']['meets_observed_upstream_floor']=False
  with self.assertRaises(RuntimeError):worker.admit_observed(bad,cfg,'x',{})
 def test_closed_archive_census(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);p=root/'greedy-raw-rows.zip'
   with zipfile.ZipFile(p,'w')as z:z.writestr('greedy.0.logits.f32',b'\0'*(248320*4))
   arm={'raw_rows':dict(archive=dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()),rows=[dict(bytes=248320*4)])};worker.archive(root,arm)
   arm['raw_rows']['archive']['file']='../escape'
   with self.assertRaises(RuntimeError):worker.archive(root,arm)

if __name__=='__main__':unittest.main()
