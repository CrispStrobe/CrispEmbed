"""Model-free frozen-panel admission, gate isolation and capped-stop controls."""
import copy,json,tempfile,unittest
from pathlib import Path
import worker
class Controls(unittest.TestCase):
 def test_original_contract(self):
  cfg,selection=worker.admit(Path(__file__).with_name('protocol.json'));self.assertEqual([v['id']for v in cfg['cases']],[v['id']for v in selection['cases']]);self.assertEqual([v['baseline_tokens']for v in cfg['cases']],[468,1536,248]);self.assertEqual([v['baseline_eos']for v in cfg['cases']],[True,False,True]);self.assertEqual(cfg['changed_gates'],[worker.GATE])
 def test_mutated_protocol_rejected_before_models(self):
  cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())
  for key,value in [('cap',2048),('threads',2),('worker_seconds',25200),('qualified',True),('reference_forward_allowed',True),('source_feature_forward_allowed',True),('tuning_allowed',True)]:
   bad=copy.deepcopy(cfg);bad[key]=value
   with tempfile.TemporaryDirectory()as folder:
    p=Path(folder)/'protocol.json';p.write_text(json.dumps(bad,indent=2)+'\n')
    with self.assertRaises(RuntimeError):worker.admit(p)
 def old_gates(self):
  p='CRISPEMBED_LIGHTON4B_';r={p+k:'1'for k in ['PHYSICAL_SOURCE_PREFILL','PHYSICAL_GDN_SOURCE_OPS','PHYSICAL_Q4_REPACK','PHYSICAL_Q6_TILED','PHYSICAL_PILLOW_RESIZE','PHYSICAL_F16_VISION','PHYSICAL_F16_ATTENTION']};r.update({p+k:'0'for k in ['HEAD_BF16','BF16_LINEAR','BF16_QUANT_WEIGHTS']});return r
 def test_only_swiglu_changes_with_fixed_planner(self):
  original=self.old_gates();off,on=worker.gates(original);self.assertNotIn(worker.GATE,original);self.assertEqual({k for k in off if off[k]!=on[k]},{worker.GATE});self.assertEqual(off[worker.GATE],'0');self.assertEqual(on[worker.GATE],'1')
  for key in ['CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL','CRISPEMBED_LIGHTON4B_HEAD_BF16']:
   bad=dict(original);bad[key]='0'if original[key]=='1'else'1'
   with self.assertRaises(RuntimeError):worker.gates(bad)
 def summary(self,n,eos):return dict(generated_tokens=n,eos=eos,cap=1536,upstream_physical=True,prompt_tokens=86,image_tokens=72,physical_source_prefill_chunks=True,physical_q6_tiled=True,physical_q4_repack=True,physical_repack_entries=216,physical_repack_bytes=1662861312,converted_f32_weight_entries=0,converted_f32_weight_bytes=0,teacher_positions=1,source_fed_vision=False)
 def test_book_cap_is_exact_negative_baseline(self):
  ids=[7]*1536;worker.stop_state(self.summary(1536,False),ids,1536,False)
  with self.assertRaises(RuntimeError):worker.stop_state(self.summary(1536,False),ids,1536,True)
  with self.assertRaises(RuntimeError):worker.stop_state(self.summary(1054,False),ids[:1054],1536,False)
 def test_complete_and_cache_admission(self):
  worker.stop_state(self.summary(468,True),[7]*467+[248046],468,True)
  bad=self.summary(468,True);bad['converted_f32_weight_bytes']=4
  with self.assertRaises(RuntimeError):worker.stop_state(bad,[7]*467+[248046],468,True)
  with self.assertRaises(RuntimeError):worker.stop_state(self.summary(1537,False),[7]*1537)
 def test_observed_stop_and_sealed_counts(self):
  s=self.summary(168,True);s['cap']=512;ids=[7]*167+[248046];worker.stop_state(s,ids,168,True,512,86,72)
  for key,value in [('eos',False),('cap',1536),('upstream_physical',False),('prompt_tokens',85),('image_tokens',71),('source_fed_vision',True)]:
   bad=dict(s);bad[key]=value
   with self.assertRaises(RuntimeError):worker.stop_state(bad,ids,168,True,512,86,72)
if __name__=='__main__':unittest.main()
