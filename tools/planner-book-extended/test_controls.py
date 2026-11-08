import ast,copy,json,unittest
from pathlib import Path
import worker
class Controls(unittest.TestCase):
 def test_fixed_resource_scope(self):
  cfg=json.loads((worker.ROOT/'docs/lighton4b-planner-book-extended-v1.json').read_text());worker.admit(cfg)
  for k,v in [('cap',2048),('threads',2),('case','inventory-table'),('worker_seconds',0),('arm_seconds',1200),('qualified',True),('old_baseline_decode_allowed',True)]:
   bad=copy.deepcopy(cfg);bad[k]=v
   with self.assertRaises(ValueError):worker.admit(bad)
 def test_pending_bindings_fail_closed(self):
  cfg=json.loads((worker.ROOT/'docs/lighton4b-planner-book-extended-v1.json').read_text())
  with self.assertRaises(ValueError):worker.bind_complete(cfg)
  for key in['timeout_audit_sha256','inventory_receipt_sha256','runtime_manifest_sha256']:cfg[key]='1'*64
  worker.bind_complete(cfg)
  for value in['','pending','X'*64,'a'*63]:
   bad=dict(cfg);bad['runtime_manifest_sha256']=value
   with self.assertRaises(ValueError):worker.bind_complete(bad)
 def test_exact_complete_prefix_metadata(self):
  cfg={'artifacts':{f'corrected-composed/greedy.{i}.logits.f32':dict(bytes=248320*4,sha256=f'{i:064x}')for i in range(1054)}};self.assertEqual(len(worker.prefix_keys(cfg)),1054)
  cfg['artifacts']['corrected-composed/greedy.1053.logits.f32']['bytes']=4
  with self.assertRaises(ValueError):worker.prefix_keys(cfg)
 def test_no_inference_before_admission_or_forced_prefix(self):
  text=Path(worker.__file__).read_text();ast.parse(text);self.assertNotIn("'cmake'",text);self.assertNotIn("'--source-vision'",text);self.assertIn("teacher.write_text('248046\\n')",text);self.assertIn("checked[0]<min(1054,n-1)",text);self.assertLess(text.index("bind_complete(cfg)"),text.index("import lighton4b_upstream_physical_pack"));self.assertLess(text.index("strict earlier1054 raw-row prefix"),text.index("r['strict_1054_prefix_and_stages_reproduced']=True"))
 def test_deadline_is_not_swallowed(self):
  text=Path(worker.__file__).read_text();self.assertIn('except WorkerDeadline:raise',text);self.assertNotIn('except Exception',text);self.assertIn('signal.SIGTERM,signal.SIGINT',text);self.assertIn('deadline.close()',text)
if __name__=='__main__':unittest.main()
