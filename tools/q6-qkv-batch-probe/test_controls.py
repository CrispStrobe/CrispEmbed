import copy,json,unittest
from pathlib import Path
import worker
class Controls(unittest.TestCase):
 def test_immutable_scope(self):
  cfg=json.loads((Path(__file__).parent/'protocol.json').read_text());worker.admit(cfg)
  for key,value in [('source_tokens',139),('model_forward_allowed',True),('threads',[4]),('batches',[16])]:
   bad=copy.deepcopy(cfg);bad[key]=value
   with self.assertRaises(RuntimeError):worker.admit(bad)
 def test_geometry(self):
  plans=worker.batch_plans([1,4,12,16,32,64,140,144]);self.assertEqual(plans['source140'],[(4,144)]);self.assertEqual(plans['mixed144-chunk16'][0],(0,16))
  for name,slices in plans.items():
   if name.startswith('image'):self.assertEqual([x for a,b in slices for x in range(a,b)],list(range(4,144)))
   if name.startswith('mixed'):self.assertEqual([x for a,b in slices for x in range(a,b)],list(range(144)))
 def test_map(self):
  mapping=worker.base.row_map('qkv',8192);self.assertEqual(sorted(mapping),list(range(8192)));self.assertEqual(mapping[:4096],list(range(4096)))
 def test_lifecycle_and_unwritten_output(self):
  text=(Path(__file__).parent/"worker.py").read_text();self.assertIn("signal.SIGTERM,signal.SIGINT",text);self.assertIn("physical.memory()",text);self.assertIn("signal.signal(sig,handler)",text);self.assertIn("quiet_NaN()",(Path(__file__).parent/"probe.cpp").read_text())
 def test_closures_and_profiles(self):
  text=(Path(__file__).parent/'worker.py').read_text();self.assertIn('full actual source Q6 batch closure',text);self.assertIn('actual native first16 closure',text);self.assertIn("['ON','OFF']",text);self.assertNotIn('except Exception',text)
if __name__=='__main__':unittest.main()
