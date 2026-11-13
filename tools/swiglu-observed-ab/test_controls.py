"""Model-free protocol admission and gate-isolation controls (stdlib only)."""
import ast,hashlib,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).parent

def functions():
 tree=ast.parse((ROOT/'worker.py').read_text());nodes=[]
 for node in tree.body:
  if isinstance(node,ast.Assign)and any(isinstance(t,ast.Name)and t.id in['GATE','CFG_SHA']for t in node.targets):nodes.append(node)
  elif isinstance(node,ast.FunctionDef)and node.name in['admit','gates']:nodes.append(node)
 def require(ok,message):
  if not ok:raise ValueError(message)
 env={'require':require,'sha':lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),'json':json,'Path':Path};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<selected-admission-functions>','exec'),env);return env

class Controls(unittest.TestCase):
 def test_frozen_protocol(self):
  f=functions();cfg=f['admit'](ROOT/'protocol.json');self.assertEqual(cfg['cases'],['simple_table','simple_form','arabic_printed_line']);self.assertEqual(cfg['historical_off_tokens'],[309,168,6]);self.assertEqual(cfg['cap'],512);self.assertEqual(cfg['candidate_source'],'519c3d9765c234cd93e9069d5efdc239e3cad3f2');self.assertFalse(cfg['heldout_data_allowed']);self.assertFalse(cfg['source_reference_forward_allowed']);self.assertFalse(cfg['runtime_rebuild_allowed']);self.assertFalse(cfg['quality_qualified'])
 def test_modified_protocol_rejected(self):
  f=functions();cfg=json.loads((ROOT/'protocol.json').read_text());cfg['cap']=1536
  with tempfile.TemporaryDirectory()as directory:
   p=Path(directory)/'protocol.json';p.write_text(json.dumps(cfg,indent=2)+'\n')
   with self.assertRaises(ValueError):f['admit'](p)
 def test_only_new_gate_changes(self):
  f=functions();original={'CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL':'1','OTHER_FIXED':'unchanged'};off,on=f['gates'](original);self.assertNotIn(f['GATE'],original);self.assertEqual({k for k in off if off[k]!=on[k]},{f['GATE']});self.assertEqual(off[f['GATE']],'0');self.assertEqual(on[f['GATE']],'1');self.assertEqual(off['CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL'],'1');self.assertEqual(on['OTHER_FIXED'],'unchanged')
 def test_source_bound_probe_is_frozen(self):
  cfg=functions()['admit'](ROOT/'protocol.json');self.assertEqual(cfg['source_form_probe_receipt_sha256'],'2723be06d2b1a4836d51b75d50a23f61a07972127c907e01d0505c778dcf7260');self.assertEqual(cfg['model_sha256'],'843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43')
if __name__=='__main__':unittest.main()
