#!/usr/bin/env python3
"""v3 observer protocol controls; no build/model."""
import copy,importlib.util,json,pathlib,re,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/wasm-live-norm-v1')]
import live_contract as K

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

V1H=(ROOT/'tests/wasm-live-norm/observer.h').read_text()
V3H=(ROOT/'tests/wasm-live-norm-v3/observer.h').read_text()

class Observer(unittest.TestCase):
 def test_v1_observer_unchanged(self):
  self.assertEqual(K.git('rev-parse','ff3b8a0171d2cdb5cfcdd65701fadaa49a99d265:tests/wasm-live-norm/observer.h'),K.git('rev-parse','HEAD:tests/wasm-live-norm/observer.h'))
 def test_no_numeric_graph_nodes(self):
  for forbidden in ['ggml_set_name','ggml_view_','ggml_set_output','ggml_add(','ggml_mul(','ggml_norm(']:self.assertNotIn(forbidden,V3H)
  self.assertIn('output->src[0] != input',V3H);self.assertIn('GGML_OP_MAP_CUSTOM1',V3H)
 def test_rejections_are_logged(self):
  self.assertIn('live_norm_reject: add_norm',V3H);self.assertIn('describe("geometry"',V3H);self.assertIn('describe("duplicate"',V3H)

class Overlay(unittest.TestCase):
 def test_v3_overlay_equals_v1_except_header(self):
  O1=load('o1','tools/wasm-live-norm-v1/overlay.py');O3=load('o3','tools/wasm-live-norm-v3/overlay.py')
  outs=[]
  for O in [O1,O3]:
   with tempfile.TemporaryDirectory()as tmp:
    root=pathlib.Path(tmp)
    for name in ['src/crispembed.cpp','wasm/embed_wrapper.c','build-embed-wasm.sh']:
     (root/name).parent.mkdir(parents=True,exist_ok=True);(root/name).write_bytes((ROOT/name).read_bytes())
    changes=O.apply(root);outs.append((changes,(root/'src/crispembed.cpp').read_bytes(),(root/'src/live_norm_observer.h').read_bytes()))
  self.assertEqual(outs[0][1],outs[1][1])
  self.assertEqual(outs[1][2],V3H.encode());self.assertEqual(outs[0][2],V1H.encode())
  self.assertEqual({k:v for k,v in outs[0][0].items()if k!='src/live_norm_observer.h'},{k:v for k,v in outs[1][0].items()if k!='src/live_norm_observer.h'})

class Caller(unittest.TestCase):
 def test_profile_and_routing(self):
  D=load('v3','tools/ci-heavy/wasm_live_norm_v3.py');p=K.read(D.PROFILE);D.validate(p)
  for path,digest in p['owned_sha256'].items():self.assertEqual(K.sha(ROOT/path),digest,path)
  for key,value in [('scope_unchanged_from_v1',False),('runtime_change',True),('component_admission','v1'),('workflow','.github/workflows/wasm-live-norm-operands-v2.yml')]:
   q=copy.deepcopy(p);q[key]=value
   with self.assertRaises(ValueError):D.validate(q)
  self.assertIs(D.V1.source,D.source);self.assertIs(D.V2.source,D.source);self.assertIs(D.V1.apply,D.OVERLAY.apply)
  self.assertEqual(D.V1.__file__,str(pathlib.Path(D.__file__).resolve()))
  self.assertIs(D.PHASES['component-build'],D.V2.component_build);self.assertIs(D.PHASES['replay'],D.V1.replay)

if __name__=='__main__':unittest.main()
