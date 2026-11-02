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
 def apply_both(self):
  O1=load('o1','tools/wasm-live-norm-v1/overlay.py');O3=load('o3','tools/wasm-live-norm-v3/overlay.py')
  outs=[]
  for O in [O1,O3]:
   with tempfile.TemporaryDirectory()as tmp:
    root=pathlib.Path(tmp)
    for name in ['src/crispembed.cpp','wasm/embed_wrapper.c','build-embed-wasm.sh']:
     (root/name).parent.mkdir(parents=True,exist_ok=True);(root/name).write_bytes((ROOT/name).read_bytes())
    changes=O.apply(root);outs.append((changes,(root/'src/crispembed.cpp').read_text(),(root/'src/live_norm_observer.h').read_bytes()))
  return O3,outs
 def test_v3_overlay_is_v1_plus_three_reservation_brackets(self):
  O3,outs=self.apply_both()
  self.assertEqual(outs[1][1],O3.bracket(outs[0][1]))
  self.assertEqual(outs[1][1].count('ctx->live_norm.measuring = true;'),3);self.assertEqual(outs[1][1].count('ctx->live_norm.measuring = false;'),3)
  stripped='\n'.join(l for l in outs[1][1].split('\n')if 'ctx->live_norm.measuring = 'not in l)
  self.assertEqual(stripped,outs[0][1])
  self.assertEqual(outs[1][2],V3H.encode());self.assertEqual(outs[0][2],V1H.encode())
  self.assertEqual({k:v for k,v in outs[0][0].items()if k not in('src/live_norm_observer.h','src/crispembed.cpp')},{k:v for k,v in outs[1][0].items()if k not in('src/live_norm_observer.h','src/crispembed.cpp')})
 def test_bracket_rejects_changed_reservation_inventory(self):
  O3=load('o3','tools/wasm-live-norm-v3/overlay.py');text=(ROOT/'src/crispembed.cpp').read_text()
  line=next(l for l in text.split('\n')if 'measure_gf = build_encoder_graph' in l)
  with self.assertRaises(ValueError):O3.bracket(text.replace(line,'',1))
  with self.assertRaises(ValueError):O3.bracket(text+'\n'+line+'\n')

class Diagnosis(unittest.TestCase):
 def test_actual_41421952964_reject_is_the_unevaluated_bucket_graph(self):
  log=(ROOT/'tests/wasm-live-norm-v3/actual-41421952964-native1-observed.log').read_text()
  rows=re.findall(r'^live_norm_reject: geometry name=(\S+) op=(\S+) type=(\S+) contiguous=(\d) ne=\[(\d+),(\d+),(\d+),(\d+)\]',log,re.M)
  self.assertEqual(rows,[('norm00_input','ADD','f32','1','384','16','1','1')])
  self.assertIn('GGML_ASSERT(ctx->live_norm.add_norm(input, result)) failed',log)
  # 16 = bucket_seq_len(12): the reservation build precedes the evaluated build in encode_tokens.
  src=(ROOT/'src/crispembed.cpp').read_text();at=src.index('int T_bucket = bucket_seq_len(T);')
  self.assertLess(src.index('build_encoder_graph(ctx, T_bucket)',at),src.index('build_encoder_graph(ctx, T);',at))
 def test_observer_measuring_contract(self):
  self.assertIn('if (measuring) { ++measured_builds; return; }',V3H)
  self.assertEqual(V3H.count('if (measuring || !active()) return true;'),2)
  # v1 geometry rule (12 rows) unchanged for evaluated graphs
  self.assertIn('tensor->ne[1] != 12',V3H)
  self.assertIn('expect(!o.add_norm(x16, ggml_norm(g, x16, 1e-12f))',(ROOT/'tests/wasm-live-norm-v3/observer_controls.cpp').read_text())

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
