"""Source/stdlib-only controls: no model, package import, compile or inference."""
import ast,copy,hashlib,importlib.util,json,os,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/moedern-router-real-v1'))
import common as c
import prepare
class Controls(unittest.TestCase):
 def test_exact_scope_and_all_call_counts(self):
  p=c.declaration();events=[(phase,l,b,entry,t,r)for phase in ['own','cross']for l in range(2)for b in ['native','torch']for entry in ['input','logits','probs']for t in [1,2]for r in [0,1]]
  self.assertEqual(len(events),96);self.assertEqual(sum(e[3]=='input'for e in events),32);self.assertEqual(sum(e[3]!='probs'for e in events),64)
  self.assertEqual(p['counts']['primary_component_batch_max'],dict(linear=32,softmax=64,topk=len(events),renormalization=len(events)))
  self.assertEqual(3*2*2*2*(3+2+1),p['counts']['float_ref_max']);self.assertEqual(3*2*2*2*3,p['counts']['integer_comparisons_max'])
 def test_changed_capture_precision_shapes_counts_or_gates_reject(self):
  cfg=c.declaration()
  for mutate in [lambda p:p['capture'].__setitem__('source','0'*40),lambda p:p['capture'].__setitem__('script_sha256','f'*64),
   lambda p:p['layers'][0]['original']['input'].__setitem__('shape',[1,307,128]),lambda p:p['layers'][0]['native']['indices'].__setitem__('sha256','a'*64),
   lambda p:p['layers'][0].__setitem__('weight_sha256','0'*64),lambda p:p['counts'].__setitem__('original_full_model_forwards',1),
   lambda p:p['gates'].__setitem__('min_row_cosine',.999),lambda p:p['native_integer_admission'].__setitem__('pre_reviewed_sha256',True)]:
   bad=copy.deepcopy(cfg);mutate(bad)
   with self.assertRaises(ValueError):c.validate(bad)
 def test_all_selected_and_source_hashes_explicit(self):
  cfg=c.declaration();self.assertEqual(len(cfg['capture']['pre_pinned_members']),29)
  for group in ['protected_sha256','owned_sha256']:
   for name,pin in cfg[group].items():self.assertEqual(c.sha(ROOT/name),pin,name)
  self.assertIsNone(cfg['layers'][1]['native']['indices']['sha256'])
 def integers(self,delta=False):
  rows=[(0,1)]*307;rows[58]=(2,6)if delta else(2,4);return b''.join(struct.pack('<2q',*r)for r in rows)
 def test_full_native_integer_pointwise_binding_not_invented_hash(self):
  original=self.integers();native=self.integers(True)
  a=prepare.native_ids(original,original,0);b=prepare.native_ids(original,native,1)
  self.assertEqual(a['changed_rows'],[]);self.assertEqual(b['changed_rows'],[58]);self.assertTrue(b['first_observed_sha256']);self.assertFalse(b['pre_reviewed_sha256'])
  self.assertEqual(b['sha256'],hashlib.sha256(native).hexdigest());self.assertEqual(b['semantic_rows_validated'],307)
 def test_integer_schema_wrong_row_range_duplicate_and_torn_reject(self):
  a=self.integers();n=self.integers(True)
  for bad in [n[:-1],b'\0'*4912,struct.pack('>614q',*sum(([0,1]for _ in range(307)),[]))]:
   with self.assertRaises(ValueError):prepare.native_ids(a,bad,1)
  for row,values in [(0,(1,2)),(58,(2,5)),(17,(0,8)),(18,(1,1))]:
   bad=bytearray(n);struct.pack_into('<2q',bad,row*16,*values)
   with self.assertRaises(ValueError):prepare.native_ids(a,bad,1)
 def test_native_source_fragment_is_exact_not_hand_algorithm(self):
  cfg=c.declaration();text=(ROOT/'src/moedern_text_engine.cpp').read_text();f=c.fragment(text)
  self.assertIn('std::stable_sort',f);self.assertIn('Exact router tie is unqualified',f);self.assertIn(' + ROUTER_EPS;',f)
  self.assertEqual(hashlib.sha256(f.encode()).hexdigest(),cfg['native_fragment']['sha256'])
  with tempfile.TemporaryDirectory()as d:
   path=c.extract(Path(d)/'generated',cfg);self.assertEqual(path.read_text(),f)
  with self.assertRaises(ValueError):c.fragment(text+f)
 def test_frozen_cpu_lock_and_historical_production_bytes(self):
  cfg=c.declaration()
  for name in ['requirements.txt','wheels.json']:
   self.assertEqual((ROOT/'tools/moedern-router-real-v1'/name).read_bytes(),c.git('show',cfg['torch']['lock_source']+':tools/moedern-router-topk-v1/'+name))
  self.assertEqual(c.git('rev-parse','HEAD:src'),c.git('rev-parse',c.BASE+':src'))
 def test_safe_selected_files_and_wrong_byte_auth_reject_before_import(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'member').write_bytes(b'known')
   for path in ['../member','/member','a\\b']:
    with self.assertRaises(ValueError):prepare.safe(root,path)
   (root/'link').symlink_to(root/'member')
   with self.assertRaises(ValueError):prepare.safe(root,'link')
   cfg=dict(capture=dict(pre_pinned_members={'member':dict(bytes=5,sha256='0'*64)}))
   with self.assertRaisesRegex(ValueError,'selected bytes'):prepare.authenticate(root,root/'output',cfg)
  for name in ['numpy','torch','transformers']:self.assertNotIn(name,sys.modules)
 def test_direct_captured_operators_and_no_original_forward(self):
  source=(ROOT/'tools/moedern-router-real-v1/worker.py').read_text();ast.parse(source)
  for exact in ['torch.nn.functional.linear(x,weight,None)','torch.ops.aten.softmax.int(p,-1)','torch.ops.aten.topk.default(p,2)','torch.ops.aten.div.Tensor(values,denominator)']:
   self.assertIn(exact,source)
  for forbidden in ['load_release_model','model.encode','original.run(','transformers']:
   self.assertNotIn(forbidden,source)
  self.assertLess(source.index('prepare.verify_prepared'),source.index(' import numpy as np'))
  self.assertLess(source.index("if not report['own_anchors_byte_exact']"),source.index("refs('same_input')"))
 def test_native_helpers_and_only_test_fragment(self):
  text=(ROOT/'tests/moedern-router-real-v1/probe.cpp').read_text()
  for exact in ['#include "../../src/moedern_text_engine.cpp"','native::linear(','native::softmax(','#include "native_router.inc"','core_util::clean_exit']:
   self.assertIn(exact,text)
  self.assertNotIn('native::encode_tokens(',text)
  cmake=(ROOT/'tests/moedern-router-real-v1/CMakeLists.txt').read_text();self.assertIn('-ffp-contract=off ${ARCH_FLAGS}',cmake);self.assertNotIn('CRISPEMBED_MOEDERN_COMPOSED_TESTING',cmake)
 def test_help_under_isolation_uses_no_framework(self):
  for name in ['tools/ci-heavy/moedern_router_real_v1.py','tools/moedern-router-real-v1/worker.py']:
   result=subprocess.run([sys.executable,'-I',str(ROOT/name),'--help'],cwd=tempfile.gettempdir(),capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr)
  self.assertNotIn('numpy',sys.modules);self.assertNotIn('torch',sys.modules)
 def test_genuine_provider_does_not_accept_github_spoof(self):
  spec=importlib.util.spec_from_file_location('real_router_owner',ROOT/'tools/ci-heavy/moedern_router_real_v1.py');owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
  with patch.dict(os.environ,{'GITHUB_ACTIONS':'true','KAGGLE_KERNEL_RUN_TYPE':'Batch'}):
   with self.assertRaises(ValueError):owner.provider('a'*40)
if __name__=='__main__':unittest.main()
