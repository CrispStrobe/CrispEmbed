#!/usr/bin/env python3
"""Small stdlib source/protocol controls; no backend, arrays or model execution."""
import ast,importlib.util,json,math,re,struct,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/wasm-norm-order'))
import protocol as P
class Controls(unittest.TestCase):
 def text(self,name):return(ROOT/name).read_text()
 def test_group8_rounding_tree(self):
  # Eight authored scalar squares expose intermediate F32 chunk rounding.
  v=[16777216.,1.,1.,1.,0.,0.,0.,0.]
  self.assertEqual(P.group8(v),16777218.)
  self.assertEqual(struct.unpack('<f',struct.pack('<f',sum(v)))[0],16777220.)
  self.assertEqual(P.group8([1.]*8),8.)
 def test_group8_extent_nonfinite(self):
  for v in [[1.]*7,[1.]*9,[math.inf]+[0.]*7,[math.nan]+[0.]*7]:
   with self.assertRaises(ValueError):P.group8(v)
 def test_actual_workspace_block_layout(self):
  block=struct.pack('<e',.5)+bytes(range(32));rows=P.workspace(block*144)
  self.assertEqual(len(rows),144);self.assertEqual(rows[12]['token'],1);self.assertEqual(rows[143]['block'],11)
  self.assertEqual(rows[0]['scale'],.5);self.assertEqual(rows[0]['signed_sha256'],P.sha(bytes(range(32))))
 def test_workspace_rejects_wrong_extent_or_nonfinite_scale(self):
  for raw in [bytes(4895),bytes(4897),struct.pack('<e',math.nan)+bytes(4894),struct.pack('<e',math.inf)+bytes(4894)]:
   with self.assertRaises(ValueError):P.workspace(raw)
 def test_cpp_has_no_oracle_argument(self):
  cpp=self.text('tests/wasm-norm-order/operator.cpp');signature=re.split(r'\)\s*\{',cpp.split('extern "C" int crispembed_norm_order(',1)[1],maxsplit=1)[0]
  self.assertIn('const float * x',signature)
  for forbidden in ['workspace','reference','expected','norm']:self.assertNotIn(forbidden,signature)
  cpp=re.sub(r'\s+','',cpp);self.assertIn('computed=native_order(x)',cpp);self.assertIn('mode?computed.data():x',cpp)
  self.assertIn('ggml_mul_mat(o.ctx,w,affine)',cpp);self.assertIn('GGML_TYPE_IQ4_NL',cpp)
 def test_portable_tree_and_finite_output_commit(self):
  cpp=self.text('tests/wasm-norm-order/operator.cpp');cm=self.text('tests/wasm-norm-order/CMakeLists.txt')
  cpp=re.sub(r'\s+','',cpp);self.assertIn('volatilefloat',cpp);self.assertIn('add(add(a0,a2),add(a1,a3))',cpp)
  self.assertIn('square_sum/double(H)',cpp);self.assertIn('std::sqrt(add(variance,1e-12f))',cpp)
  self.assertLess(cpp.index('!finite(pending.data(),required)'),cpp.index('std::copy(pending.begin()'))
  self.assertIn('target_compile_options(wasm-norm-order PRIVATE -ffp-contract=off)',cm)
 def test_actual_backend_context_and_conversion_capture(self):
  cpp=self.text('tests/wasm-norm-order/operator.cpp');driver=self.text('tools/ci-heavy/wasm_norm_order_v1.py')
  self.assertIn('#ifdef__EMSCRIPTEN__ggml_backend_cpu_set_wasm_q8_0_rounding(backend.get(),true);',re.sub(r'\s+','',cpp))
  self.assertIn("k['weight_type']==20 and k['vec_dot_type']==8",driver)
  self.assertIn("row['kernel']['wasm_round_plan']==int(backend=='wasm')",driver)
  self.assertIn("'.input.f32')).read_bytes()==raw[2*18432:3*18432]",driver)
 def test_macro_guard_and_original_compile_profile(self):
  s=self.text('tools/ci-heavy/wasm_norm_order_v1.py')
  for token in ["'-dM','-E'","'__AVX2__'in definitions","'__FMA__'in definitions","'__AVX512F__'not in definitions","'__FAST_MATH__'not in definitions","'__wasm_simd128__'in definitions",'old.compile_profile(out)']:self.assertIn(token,s)
 def test_barriers_precede_any_projection(self):
  s=self.text('tools/ci-heavy/wasm_norm_order_v1.py');js=self.text('tests/wasm-norm-order/browser.cjs')
  self.assertIn("if phase=='norm':baseline(out)",s);self.assertIn("if phase=='chain':recovery(out)",s)
  self.assertIn("row['stages']['norm']['sha256']==p['anchors']['native1']['norm']['sha256']",s)
  self.assertIn("len(checked)==30",s);self.assertIn("len(checked)==12",s)
  self.assertIn("phase==='chain'?'norm-recovery.json':'norm-baseline.json'",js)
  self.assertIn("phase!=='baseline'",js)
 def test_independent_and_actual_ref_all56_preserved(self):
  s=self.text('tools/ci-heavy/wasm_norm_order_v1.py')
  for token in ['old.metrics(np.memmap','old.write_pair(','old.collect(','len(pairs)==56',"'scale_blocks_changed'", "'signed_bytes_changed'"]:
   self.assertIn(token.strip("'"),s)
  self.assertLess(s.index("save(out/'review/norm-risk.json'"),s.index("require(cpp['status']=='passed'"))
 def test_browser_isolation_and_ownership(self):
  s=self.text('tests/wasm-norm-order/browser.cjs')
  for token in ['Cross-Origin-Opener-Policy','same-origin','Cross-Origin-Embedder-Policy','require-corp','crossOriginIsolated','SharedArrayBuffer','ordinary_dump_bit_exact','inputs_unchanged','m._free(output)','directory(null)']:self.assertIn(token,s)
 def test_workflow_phase_order_and_failure_retention(self):
  s=self.text('.github/workflows/wasm-native-order-norm-risk-v1.yml')
  phases=[' native-baseline ',' baseline\n',' baseline --output',' native-norm ',' norm\n',' recovery --output',' native-chain ',' chain\n',' compare --output']
  positions=[s.index(p)for p in phases];self.assertEqual(positions,sorted(positions))
  for token in ['fetch-depth: 0','timeout-minutes: 60',"CMAKE_BUILD_PARALLEL_LEVEL: '2'","version: '6.0.2'",'playwright@1.56.0','numpy==2.2.6']:self.assertIn(token,s)
  self.assertEqual(s.count('if: always()'),2);self.assertEqual(s.count('retention-days: 90'),2)
 def test_python_ast_only(self):
  for path in ['tools/wasm-norm-order/protocol.py','tools/ci-heavy/wasm_norm_order_v1.py']:ast.parse(self.text(path))
 def test_formatting_only_cpp_token_identity(self):
  import subprocess
  before=subprocess.check_output(['git','show','f78131c4616ece9f236ddab7695db6cd32a5521f:tests/wasm-norm-order/operator.cpp'],cwd=ROOT)
  actual=(ROOT/'tests/wasm-norm-order/operator.cpp').read_bytes()
  self.assertEqual(P.cpp_tokens(before),P.cpp_tokens(actual))
  self.assertNotEqual(P.cpp_tokens(actual),P.cpp_tokens(actual.replace(b'1e-12f',b'1e-5f',1)))
 def test_exact_source_closure(self):
  cfg=P.source();self.assertEqual(cfg['gates'],{'max_absolute':.002,'minimum_row_cosine':.99999})
  self.assertEqual(cfg['operator']['run'],40771950208);self.assertEqual(cfg['operator']['report_sha256'],'fc9b8075f584b594b11fecc12c2e8f7e323aa39428b32dd841b3d64c0acd3ed6')
  self.assertEqual(cfg['weight_sha256'],'1707be1e50373fe0945e338a263ed630968e9150197b7c08af1957352d1f319f')
  self.assertFalse(any(p.startswith('src/')or p=='CMakeLists.txt'for p in cfg['owned_sha256']))
if __name__=='__main__':unittest.main()
