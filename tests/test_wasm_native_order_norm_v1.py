"""Source/geometry/metric controls only. Actual C++/WASM runs are hosted."""
import ast
import base64
import hashlib
import json
import struct
import subprocess
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/wasm-native-order-norm-v1'))
import contract as K
spec=importlib.util.spec_from_file_location('norm_driver',ROOT/'tools/ci-heavy/wasm_native_order_norm_v1.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
class Controls(unittest.TestCase):
 def test_declared_scope(self):
  p=K.validate(K.read(K.PROFILE));self.assertEqual(p['base'],K.BASE);self.assertFalse(p['default']);self.assertFalse(p['qualification'])
  for name,value in [('held_q8_rounding','0'),('frontend_simd',True),('max_absolute',.003),('minimum_row_cosine',.99)]:
   bad=copy.deepcopy(p);bad[name]=value
   with self.assertRaises(ValueError):K.validate(bad)
 def test_complete_reference_and_build_closure_required(self):
  p=K.read(K.PROFILE);p['protected_git_objects'].pop('tools/moedern-native-v2/diff_adapter.py')
  with self.assertRaises(ValueError):K.validate(p)
 def test_exact_runtime_delta(self):
  p=K.read(K.PROFILE);self.assertEqual(set(p['runtime_delta']),K.DELTA)
  for name,pin in p['runtime_delta'].items():self.assertEqual(K.sha(ROOT/name),pin['sha256'])
  for name,pin in p['owned_sha256'].items():self.assertEqual(K.sha(ROOT/name),pin)
 def test_live_graph_no_oracle_argument(self):
  source=(ROOT/'src/core/wasm_native_order_norm.cpp').read_text()
  self.assertIn('ggml_map_custom1',source);self.assertIn('source->data',source);self.assertIn('row += nth',source)
  self.assertNotIn('fopen(',source);self.assertNotIn('getenv(',source);self.assertNotIn('workspace',source)
  self.assertIn('float centered[width]',source);self.assertIn('volatile float',source)
 def test_context_gate_and_only_three_routes(self):
  source=(ROOT/'src/crispembed.cpp').read_text()
  self.assertEqual(source.count('cur = bert_norm(cur);'),3)
  self.assertEqual(source.count('core_env::on("CRISPEMBED_WASM_NATIVE_ORDER_NORM")'),1)
  self.assertIn('wasm_native_order_norm::parameters wasm_norm_parameters;',source)
  self.assertIn('make(gctx, input, &ctx->wasm_norm_parameters)',source)
  self.assertIn('return ggml_norm(gctx, input, ln_eps);',source)
 def test_metric_rows_do_not_hide_one_bad_token(self):
  a=[1.,0.,0.,1.];b=[1.,0.,0.,-1.]
  r=D.independent(a,b,[2,2]);self.assertEqual(r['min_row_cosine'],-1.);self.assertFalse(r['passed']);self.assertEqual(r['max_absolute'],2.)
  self.assertTrue(D.independent(a,a,[2,2])['passed'])
 def test_metric_shape_and_finite_reject(self):
  for args in [([1.],[1.,2.],[1,1]),([float('nan')],[1.],[1,1]),([1.],[1.],[1])]:
   with self.assertRaises(ValueError):D.independent(*args)
 def test_actual_native_order_tree_and_eps(self):
  source=(ROOT/'src/core/wasm_native_order_norm.cpp').read_text()
  self.assertIn('square_sum += double(add(add(a0, a2), add(a1, a3)))',source)
  self.assertIn('float(square_sum / double(width))',source)
  self.assertIn('epsilon != 1e-12f',source)
  self.assertIn('-ffp-contract=off',(ROOT/'CMakeLists.txt').read_text())
 def test_no_old_guard_monkeypatch(self):
  tree=ast.parse((ROOT/'tools/ci-heavy/wasm_native_order_norm_v1.py').read_text())
  for n in ast.walk(tree):
   if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='old':
    self.assertIn(n.func.attr,{'service','zip_files'})
 def test_on_has_off_barrier_and_context_isolation(self):
  text=(ROOT/'tests/wasm-native-order-norm/panel.cjs').read_text()
  self.assertLess(text.index('All OFF bytes must pass first'),text.index('chromium.launch'))
  self.assertIn("for(const gate of ['0','1','0'])",text)
  self.assertIn("set('CRISPEMBED_WASM_Q8_0_ROUND','1')",text)
  self.assertIn('repeat<2',text)
 def test_workflow_order_and_failure_artifacts(self):
  text=(ROOT/'.github/workflows/wasm-native-order-norm-candidate-v1.yml').read_text()
  self.assertLess(text.index(' component-check '),text.index(' prepare-panel '))
  self.assertLess(text.index(' off-check '),text.index('panel.cjs "$RUNNER_TEMP/norm-candidate/browser" candidate'))
  self.assertEqual(text.count('if: always()'),2);self.assertIn('retention-days: 90',text)
 def test_raw_vector_preserves_negative_zero(self):
  negative=struct.pack('<I',0x80000000)+bytes(1532);positive=bytes(1536)
  row=dict(vector=[0.]*384,raw_base64=base64.b64encode(negative).decode(),raw_sha256=hashlib.sha256(negative).hexdigest())
  self.assertEqual(D.actual_vector(json.loads(json.dumps(row))),negative)
  self.assertNotEqual(D.actual_vector(row),positive)
  self.assertNotEqual(row['raw_sha256'],hashlib.sha256(positive).hexdigest())
  bad=copy.deepcopy(row);bad['raw_sha256']=hashlib.sha256(positive).hexdigest()
  with self.assertRaises(ValueError):D.actual_vector(bad)
  bad=copy.deepcopy(row);bad['vector'][0]=1.
  with self.assertRaises(ValueError):D.actual_vector(bad)
 def test_javascript_raw_capture_survives_json_zero_loss(self):
  helper=ROOT/'tests/wasm-native-order-norm/vector_bytes.cjs'
  script="const {retainVector}=require(process.argv[1]);const b=Buffer.alloc(1536);b.writeUInt32LE(0x80000000,0);const d=Array(384).fill(0);d[0]=-0;process.stdout.write(JSON.stringify({...retainVector(Array.from(b),d),vector:d}));"
  row=json.loads(subprocess.check_output(['node','-e',script,str(helper)],text=True))
  self.assertEqual(D.actual_vector(row)[:4],bytes.fromhex('00000080'))
  self.assertEqual(row['vector'][0],0)
 def test_compile_flag_change_is_visible(self):
  a=D.arithmetic_flags('c++ -O3 -pthread -DGGML_FMA -I/x file.cpp')
  b=D.arithmetic_flags('c++ -O3 -pthread -DGGML_FMA -I/y file.cpp')
  self.assertEqual(a,b);self.assertNotEqual(a,D.arithmetic_flags('c++ -O3 -msimd128 -DGGML_FMA file.cpp'))
if __name__=='__main__':unittest.main()
