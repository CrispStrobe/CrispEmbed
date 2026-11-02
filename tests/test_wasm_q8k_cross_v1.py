#!/usr/bin/env python3
"""Small stdlib controls only; no packages, arrays, compiler or inference."""
import ast,hashlib,importlib.util,json,math,pathlib,struct,subprocess,sys,tempfile,unittest
from unittest import mock
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/wasm-q8k-cross'))
import protocol as P

def block(scale,values):
 return struct.pack('<f256b16h',scale,*values,*(sum(values[k:k+16])for k in range(0,256,16)))

def work(scale=1,sign=1):return block(scale,[sign*((i%11)-5)for i in range(256)])*72

class Controls(unittest.TestCase):
 def test_workspace_original_layout_and_consistent_sums(self):
  self.assertEqual(P.valid_workspace(work()),work())
  self.assertEqual(len(work()),21024)
 def test_extent_scale_and_sum_rejections(self):
  a=work();b=bytearray(a);struct.pack_into('<f',b,0,float('nan'))
  c=bytearray(a);c[260]^=1
  for value in [a[:-1],a+b'0',bytes(b),bytes(c)]:
   with self.assertRaises(ValueError):P.valid_workspace(value)
 def test_hybrid_preserves_exact_scale_and_signed_sum_bytes(self):
  a,b=work(.25),work(-.5,-1);v=P.hybrid(a,b)
  for k in range(0,21024,292):self.assertEqual(v[k:k+4],a[k:k+4]);self.assertEqual(v[k+4:k+292],b[k+4:k+292])
  self.assertEqual(P.hybrid(a,a),a);self.assertNotEqual(v,a);self.assertNotEqual(v,b)
 def test_hybrid_rejects_invalid_source_instead_of_repair(self):
  a=work();b=bytearray(a);b[-1]^=1
  with self.assertRaises(ValueError):P.hybrid(a,bytes(b))
 def test_frozen_dependency_and_owned_closure_complete(self):
  cfg=json.loads(P.PATH.read_text());old=json.loads((ROOT/'docs/wasm-f32-operators-v1.json').read_text())
  required=set(old['source_sha256'])|{'docs/wasm-f32-operators-v1.json','tests/wasm-layer3/kernel_capture.inc','tests/results/packaging/wasm-f32-operators-v1.json'}
  self.assertEqual(set(cfg['protected_sha256']),required)
  self.assertEqual(set(cfg['owned_sha256']),P.OWNED-{'docs/wasm-q8k-cross-v1.json'})
  for paths in [cfg['protected_sha256'],cfg['owned_sha256']]:
   for name,h in paths.items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),h,name)
 def test_previous64_and48_proof_pins_not_pending(self):
  cfg=json.loads(P.PATH.read_text());self.assertEqual(cfg['operator_review'],dict(id=12856145091,bytes=61468,sha256='c6aa6bb2430e0dfa287324a7e0735940b2895a7f4e3c51395cc0c8d6b765e58f'))
  self.assertEqual(cfg['operator_report_sha256'],'fc9b8075f584b594b11fecc12c2e8f7e323aa39428b32dd841b3d64c0acd3ed6')
  self.assertEqual(cfg['gates'],{'max_absolute':.002,'minimum_row_cosine':.99999});self.assertFalse(cfg['publication_qualified'])
 def test_arms_and44_measurements_are_predeclared(self):
  cfg=json.loads(P.PATH.read_text());self.assertEqual(cfg['arms'],list(P.ARMS));self.assertEqual(cfg['comparisons'],8+3*6*2)
  self.assertEqual(cfg['required_byte_anchors'],18)
 def test_actual_conversion_capture_precedes_direct_route(self):
  s=(ROOT/'tools/ci-heavy/wasm_q8k_cross_v1.py').read_text();self.assertIn("'Unique actual conversion barrier required'",s)
  self.assertIn("tests/wasm-layer3/kernel_capture.inc",s);self.assertIn("'Actual native conversion workspace differs'",s)
  cpp=(ROOT/'tests/wasm-q8k-cross/operator.cpp').read_text();self.assertIn('mode ? GGML_TYPE_Q8_K : GGML_TYPE_F32',cpp)
  self.assertIn('ggml_mul_mat(o.ctx, w, x)',cpp);self.assertNotIn('ggml_cast',cpp)
 def test_barrier_fail_closed_before_all_metrics(self):
  s=(ROOT/'tools/ci-heavy/wasm_q8k_cross_v1.py').read_text();self.assertLess(s.index("'Exact18 original normal/direct anchors required'"),s.index('import numpy as np'))
  self.assertIn("if a.phase=='verify':verify(a.output,True)",s)
  self.assertIn("'Native thread-count output differs'",s)
  self.assertIn('if interventions:anchors(out)',s)
  self.assertIn("'Exact18 baseline anchors before interventions'",s)
 def test_actual_browser_isolation_and_controls(self):
  s=(ROOT/'tests/wasm-q8k-cross/browser.cjs').read_text()
  for value in ['Cross-Origin-Opener-Policy','Cross-Origin-Embedder-Policy','crossOriginIsolated','SharedArrayBuffer','Nonfinite Q8_K scale rejection','Inconsistent sum rejection','Actual recovery failed','Normal/direct route anchor failed']:self.assertIn(value,s)
 def test_same_baseline_flags_and_no_production_link(self):
  cmake=(ROOT/'tests/wasm-q8k-cross/CMakeLists.txt').read_text();self.assertIn('CMAKE_POSITION_INDEPENDENT_CODE ON',cmake)
  self.assertIn('GGML_LLAMAFILE ON',cmake);self.assertIn('GGML_LLAMAFILE OFF',cmake)
  self.assertIn('wasm-f32-operators/diff.cpp',cmake)
  worker=(ROOT/'tools/ci-heavy/wasm_q8k_cross_v1.py').read_text();self.assertIn('old.compile_profile(out)',worker)
 def test_workflow_preserves_evidence_and_budgets(self):
  s=(ROOT/'.github/workflows/wasm-q8k-cross-v1.yml').read_text()
  self.assertIn('timeout-minutes: 60',s);self.assertEqual(s.count('if: always()'),2);self.assertIn('fetch-depth: 0',s)
  self.assertLess(s.index(' prepare --output'),s.index('setup-emsdk'));self.assertNotIn('continue-on-error',s)
 def test_empty_missing_false_or_integer_controls_reject(self):
  keys=['positive','wrong_extent_rejected','nonfinite_rejected','invalid_workspace_rejected','output_preserved_on_failure','recovery']
  valid=dict.fromkeys(keys,True);self.assertEqual(P.valid_controls(valid),valid)
  for value in [{},{**valid,'positive':False},{**valid,'positive':1},{k:v for k,v in valid.items()if k!='recovery'}]:
   with self.assertRaises(ValueError):P.valid_controls(value)
 def test_actual_global_anchor_barrier_rejects_changed_bytes(self):
  spec=importlib.util.spec_from_file_location('cross_worker_guard',ROOT/'tools/ci-heavy/wasm_q8k_cross_v1.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
  with tempfile.TemporaryDirectory()as directory:
   out=pathlib.Path(directory);(out/'review').mkdir();(out/'browser').mkdir()
   keys=['f32_native','f32_candidate',*P.ARMS];packets={};native={'native1':{},'native2':{}};browser=dict(baseline_completed=True,isolated=True,pthreads=True,page_errors=[],outputs={})
   controls=dict.fromkeys(['positive','wrong_extent_rejected','nonfinite_rejected','invalid_workspace_rejected','output_preserved_on_failure','recovery'],True);browser['controls']=controls
   value=struct.pack('<f',.25)*4608;qh=P.sha(work())
   for key in keys:
    if key in P.ARMS[-2:]:continue
    mode=0 if key.startswith('f32_')else 1
    packets[key]=dict(mode=mode,input={'sha256':'a'*64},workspace={'sha256':qh},expected_workspace_sha256=qh,expected_outputs={'native':{'sha256':P.sha(value)},'wasm':{'sha256':P.sha(value)}})
    for backend,rows in [('native1',native['native1']),('native2',native['native2']),('wasm',browser['outputs'])]:
     path=out/backend/key/'output.f32';path.parent.mkdir(parents=True);path.write_bytes(value)
     capture=path.parent/('converted.bin'if backend=='wasm'else P.WEIGHT+'.converted.bin');capture.write_bytes(work())
     rows[key]=dict(file=str(path.relative_to(out)),sha256=P.sha(value),repeat_bit_exact=True,input_sha256='a'*64,workspace_sha256=qh)
   for name,data in [('browser/q8k-packets.json',packets),('review/native-cross.json',native),('browser/q8k-report.json',browser),('review/native-controls.json',dict(controls,threads=[1,2]))]:
    (out/name).write_text(json.dumps(data))
   with mock.patch.object(worker,'verify'),mock.patch.object(worker.old,'source_identity',return_value={'source':'b'*40}):
    worker.anchors(out);self.assertEqual(len(json.loads((out/'review/cross-anchors.json').read_text())['anchors']),18)
    path=out/native['native2']['native_candidate']['file'];path.write_bytes(value[:-1]+bytes([value[-1]^1]))
    with self.assertRaises(ValueError):worker.anchors(out)
 def test_python_source_only_syntax(self):
  for name in ['tools/wasm-q8k-cross/protocol.py','tools/ci-heavy/wasm_q8k_cross_v1.py']:ast.parse((ROOT/name).read_text())

if __name__=='__main__':unittest.main()
