#!/usr/bin/env python3
"""Tiny authored format/source/protocol controls; no packages/model payloads."""
import ast,importlib.util,json,struct,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cast_protocol',ROOT/'tools/wasm-layer3-f32/protocol.py');P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
class Controls(unittest.TestCase):
 def test_closed_source(self):self.assertEqual(P.source()['comparisons'],25)
 def test_literal_decoder_block_order_and_scales(self):
  # d=1, all scales=33 -> multiplier1. Independently authored nibbles.
  block=struct.pack('<eH',1.,0xaaaa)+bytes([0x11]*4)+bytes([0xf8]*128)
  values=struct.unpack('<256f',P.decode(block))
  self.assertEqual(values[:16],(1.,)*16);self.assertEqual(values[16:32],(113.,)*16)
  self.assertEqual(values[32:48],(1.,)*16)
 def test_negative_scale_and_signed_zero(self):
  b=struct.pack('<eH',-1.,0xaaaa)+bytes([0x11]*4)+bytes([0x88]*128)
  self.assertEqual(struct.unpack('<f',P.decode(b)[:4])[0],-1.)
  zero=struct.pack('<eH',-0.,0xaaaa)+bytes([0x11]*4)+bytes([0x88]*128)
  self.assertEqual(P.decode(zero)[:4],struct.pack('<f',-0.))
 def test_nonfinite_and_extent(self):
  for raw in [b'',b'x'*135,struct.pack('<e',float('inf'))+bytes(134),struct.pack('<e',float('nan'))+bytes(134)]:
   with self.assertRaises(ValueError):P.decode(raw)
 def test_cpp_actual_cast_and_f32_operand(self):
  s=(ROOT/'tests/wasm-layer3-f32/operator.cpp').read_text()
  self.assertIn('ggml_cast(o.ctx, w, GGML_TYPE_F32)',s);self.assertIn('ggml_get_type_traits_cpu(operand->type)->vec_dot_type != GGML_TYPE_F32',s)
  self.assertIn('ggml_mul_mat(o.ctx, operand, x)',s);self.assertNotIn('setenv(',s)
 def test_source_only_no_production_link(self):
  s=(ROOT/'tests/wasm-layer3-f32/CMakeLists.txt').read_text()
  self.assertIn('GGML_LLAMAFILE ON',s);self.assertIn('GGML_LLAMAFILE OFF',s);self.assertIn('CMAKE_POSITION_INDEPENDENT_CODE ON',s)
  self.assertNotIn('src/crispembed.cpp',s);self.assertIn('-sPTHREAD_POOL_SIZE=2',s)
 def test_harness_precedes_candidate_with_six_anchors(self):
  s=(ROOT/'tools/ci-heavy/wasm_layer3_f32_v1.py').read_text();tree=ast.parse(s)
  f=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='native')
  self.assertLess(ast.get_source_segment(s,f).index('if candidate:anchors(out)'),ast.get_source_segment(s,f).index('lib=C.CDLL'))
  self.assertIn("require(len(pairs)==25",s);self.assertIn("old.collect(process.returncode",s)
  self.assertIn("require(cpp['status']=='passed'",s)
 def test_browser_isolation_and_failure_outputs(self):
  s=(ROOT/'tests/wasm-layer3-f32/browser.cjs').read_text()
  for part in ['Cross-Origin-Opener-Policy','Cross-Origin-Embedder-Policy','crossOriginIsolated','SharedArrayBuffer','coefficient_output_preserved','Six original anchors required']:
   self.assertIn(part,s)
 def test_workflow_failure_retention_and_order(self):
  s=(ROOT/'.github/workflows/wasm-layer3-f32-v1.yml').read_text()
  self.assertEqual(s.count('if: always()'),2);self.assertIn('fetch-depth: 0',s);self.assertIn('timeout-minutes: 60',s)
  self.assertLess(s.index('Six byte-exact quantized anchors'),s.index('Native1/2 actual graph F32 cast'))
 def test_frozen_source_bodies_unchanged(self):
  cfg=json.loads(P.PATH.read_text())
  for name in cfg['protected_sha256']:
   self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show',P.BASE+':'+name],cwd=ROOT))
if __name__=='__main__':unittest.main()
