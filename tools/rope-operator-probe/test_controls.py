#!/usr/bin/env python3
import hashlib,importlib.util,json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
import observe

class Controls(unittest.TestCase):
 def test_observer_reverses_to_exact_production_bytes(self):
  original=(ROOT/'src/qwen35_attention.cpp').read_text();cfg=json.loads((HERE/'protocol.json').read_text());self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),cfg['native_attention_sha256'])
  generated=observe.generate(original)
  prefix='#include <fstream>\n#include <cstdlib>\n#include <stdexcept>\n#include <string>\n'
  self.assertTrue(generated.startswith(prefix));restored=generated[len(prefix):].replace(observe.HOOK+'\n'+observe.ANCHOR,observe.ANCHOR)
  self.assertEqual(restored,original)
  self.assertLess(generated.index('append("queries.actual.f32"'),generated.index(observe.ANCHOR))
  with self.assertRaises(ValueError):observe.generate(original+observe.ANCHOR)
 def test_closed_operator_scope_and_profile(self):
  cfg=json.loads((HERE/'protocol.json').read_text());self.assertEqual(cfg['positions'],[4,252,6,4]);self.assertEqual(cfg['heads'],[16,4]);self.assertEqual(len(cfg['tensors']),11)
  self.assertFalse(any(cfg[k]for k in['model_forward_allowed','source_capture_repeat_allowed','whole_f32_model_allowed']))
  self.assertEqual(cfg['rope']['mode'],40);self.assertEqual(cfg['rope']['n_dims'],64);self.assertEqual(cfg['rope']['sections'],[11,11,10,0]);self.assertEqual(cfg['rope']['ext_factor'],0)
  text=(HERE/'CMakeLists.txt').read_text();self.assertIn('target_compile_options(rope-native-capture PRIVATE -ffp-contract=off)',text)
  self.assertNotIn('add_library(crispembed-core',text)
 def test_all22_native_stage_pairs_are_required(self):
  code=(HERE/'worker.py').read_text();self.assertIn('len(neutral)==22',code);self.assertIn("a.read_bytes()==b.read_bytes()",code);self.assertIn("require(row['bit_exact'],'actual source RoPE byte closure')",code)
  self.assertNotIn('n_predict',code)

if __name__=='__main__':unittest.main()
