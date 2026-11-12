#!/usr/bin/env python3
import ast,hashlib,json,sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

class Controls(unittest.TestCase):
 def test_closed_candidate_and_scope(self):
  cfg=json.loads((HERE/'protocol.json').read_text());self.assertEqual(cfg['candidate_source'],'a8a22749ab0b757ae1f317b6fa5b84ffd148b939')
  self.assertFalse(any(cfg[k]for k in ['model_forward_allowed','source_capture_repeat_allowed','full_f32_model_allowed']));self.assertEqual(len(cfg['tensors']),11);self.assertEqual(cfg['gates']['CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU'],'1')
 def test_real_source_stage_names(self):
  file=ROOT/'tests/results/qwen35-ocr/lighton4b-attention252-stages.json'
  if file.is_file():d=json.loads(file.read_text())
  else:
   rows=[p for p in Path('/kaggle/input').rglob('attention252.json')if p.is_file()and p.stat().st_size<1<<20 and hashlib.sha256(p.read_bytes()).hexdigest()=='f26070189c1fe71074a149814d6e90607d7fb31c7bbea03e79eb3447edb576ba'];self.assertEqual(len(rows),1);d=json.loads(rows[0].read_text())
  needed={s+'-3'for s in ['attn_norm','Qcur_full','Kcur_raw','Vcur_raw','Qcur_normed','Kcur_normed','Qcur','Kcur','gate_reshaped','attn_gated','attn_output','attn_residual','attn_post_norm','ffn_gate','ffn_up','ffn_swiglu','ffn_out','l_out']}
  for g in range(1,5):self.assertTrue(needed<={r['name']for r in d['source_graph_indexes'][g]['records']})
 def test_component_mode_and_identity_contract(self):
  source=(HERE/'component.cpp').read_text();self.assertIn('b.cache.physical_rope==mode',source);self.assertIn('actual context mode toggle rejected',source);self.assertIn('actual cache recovery after rejected toggle',source);self.assertIn('OFF frozen native layer3 byte closure',source);self.assertIn('ON source-exact Q/K RoPE',source)
  worker=(HERE/'worker.py').read_text();self.assertIn("r['passed']=r['all_source_stages_exact']",worker);self.assertIn("'test-lighton4b-decoder-compose'",worker);self.assertNotIn('n_predict',worker)
  cmake=(HERE/'CMakeLists.txt').read_text();self.assertIn('physical-rope-wasm-guard',cmake);self.assertIn('-ffp-contract=off',cmake)

if __name__=='__main__':unittest.main()
