import hashlib,importlib.util,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_gdn_out_pack as pack
class Controls(unittest.TestCase):
 def test_exact_whole_tensor_selector(self):
  selected=[f'l.layers.{i}.linear_attn.out_proj.weight'for i in range(32)if i%4!=3]
  self.assertEqual(len(selected),24)
  for i in range(32):
   for suffix in ('linear_attn.out_proj.weight','linear_attn.in_proj_qkv.weight','linear_attn.in_proj_z.weight','self_attn.o_proj.weight','mlp.down_proj.weight'):
    name=f'l.layers.{i}.{suffix}';self.assertEqual(pack.upgraded(name,{'stored_shape':[4096,2560]}),name in selected)
  for n in ('l.layers.32.linear_attn.out_proj.weight','l.layers.-1.linear_attn.out_proj.weight','x.l.layers.0.linear_attn.out_proj.weight'):
   self.assertFalse(pack.upgraded(n,{'stored_shape':[4096,2560]}))
  self.assertFalse(pack.upgraded(selected[0],{'stored_shape':[4096]}))
 def test_frozen_manifest_and_exact_metadata(self):
  p=json.loads((ROOT/'docs/lighton4b-q4k-gdn-out-q8-v1.json').read_text());rows=p['selected_tensors']
  self.assertEqual(len(rows),24);self.assertEqual(hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),pack.MANIFEST_SHA)
  fields=pack.metadata_fields();self.assertEqual(fields['qwen35_ocr.mixed_quant_selected_count'],24);self.assertEqual(fields['qwen35_ocr.mixed_quant_payload_delta'],125829120)
  self.assertEqual(fields[pack.KEY],p['recipe']);self.assertNotIn('Q4_K_M',fields.values());self.assertEqual(len(fields),6)
if __name__=='__main__':unittest.main()
