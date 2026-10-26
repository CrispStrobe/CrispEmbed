#!/usr/bin/env python3
"""Small controls only: no frameworks, inference, model download or builds."""
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from successor_wasm_trace_contract import Reader,digest,floats,f32_bytes,gguf,metrics,overlay,tokens,replace_one

class Contract(unittest.TestCase):
 def test_tokens_require_one_actual_line(self):
  self.assertEqual(tokens('crispembed: token_ids (n=3): 101 1 102\n'),[101,1,102])
  for bad in ('', 'crispembed: token_ids (n=2): 101\n','crispembed: token_ids (n=1): 30522\n',
   'crispembed: token_ids (n=1): 101\ncrispembed: token_ids (n=1): 101\n'):
   with self.assertRaises(ValueError):tokens(bad)
 def test_magnitudes_and_shape_controls(self):
  m=metrics([1.,2.],[2.,4.])
  self.assertEqual(m['max_abs'],2);self.assertAlmostEqual(m['cosine'],1);self.assertAlmostEqual(m['wasm_l2']/m['native_l2'],2)
  for a,b in (([1.],[]),([0.],[0.]),([1.],[math.inf])):
   with self.assertRaises(ValueError):metrics(a,b)
 def test_f32_rejects_truncated_and_nonfinite(self):
  self.assertEqual(floats(f32_bytes([1.,2.])),[1.,2.])
  for b in (b'',b'abc',struct.pack('<f',math.nan)):
   with self.assertRaises(ValueError):floats(b)
 def test_gguf_inventory_uses_actual_tensor_type(self):
  def string(s):b=s.encode();return struct.pack('<Q',len(b))+b
  header=b'GGUF'+struct.pack('<IQQ',3,1,0)
  info=string('enc.0.attn_q.weight')+struct.pack('<IQQIQ',2,384,384,20,0)
  data=header+info;data+=bytes((-len(data))%32)+b'\x01'
  _,rows=gguf(data)
  self.assertEqual(rows[0]['type'],20);self.assertEqual(rows[0]['shape'],[384,384])
  for bad in (data[:10],data[:-1],data.replace(b'GGUF',b'XXXX',1)):
   with self.assertRaises(ValueError):gguf(bad)
  with self.assertRaises(ValueError):gguf(data,True)
 def test_overlay_changes_only_disposable_three_files(self):
  paths=('src/crispembed.cpp','wasm/embed_wrapper.c','build-embed-wasm.sh')
  originals={name:(ROOT/name).read_bytes() for name in paths}
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)
   for name,b in originals.items():
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
   changes=overlay(root)
   self.assertEqual(set(changes),set(paths))
   for name in paths:self.assertEqual(changes[name]['before'],digest(originals[name]))
   text=(root/'src/crispembed.cpp').read_text()
   self.assertEqual(text.count('Test-only actual pooling buffer capture'),2)
   self.assertIn('-j2',(root/'build-embed-wasm.sh').read_text())
   self.assertIn("'_wasm_trace_q8',",(root/'build-embed-wasm.sh').read_text())
   self.assertIn('quantize_row_q8_0_ref',(root/'wasm/embed_wrapper.c').read_text())
   with self.assertRaises(ValueError):overlay(root)
  for name,b in originals.items():self.assertEqual((ROOT/name).read_bytes(),b)
 def test_exact_source_anchor_controls(self):
  for text in ('missing','xx'):
   with self.assertRaises(ValueError):replace_one(text,'x','y')
  self.assertEqual(replace_one('a x b','x','y'),'a y b')
if __name__=='__main__':unittest.main()
