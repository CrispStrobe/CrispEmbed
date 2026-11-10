import importlib.util,json,struct,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_hf_tiled_out_pack as pack
from lighton4b_q6_convert import text
from gguf_file_type_canonicalize import parse
class Tests(unittest.TestCase):
 def fixture(self,path):
  fields={'qwen35_ocr.mixed_quant_recipe':'lighton4b-weighted-q4km-style-q6head-v1','qwen35_ocr.quant_coordinate_profile':'hf-grouped-v1','crispembed.quantization.imatrix_profile':'lighton4b-text-inputs-hf-grouped-v1','unrelated':'must remain exact'}
  records=[text(k)+struct.pack('<I',8)+text(v)for k,v in fields.items()];records.append(text('qwen35_ocr.zero_centered_text_norm')+struct.pack('<IB',7,1));descriptor=text('scalar')+struct.pack('<I',1)+struct.pack('<QIQ',1,0,0);header=b'GGUF'+struct.pack('<IQQ',3,1,len(records))+b''.join(records)+descriptor;header+=bytes((-len(header))%32);path.write_bytes(header+struct.pack('<f',0.125));return parse(path,pack.TYPES)
 def test_only_closed_metadata_changes(self):
  with tempfile.TemporaryDirectory()as d:
   a=Path(d)/'a';base=self.fixture(a);header,changed=pack.prepare_header(base,a);b=Path(d)/'b';b.write_bytes(header+struct.pack('<f',0.125));after=parse(b,pack.TYPES)
   self.assertEqual(changed,{'qwen35_ocr.mixed_quant_recipe','qwen35_ocr.quant_coordinate_profile','crispembed.quantization.imatrix_profile'})
   for k,v in base['metadata'].items():
    if k not in changed:self.assertEqual(v['raw_sha256'],after['metadata'][k]['raw_sha256'])
   for k,v in pack.strings().items():self.assertEqual(after['metadata'][k]['value'],v)
   for k,v in pack.numbers().items():self.assertEqual(after['metadata'][k]['value'],v)
   self.assertEqual(b.read_bytes()[after['data_offset']:],a.read_bytes()[base['data_offset']:])
   with self.assertRaises(ValueError):pack.prepare_header(after,b)
 def test_existing_receipt_preserved(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);receipt=p/'receipt.json';receipt.write_bytes(b'preserve-original');
   with self.assertRaises(ValueError):pack.pack(p/'absent',p/'absent',p/'absent',p/'new',receipt)
   self.assertEqual(receipt.read_bytes(),b'preserve-original');self.assertFalse((p/'new').exists())
 def test_proof_identity_fails_before_large_source(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);paired=p/'wrong';paired.write_text('{}');receipt=p/'receipt.json'
   with self.assertRaisesRegex(ValueError,'exact measured original-F16 receipt required'):pack.pack(p/'absent',paired,p/'absent',p/'new',receipt)
   self.assertEqual(json.loads(receipt.read_text())['status'],'failed');self.assertFalse((p/'new').exists())
if __name__=='__main__':unittest.main()
