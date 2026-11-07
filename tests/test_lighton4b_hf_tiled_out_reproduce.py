import importlib.util,json,math,struct,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('reproducer',ROOT/'tools/lighton4b_hf_tiled_out_reproduce.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
 def test_physical_statistics_not_inverse_permuted(self):
  values=[float(i+1)*66048 for i in range(4096)];means=struct.unpack('<4096f',m.physical_mean(values,[66048.]));self.assertEqual(means[128],129.);self.assertEqual(means[2048],2049.);self.assertNotEqual(means[128],2049.)
 def test_bad_statistics_fail_closed(self):
  for values,count in [([0.]*4096,[66048]),([1.]*4095,[66048]),([1.]*4096,[1]),([math.nan]+[1.]*4095,[66048]),([-1.]+[1.]*4095,[66048])]:
   with self.assertRaises(ValueError):m.physical_mean(values,count)
 def test_protocol_rejects_missing_or_wrong_tensor_domain(self):
  good=m.protocol()
  for mutate in ('missing','duplicate','wrong-geometry','wrong-source'):
   bad=json.loads(json.dumps(good))
   if mutate=='missing':bad['selected'].pop()
   elif mutate=='duplicate':bad['selected'][0]=bad['selected'][1]
   elif mutate=='wrong-geometry':bad['selected'][0]['source_type']=12
   else:bad['source_original_f16']['sha256']='0'*64
   with tempfile.TemporaryDirectory()as folder:
    p=Path(folder)/'protocol.json';p.write_text(json.dumps(bad))
    with patch.object(m,'PROTOCOL',p),self.assertRaises(ValueError):m.protocol()
 def test_existing_receipt_preserved(self):
  with tempfile.TemporaryDirectory()as folder:
   root=Path(folder);receipt=root/'receipt.json';receipt.write_bytes(b'keep exact prior evidence')
   with self.assertRaises(ValueError):m.reproduce(root/'source',root/'stats',root/'helper',root/'outputs',receipt)
   self.assertEqual(receipt.read_bytes(),b'keep exact prior evidence');self.assertFalse((root/'outputs').exists())
 def test_quantized_or_arbitrary_source_never_reaches_helper(self):
  with tempfile.TemporaryDirectory()as folder:
   root=Path(folder);source=root/'fake.gguf';source.write_bytes(b'GGUF'+b'\0'*60)
   with self.assertRaises(ValueError):m.reproduce(source,root/'stats',root/'helper',root/'outputs',root/'receipt.json')
   r=json.loads((root/'receipt.json').read_text());self.assertFalse(r['quantization_attempted']);self.assertFalse(r['candidate_conversion_allowed']);self.assertFalse(r['full_f32_model_instantiated'])
if __name__=='__main__':unittest.main()
