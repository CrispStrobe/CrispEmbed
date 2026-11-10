import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('decode',ROOT/'tools/ci-heavy/lighton4b_imatrix_q4km_decode.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Admission(unittest.TestCase):
 def test_authenticated_but_failed_or_wrong_model_cannot_decode(self):
  good={'source_sha':'a'*40,'status':'measured','actual248_applied':True,'both_frozen_null_scores_exact':True,'post_merge_model_free_controls_passed':True,'candidate':{'types':{'0':474,'12':216,'14':33},'all474_f32_exact':True,'actual_imatrix_applied_entries':248},'cases':{'simple_table':{},'arabic_printed_line':{}}}
  with tempfile.TemporaryDirectory()as folder:
   p=Path(folder)/'receipt.json';p.write_text(json.dumps(good));self.assertEqual(m.admit_producer(p,m.sha(p),'a'*40),good)
   for key,value in [('status','failed'),('actual248_applied',False),('both_frozen_null_scores_exact',False),('post_merge_model_free_controls_passed',False),('source_sha','b'*40),('cases',{'form':{}}),('candidate',dict(good['candidate'],types={'0':474,'12':217,'14':32}))]:
    bad=copy.deepcopy(good);bad[key]=value;p.write_text(json.dumps(bad))
    with self.subTest(key=key),self.assertRaises(ValueError):m.admit_producer(p,m.sha(p),'a'*40)
 def test_cpu_identity_whitespace(self):
  x=m.cpu_profile('vendor_id\t: GenuineIntel\nmodel name\t: fixture CPU\nflags\t\t: avx2 fma\n\nvendor_id: ignored')
  self.assertEqual(x['vendor_id'],'GenuineIntel');self.assertEqual(x['model name'],'fixture CPU');self.assertEqual(x['flags'],'avx2 fma');self.assertTrue(x['identity_fields_available'])
 def test_loader_guards_before_model_staging(self):
  with tempfile.TemporaryDirectory()as folder:
   root=Path(folder);(root/'bin').mkdir();(root/'lib').mkdir()
   with self.assertRaises(ValueError):m.check_runtime(root)
   for name in ('lighton4b-native-document','lighton4b-native-case-preflight'):
    p=root/'bin'/name;p.write_text('#!'+sys.executable+'\nraise SystemExit(2)\n');p.chmod(0o755)
   self.assertEqual(len(m.check_runtime(root)),2)
   p.write_text('#!'+sys.executable+'\nraise SystemExit(127)\n')
   with self.assertRaises(ValueError):m.check_runtime(root)
 def test_tampering_or_unsealed_receipt_cannot_decode(self):
  with tempfile.TemporaryDirectory()as folder:
   p=Path(folder)/'receipt.json';p.write_text('{}')
   for digest,source in [('0'*64,'a'*40),(m.sha(p),'unsealed')]:
    with self.assertRaises(ValueError):m.admit_producer(p,digest,source)
if __name__=='__main__':unittest.main()
