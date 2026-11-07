"""Dense input-statistics adapter controls; no model/weight quantization."""
import copy,hashlib,importlib.util,math,struct,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_imatrix_adapter as m
REAL=None
if '--source' in sys.argv:
 i=sys.argv.index('--source');REAL=Path(sys.argv[i+1]);del sys.argv[i:i+2]
def fixture():return {k:([float(i%97+1)for i in range(v['width'])],[66048.0])for k,v in m.inventory().items()}
class Controls(unittest.TestCase):
 def test_inverse_head_coordinate_scatter(self):
  tiled=[float((key*2+sub)*128+d)for sub in range(2)for key in range(16)for d in range(128)];self.assertEqual(m.native_order(tiled),list(map(float,range(4096))))
  with self.assertRaises(ValueError):m.native_order(tiled+[0])
 def test_complete_dense_coverage_and_normalization(self):
  source=fixture();converted,rows=m.validate_statistics(source);self.assertEqual(len(converted),248);self.assertEqual(sum(r['inverse_v_head_columns']for r in rows),24)
  for row in rows:
   raw,count=converted[row['target']];self.assertEqual(count,66048);expected=m.native_order(source[row['source']][0])if row['inverse_v_head_columns']else source[row['source']][0];self.assertEqual(raw,expected);mean=[m.f32(x/count)for x in expected];self.assertEqual(hashlib.sha256(struct.pack('<'+'f'*len(mean),*mean)).hexdigest(),row['normalized_f32_sha256'])
 def test_missing_extra_foreign_domain(self):
  src=fixture();name=next(iter(src))
  for bad in ({k:v for k,v in src.items()if k!=name},{**src,'v.vision.weight':([1],[66048])}):
   with self.assertRaises(ValueError):m.validate_statistics(bad)
  with self.assertRaises(ValueError):m.validate_statistics(src,128,512)
 def test_bad_counts_energy_and_padding(self):
  src=fixture();name=next(iter(src));values=src[name][0]
  for counts in ([0],[-1],[math.nan],[math.inf],[66048.5],[2**25],[129],[66048,66048]):
   with self.assertRaises(ValueError):m.validate_statistics({**src,name:(values,counts)})
  for bad in (values+[1],[0]*len(values),[math.nan]+values[1:],[-1]+values[1:],[math.inf]+values[1:]):
   with self.assertRaises(ValueError):m.validate_statistics({**src,name:(bad,[66048])})
 def test_format5_header_fixture_flattens_only_conv3d(self):
  spec=importlib.util.spec_from_file_location('controller',ROOT/'tools/ci-heavy/lighton4b_imatrix_adapter_controls.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
  with tempfile.TemporaryDirectory()as d:
   f=Path(d)/'header';c.target_header(f);raw=f.read_bytes();at=24
   def text():
    nonlocal at
    n=struct.unpack_from('<Q',raw,at)[0];at+=8;v=raw[at:at+n].decode();at+=n;return v
   for _ in range(2):text();kind=struct.unpack_from('<I',raw,at)[0];at+=4;self.assertEqual(kind,8);text()
   found={}
   for _ in range(723):
    name=text();rank=struct.unpack_from('<I',raw,at)[0];at+=4;self.assertLessEqual(rank,4);shape=list(struct.unpack_from('<'+'Q'*rank,raw,at));at+=8*rank;kind,offset=struct.unpack_from('<IQ',raw,at);at+=12;self.assertEqual(kind,0);found[name]=shape
   self.assertEqual(found['v.patch_embed.proj.weight'],[1536,1024]);self.assertEqual(found['l.layers.0.mlp.gate_proj.weight'],[2560,9216])
 def test_legacy_fixture_has_canonical_payload_padding(self):
  spec=importlib.util.spec_from_file_location('legacy_controller',ROOT/'tools/ci-heavy/lighton4b_imatrix_adapter_controls.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
  with tempfile.TemporaryDirectory()as d:
   f=Path(d)/'legacy';c.legacy_header_fixture(f);r=m.parse(f,{0:(1,4)});self.assertEqual((f.stat().st_size-r['data_offset'])%32,0);self.assertEqual(r['metadata']['count.legacy.weight']['kind'],10);self.assertEqual(r['metadata']['count.legacy.weight']['value'],2);self.assertEqual(r['tensors']['legacy.weight']['stored_shape'],[2])
 def test_no_clobber_wrong_source_or_path_alias(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);wrong=p/'wrong';wrong.write_bytes(b'not approved');out=p/'out';report=p/'report'
   with self.assertRaises(ValueError):m.adapt(wrong,out,report)
   self.assertFalse(out.exists());self.assertEqual(__import__('json').loads(report.read_text())['status'],'failed')
   with self.assertRaises(ValueError):m.adapt(wrong,out,out)
   self.assertFalse(out.exists())
 @unittest.skipUnless(REAL,'real public fixture supplied by scoped worker')
 def test_real_approved_file_and_consumer_means(self):
  self.assertEqual(hashlib.sha256(REAL.read_bytes()).hexdigest(),m.SOURCE_SHA)
  with tempfile.TemporaryDirectory()as d:
   p=Path(d);report=m.adapt(REAL,p/'out.gguf',p/'receipt.json');self.assertEqual(report['output_sha256'],'105aba299aff48c63b43323bf6ec4f02d20f2fbb726b1ece0728e9e7fbb24663');self.assertEqual(report['entries'],248);self.assertEqual(report['inverse_v_head_vectors'],24);self.assertFalse(report['quantization_executed']);self.assertEqual(len(list(p.glob('*.partial'))),0)
if __name__=='__main__':unittest.main()
