#!/usr/bin/env python3
"""Bounded source/stdlib statistics and negative admission controls; no frameworks/build/model."""
import ast,copy,hashlib,json,pathlib,struct,subprocess,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/josev-ffn-imatrix-v1'),str(ROOT/'tools')]
import josev_ffn_statistics as S
import overlay
import josev2b_final_pair_imatrix_contract as K
import quantize_josev2b_final_pair_imatrix as P

class Controls(unittest.TestCase):
 def test_explicit_protocol_and_source_inventory(self):
  p=json.loads(K.PATH.read_text());K.validate(p)
  for name,pin in p['protected_sha256'].items():self.assertEqual(K.sha(ROOT/name),pin,name)
  for name,pin in p['owned_sha256'].items():self.assertEqual(K.sha(ROOT/name),pin,name)
  self.assertTrue(all(not n.startswith('src/')for n in p['owned_sha256']))
 def test_no_frozen_runtime_or_old_worker_changes(self):
  for name in ['src','CMakeLists.txt','ggml','tools/ci-heavy/josev2b_last_projection.py','tests/josev-packed/probe.cpp','tests/josev-q4k/CMakeLists.txt','tools/quantize.cpp']:
   self.assertEqual(K.git('rev-parse','3cc96dbabcc75b7c7665164e70a26db93da653e7:'+name),K.git('rev-parse','HEAD:'+name),name)
 def test_role_plan_is_same_two_q4_no_q8(self):
  plan=P.plan(P.previous.schema());self.assertEqual({t:sum(k==t for k in plan.values())for t in P.COUNTS},P.COUNTS)
  self.assertEqual({n for n,t in plan.items()if t==12},set(S.NAMES));self.assertNotIn(8,plan.values())
 def test_scope_gates_holdout_and_counts_cannot_drift(self):
  p=json.loads(K.PATH.read_text())
  for key,value in [('holdout_access',True),('python_activation_gold',True),('model_forward_maximum',36),('recipes',['calibrated-pair']),('gates',{})]:
   q=copy.deepcopy(p);q[key]=value
   with self.assertRaises(ValueError):K.validate(q)
  q=copy.deepcopy(p);q['calibration']['total_rows']=2386
  with self.assertRaises(ValueError):K.validate(q)
 def test_statistics_are_raw_sums_not_means_with_one_row_count(self):
  data=struct.pack('<4f',1,2,3,4);sums,count=S.statistics([data],width=2)
  self.assertEqual(count,2);self.assertEqual(struct.unpack('<2f',sums),(10.,20.))
  sums2,count2=S.statistics([data,data],width=2);self.assertEqual(count2,4);self.assertEqual(struct.unpack('<2f',sums2),(20.,40.))
  for bad in [b'',b'123',struct.pack('<2f',float('nan'),1),struct.pack('<2f',float('inf'),1)]:
   with self.assertRaises(ValueError):S.statistics([bad],width=2)
 def test_imatrix_roundtrip_exact_two_names_u64_count_and_loader_means(self):
  with tempfile.TemporaryDirectory()as tmp:
   path=pathlib.Path(tmp)/'importance.gguf';data=struct.pack('<2048f',*([10.]*2048));proof=S.write(path,data,2)
   self.assertEqual(proof['row_count'],2);self.assertEqual(proof['columns'],2048);self.assertTrue(proof['vectors_identical'])
   self.assertEqual(proof['loader_mean_squared_f32_sha256'],hashlib.sha256(struct.pack('<2048f',*([5.]*2048))).hexdigest())
   self.assertEqual(S.verify(path,2),proof);self.assertEqual(proof['positive_columns'],2048);self.assertEqual(proof['zero_columns'],0)
   fractional=pathlib.Path(tmp)/'fractional.gguf';fractional_proof=S.write(fractional,data,3)
   self.assertEqual(fractional_proof['loader_mean_squared_f32_sha256'],hashlib.sha256(struct.pack('<2048f',*([10.*(1./3.)]*2048))).hexdigest())
   with self.assertRaises(ValueError):S.verify(path,3)
   with self.assertRaises(ValueError):S.write(pathlib.Path(tmp)/'zero.gguf',data,0)
   with self.assertRaises(ValueError):S.write(pathlib.Path(tmp)/'names.gguf',data,2,names=[S.NAMES[0],S.NAMES[0]])
 def test_nonfinite_negative_and_malformed_imatrix_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   path=pathlib.Path(tmp)/'bad.gguf'
   for value in [float('nan'),float('inf'),-1.]:
    data=struct.pack('<2048f',*([value]*2048))
    with self.assertRaises(ValueError):S.write(path,data,1)
   data=struct.pack('<2048f',*([1.]*2048));S.write(path,data,1);raw=path.read_bytes();path.write_bytes(raw[:-1])
   with self.assertRaises(ValueError):S.verify(path)
 def test_loader_nonfatal_fallback_skips_missing_or_unexpected_use_rejected(self):
  good='imatrix: loaded importance vectors for 2 tensors from file\n'
  good+='\n'.join('[1/442] '+n+' - f32, (imatrix) q4_k'for n in S.NAMES)+'\ndone, 2 with imatrix\n'
  self.assertEqual(S.usage(good)['used'],2)
  for bad in [good.replace('for 2 tensors','for 0 tensors'),good.replace(S.NAMES[1],'wrong.weight'),good.replace('(imatrix)','(imatrix shape 2047!=2048, skipped)',1),good+'(fallback Q4_0)',good.replace(', 2 with imatrix',', 1 with imatrix')]:
   with self.assertRaises(ValueError):S.usage(bad)
 def capture(self,folder,rows=1):
  packet=dict(schema='crispembed-josev-native-f32-pre-ffn23-v1',calls=1,layer=23,batch=1,past=0,rows=rows,columns=2048,
   epsilon_bits=struct.unpack('<I',struct.pack('<f',1e-6))[0],gate=S.NAMES[0],up=S.NAMES[1],post_norm='l.layers.23.post_attention_layernorm.weight',stored_types=[0,0,0],actual_native_f32_operand=True,python_reference_activation=False)
  (folder/'pre_ffn23.json').write_text(json.dumps(packet));(folder/'pre_ffn23.f32').write_bytes(struct.pack('<2048f',*([.5]*2048)));return packet
 def test_capture_actual_geometry_count_and_native_provenance_admission(self):
  with tempfile.TemporaryDirectory()as tmp:
   folder=pathlib.Path(tmp);p=self.capture(folder);self.assertEqual(len(S.capture(folder,1)),8192)
   for key,value in [('calls',0),('rows',2),('columns',6144),('layer',22),('python_reference_activation',True),('stored_types',[1,0,0])]:
    q=copy.deepcopy(p);q[key]=value;(folder/'pre_ffn23.json').write_text(json.dumps(q))
    with self.assertRaises(ValueError):S.capture(folder,1)
 def test_disposable_overlay_has_one_correct_live_source_and_only_observed_macro(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=pathlib.Path(tmp)
   names=['src/qwen35_decoder.cpp','tests/josev-packed/probe.cpp','tests/josev-q4k/CMakeLists.txt']
   for name in names:(root/name).parent.mkdir(parents=True,exist_ok=True);(root/name).write_bytes((ROOT/name).read_bytes())
   changes=overlay.apply(root);self.assertEqual(set(changes),set(names))
   text=(root/'src/qwen35_decoder.cpp').read_text();self.assertEqual(text.count('josev_ffn_imatrix_test::capture('),1)
   point=text.index('josev_ffn_imatrix_test::capture(');self.assertLess(text.rfind('rms(hidden, layer.post_norm',0,point),point);self.assertLess(point,text.index('project_cpu({ layer.gate, layer.up }'))
   self.assertIn('normalized.data(), width, batch * tokens',text)
   cmake=(root/names[2]).read_text();self.assertIn('target_compile_definitions(josev-q4k-observed PRIVATE CRISPEMBED_JOSEV_FFN_IMATRIX_TESTING)',cmake)
   self.assertNotIn('target_compile_definitions(josev-q4k-plain PRIVATE CRISPEMBED_JOSEV_FFN_IMATRIX_TESTING)',cmake)
   probe=(root/names[1]).read_text();self.assertLess(probe.index('malformed request did not clear'),probe.index('josev_ffn_imatrix_begin(directory'))
 def test_f32_barrier_precedes_calibration_and_every_quantizer(self):
  source=(ROOT/'tools/ci-heavy/josev2b_final_pair_imatrix.py').read_text()
  a=source.index("control=execute('f32'");b=source.index("require(control['numeric_and_complete_typed_parity']",a);c=source.index('statistics.collect(',b);d=source.index('packer.quantize(',c)
  self.assertLess(a,b);self.assertLess(b,c);self.assertLess(c,d)
  self.assertIn("for identity in ('uncalibrated-pair','calibrated-pair')",source)
  self.assertIn("label=='f32'and mode in ('observed','repeat')",source)
  self.assertIn('model_prefills=40,primary_prefills=36,recovery_prefills=4',source)
 def test_original_comparison_and_numeric_judgment_are_unchanged(self):
  before=(ROOT/'tools/ci-heavy/josev2b_last_projection.py').read_text();after=(ROOT/'tools/ci-heavy/josev2b_final_pair_imatrix.py').read_text()
  def block(s):return s.split('            comparison=panel.compare(',1)[1].split("        result['phase']='strict-original-F32-control'",1)[0]
  self.assertEqual(block(before),block(after))
  self.assertNotIn('old.closure=',after);self.assertNotIn('old.GATES=',after)
 def test_workflow_locked_environment_retention_and_no_model_dispatch_in_tests(self):
  text=(ROOT/'.github/workflows/josev-final-pair-imatrix-v1.yml').read_text()
  self.assertIn('--require-hashes',text);self.assertIn("python-version: '3.13'",text);self.assertEqual(text.count('if: always()'),2);self.assertIn('retention-days: 90',text)
  for path in [ROOT/'tools/ci-heavy/josev2b_final_pair_imatrix.py',ROOT/'tools/quantize_josev2b_final_pair_imatrix.py',ROOT/'tools/josev2b_final_pair_imatrix_contract.py',*list((ROOT/'tools/josev-ffn-imatrix-v1').glob('*.py'))]:ast.parse(path.read_text())
if __name__=='__main__':unittest.main()
