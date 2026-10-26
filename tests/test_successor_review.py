import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ci-heavy'))
import successor_review as review
from probe_posformer_release import MODEL_SHA, MANIFEST_SHA, compare_release_records

class ReviewControls(unittest.TestCase):
 def fixture(self,root):
  run='123';audit='b'*40
  base={'schema':'crispembed-successor-fresh-consumer-v2','source_commit':'a'*40,'consumer_source':audit,
   'runtime_source_closure':{'scope':'Exact production/build/binding Git objects; audit helpers may differ',
    'consumer_source':audit,'objects':['040000 tree '+'c'*40+'\tsrc']}}
  def save(artifact,name,data):
   path=root/(artifact+'-'+run)/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
  metric={'passed':True,'max_abs':0.0,'cosine':1.0}
  desktops={}
  for platform in review.DESKTOPS:
   data=dict(base,desktop_bounded_consumer_positive=True,empty_cache_download=True,empty_cache_offline_rejected=True,
    offline_reuse=True,comparisons={k:metric for k in ('cli_to_cabi','offline_cli','python_to_cabi','compiled_C_to_cabi')},
    vector=[1]+[0]*383,model_sha256=review.MINILM_MODEL_SHA,text=review.MINILM_TEXT,compiled_hparams_size=40,abi_exports=['crispembed_init','crispembed_encode','crispembed_free','crispembed_resolve_model','crispembed_set_offline'],
    payload_artifact={'id':1,'name':platform,'size_in_bytes':1000,'digest':'sha256:'+'d'*64},
    archives=[{'name':'crispembed-'+platform+'.tar.gz','sha256':'e'*64,'member_count':10}])
   data.update({k:hashlib.sha256((platform+k).encode()).hexdigest() for k in ('library_sha256','cli_binary_sha256','compiled_c_consumer_sha256')})
   save('successor-consumer-'+platform,'consumer.json',data);desktops[platform]=data
  for platform in review.MOBILES:
   save('successor-mobile-'+platform,'consumer.json',dict(base,mobile_abi=platform,cross_link_passed=True,cross_link_exit_code=0,device_execution_performed=False))
  # Actual browser/summary schemas carry source identity without a prepare closure.
  identity={'source_commit':base['source_commit'],'consumer_source':audit}
  save('successor-wasm-consumer','browser.json',dict(identity,schema='crispembed-successor-wasm-consumer-v1',
   page_errors=[],ocr={'authored_black_blob_cleared_rgb_exact':True},embed={'vector':[1]+[0]*383},max_abs=0,cosine=1))
  mac=desktops['macos-arm64']
  candidate=dict(base,payload_artifact=mac['payload_artifact'],archives=mac['archives'])
  save('successor-apple-released','apple-candidate/consumer.json',candidate)
  arms={}
  for name,gate in (('released',None),('candidate-unset',None),('candidate-one','1'),('candidate-zero','0')):
   cases=[]
   for i in range(50):
    confidence=[.8,.9] if name=='released' else [.8001,.9001]
    cases.append({'id':'case-'+str(i),'image_sha256':hashlib.sha256(str(i).encode()).hexdigest(),
     'text':('legacy ' if name=='candidate-zero' else '')+str(i),'token_ids':[10+i,2],'stop':'eos',
     'confidence':confidence,'mean_confidence':sum(confidence)/len(confidence),'elapsed_seconds':.1,
     'log_sha256':hashlib.sha256(('log'+str(i)).encode()).hexdigest()})
   item={'library':'libcrispembed.dylib',
    'library_source':'78493a31fc3043f3af4ab6fd2f197a32da6f07f0' if name=='released' else base['source_commit'],
    'library_sha256':'f'*64 if name=='released' else mac['library_sha256'], 'tool_sha256':'e'*64,
    'model_sha256':MODEL_SHA,'manifest_sha256':MANIFEST_SHA,'threads':2,'trained_forward_env':gate,
    'completed':True,'cases':cases,'peak_rss_mib':100.0}
   save('successor-apple-released','apple-released/'+name+'.json',item);arms[name]=item
  comparisons={name:compare_release_records(arms['released'],arms[name]) for name in ('candidate-unset','candidate-one')}
  save('successor-apple-released','apple-released/comparison.json',dict(identity,
   schema='crispembed-successor-apple-released-comparison-v1',candidate_unset_one_exact=True,legacy_zero_preserved=True,
   comparisons=comparisons,released_archive_sha256='d'*64))
  binding={'passed':True,'comparisons':[metric]*12}
  save('successor-prebuilt-bindings','successor-bindings.json',dict(audit_source=audit,passed=True,producer=base,native_build_fallback=False,offline_cache_unchanged=True,surfaces={'rust':binding,'dart':binding}))
  return run,audit
 def mutate(self,root,run,artifact,name,change):
  path=root/(artifact+'-'+run)/name
  data=json.loads(path.read_text());change(data);path.write_text(json.dumps(data))
 def test_complete_scoped_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root);result=review.review(root,run,audit)
   self.assertTrue(result['desktop_browser_apple50_and_linux_bindings_positive'])
   self.assertFalse(result['mobile_device_execution_verified'])
   self.assertEqual(result['desktop_reference_platform'],'linux-x86_64')
   self.assertEqual(set(result['desktop_vector_comparisons']),set(review.DESKTOPS))
   self.assertTrue(all(v['passed'] and v['max_abs']==0 and v['rms']==0 for v in result['desktop_vector_comparisons'].values()))
 def test_orthogonal_desktop_cannot_hide_behind_positive_self_comparisons(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root)
   self.mutate(root,run,'successor-consumer-linux-arm64','consumer.json',lambda x:x.update(vector=[0,1]+[0]*382))
   with self.assertRaisesRegex(ValueError,'Cross-platform desktop vector parity failed: linux-arm64'):
    review.review(root,run,audit)
 def test_every_desktop_requires_shared_model_and_text(self):
  for platform in review.DESKTOPS:
   for key,value in (('model_sha256','f'*64),('model_sha256',None),('text','changed'),('text',None)):
    with self.subTest(platform=platform,field=key),tempfile.TemporaryDirectory() as d:
     root=Path(d);run,audit=self.fixture(root)
     self.mutate(root,run,'successor-consumer-'+platform,'consumer.json',lambda x:x.update({key:value}))
     with self.assertRaisesRegex(ValueError,'shared model/text/finite384-vector'):review.review(root,run,audit)
 def test_every_desktop_vector_shape_finite_type_and_magnitude(self):
  variants=[None,{},[1]*383,[1]*385,[0]*384,[True]+[0]*383,
   ['1']+[0]*383,[None]+[0]*383,[float('nan')]+[0]*383,[float('inf')]+[0]*383]
  for platform in review.DESKTOPS:
   for vector in variants:
    with self.subTest(platform=platform,vector=str(vector)[:30]),tempfile.TemporaryDirectory() as d:
     root=Path(d);run,audit=self.fixture(root)
     self.mutate(root,run,'successor-consumer-'+platform,'consumer.json',lambda x:x.update(vector=vector))
     with self.assertRaises(ValueError):review.review(root,run,audit)
 def test_absolute_and_cosine_gates_are_both_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root)
   delta=.0021
   self.mutate(root,run,'successor-consumer-windows-x86_64','consumer.json',lambda x:x.update(vector=[math.sqrt(1-delta*delta),delta]+[0]*382))
   with self.assertRaisesRegex(ValueError,'Cross-platform desktop vector parity failed'):review.review(root,run,audit)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root);baseline=[1/math.sqrt(384)]*384
   for platform in review.DESKTOPS:
    self.mutate(root,run,'successor-consumer-'+platform,'consumer.json',lambda x:x.update(vector=baseline))
   drift=[v+(.001 if i%2 else -.001) for i,v in enumerate(baseline)]
   measured=review.compare_desktop_vectors(baseline,drift)
   self.assertLess(measured['max_abs'],.002);self.assertLess(measured['cosine'],.99999)
   self.mutate(root,run,'successor-consumer-macos-arm64','consumer.json',lambda x:x.update(vector=drift))
   with self.assertRaisesRegex(ValueError,'Cross-platform desktop vector parity failed'):review.review(root,run,audit)
 def test_small_cross_platform_drift_is_measured_and_allowed(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root);delta=.002
   self.mutate(root,run,'successor-consumer-linux-arm64','consumer.json',lambda x:x.update(vector=[math.sqrt(1-delta*delta),delta]+[0]*382))
   result=review.review(root,run,audit);measured=result['desktop_vector_comparisons']['linux-arm64']
   self.assertEqual(measured['max_abs'],.002);self.assertTrue(measured['passed'])
   self.assertGreater(measured['rms'],0);self.assertAlmostEqual(measured['native_norm'],1)
 def test_flat_apple_layout_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root)
   artifact=root/('successor-apple-released-'+run)
   (artifact/'apple-released/comparison.json').rename(artifact/'comparison.json')
   with self.assertRaises(FileNotFoundError):review.review(root,run,audit)
 def test_missing_platform_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root)
   (root/('successor-mobile-ios-arm64-'+run)/'consumer.json').unlink()
   with self.assertRaises(FileNotFoundError):review.review(root,run,audit)
 def test_mixed_producer_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root)
   self.mutate(root,run,'successor-consumer-macos-arm64','consumer.json',lambda x:x.update(source_commit='d'*40))
   with self.assertRaises(ValueError):review.review(root,run,audit)
 def test_prepare_closure_required_for_every_prepared_role(self):
  roles=[('successor-consumer-linux-x86_64','consumer.json',False),
   ('successor-mobile-ios-arm64','consumer.json',False),
   ('successor-apple-released','apple-candidate/consumer.json',False),
   ('successor-prebuilt-bindings','successor-bindings.json',True)]
  for artifact,name,nested in roles:
   for variant in ('missing','empty','different','old-schema','wrong-type','wrong-consumer'):
    with self.subTest(role=artifact,variant=variant),tempfile.TemporaryDirectory() as d:
     root=Path(d);run,audit=self.fixture(root)
     def change(item):
      if nested:item=item['producer']
      if variant=='missing':item.pop('runtime_source_closure')
      elif variant=='empty':item['runtime_source_closure']['objects']=[]
      elif variant=='different':item['runtime_source_closure']['objects']=['different source object']
      elif variant=='wrong-type':item['runtime_source_closure']['objects']='a non-list closure'
      elif variant=='wrong-consumer':item['runtime_source_closure']['consumer_source']='d'*40
      else:item['schema']='old-prepare'
     self.mutate(root,run,artifact,name,change)
     with self.assertRaises(ValueError):review.review(root,run,audit)
 def test_raw_apple_content_cannot_hide_behind_positive_summary(self):
  for key,value in [('token_ids',[99,2]),('text','changed'),('stop','length'),
                    ('confidence',[.81,.91]),('mean_confidence',.86),('confidence',[float('nan'),.9])]:
   with self.subTest(field=key,value=value),tempfile.TemporaryDirectory() as d:
    root=Path(d);run,audit=self.fixture(root)
    def change(item):item['cases'][0][key]=value
    self.mutate(root,run,'successor-apple-released','apple-released/candidate-unset.json',change)
    with self.assertRaises(ValueError):review.review(root,run,audit)
 def test_apple_raw_source_gate_library_and_summary_identity(self):
  variants=[('candidate-one','library_source','d'*40),('candidate-zero','trained_forward_env','1'),
   ('candidate-unset','library_sha256','f'*64),('released','library_source','d'*40)]
  for arm,key,value in variants:
   with self.subTest(arm=arm,field=key),tempfile.TemporaryDirectory() as d:
    root=Path(d);run,audit=self.fixture(root)
    self.mutate(root,run,'successor-apple-released','apple-released/'+arm+'.json',lambda x:x.update({key:value}))
    with self.assertRaises(ValueError):review.review(root,run,audit)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root)
   def change(item):item['comparisons']['candidate-one']['max_confidence_abs']=0.0
   self.mutate(root,run,'successor-apple-released','apple-released/comparison.json',change)
   with self.assertRaises(ValueError):review.review(root,run,audit)
 def test_legacy_diagnostic_payload_is_retained_without_quality_waiver(self):
  for field,value in [('id','other-case'),('image_sha256','0'*64),('token_ids',[]),
                      ('confidence',[float('nan')]),('stop','other')]:
   with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
    root=Path(d);run,audit=self.fixture(root)
    def change(item):item['cases'][0][field]=value
    self.mutate(root,run,'successor-apple-released','apple-released/candidate-zero.json',change)
    with self.assertRaises(ValueError):review.review(root,run,audit)
 def test_apple_candidate_archive_must_match_actual_mac_desktop(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run,audit=self.fixture(root)
   def change(item):item['payload_artifact']['id']=2
   self.mutate(root,run,'successor-apple-released','apple-candidate/consumer.json',change)
   with self.assertRaises(ValueError):review.review(root,run,audit)
if __name__=='__main__':unittest.main()
