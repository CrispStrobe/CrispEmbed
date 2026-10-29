#!/usr/bin/env python3
"""Bounded stdlib source, observer schema and artifact controls; no model/build."""
import ast,copy,hashlib,importlib.util,json,os,pathlib,struct,subprocess,sys,tempfile,unittest,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/wasm-live-norm-v1'),str(ROOT/'tools')]
from unittest.mock import patch
import live_contract as K
import overlay
spec=importlib.util.spec_from_file_location('live_driver',ROOT/'tools/ci-heavy/wasm_live_norm_v1.py')
D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)

class Controls(unittest.TestCase):
 def test_frozen_runtime_and_old_owner(self):
  for path in ['src','CMakeLists.txt','tools/ci-heavy/wasm_native_order_norm_v1.py','docs/wasm-native-order-norm-candidate-v1.json','tests/wasm-native-order-norm']:
   self.assertEqual(K.git('rev-parse',K.CANDIDATE+':'+path),K.git('rev-parse','HEAD:'+path))
 def test_scope_rejects_extra_forward_and_tolerances(self):
  p=K.read(K.PROFILE);K.validate(p)
  for key,value in [('cases',['original','empty']),('tokens',13),('norm_nodes',12),('max_absolute',.003),('held_q8_rounding','0'),('runtime_change',True)]:
   q=copy.deepcopy(p);q[key]=value
   with self.assertRaises(ValueError):K.validate(q)
 def test_explicit_source_closure(self):
  p=K.read(K.PROFILE)
  self.assertEqual(set(p['protected_git_objects']),set(K.PROTECTED))
  self.assertIn('src',p['protected_git_objects']);self.assertIn('tools/wasm_layer3_attribution_v1.py',p['protected_git_objects'])
  for path,pin in p['protected_git_objects'].items():self.assertEqual(K.git('rev-parse',K.BASE+':'+path),pin)
  for path,digest in p['owned_sha256'].items():self.assertEqual(K.sha(ROOT/path),digest,path)
 def test_disposable_overlay_exact_anchor_and_no_numeric_observer_nodes(self):
  source=(ROOT/'src/crispembed.cpp').read_text();changed=overlay.encoder(source)
  self.assertIn('ctx->live_norm.add_norm(input, result)',changed)
  self.assertIn('ctx->live_norm.begin(ctx->sched)',changed)
  for name in D.NODES:self.assertIn('layer3("'+name+'"',changed)
  # Observer creates no graph/view/identity node, never renames the live tensor.
  h=(ROOT/'tests/wasm-live-norm/observer.h').read_text()
  for forbidden in ['ggml_set_name','ggml_view_','ggml_set_output','ggml_add(','ggml_mul(','ggml_norm(']:self.assertNotIn(forbidden,h)
  self.assertIn('output->src[0] != input',h);self.assertIn('GGML_OP_MAP_CUSTOM1',h)
  with self.assertRaises(ValueError):overlay.encoder(source.replace('    auto bert_norm = [&](ggml_tensor * input) {','wrong'))
 def test_overlay_applies_only_in_task_owned_copy(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=pathlib.Path(tmp)
   for name in ['src/crispembed.cpp','wasm/embed_wrapper.c','build-embed-wasm.sh']:
    (root/name).parent.mkdir(parents=True,exist_ok=True);(root/name).write_bytes((ROOT/name).read_bytes())
   changes=overlay.apply(root)
   self.assertEqual(set(changes),{'src/crispembed.cpp','src/live_norm_observer.h','wasm/embed_wrapper.c','build-embed-wasm.sh'})
   self.assertEqual(changes['src/crispembed.cpp']['before'],K.sha(ROOT/'src/crispembed.cpp'))
   self.assertEqual(changes['src/crispembed.cpp']['after'],K.sha(root/'src/crispembed.cpp'))
   self.assertIn("'_crispembed_live_norm_finish'",(root/'build-embed-wasm.sh').read_text())
 def fixture(self,folder):
  rows=[]
  for name in sorted({f'norm{i:02d}_{kind}'for i in range(13)for kind in ['input','output']}|{'layer3_'+n for n in D.NODES}):
   width=1536 if name in ['layer3_fc1_raw','layer3_fc1','layer3_gelu']else 384
   data=struct.pack('<f',.5)*(12*width);(folder/(name+'.f32')).write_bytes(data)
   rows.append(dict(name=name,hits=1,shape=[12,width],bytes=len(data)))
  p=dict(schema='crispembed-wasm-live-norm-capture-v1',architecture='bert',width=384,layers=6,heads=12,intermediate=1536,pre_ln=False,cpu=True,enabled=True,
   epsilon_bits=struct.unpack('<I',struct.pack('<f',1e-12))[0],connected_norm_nodes=13,custom_norm_nodes=13,files=rows)
  K.write(folder/'capture.json',p);return p
 def test_complete_realistic_capture_and_sensitive_geometry(self):
  with tempfile.TemporaryDirectory()as tmp:
   folder=pathlib.Path(tmp);p=self.fixture(folder);self.assertEqual(len(D.capture_rows(folder,True)),49)
   for key,value in [('connected_norm_nodes',0),('custom_norm_nodes',12),('enabled',False),('heads',8),('epsilon_bits',0),('cpu',False)]:
    q=copy.deepcopy(p);q[key]=value;K.write(folder/'capture.json',q)
    with self.assertRaises(ValueError):D.capture_rows(folder,True)
 def test_missing_duplicate_wrong_shape_and_nonfinite_capture(self):
  with tempfile.TemporaryDirectory()as tmp:
   folder=pathlib.Path(tmp);p=self.fixture(folder)
   for change in ['missing','duplicate','shape','hits']:
    q=copy.deepcopy(p)
    if change=='missing':q['files'].pop()
    elif change=='duplicate':q['files'][-1]=q['files'][0]
    elif change=='shape':q['files'][0]['shape']=[384,12]
    else:q['files'][0]['hits']=2
    K.write(folder/'capture.json',q)
    with self.assertRaises(ValueError):D.capture_rows(folder,True)
   K.write(folder/'capture.json',p);file=folder/(p['files'][0]['name']+'.f32');data=bytearray(file.read_bytes());data[:4]=struct.pack('<f',float('nan'));file.write_bytes(data)
   with self.assertRaises(ValueError):D.capture_rows(folder,True)
 def test_native_observer_environment_snapshotted_before_init_and_recovers(self):
  snapshots=[]
  class FakeLibrary:
   def crispembed_init(self,path,threads):
    snapshots.append((os.environ.get('CRISPEMBED_DUMP_LAYERS'),os.environ.get('CRISPEMBED_LIVE_NORM_CAPTURE'),threads))
    return 123
  with patch.dict(os.environ,{'CRISPEMBED_DUMP_LAYERS':'stale','CRISPEMBED_LIVE_NORM_CAPTURE':'stale'}):
   for mode in ['ordinary','observed','ordinary']:self.assertEqual(D.native_context(FakeLibrary(),'synthetic-model-name',2,mode),123)
  self.assertEqual(snapshots,[(None,None,2),('1',None,2),(None,None,2)])
 def test_native_ids_multiple_public_forwards_exact(self):
  ids=[101,7,102];line='crispembed: token_ids (n=3): 101 7 102\n';log=line*8
  self.assertTrue(D.native_tokens(log,ids,4));self.assertFalse(D.native_tokens(log+line,ids,4))
  self.assertFalse(D.native_tokens(log.replace('101 7 102','101 8 102',1),ids,4))
 def test_independent_metrics_reject_numeric_error(self):
  a=[float(i)/16 for i in range(16)];self.assertTrue(D.independent(a,a,[2,8])['passed'])
  b=list(a);b[0]+=.01;self.assertFalse(D.independent(a,b,[2,8])['passed'])
 def test_workflow_order_scope_and_retention(self):
  w=(ROOT/'.github/workflows/wasm-live-norm-operands-v1.yml').read_text()
  names=['.py preflight ','.py prepare ','.py component-build ','.py component-check ','.py ref-controls ','.py model ','.py build-full ','.py native ','browser\" capture\n','.py prepare-replay ','browser\" replay\n','.py replay --output']
  positions=[w.index(n)for n in names];self.assertEqual(positions,sorted(positions))
  self.assertIn('retention-days: 90',w);self.assertIn('fetch-depth: 0',w);self.assertEqual(w.count('if: always()'),2)
  s=(ROOT/'tools/ci-heavy/wasm_live_norm_v1.py').read_text();self.assertNotIn('old.source(',s);self.assertNotIn('old.K=',s)
  self.assertIn('numeric_nodes_added=0',s);self.assertIn("'crispembed_live_norm_finish'in symbols",s)
 def test_archive_retains_more_than_legacy200_members_and_rejects_unsafe(self):
  with tempfile.TemporaryDirectory()as tmp:
   path=pathlib.Path(tmp)/'many.zip'
   with zipfile.ZipFile(path,'w')as z:
    for i in range(300):z.writestr('arrays/f'+str(i)+'.f32',b'1234')
   self.assertEqual(len(D.zip_files(path)),300)
   with self.assertRaises(ValueError):D.zip_files(path,member_limit=299)
   with self.assertRaises(ValueError):D.zip_files(path,byte_limit=1199)
   with zipfile.ZipFile(path,'w')as z:z.writestr('../outside',b'x')
   with self.assertRaises(ValueError):D.zip_files(path)
 def test_python_source_parses(self):
  for p in [ROOT/'tools/ci-heavy/wasm_live_norm_v1.py',ROOT/'tools/wasm-live-norm-v1/live_contract.py',ROOT/'tools/wasm-live-norm-v1/overlay.py']:ast.parse(p.read_text())

if __name__=='__main__':unittest.main()
