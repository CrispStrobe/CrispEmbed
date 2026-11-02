#!/usr/bin/env python3
"""v4 Ref adapter: bit-for-bit equal to the frozen adapter on <=512 shapes,
exact allowlist incl. live [12,1536], full-row metrics. No build/model."""
import copy,importlib.util,json,math,pathlib,random,struct,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/wasm-live-norm-v1'),str(ROOT/'tools/moedern-native-v2')]
import live_contract as K
import diff_adapter as OLD

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
A=load('diff_adapter_v4','tools/wasm-live-norm-v4/diff_adapter_v4.py')
FF3='ff3b8a0171d2cdb5cfcdd65701fadaa49a99d265'

def floats(n,seed):
 r=random.Random(seed);return [struct.unpack('<f',struct.pack('<f',r.uniform(-8,8)))[0]for _ in range(n)]

def files(folder,name,shape,a,b):
 pair=dict(name=name)
 for side,v in [('original',a),('native',b)]:
  p=folder/(name+'-'+side+'.f32');p.write_bytes(struct.pack('<%df'%len(v),*v))
  pair[side]=dict(path=str(p),shape=list(shape),sha256=K.sha(p))
 return pair

def independent(a,b,shape):
 """Same arithmetic as wasm_native_order_norm_v1.independent (full rows of width shape[-1])."""
 d=[y-x for x,y in zip(a,b)];width=shape[-1];cos=[]
 for at in range(0,len(a),width):
  x,y=a[at:at+width],b[at:at+width];aa=sum(v*v for v in x);bb=sum(v*v for v in y)
  cos.append(sum(v*w for v,w in zip(x,y))/math.sqrt(aa*bb)if aa>1e-18 and bb>1e-18 else 1. if aa<=1e-18 and bb<=1e-18 else 0.)
 row=dict(shape=list(shape),elements=len(a),max_absolute=max(abs(x)for x in d),mean_absolute=sum(abs(x)for x in d)/len(d),
  mean_error=sum(d)/len(d),rms=math.sqrt(sum(x*x for x in d)/len(d)),min_row_cosine=min(cos),
  original_l2=math.sqrt(sum(x*x for x in a)),native_l2=math.sqrt(sum(x*x for x in b)))
 for label,v in [('original',a),('native',b)]:row.update({label+'_min':min(v),label+'_max':max(v),label+'_abs_min':min(abs(x)for x in v),label+'_abs_max':max(abs(x)for x in v)})
 row['passed']=row['max_absolute']<=.002 and row['min_row_cosine']>=.99999;return row

def report(measured):
 rows=[];bad=False
 for name,m in measured.items():
  r={k:m[k]for k in OLD.FIELDS};gate=m['max_absolute']<=.002 and m['min_row_cosine']>=.99999;bad|=not gate
  rows.append(dict(name=name,ggml_shape=list(reversed(m['shape'])),row_axis=0,found=True,elements=m['elements'],passed=gate,**r))
 return dict(schema='crispembed-moedern-native-ref-v2',status='numeric_rejection'if bad else'passed',max_absolute_budget=.002,minimum_row_cosine=.99999,stages=rows),1 if bad else 0

class Frozen(unittest.TestCase):
 def test_frozen_adapter_unchanged(self):
  self.assertEqual(K.git('rev-parse',FF3+':tools/moedern-native-v2/diff_adapter.py'),K.git('rev-parse','HEAD:tools/moedern-native-v2/diff_adapter.py'))
  self.assertEqual(K.sha(A.FROZEN),A.FROZEN_SHA256)
  self.assertEqual((A.ABSOLUTE,A.COSINE,A.METRIC_RELATIVE,A.FIELDS),(OLD.ABSOLUTE,OLD.COSINE,OLD.METRIC_RELATIVE,OLD.FIELDS))
 def test_only_shape_differs(self):
  self.assertIsNot(A.write_gguf,OLD.write_gguf)  # private copy, not the frozen module object
  for name in ['source_bytes','write_gguf','read_gguf','write_pair','collect','sha','_string']:
   self.assertEqual(getattr(A._M,name).__code__.co_code,getattr(OLD,name).__code__.co_code,name)
  self.assertIs(A._M.shape,A.shape)

class BitForBit(unittest.TestCase):
 def run_both(self,shape,perturb):
  outs=[]
  for mod in [OLD,A]:
   with tempfile.TemporaryDirectory()as tmp:
    folder=pathlib.Path(tmp);n=math.prod(shape);a=floats(n,1);b=list(a)
    for i in perturb:b[i]=struct.unpack('<f',struct.pack('<f',b[i]+.01))[0]
    pairs=[files(folder,'p%d'%k,shape,a,b if k else a)for k in range(3)]
    for p in pairs:  # identical paths across modules except the temp root
     for side in ['original','native']:p[side]['path']=str(folder/pathlib.Path(p[side]['path']).name)
    adapters=mod.write_pair(folder/'left.gguf',folder/'right.gguf',pairs)
    raw=((folder/'left.gguf').read_bytes(),(folder/'right.gguf').read_bytes())
    measured={p['name']:independent(a,b if k else a,shape)for k,p in enumerate(pairs)}
    rep,code=report(measured);(folder/'ref.json').write_text(json.dumps(rep))
    m=copy.deepcopy(measured);summary=mod.collect(code,folder/'ref.json',m)
    outs.append((adapters,raw,mod.read_gguf(folder/'left.gguf'),summary,m))
  return outs
 def test_all_le512_allowlisted_shapes_bit_for_bit(self):
  shapes=[s for s in A.ALLOWED_SHAPES if max(s)<=512];self.assertEqual(sorted(shapes),[(2,8),(12,384)])
  for shape in shapes:
   for perturb in [[],[3]]:  # passing and numeric-rejection records
    old,new=self.run_both(shape,perturb)
    self.assertEqual(old[1],new[1],shape)        # GGUF bytes
    self.assertEqual(old[0],new[0],shape);self.assertEqual(old[2],new[2],shape)
    self.assertEqual(old[3],new[3],shape);self.assertEqual(json.dumps(old[4],sort_keys=True),json.dumps(new[4],sort_keys=True))
    self.assertEqual(new[3]['exit_code'],1 if perturb else 0)

class Live1536(unittest.TestCase):
 def test_old_rejects_v4_admits_live_fc1_rows(self):
  with tempfile.TemporaryDirectory()as tmp:
   folder=pathlib.Path(tmp);a=floats(12*1536,2);pair=files(folder,'layer3_fc1',(12,1536),a,a)
   with self.assertRaisesRegex(ValueError,'Bounded exact NumPy shape required'):OLD.write_pair(folder/'o1',folder/'o2',[pair])
   out=A.write_pair(folder/'l',folder/'r',[pair])
   self.assertEqual(out['original']['tensors']['layer3_fc1']['shape'],[12,1536])
   self.assertEqual(A.read_gguf(folder/'l')['layer3_fc1']['bytes'],12*1536*4)
 def test_full_row_metric_not_chunked(self):
  a=floats(12*1536,3);b=list(a)
  for i in range(1024,1536):b[i]=-b[i]  # second/third 512-chunks of row 0 flipped
  full=independent(a,b,[12,1536])['min_row_cosine']
  chunked=min(independent(a[k:k+512],b[k:k+512],[1,512])['min_row_cosine']for k in range(0,1536,512))
  self.assertLess(chunked,-0.99);self.assertGreater(full,0.0)  # 512 chunks would report a different statistic
  self.assertAlmostEqual(full,(sum(x*x for x in a[:1024])-sum(x*x for x in a[1024:1536]))/sum(x*x for x in a[:1536]),places=6)

class Allowlist(unittest.TestCase):
 def test_rejects_unexpected_shapes(self):
  for bad in [(12,385),(16,384),(12,1537),(1536,12),(384,12),(12,1536,1),(1,12,1536),(12,),(384,),(4,4),(12,512),[12.0,384],[True,8],'12x384',None,(13,384)]:
   with self.assertRaises(ValueError,msg=repr(bad)):A.shape(bad)
  for good in [(12,384),[12,1536],(2,8)]:self.assertEqual(A.shape(good),tuple(good))
 def test_readback_rejects_unallowlisted_file(self):
  with tempfile.TemporaryDirectory()as tmp:
   folder=pathlib.Path(tmp);p=folder/'x.f32';p.write_bytes(struct.pack('<16f',*range(16)))
   OLD.write_gguf(folder/'x.gguf',[dict(name='control',path=str(p),shape=[4,4],sha256=K.sha(p))])
   with self.assertRaisesRegex(ValueError,'Allowlisted exact v4 shape'):A.read_gguf(folder/'x.gguf')
 def test_pair_shape_mismatch_still_rejected(self):
  with tempfile.TemporaryDirectory()as tmp:
   folder=pathlib.Path(tmp);pair=files(folder,'x',(12,384),floats(4608,4),floats(4608,5));pair['native']['shape']=[12,1536]
   with self.assertRaises(ValueError):A.write_pair(folder/'l',folder/'r',[pair])

class Caller(unittest.TestCase):
 def test_profile_and_routing(self):
  D=load('v4','tools/ci-heavy/wasm_live_norm_v4.py');p=K.read(D.PROFILE);D.validate(p)
  for path,digest in p['owned_sha256'].items():self.assertEqual(K.sha(ROOT/path),digest,path)
  for key,value in [('scope_unchanged_from_v1',False),('runtime_change',True),('observer','v1'),('workflow','.github/workflows/wasm-live-norm-operands-v3.yml')]:
   q=copy.deepcopy(p);q[key]=value
   with self.assertRaises(ValueError):D.validate(q)
  q=copy.deepcopy(p);q['ref_adapter']['allowed_shapes'].append([12,4096])
  with self.assertRaises(ValueError):D.validate(q)
  q=copy.deepcopy(p);q['ref_adapter']['budgets']['max_absolute']=.003
  with self.assertRaises(ValueError):D.validate(q)
  self.assertIs(D.V1.frozen_actual_ref,D.frozen_actual_ref)
  for m in [D.V1,D.V2,D.V3]:self.assertIs(m.source,D.source)
  self.assertEqual(D.V1.__file__,str(pathlib.Path(D.__file__).resolve()))
  self.assertIs(D.PHASES['replay'],D.V1.replay);self.assertIs(D.PHASES['observer-controls'],D.V3.observer_controls)
 def test_v4_ref_body_is_frozen_body_with_adapter_swapped(self):
  import ast,inspect
  D=load('v4b','tools/ci-heavy/wasm_live_norm_v4.py')
  fbody=ast.parse(inspect.getsource(D.OLD.actual_ref)).body[0].body
  self.assertEqual([ast.unparse(s)for s in fbody[:2]],["sys.path.insert(0, str(ROOT / 'tools/moedern-native-v2'))",'import diff_adapter'])
  frozen='\n'.join(ast.unparse(s)for s in fbody[2:])  # skip the local import line
  text='\n'.join(ast.unparse(s)for s in ast.parse(inspect.getsource(D.frozen_actual_ref)).body[0].body[1:])  # skip the docstring
  for a,b in [('OLD.independent(','independent('),('OLD.f32_bytes(','f32_bytes('),('OLD.command(','command('),('A.write_pair(','diff_adapter.write_pair('),('A.collect(','diff_adapter.collect(')]:
   self.assertIn(a,text);text=text.replace(a,b)
  self.assertNotIn('OLD.',text);self.assertNotIn(' A.',text)
  self.assertEqual(text,frozen)

if __name__=='__main__':unittest.main()
