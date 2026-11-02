#!/usr/bin/env python3
"""v2 component-inventory admission: regression for run 40867038915; no build/model."""
import copy,json,pathlib,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/wasm-live-norm-v2'),str(ROOT/'tools/wasm-live-norm-v1')]
import inventory as I
import live_contract as K
FIX=ROOT/'tests/wasm-live-norm-v2'
FROZEN_ROOT='/home/runner/work/CrispEmbed/CrispEmbed'
NEW_ROOT='/w/src'

def frozen(kind):return json.loads((FIX/f'frozen-40852835039-{kind}-compile.json').read_text())

def synthetic(kind,build='/w/out/component-'):
 """Actual live CMake layout (as in run 41416683981): frozen controls under frozen-controls/
 with top-level-relative `output`; live target at the build root."""
 build=build+kind;rows=[]
 for r in frozen(kind):
  r=copy.deepcopy(r)
  for k in ['command','file','output']:r[k]=r[k].replace(FROZEN_ROOT,NEW_ROOT)
  sub=r['directory'].split('component-'+kind,1)[1].lstrip('/')
  r['directory']=build+'/frozen-controls'+('/'+sub if sub else '')
  minus_o=r['output'];r['output']='frozen-controls/'+(sub+'/' if sub else '')+minus_o
  rows.append(r)
  if minus_o.startswith('CMakeFiles/norm-candidate-controls.dir/'):
   live=copy.deepcopy(r);helper=r['file'].endswith('wasm_native_order_norm.cpp')
   src=r['file']if helper else NEW_ROOT+'/tests/wasm-live-norm/replay.cpp'
   out=minus_o.replace('norm-candidate-controls.dir','live-norm-replay.dir')
   if not helper:out=out.replace('controls.cpp.o','replay.cpp.o')
   live['command']=r['command'].replace(minus_o,out).replace(r['file'],src)
   live['file']=src;live['output']=out;live['directory']=build
   rows.append(live)
 return rows,pathlib.Path(build)

class Inventory(unittest.TestCase):
 def check(self,kind,rows,build):return I.check(kind,rows,frozen(kind),NEW_ROOT,build,FROZEN_ROOT)
 def test_v1_single_entry_assumption_is_the_40867038915_failure(self):
  for kind in ['native','wasm']:
   rows,_=synthetic(kind);own=[r for r in rows if pathlib.Path(r['file']).name=='wasm_native_order_norm.cpp']
   self.assertEqual(len(own),2)  # v1 required exactly one -> 'Actual component rounding flags differ'
   self.assertTrue(all('-ffp-contract=off'in r['command']and'-ffast-math'not in r['command']for r in own))
 def test_declared_inventory_admitted_both_backends(self):
  for kind in ['native','wasm']:
   rows,build=synthetic(kind);report=self.check(kind,rows,build)
   self.assertEqual(report['declared_entries'],5);self.assertEqual(report['helper_entries'],2)
   self.assertEqual(report['ggml_backend_entries'],len(frozen(kind))-3);self.assertFalse(report['relaxed_flags'])
 def mutate(self,kind,fn,message):
  rows,build=synthetic(kind);fn(rows)
  with self.assertRaisesRegex(ValueError,message):self.check(kind,rows,build)
 def helper(self,rows,target='live-norm-replay'):
  return next(r for r in rows if r['file'].endswith('wasm_native_order_norm.cpp')and target in r['output'])
 def test_flag_negatives(self):
  for kind in ['native','wasm']:
   for bad in ['-ffast-math','-Ofast','-funsafe-math-optimizations','-fassociative-math','-ffp-model=fast','-msimd128']:
    self.mutate(kind,lambda rows:self.helper(rows).__setitem__('command',self.helper(rows)['command'].replace(' -c ',' '+bad+' -c ')),'rounding flags differ')
   self.mutate(kind,lambda rows:self.helper(rows).__setitem__('command',self.helper(rows)['command'].replace('-ffp-contract=off','')),'rounding flags differ')
   self.mutate(kind,lambda rows:self.helper(rows).__setitem__('command',self.helper(rows)['command'].replace(' -c ',' -ffp-contract=fast -c ')),'rounding flags differ')
   def ref(rows):
    r=next(r for r in rows if 'norm-candidate-ref' in r['output']);r['command']=r['command'].replace('-ffp-contract=off','-ffp-contract=on')
   self.mutate(kind,ref,'rounding flags differ')
 def test_profile_negatives(self):
  self.mutate('native',lambda rows:self.helper(rows).__setitem__('command',self.helper(rows)['command'].replace('-O3','-O2')),'profile differs')
  self.mutate('native',lambda rows:self.helper(rows).__setitem__('command',self.helper(rows)['command'].replace(' -c ',' -mavx2 -c ')),'profile differs')
  self.mutate('wasm',lambda rows:self.helper(rows).__setitem__('command',self.helper(rows)['command'].replace('-pthread','')),'profile differs')
  def ggml(rows):
   r=next(r for r in rows if r['file'].endswith('ggml-cpu.c'));r['command']=r['command'].replace('-O3','-O1')
  self.mutate('native',ggml,'GGML backend profile differs')
  def simd(rows):
   r=next(r for r in rows if r['file'].endswith('ggml-cpu.c'));r['command']=r['command'].replace('-msimd128','')
  self.mutate('wasm',simd,'GGML backend profile differs')
 def test_inventory_count_negatives(self):
  for kind in ['native','wasm']:
   self.mutate(kind,lambda rows:rows.append(copy.deepcopy(self.helper(rows))),'Duplicate declared')
   self.mutate(kind,lambda rows:rows.remove(self.helper(rows)),'inventory differs')
   def extra(rows):
    r=copy.deepcopy(self.helper(rows));r['output']=r['output'].replace('live-norm-replay','other');r['command']=r['command'].replace('live-norm-replay','other');rows.append(r)
   self.mutate(kind,extra,'inventory differs')
   def source(rows):
    r=self.helper(rows);r['file']=r['file'].replace('src/core','src/x');r['command']=r['command'].replace('src/core','src/x')
   self.mutate(kind,source,'inventory differs')
   self.mutate(kind,lambda rows:rows.pop(next(i for i,r in enumerate(rows)if '/ggml/' in r['file'])),'GGML backend inventory differs')
   def outside(rows):
    r=copy.deepcopy(self.helper(rows));r['file']='/elsewhere/x.cpp';rows.append(r)
   self.mutate(kind,outside,'Unexpected compile entries')
   self.mutate(kind,lambda rows:self.helper(rows).__setitem__('directory','/w/out/elsewhere'),'directory differs')
 def test_actual_41416683981_native_inventory_admitted(self):
  # Full inventory saved by the first v2 run before its parser rejection.
  rows=json.loads((FIX/'actual-41416683981-native-compile.json').read_text())
  report=I.check('native',rows,frozen('native'),FROZEN_ROOT,'/home/runner/work/_temp/live-norm/component-native',FROZEN_ROOT)
  self.assertEqual(report['declared_entries'],5);self.assertEqual(report['total_entries'],30)
  r=next(r for r in rows if 'live-norm-replay.dir' in r['output']and r['file'].endswith('wasm_native_order_norm.cpp'))
  r['command']=r['command'].replace(' -c ',' -ffast-math -c ')
  with self.assertRaisesRegex(ValueError,'rounding flags differ'):I.check('native',rows,frozen('native'),FROZEN_ROOT,'/home/runner/work/_temp/live-norm/component-native',FROZEN_ROOT)
 def test_output_path_negative(self):
  def bad(rows):
   r=next(r for r in rows if 'norm-candidate-controls.dir' in r['output']);r['output']=r['output'].replace('frozen-controls/','')
  self.mutate('native',bad,'output path differs')
 def test_compiler_backend(self):
  self.mutate('wasm',lambda rows:self.helper(rows).__setitem__('command','/usr/bin/c++'+self.helper(rows)['command'].split('em++',1)[1]),'Compiler differs')
  self.mutate('native',lambda rows:self.helper(rows).__setitem__('command','/x/em++'+self.helper(rows)['command'].split('c++',1)[1]),'Compiler differs')

class Closure(unittest.TestCase):
 def test_v1_and_frozen_owner_bytes_unchanged(self):
  p=K.read(K.PROFILE)
  for path,digest in p['owned_sha256'].items():self.assertEqual(K.sha(ROOT/path),digest,path)
  self.assertEqual(K.git('rev-parse','ff3b8a0171d2cdb5cfcdd65701fadaa49a99d265:tools/ci-heavy/wasm_live_norm_v1.py'),K.git('rev-parse','HEAD:tools/ci-heavy/wasm_live_norm_v1.py'))
  self.assertEqual(K.git('rev-parse','ff3b8a0171d2cdb5cfcdd65701fadaa49a99d265:docs/wasm-live-norm-operands-v1.json'),K.git('rev-parse','HEAD:docs/wasm-live-norm-operands-v1.json'))
 def test_v2_profile_scope(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('v2',ROOT/'tools/ci-heavy/wasm_live_norm_v2.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
  p=K.read(D.PROFILE);D.validate(p)
  for path,digest in p['owned_sha256'].items():self.assertEqual(K.sha(ROOT/path),digest,path)
  for key,value in [('scope_unchanged_from_v1',False),('runtime_change',True),('forbidden_flags',['-ffast-math']),('workflow','.github/workflows/wasm-live-norm-operands-v1.yml')]:
   q=copy.deepcopy(p);q[key]=value
   with self.assertRaises(ValueError):D.validate(q)
  for kind in ['native','wasm']:
   self.assertEqual(K.sha(FIX/f'frozen-40852835039-{kind}-compile.json'),p['frozen_component_inventory']['sha256'][kind])
  # v2 routes every v1 phase through its own guard and worker path.
  self.assertIs(D.V1.source,D.source);self.assertEqual(D.V1.__file__,str(pathlib.Path(D.__file__).resolve()))
  self.assertIs(D.PHASES['replay'],D.V1.replay);self.assertIs(D.PHASES['component-check'],D.V1.component_check)

if __name__=='__main__':unittest.main()
