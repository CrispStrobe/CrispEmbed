#!/usr/bin/env python3
"""Hosted TEST-only layer3 F32 weightcast risk; no encoder/model forward."""
import argparse,ctypes as C,json,os,shutil,struct,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/wasm-layer3-f32'),str(ROOT/'tools/ci-heavy')]
import protocol as P
import wasm_f32_operators_v1 as old
save,require,digest=old.save,P.require,old.digest
CONTROLS={'positive','wrong_mode','wrong_extents','nonfinite_input','nonfinite_weight','output_preserved','coefficient_output_preserved','recovery'}

def hosted():
 cfg=P.source();old.hosted()
 require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted'and sys.platform=='linux'and os.environ.get('RUNNER_ARCH')=='X64','Genuine hosted Linux x64 required')
 return cfg

def prepare(out):
 cfg=hosted()
 mem=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
 require(mem>=4*2**30 and shutil.disk_usage(out.parent).free>=3*2**30 and (os.cpu_count()or 0)>=2,'Hosted4GiB RAM/3GiB disk/twoCPU required before transfer')
 old.prepare(out)
 current=json.loads((out/'review/prepared.json').read_text())['service']
 require(current['path']=='.github/workflows/wasm-layer3-f32-v1.yml'and current['event']=='workflow_dispatch','Exact cast workflow dispatch required')
 frozen=cfg['causal'];service=old.api('actions/runs/'+str(frozen['run']))
 require((service['head_sha'],service['path'],service['status'],service['conclusion'],service['run_attempt'])==(frozen['source'],'.github/workflows/wasm-q8k-cross-v1.yml','completed','success',1),'Causal service identity differs')
 for kind in ['review','raw']:old.download(frozen['run'],'wasm-q8k-cross-v1-'+kind+'-'+str(frozen['run']),out/('causal-'+kind+'.zip'),frozen[kind])
 review=old.safe_zip(out/'causal-review.zip',16*2**20);raw=old.safe_zip(out/'causal-raw.zip',64*2**20)
 require(P.sha(review['review/cross.json'])==frozen['report_sha256'],'Causal report SHA differs')
 report=json.loads(review['review/cross.json']);require(report['completed']and report['source']==frozen['source']and len(report['anchors'])==18 and report['comparisons']==44 and report['cpp']['all_metrics_agree'],'Reviewed causal completion differs')
 manifest=json.loads(review['review/raw-manifest.json'])
 require(all(n in raw and len(raw[n])==v['bytes']and P.sha(raw[n])==v['sha256']for n,v in manifest.items()),'Causal raw member authentication failed')
 packets={'inputs':{},'targets':{k:{}for k in ['native1','native2','wasm']}};files={}
 def retain(name,b):
  path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
  row=dict(bytes=len(b),sha256=P.sha(b));files[name]=row;return dict(file=str(path.relative_to(out/'browser')),**row)if path.is_relative_to(out/'browser')else dict(file=name,**row)
 weight=raw['browser/q8k-inputs/weight.bin'];require(len(weight)==313344 and P.sha(weight)==cfg['weight_sha256'],'Frozen IQ4_XS coefficient differs')
 packets['weight']=retain('browser/cast-inputs/weight.bin',weight)
 for axis in ['native','candidate']:
  name='browser/q8k-inputs/'+axis+'.f32';require(len(raw[name])==73728 and P.sha(raw[name])==cfg['input_sha256'][axis],'Frozen natural GELU differs')
  packets['inputs'][axis]=retain('browser/cast-inputs/'+axis+'.f32',raw[name])
  for backend in ['native1','native2','wasm']:
   name='raw/'+backend+'/f32_'+axis+'/output.f32';require(len(raw[name])==18432,'Frozen projection extent differs')
   packets['targets'][backend][axis]=retain('browser/cast-targets/'+backend+'-'+axis+'.f32',raw[name])
   expected=json.loads(review['browser/q8k-report.json'])['outputs']['f32_'+axis]if backend=='wasm'else json.loads(review['review/native-cross.json'])[backend]['f32_'+axis]
   require(P.sha(raw[name])==expected['sha256'],'Frozen projection report/member differs')
 # A distinct literal decoder establishes coefficient identity before any cast.
 coefficients=P.decode(weight);require(len(coefficients)==2359296,'Independent coefficient extent differs')
 retain('raw/independent/coefficients.f32',coefficients);packets['coefficient_sha256']=P.sha(coefficients)
 save(out/'browser/cast-packets.json',packets)
 save(out/'review/cast-prepared.json',dict(schema='crispembed-wasm-layer3-f32-prepared-v1',source=old.source_identity()['source'],protocol_sha256=digest(P.PATH),causal_service={k:service[k]for k in ['id','head_sha','path','conclusion','run_attempt']},files=files,packets_sha256=digest(out/'browser/cast-packets.json'),coefficient_sha256=P.sha(coefficients),derived_coefficient_source='Independent literal IQ4_XS decoding, not an altered model',qualification=False))


def verify(out,binaries=False):
 cfg=hosted();old.prepared(out);proof=json.loads((out/'review/cast-prepared.json').read_text())
 require(proof['source']==old.source_identity()['source']and proof['protocol_sha256']==digest(P.PATH)and proof['packets_sha256']==digest(out/'browser/cast-packets.json'),'Preparation changed')
 for name,row in proof['files'].items():
  p=out/name;require(p.is_file()and not p.is_symlink()and p.stat().st_size==row['bytes']and digest(p)==row['sha256'],'Prepared member changed: '+name)
 for kind in ['review','raw']:require(digest(out/('causal-'+kind+'.zip'))==cfg['causal'][kind]['sha256'],'Causal whole archive changed')
 if binaries:old.verified_build(out)
 return json.loads((out/'browser/cast-packets.json').read_text())


def build(out):
 verify(out);source=out/'source'
 old.command(['git','worktree','add','--detach',source,old.BASE],out,'source-worktree')
 old.command(['git','-C',source,'submodule','update','--init','--depth','1','ggml'],out,'source-submodule')
 require(old.git(source,'rev-parse','HEAD')==old.BASE and old.git(source/'ggml','rev-parse','HEAD')==old.GGML and not old.git(source,'status','--porcelain')and not old.git(source/'ggml','status','--porcelain'),'Frozen blueprint/backend changed')
 for label in ['native','wasm']:
  args=[str(x).replace(str(ROOT/'tests/wasm-f32-operators'),str(ROOT/'tests/wasm-layer3-f32'))for x in old.configuration(out,source,label)]
  old.command(args,out,label+'-configure');shutil.copy2(out/(label+'-build/compile_commands.json'),out/'review'/('compile-'+label+'.json'))
  old.command(['cmake','--build',out/(label+'-build'),'--parallel','2'],out,label+'-build',1800)
 save(out/'review/compile-profile.json',old.compile_profile(out))
 destination=out/'browser/module';destination.mkdir()
 for ext in ['js','wasm']:shutil.copy2(out/('wasm-build/wasm-layer3-f32.'+ext),destination)
 save(out/'review/build-artifacts.json',{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=digest(p))for folder in [out/'native-build',destination]for p in folder.rglob('*')if p.is_file()and('.so'in p.name or p.suffix in ['.js','.wasm'])})
 old.ref_controls(out)


def native(out,candidate=False):
 p=verify(out,True)
 if candidate:anchors(out)
 lib=C.CDLL(str(out/'native-build/libwasm-layer3-f32.so'))
 declared=json.loads((out/'review/build-artifacts.json').read_text())
 mappings={line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if '/'in line and 'libggml'in line.split()[-1]}
 require(mappings,'Actual GGML mapping required')
 for name in mappings:
  path=Path(name).resolve();require(path.is_relative_to((out/'native-build').resolve()),'Unrelated GGML mapping')
  row=declared.get(str(path.relative_to(out.resolve())));require(row and digest(path)==row['sha256'],'Mapped provider changed')
 save(out/'review/native-mappings.json',dict(providers=sorted(mappings),all_exact_frozen_build=True))
 f=lib.crispembed_layer3_f32;f.argtypes=[C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_int];f.restype=C.c_int
 report=json.loads((out/'review/native-cast.json').read_text())if candidate else {};weight=C.create_string_buffer((out/'browser'/p['weight']['file']).read_bytes())
 for threads in [1,2]:
  rows=report.setdefault('native'+str(threads),{})
  for axis in ['native','candidate']:
   mode=int(candidate);key=('cast_'if candidate else'quant_')+axis;inputs=(out/'browser'/p['inputs'][axis]['file']).read_bytes();x=C.create_string_buffer(inputs);y=(C.c_float*4608)();d=(C.c_float*589824)()
   def call(observe=False):return f(mode,weight,313344,x,18432,y,4608,d if observe else None,589824 if observe else 0,threads)
   require(call()==4608,'Ordinary native graph failed');value=C.string_at(y,18432)
   require(call(candidate)==4608 and C.string_at(y,18432)==value,'Observed native output changed')
   if candidate:
    derived=C.string_at(d,2359296);require(P.sha(derived)==p['coefficient_sha256'],'Native cast coefficient differs from independent decoder')
    folder=out/'raw'/('native'+str(threads));folder.mkdir(parents=True,exist_ok=True);(folder/'coefficients.f32').write_bytes(derived)
   require(call()==4608 and C.string_at(y,18432)==value,'Native repeat changed')
   require(C.string_at(x,len(inputs))==inputs and P.sha(C.string_at(weight,313344))==p['weight']['sha256'],'Native mutated inputs')
   if not candidate:require(P.sha(value)==p['targets']['native'+str(threads)][axis]['sha256'],'Old native quantized anchor changed')
   file='raw/native'+str(threads)+'/'+key+'.f32';(out/file).parent.mkdir(parents=True,exist_ok=True);(out/file).write_bytes(value)
   rows[key]=dict(file=file,bytes=18432,sha256=P.sha(value),input_sha256=p['inputs'][axis]['sha256'],coefficient_sha256=p['coefficient_sha256']if candidate else None,ordinary_observed_bit_exact=True,repeat_bit_exact=True,inputs_unchanged=True)
   save(out/'review/native-cast.json',report)
  if candidate:
   zeros=(C.c_float*18432)();y=(C.c_float*4608)(*([17.]*4608));d=(C.c_float*589824)(*([17.]*589824));ys=C.string_at(y,18432);ds=P.sha(C.string_at(d,2359296))
   def reject(mode=1,wb=313344,nx=18432,nd=589824):
    require(f(mode,weight,wb,zeros,nx,y,4608,d,nd,threads)<0 and C.string_at(y,18432)==ys and P.sha(C.string_at(d,2359296))==ds,'Failed native call changed output/coefficients')
   reject(mode=3);reject(wb=313343);reject(nx=18431);reject(nd=589823);zeros[0]=float('nan');reject();zeros[0]=0
   saved=weight.raw[:2];weight[0]=b'\0';weight[1]=b'\x7c';reject();weight[0]=saved[:1];weight[1]=saved[1:2]
   require(f(1,weight,313344,zeros,18432,y,4608,d,589824,threads)==4608 and all(v==0 for v in y)and P.sha(C.string_at(d,2359296))==p['coefficient_sha256'],'Native recovery failed')
 if candidate:save(out/'review/native-controls.json',dict(threads=[1,2],**{k:True for k in CONTROLS}))


def anchors(out):
 p=verify(out,True);n=json.loads((out/'review/native-cast.json').read_text());w=json.loads((out/'browser/cast-report.json').read_text())
 require(w['baseline_completed']and w['isolated']and w['pthreads']and not w['page_errors'],'Browser baseline incomplete')
 checked={}
 for backend,rows in dict(n,wasm=w['outputs']).items():
  for axis in ['native','candidate']:
   row=rows['quant_'+axis];target=p['targets'][backend][axis]
   require((out/row['file']).stat().st_size==18432 and digest(out/row['file'])==row['sha256']==target['sha256']and row['input_sha256']==p['inputs'][axis]['sha256']and row['repeat_bit_exact'],'Exact original quantized baseline changed')
   checked[backend+'/'+axis]=row['sha256']
 require(len(checked)==6,'Six byte anchors required')
 save(out/'review/cast-anchors.json',dict(passed=True,anchors=checked,qualification=False))


def compare(out):
 p=verify(out,True);anchors(out);native=json.loads((out/'review/native-cast.json').read_text());browser=json.loads((out/'browser/cast-report.json').read_text())
 require(browser['completed']and not browser['page_errors']and browser['isolated']and browser['pthreads'],'Browser candidate incomplete')
 require(set(browser['controls'])==CONTROLS and all(browser['controls'].values()),'Browser controls incomplete')
 controls=json.loads((out/'review/native-controls.json').read_text());require(controls.pop('threads')==[1,2]and set(controls)==CONTROLS and all(controls.values()),'Native controls incomplete')
 outputs=dict(native,wasm=browser['outputs']);keys={'quant_native','quant_candidate','cast_native','cast_candidate'}
 for backend,rows in outputs.items():
  require(set(rows)==keys,'Four-arm coverage differs')
  for key,row in rows.items():
   require(digest(out/row['file'])==row['sha256']and(out/row['file']).stat().st_size==18432 and all(row[k]for k in ['ordinary_observed_bit_exact','repeat_bit_exact','inputs_unchanged']),'Output/neutrality changed')
   if key.startswith('cast_'):require(row['coefficient_sha256']==p['coefficient_sha256'],'Derived coefficient changed')
  require(digest(out/('raw/'+backend+'/coefficients.f32'))==p['coefficient_sha256'],'Actual derived coefficient payload changed')
 import numpy as np
 pairs=[];independent={}
 def pair(name,a,b):
  a=out/a;b=out/b;shape=[12,384]
  pairs.append(dict(name=name,original=dict(path=str(a),shape=shape,sha256=digest(a)),native=dict(path=str(b),shape=shape,sha256=digest(b))))
  independent[name]=old.metrics(np.memmap(a,dtype='<f4',mode='r',shape=tuple(shape)),np.memmap(b,dtype='<f4',mode='r',shape=tuple(shape)))
 for axis in ['native','candidate']:
  pair('backend.'+axis,native['native1']['cast_'+axis]['file'],browser['outputs']['cast_'+axis]['file'])
  pair('threads.'+axis,native['native1']['cast_'+axis]['file'],native['native2']['cast_'+axis]['file'])
 for backend,rows in outputs.items():
  for axis in ['native','candidate']:
   candidate=rows['cast_'+axis]['file'];pair(backend+'.same_input.'+axis,rows['quant_'+axis]['file'],candidate)
   for target in ['native1','native2']:pair(backend+'.old_'+target+'.'+axis,'browser/'+p['targets'][target]['native']['file'],candidate)
  pair(backend+'.natural_input_sensitivity',rows['cast_native']['file'],rows['cast_candidate']['file'])
 require(len(pairs)==25,'Exact25 comparisons required')
 adapters=old.write_pair(out/'control.gguf',out/'replay.gguf',pairs)
 process=subprocess.run([out/'native-build/wasm-f32-diff',out/'control.gguf',out/'replay.gguf',out/'review/ref.json'],capture_output=True)
 (out/'ref.log').write_bytes(process.stdout+process.stderr);cpp=old.collect(process.returncode,out/'review/ref.json',independent)
 coeff=np.fromfile(out/'raw/independent/coefficients.f32',dtype='<f4').reshape(384,1536).astype(np.float64);f64={}
 for axis in ['native','candidate']:
  x=np.fromfile(out/'browser'/p['inputs'][axis]['file'],dtype='<f4').reshape(12,1536).astype(np.float64);reference=x@coeff.T
  f64[axis]={backend:old.metrics(reference,np.memmap(out/rows['cast_'+axis]['file'],dtype='<f4',mode='r',shape=(12,384)))for backend,rows in outputs.items()}
 save(out/'review/cast.json',dict(schema='crispembed-wasm-layer3-f32-v1',completed=True,source=old.source_identity()['source'],protocol_sha256=digest(P.PATH),anchors=6,comparisons=25,coefficient_sha256=p['coefficient_sha256'],coefficient_bytes=2359296,all_derived_coefficients_byte_exact=True,cpp=cpp,independent=independent,diagnostic_f64_dot=f64,adapters=adapters,operator_supported=cpp['status']=='passed',qualification=False,production_change=False,whole_model_negative_preserved=True,original_rerun=False,limits='One frozen projection/two natural GELU inputs. F64 dot is a diagnostic, not a new golden reference. All old native targets and strict gates remain unchanged; no complete encoder forward.'))
 save(out/'review/raw-manifest.json',{str(v.relative_to(out)):dict(bytes=v.stat().st_size,sha256=digest(v))for folder in [out/'raw',out/'browser/cast-inputs',out/'browser/cast-targets',out/'browser/module']for v in folder.rglob('*')if v.is_file()})
 require(cpp['status']=='passed','F32 cast risk replay rejected unchanged old-target/crossbackend gates; all outcomes retained')


def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','verify','build','native','anchors','native-candidate','compare']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 try:
  if a.phase=='verify':verify(a.output,True)
  elif a.phase=='native-candidate':native(a.output,True)
  else:globals()[a.phase](a.output)
 except Exception as e:save(a.output/'review'/('cast-rejection-'+a.phase+'.json'),dict(phase=a.phase,error=str(e),qualification=False));raise
if __name__=='__main__':main()
