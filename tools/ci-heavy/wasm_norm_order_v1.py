#!/usr/bin/env python3
"""Hosted test-only native norm order then computed-activation projection risk."""
import argparse,ctypes as C,json,os,shlex,shutil,struct,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/wasm-norm-order'),str(ROOT/'tools/ci-heavy')]
import protocol as P
import wasm_f32_operators_v1 as old
save,require,digest=old.save,P.require,old.digest

def hosted():
 cfg=P.source();old.hosted();require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted'and os.environ.get('RUNNER_ARCH')=='X64'and sys.platform=='linux','Genuine Linux x64 hosted risk study required');return cfg

def prepare(out):
 cfg=hosted();mem=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
 require(mem>=4*2**30 and shutil.disk_usage(out.parent).free>=3*2**30,'4GiB RAM/3GiB disk before payloads')
 old.prepare(out);proof=json.loads((out/'review/prepared.json').read_text());require(proof['epsilon']['value']==9.999999960041972e-13,'Exact F32 model epsilon required');require(proof['service']['path']=='.github/workflows/wasm-native-order-norm-risk-v1.yml','Exact risk workflow required')
 pin=cfg['operator'];service=old.api('actions/runs/'+str(pin['run']))
 require((service['head_sha'],service['path'],service['status'],service['conclusion'],service['run_attempt'])==(pin['source'],'.github/workflows/wasm-f32-operators-v1.yml','completed','success',1),'Reviewed operator service differs')
 for kind in ['review','raw']:old.download(pin['run'],'wasm-f32-operators-v1-'+kind+'-'+str(pin['run']),out/('operator-'+kind+'.zip'),pin[kind])
 review=old.safe_zip(out/'operator-review.zip',16*2**20);raw=old.safe_zip(out/'operator-raw.zip',128*2**20)
 require(P.sha(review['review/operators.json'])==pin['report_sha256'],'Exact operator report SHA differs');report=json.loads(review['review/operators.json'])
 require(report['completed']and report['own_anchors']['exact_own_input_output_anchors']==64 and report['cpp']['stages']==48 and report['all_boundaries_within_frozen_budgets'],'Reviewed64/48 positive required')
 manifest=json.loads(review['review/raw-manifest.json']);binary=json.loads(review['review/build-artifacts.json']);expected=dict(manifest,**binary)
 for name,side in [('control.gguf','original'),('replay.gguf','native')]:expected[name]={k:report['adapters'][side][k]for k in ['bytes','sha256']}
 require(set(raw)==set(expected),'Exact operator raw inventory required')
 for name,row in expected.items():require(len(raw[name])==row['bytes']and P.sha(raw[name])==row['sha256'],'Operator raw member differs: '+name)
 original_raw=old.safe_zip(out/'old-raw.zip',96*2**20);original_review=old.safe_zip(out/'old-review.zip',16*2**20);trace=json.loads(original_review['review/layer3.json'])
 files={};packets={};source_packets=json.loads((out/'browser/packets.json').read_text())
 def retain(name,data):
  path=out/'browser'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);row=dict(file=name,bytes=len(data),sha256=P.sha(data));files['browser/'+name]={k:row[k]for k in ['bytes','sha256']};return row
 weight=original_raw['native/1/capture/'+P.WEIGHT+'.weight.bin'];require(len(weight)==331776 and P.sha(weight)==cfg['weight_sha256']and trace['weights'][P.WEIGHT]['type']==20,'Exact stored IQ4_NL FC1 weight required')
 weight_row=retain('norm-inputs/weight.bin',weight)
 def stage(axis,name):
  member=old.STAGES['native'if axis=='native'else'wasm1'];row=old.layer_rows(original_raw[member])['layer3_'+name];at=row['data_offset'];data=original_raw[member][at:at+4*__import__('math').prod(row['shape'])];require(P.sha(data)==row['sha256'],'Original stage payload differs');return data
 for axis,source_axis in [('native','native'),('candidate','wasm1')]:
  for label,op in [('attention','attention_norm'),('ffn','ffn_norm')]:
   item=source_packets[source_axis+'/'+op];inputs=item['inputs']['a'];x=(out/'browser'/inputs['file']).read_bytes();p=dict(kind=0,input=retain('norm-inputs/'+label+'-'+axis+'.f32',x),anchors={})
   for backend in ['native1','native2','wasm']:
    name='raw/'+backend+'/'+source_axis+'/'+op+'.f32';p['anchors'][backend]={'norm':retain('norm-targets/'+backend+'-'+label+'-'+axis+'.f32',raw[name])}
   packets[label+'_'+axis]=p
  p=dict(kind=1,input=packets['attention_'+axis]['input'],weight=weight_row,anchors={backend:dict(packets['attention_'+axis]['anchors'][backend])for backend in ['native1','native2','wasm']})
  for label,op in [('gamma','attention_mul'),('beta','attention_bias')]:
   row=source_packets[source_axis+'/'+op]['inputs']['b'];p[label]=retain('norm-inputs/'+label+'.f32',(out/'browser'/row['file']).read_bytes())if axis=='native'else packets['chain_native'][label]
  for backend in (['native1','native2']if axis=='native'else['wasm']):
   for label,target in [('scaled','attention_scaled'),('affine','attention_output'),('fc1','fc1_raw')]:p['anchors'][backend][label]=retain('norm-targets/'+backend+'-'+label+'-'+axis+'.f32',stage(axis,target))
   row=trace['kernels']['native1'if axis=='native'else'wasm1'][P.WEIGHT];require(row['metadata']['vec_dot_type']==8 and row['metadata']['converted_bytes']==4896 and row['weight_sha256']==P.sha(weight),'Actual Q8_0 capture policy differs')
   prefix='native/1/capture/'if axis=='native'else'browser/arrays/wasm1/';q=original_raw[prefix+P.WEIGHT+'.converted.bin'];require(P.sha(q)==row['converted_sha256'],'Original workspace differs');P.workspace(q)
   p['anchors'][backend]['workspace']=retain('norm-targets/'+backend+'-workspace-'+axis+'.bin',q)
  packets['chain_'+axis]=p
 require(set(packets)==set(P.PACKETS),'Six declared packets required')
 save(out/'browser/norm-packets.json',packets);save(out/'review/norm-prepared.json',dict(protocol_sha256=digest(P.PATH),files=files,packets_sha256=digest(out/'browser/norm-packets.json'),operator_service={k:service[k]for k in ['id','head_sha','path','conclusion','run_attempt']},weight=trace['weights'][P.WEIGHT],epsilon=proof['epsilon'],production_change=False))


def verify(out,binaries=False):
 cfg=hosted();old.prepared(out);proof=json.loads((out/'review/norm-prepared.json').read_text());require(proof['protocol_sha256']==digest(P.PATH)and proof['packets_sha256']==digest(out/'browser/norm-packets.json'),'Preparation changed')
 for name,row in proof['files'].items():require((out/name).is_file()and digest(out/name)==row['sha256']and(out/name).stat().st_size==row['bytes'],'Packet member changed: '+name)
 for kind in ['review','raw']:require(digest(out/('operator-'+kind+'.zip'))==cfg['operator'][kind]['sha256'],'Retained operator ZIP changed')
 if binaries:old.verified_build(out)
 return json.loads((out/'browser/norm-packets.json').read_text())


def build(out):
 verify(out);source=out/'source';old.command(['git','worktree','add','--detach',source,old.BASE],out,'source-worktree');old.command(['git','-C',source,'submodule','update','--init','--depth','1','ggml'],out,'source-submodule')
 require(old.git(source/'ggml','rev-parse','HEAD')==old.GGML,'Exact a06 backend required')
 path=source/'ggml/src/ggml-cpu/ggml-cpu.c';text=path.read_text();require(digest(path)=='b8c04a65c4ee295428501c0951c52b334920fef9101cb100061731b1b9fbee94','Exact original CPU source required')
 anchor='    ggml_barrier(params->threadpool);\n\n#if GGML_USE_LLAMAFILE\n    if (src1->type != vec_dot_type) {';require(text.count(anchor)==1,'Unique actual conversion barrier required');capture=(ROOT/'tests/wasm-layer3/kernel_capture.inc').read_text();path.write_text(text.replace(anchor,'    ggml_barrier(params->threadpool);\n\n'+capture+'\n#if GGML_USE_LLAMAFILE\n    if (src1->type != vec_dot_type) {'));save(out/'review/overlay.json',dict(test_only=True,source='ggml/src/ggml-cpu/ggml-cpu.c',capture_sha256=P.sha(capture.encode()),after_sha256=digest(path),math_changed=False))
 macros={}
 for label in ['native','wasm']:
  args=[str(x).replace(str(ROOT/'tests/wasm-f32-operators'),str(ROOT/'tests/wasm-norm-order'))for x in old.configuration(out,source,label)];old.command(args,out,label+'-configure');shutil.copy2(out/(label+'-build/compile_commands.json'),out/'review'/('compile-'+label+'.json'))
  old.command(['cmake','--build',out/(label+'-build'),'--parallel','2'],out,label+'-build',1800)
  rows=json.loads((out/(label+'-build/compile_commands.json')).read_text());row=next(r for r in rows if r['file'].endswith('/ggml-cpu/vec.cpp'));argv=row.get('arguments')or shlex.split(row['command']);clean=[];skip=False
  for item in argv:
   if skip:skip=False;continue
   if item=='-o':skip=True;continue
   if item=='-c':continue
   clean.append(item)
  old.command([*clean,'-dM','-E'],out,label+'-vec-macros',180);definitions={line.split()[1]:line.split()[2:]for line in(out/(label+'-vec-macros.log')).read_text().splitlines()if line.startswith('#define ')}
  require('__FAST_MATH__'not in definitions,'No fast-math variance emulation policy')
  if label=='native':require('__AVX2__'in definitions and'__FMA__'in definitions and'__AVX512F__'not in definitions,'Frozen native AVX2/FMA eight-lane branch required')
  else:require('__wasm_simd128__'in definitions,'Actual WASM SIMD branch required')
  macros[label]={k:definitions.get(k)for k in ['__AVX2__','__FMA__','__AVX512F__','__SSE2__','__wasm_simd128__','__FAST_MATH__']}
 save(out/'review/compiler-macros.json',macros);save(out/'review/compile-profile.json',old.compile_profile(out));destination=out/'browser/module';destination.mkdir()
 for ext in ['js','wasm']:shutil.copy2(out/('wasm-build/wasm-norm-order.'+ext),destination)
 save(out/'review/build-artifacts.json',{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=digest(p))for folder in [out/'native-build',destination]for p in folder.rglob('*')if p.is_file()and('.so'in p.name or p.suffix in ['.js','.wasm'])});old.ref_controls(out)


def native(out,phase):
 packets=verify(out,True)
 if phase=='norm':baseline(out)
 if phase=='chain':recovery(out)
 lib=C.CDLL(str(out/'native-build/libwasm-norm-order.so'));f=lib.crispembed_norm_order;f.argtypes=[C.c_int,C.c_int,C.c_void_p,C.c_int,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_int];f.restype=C.c_int
 setter=lib.crispembed_norm_setenv;setter.argtypes=[C.c_char_p];setter.restype=C.c_int
 mappings={line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if'/'in line and'libggml'in line.split()[-1]};declared=json.loads((out/'review/build-artifacts.json').read_text());require(mappings,'Mapped exact backend required')
 for value in mappings:
  path=Path(value).resolve();require(path.is_relative_to((out/'native-build').resolve())and digest(path)==declared[str(path.relative_to(out.resolve()))]['sha256'],'Unexpected/changed GGML provider')
 save(out/'review/native-mappings.json',dict(all_frozen_build=True,providers=sorted(mappings)))
 report=json.loads((out/'review/native-norm.json').read_text())if phase!='baseline'else{}
 for threads in [1,2]:
  rows=report.setdefault('native'+str(threads),{})
  for key,p in packets.items():
   if phase=='norm'and p['kind']or phase=='chain'and not p['kind']:continue
   data={n:(out/'browser'/p[n]['file']).read_bytes()for n in ['input','gamma','beta','weight']if n in p};buf={n:C.create_string_buffer(v)for n,v in data.items()};mode=0 if phase=='baseline'else 1;count=32256 if p['kind']else 4608;y=(C.c_float*count)()
   def call():return f(mode,p['kind'],buf['input'],4608,buf.get('gamma'),buf.get('beta'),buf.get('weight'),331776 if p['kind']else 0,y,count,threads)
   setter(None);require(call()==count,'Ordinary native graph failed');raw=C.string_at(y,count*4);folder=out/'raw'/('native'+str(threads))/(phase+'.'+key);folder.mkdir(parents=True);kernel=None;workspace=None
   if p['kind']:
    require(setter(str(folder).encode())==0 and call()==count and C.string_at(y,count*4)==raw,'Kernel capture changed output');q=folder/(P.WEIGHT+'.converted.bin');kernel=json.loads((folder/(P.WEIGHT+'.json')).read_text());workspace=dict(file=str(q.relative_to(out)),bytes=q.stat().st_size,sha256=digest(q));require((folder/(P.WEIGHT+'.input.f32')).read_bytes()==raw[2*18432:3*18432],'Actual native projection did not consume computed affine')
   setter(None);require(call()==count and C.string_at(y,count*4)==raw,'Native repeat changed');require(all(C.string_at(buf[n],len(v))==v for n,v in data.items()),'Native input mutated')
   stages={};offset=0
   for label,n in ([('norm',4608),('scaled',4608),('affine',4608),('fc1',18432)]if p['kind']else[('norm',4608)]):
    path=folder/(label+'.f32');path.write_bytes(raw[offset*4:(offset+n)*4]);offset+=n;stages[label]=dict(file=str(path.relative_to(out)),bytes=n*4,sha256=digest(path))
   rows[phase+'.'+key]=dict(stages=stages,workspace=workspace,kernel=kernel,input_sha256=p['input']['sha256'],repeat_bit_exact=True,ordinary_dump_bit_exact=True,inputs_unchanged=True);save(out/'review/native-norm.json',report)
  x=(C.c_float*4608)();y=(C.c_float*4608)(*([17.]*4608));stable=lambda:all(v==17 for v in y)
  require(f(9,0,x,4608,None,None,None,0,y,4608,threads)<0 and stable()and f(1,0,x,4607,None,None,None,0,y,4608,threads)<0 and stable(),'Native mode/extent output preservation')
  x[0]=float('nan');require(f(1,0,x,4608,None,None,None,0,y,4608,threads)<0 and stable(),'Nonfinite rejection');x[0]=0;require(f(1,0,x,4608,None,None,None,0,y,4608,threads)==4608 and all(v==0 for v in y),'Native recovery')
 save(out/'review/native-controls.json',dict(wrong_mode=True,wrong_extent=True,nonfinite=True,output_preserved=True,recovery=True,threads=[1,2]))


def outputs(out):
 n=json.loads((out/'review/native-norm.json').read_text());b=json.loads((out/'browser/norm-report.json').read_text());require(b['isolated']and b['pthreads']and not b['page_errors']and all(b['controls'].values()),'Actual browser controls required');return dict(n,wasm=b['outputs'])


def check_row(out,row,p):
 require(row['input_sha256']==p['input']['sha256']and all(row[k]is True for k in ['repeat_bit_exact','ordinary_dump_bit_exact','inputs_unchanged']),'Output/input/neutrality changed')
 for value in row['stages'].values():require((out/value['file']).stat().st_size==value['bytes']and digest(out/value['file'])==value['sha256'],'Output SHA changed')
 if p['kind']:
  q=row['workspace'];require(q['bytes']==4896 and digest(out/q['file'])==q['sha256'],'Actual converted workspace changed');P.workspace((out/q['file']).read_bytes())
  k=row['kernel'];require(k['weight_type']==20 and k['vec_dot_type']==8 and k['converted_bytes']==4896 and k['input_shape']==[384,12]and k['output_shape']==[1536,12],'Actual selected Q8_0 policy differs')


def baseline(out):
 packets=verify(out,True);rows=outputs(out);checked={}
 for backend,values in rows.items():
  for key,p in packets.items():
   row=values['baseline.'+key];check_row(out,row,p)
   if p['kind']:require(row['kernel']['wasm_round_plan']==int(backend=='wasm'),'Actual context rounding policy differs')
   for name,target in p['anchors'][backend].items():
    actual=row['workspace']if name=='workspace'else row['stages'][name];require(actual['sha256']==target['sha256']and actual['bytes']==target['bytes'],'Exact old baseline byte anchor failed: '+backend+'/'+key+'/'+name);checked[backend+'/'+key+'/'+name]=actual['sha256']
 require(len(checked)==30,'Exact30 old norm/chain/workspace anchors required');save(out/'review/norm-baseline.json',dict(passed=True,anchors=checked,qualification=False))


def recovery(out):
 packets=verify(out,True);baseline(out);rows=outputs(out);checked={}
 for backend,values in rows.items():
  for key,p in packets.items():
   if p['kind']:continue
   row=values['norm.'+key];check_row(out,row,p);require(row['stages']['norm']['sha256']==p['anchors']['native1']['norm']['sha256'],'Native-order norm must recover actual native bits BEFORE candidate projection: '+backend+'/'+key);checked[backend+'/'+key]=row['stages']['norm']['sha256']
 require(len(checked)==12,'Twelve native-order exact recovery controls required');save(out/'review/norm-recovery.json',dict(passed=True,controls=checked,projection_candidate_allowed=True,production_allowed=False))


def compare(out):
 packets=verify(out,True);recovery(out);rows=outputs(out);pairs=[];independent={}
 import numpy as np
 def pair(name,a,b,width):
  a=out/a;b=out/b;shape=[12,width];pairs.append(dict(name=name,original=dict(path=a,shape=shape,sha256=digest(a)),native=dict(path=b,shape=shape,sha256=digest(b))));independent[name]=old.metrics(np.memmap(a,dtype='<f4',mode='r',shape=tuple(shape)),np.memmap(b,dtype='<f4',mode='r',shape=tuple(shape)))
 workspaces={}
 for backend,values in rows.items():
  for key,p in packets.items():
   if not p['kind']:pair(backend+'.norm.'+key,'browser/'+p['anchors']['native1']['norm']['file'],values['norm.'+key]['stages']['norm']['file'],384);continue
   row=values['chain.'+key];check_row(out,row,p);require(row['kernel']['wasm_round_plan']==int(backend=='wasm'),'Candidate context policy changed')
   for label in ['norm','scaled','affine','fc1']:pair(backend+'.old_native.'+key+'.'+label,'browser/'+packets['chain_native']['anchors']['native1'][label]['file'],row['stages'][label]['file'],1536 if label=='fc1'else 384)
   raw=(out/row['workspace']['file']).read_bytes();reference=(out/('browser/'+packets['chain_native']['anchors']['native1']['workspace']['file'])).read_bytes();workspaces[backend+'/'+key]=dict(actual=P.workspace(raw),old_native_sha256=P.sha(reference),candidate_sha256=P.sha(raw),scale_blocks_changed=sum(raw[i:i+2]!=reference[i:i+2]for i in range(0,4896,34)),signed_bytes_changed=sum(a!=b for i in range(0,4896,34)for a,b in zip(raw[i+2:i+34],reference[i+2:i+34])))
  for label in ['norm','scaled','affine','fc1']:pair(backend+'.natural_input.'+label,values['chain.chain_native']['stages'][label]['file'],values['chain.chain_candidate']['stages'][label]['file'],1536 if label=='fc1'else 384)
 for key in ['chain_native','chain_candidate']:
  for label in ['norm','scaled','affine','fc1']:pair('backend.'+key+'.'+label,rows['native1']['chain.'+key]['stages'][label]['file'],rows['wasm']['chain.'+key]['stages'][label]['file'],1536 if label=='fc1'else 384)
 require(len(pairs)==56,'Exact56 comparisons required');adapters=old.write_pair(out/'control.gguf',out/'replay.gguf',pairs);process=subprocess.run([out/'native-build/wasm-f32-diff',out/'control.gguf',out/'replay.gguf',out/'review/ref.json'],capture_output=True);(out/'ref.log').write_bytes(process.stdout+process.stderr);cpp=old.collect(process.returncode,out/'review/ref.json',independent)
 save(out/'review/norm-risk.json',dict(schema='crispembed-wasm-native-order-norm-risk-v1',completed=True,source=old.source_identity()['source'],protocol_sha256=digest(P.PATH),baseline_byte_anchors=30,native_order_norm_bit_controls=12,comparisons=56,cpp=cpp,independent=independent,workspaces=workspaces,adapters=adapters,local_risk_supported=cpp['status']=='passed',qualification=False,production_change=False,model_forward=False,original_rerun=False,whole_model_negative_preserved=True,limits='Two retained residual inputs, two layer3 norm controls and attention-norm/affine/actual IQ4_NL-Q8_0 FC1 only. No oracle workspace or original output injected. Strict oldnative targets unchanged; whole10/eightstage replay still mandatory before production.'))
 save(out/'review/raw-manifest.json',{str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=digest(p))for folder in [out/'raw',out/'browser/norm-inputs',out/'browser/norm-targets',out/'browser/module']for p in folder.rglob('*')if p.is_file()});require(cpp['status']=='passed','Norm risk rejected unchanged gates; all outcomes retained')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','verify','build','native-baseline','native-norm','native-chain','baseline','recovery','compare']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 try:
  if a.phase.startswith('native-'):native(a.output,a.phase[7:])
  elif a.phase=='verify':verify(a.output,True)
  else:globals()[a.phase](a.output)
 except Exception as e:save(a.output/'review'/('norm-rejection-'+a.phase+'.json'),dict(phase=a.phase,error=type(e).__name__+': '+str(e),qualification=False));raise
