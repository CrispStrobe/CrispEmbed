#!/usr/bin/env python3
"""Closed source-fed GDN replay; ten operator cases, no model/decode/heldout."""
import hashlib,importlib.util,json,os,platform,shutil,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'));from runtime_guard import Deadline,command
CFG_SHA='948a8fb058465313cfad3153185d90c9acd5c2b5375be46f9d2e64f1b1a0968f'
PROFILE={'BUILD_SHARED_LIBS':'ON','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_AVX512':'OFF','GGML_CPU_REPACK':'ON','GGML_OPENMP':'ON','GGML_LLAMAFILE':'ON','GGML_BACKEND_DL':'OFF','GGML_BLAS':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=module('physical',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');require=base.require;sha=base.sha

def admit(cfg):
 require(hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()==CFG_SHA,'immutable predeclared replay protocol')
 require(cfg['schema']=='lighton4b-physical-gdn-hybrid/v1'and cfg['tokens']==[1,4,16]and(cfg['key_heads'],cfg['value_heads'],cfg['width'],cfg['conv_kernel'],cfg['threads'],cfg['build_threads'])==(16,32,128,4,4,2),'closed captured geometry')
 require(not cfg['model_forward_allowed']and not cfg['full_f32_model_allowed']and cfg['total_timeout_seconds']==1800,'operator only deadline')
 require(cfg['fused_gdn_ar']and cfg['fused_gdn_ch']and not cfg['auto_fgdn']and cfg['recurrent_snapshots']==1,'actual upstream fused K1 graph')
 require(len(cfg['tensors'])==5 and sum(r['bytes']for r in cfg['tensors'])==6030080,'five whole bounded parameters')
 require({r['file']for r in cfg['tensors']}=={'conv.f32','A.f32','dt.f32','norm.f32','out.q4k'},'exact parameter domain')
 require([c['variant']for c in cfg['cases']]==['native16','native4','zab4','continuation1']and sum(len(c['runs'])for c in cfg['cases'])==14,'predeclared mechanisms, not output selection')
def admit_compile(entries):
 selected=[e for e in entries if Path(e['file']).name in ['qwen35_delta.cpp','lighton4b_physical_gdn_source_ops.cpp']]
 expected={(name,target)for name in ['qwen35_delta.cpp','lighton4b_physical_gdn_source_ops.cpp']for target in ['gdn-hybrid-probe','gdn-hybrid-cache-controls']}
 seen=set()
 for entry in selected:
  command=entry.get('command',' '.join(entry.get('arguments',[])))
  targets=[target for target in ['gdn-hybrid-probe','gdn-hybrid-cache-controls']if 'CMakeFiles/'+target+'.dir/'in command]
  require(len(targets)==1 and '-ffp-contract=off'in command,'actual exact target/helper contraction rule')
  key=(Path(entry['file']).name,targets[0]);require(key not in seen,'unique target/helper compile entry');seen.add(key)
 require(seen==expected,'both actual helpers in both named targets')
 return selected
def row_map(role,width):
 p=lambda h:16*(h%2)+h//2
 if role=='qkv':return list(range(4096))+[4096+128*p(i//128)+i%128 for i in range(4096)]
 if role=='z':return [128*p(i//128)+i%128 for i in range(width)]
 return [p(i)for i in range(width)]
def main():
 import numpy as np
 require(np.__version__=='2.3.5'and sys.byteorder=='little'and platform.system()=='Linux'and platform.machine().lower()in['x86_64','amd64']and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'pinned CPU operator worker')
 cfgpath=ROOT/'tools/gdn-hybrid-probe/protocol.json';cfg=json.loads(cfgpath.read_text());admit(cfg);out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'gdn-trace';work.mkdir();started=time.monotonic();deadline=Deadline(1800);deadline.__enter__();report={'schema':cfg['schema'],'status':'failed','commands':{},'cases':{},'operator_compute_executed':False,'model_forward_executed':False,'full_f32_model_created':False,'decoded_generated':False,'quality_qualified':False,'heldout_used':False}
 def interrupted(sig,frame):raise SystemExit(128+sig)
 previous_signals={sig:signal.signal(sig,interrupted)for sig in [signal.SIGTERM,signal.SIGINT]}
 def save():(out/'gdn-hybrid-probe.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 def remaining():n=1800-(time.monotonic()-started);require(n>0,'whole operator worker deadline');return n
 env=dict(os.environ)
 for key in list(env):
  if key.startswith('CRISPEMBED_'):env[key]='0'
 env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='1')
 def run(label,args,limit=900,expected=0):
  report['phase']=label;report['commands'][label]={};save();print('PHASE '+label,flush=True)
  try:
   try:command(args,out/(label+'.log'),min(limit,remaining()),report['commands'][label],env=env,cwd=ROOT)
   except subprocess.CalledProcessError as error:
    require(expected!=0 and error.returncode==expected,'actual expected loader returncode');report['commands'][label]['status']='completed_expected_failure'
   else:require(expected==0,'loader must reject missing arguments')
  finally:save()
 def locate(name,digest):
  rows=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(rows)==1,'one exact retained '+name);return rows[0]
 def verify(path,row):require(path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'whole selected actual artifact '+path.name)
 def array(path,width,n):
  a=np.fromfile(path,dtype='<f4');require(a.size==width*n and np.isfinite(a).all(),'finite exact actual geometry');return a.reshape(n,width)
 def diff(candidate,reference):
  x=np.fromfile(candidate,dtype='<f4');y=np.fromfile(reference,dtype='<f4');require(x.shape==y.shape and x.size and np.isfinite(x).all()and np.isfinite(y).all(),'same finite stage arrays');d=x.astype('f8')-y.astype('f8');bad=np.flatnonzero(x.view('<u4')!=y.view('<u4'));return {'elements':int(x.size),'bit_exact':not len(bad),'different_bits':int(len(bad)),'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(abs(d).max()),'candidate_rms_magnitude':float(np.sqrt(np.mean(x.astype('f8')**2))),'reference_rms_magnitude':float(np.sqrt(np.mean(y.astype('f8')**2))),'reference_max_abs':float(abs(y).max()),'first_differences':[{'index':int(i),'candidate':float(x[i]),'reference':float(y[i])}for i in bad[:8]]}
 def fetch(row,target):
  start=row['source_offset'];size=row['bytes'];tmp=target.with_suffix(target.suffix+'.partial');count=0;h=hashlib.sha256()
  try:
   request=urllib.request.Request(cfg['source_url'],headers={'Range':f'bytes={start}-{start+size-1}'})
   with urllib.request.urlopen(request,timeout=min(120,remaining()))as response,tmp.open('xb')as stream:
    require(response.status==206 and response.headers.get('Content-Range')==f"bytes {start}-{start+size-1}/{cfg['source_bytes']}",'exact source parameter range')
    while data:=response.read(1<<20):count+=len(data);require(count<=size,'bounded parameter bytes');h.update(data);stream.write(data)
   require(count==size and h.hexdigest()==row['source_packed_sha256'],'whole original parameter hash');tmp.rename(target)
  except BaseException:
   if tmp.exists():tmp.unlink()
   raise RuntimeError('bounded source parameter acquisition rejected: '+row['file'])from None
 try:
  run('source',['git','rev-parse','HEAD'],30);report['source_sha']=(out/'source.log').read_text().strip();require(report['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact replay source');run('cpu',['lscpu'],30);flags=next(x.split(':',1)[1].split()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('flags'));require({'avx2','fma','f16c'}<=set(flags),'actual AVX2/FMA/F16C CPU');report['capacity']={'memory':base.memory(),'scratch_free':shutil.disk_usage(work).free,'output_free':shutil.disk_usage(out).free};require(report['capacity']['memory']['MemAvailable']>=cfg['minimum_available_memory_bytes']and report['capacity']['scratch_free']>=cfg['minimum_scratch_bytes']and report['capacity']['output_free']>=cfg['minimum_output_bytes'],'bounded operator resources')
  for name,digest in cfg['native_source_sha256'].items():require(sha(ROOT/name)==digest,'actual native helper bytes '+name)
  require(sha(ROOT/'tools/gdn-hybrid-probe/main.cpp')==cfg['probe_cpp_sha256']and sha(ROOT/'tools/gdn-hybrid-probe/CMakeLists.txt')==cfg['probe_cmake_sha256'],'reviewed reference adapter source')
  tp=locate('physical-f16-table512.json',cfg['table_receipt_sha256']);table=json.loads(tp.read_text());require(table['status']=='measured_extended_table'and table['source_observer_full_neutral'],'complete frozen native table');pp=locate('text-trace.json',cfg['capture_receipt_sha256']);producer=json.loads(pp.read_text());rp=locate('text-trace-analysis.json',cfg['reader_receipt_sha256']);reader=json.loads(rp.read_text());require(reader['status']=='measured_neutral_aligned_text_trace'and reader['all_producer_artifacts_rehashed']==915 and reader['analysis']['same_model_input_bits']and reader['analysis']['same_positions_first3_axes']and producer['all282_scores_observer_neutral'],'actual neutral aligned live reference')
  native=tp.parent/'native';native_arrays={}
  for name,row in cfg['native_inputs'].items():verify(native/name,row)
  for role,width in [('qkv',8192),('z',4096),('a',32),('b',32)]:native_arrays[role]=array(native/f'prefill.first_chunk.layer0.{role}.f32',width,16)
  live={}
  for role,row in cfg['live_inputs'].items():verify(pp.parent/row['file'],row);live[role]=array(pp.parent/row['file'],row['width'],4)[:,row_map(role,row['width'])].copy()
  require(live['qkv'].tobytes()==native_arrays['qkv'][:4].tobytes(),'actual source/native first4 QKV are identical before interventions')
  live_expected=work/'live-expected';live_expected.mkdir()
  for name,row in cfg['live_stages'].items():
   path=pp.parent/row['file'];verify(path,row)
   if name=='state_predelta':require(np.fromfile(path,dtype='<f4').size==524288 and np.all(np.fromfile(path,dtype='<f4')==0),'actual source pre-delta state exactly zero');continue
   width=8192 if name.startswith('conv_')else 4096 if name=='final_output'else 2560;a=array(path,width,4)
   if width in[8192,4096]:a=a[:,row_map('qkv'if width==8192 else'z',width)]
   a.astype('<f4').tofile(live_expected/(name+'.f32'))
  continuation={}
  for role,row in cfg['continuation_inputs'].items():
   path=pp.parent/row['file'];verify(path,row)
   raw=np.fromfile(path,dtype='<f4');require(np.isfinite(raw).all(),'finite actual continuation input')
   if role=='state':
    require(raw.size==524288 and np.any(raw!=0),'actual nonzero continuation state');continuation[role]=raw.reshape(32,128,128)[row_map('a',32)].transpose(0,2,1).copy()
   else:continuation[role]=raw.reshape(72,row['width'])[:1,row_map(role,row['width'])].copy()
  inputs=work/'inputs';inputs.mkdir()
  for case in cfg['cases']:
   folder=inputs/case['variant'];folder.mkdir();n=case['input_tokens']
   for role in native_arrays:
    a=continuation[role]if case['variant']=='continuation1'else live[role]if role in case['replaced_roles']else native_arrays[role][:n];a.astype('<f4').tofile(folder/f'prefill.first_chunk.layer0.{role}.f32')
   if case['variant']=='continuation1':
    continuation['state'].astype('<f4').tofile(folder/'initial_state.f32');live['qkv'].T.copy().astype('<f4').tofile(folder/'initial_history.f32')
   if n==16:shutil.copy2(native/'prefill.first_chunk.layer0.attention.f32',folder/'prefill.first_chunk.layer0.attention.f32')
  up=work/'upstream'
  for label,args in [('up-init',['git','init',up]),('up-remote',['git','-C',up,'remote','add','origin','https://github.com/ggml-org/llama.cpp']),('up-fetch',['git','-C',up,'fetch','--depth','1','origin',cfg['upstream_commit']]),('up-checkout',['git','-C',up,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',up,'rev-parse','HEAD'],text=True).strip()==cfg['upstream_commit'],'actual source ggml revision')
  for name,digest in cfg['source_function_sha256'].items():require(sha(up/name)==digest,'actual source GDN blueprint bytes '+name)
  run('native-submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['native_ggml_commit'],'actual native ggml revision');builds={}
  for kind,ggml_source in [('native',ROOT/'ggml'),('upstream',up/'ggml')]:
   build=work/(kind+'-build');args=['cmake','-S',ROOT/'tools/gdn-hybrid-probe','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DGGML_SOURCE='+str(ggml_source)]+[f'-D{k}={v}'for k,v in PROFILE.items()]
   if kind=='native':args+=['-DNATIVE_SOURCE='+str(ROOT)]
   run(kind+'-configure',args);cache=(build/'CMakeCache.txt').read_text()
   for key,value in PROFILE.items():require(f'{key}:BOOL={value}'in cache,'exact actual CPU profile '+kind+'/'+key)
   shutil.copy2(build/'CMakeCache.txt',out/(kind+'-CMakeCache.txt'))
   if kind=='native':
    report['native_helper_compile_entries']=admit_compile(json.loads((build/'compile_commands.json').read_text()))
   run(kind+'-build',['cmake','--build',build,'--target','gdn-hybrid-probe','--parallel','2']);binary=build/'gdn-hybrid-probe';run(kind+'-noargs',[binary],30,1);require('MODE TOKENS INPUT_TOKENS ACTUAL_INPUT_DIR WEIGHTS_DIR OUTPUT_DIR'in(out/(kind+'-noargs.log')).read_text(),'actual source adapter loader contract');synthetic=work/(kind+'-synthetic');synthetic.mkdir();run(kind+'-backend-control',[binary,'--backend-control',synthetic],30);require('GDN_TRACE_BACKEND PASS model=0 raw_preferred=1'in(out/(kind+'-backend-control.log')).read_text(),'actual zero-quantized raw/preferred backend setup');builds[kind]=binary
  run('hybrid-cache-build',['cmake','--build',work/'native-build','--target','gdn-hybrid-cache-controls','--parallel','2']);run('hybrid-cache-controls',[work/'native-build/gdn-hybrid-cache-controls'],120);require('PHYSICAL_GDN_SOURCE_OPS_CONTROLS PASS'in(out/'hybrid-cache-controls.log').read_text(),'actual full block cache/gate controls')
  weights=work/'weights';weights.mkdir();report['phase']='five-source-parameter-ranges';save()
  for row in cfg['tensors']:fetch(row,weights/row['file'])
  report['whole_parameter_bytes']=6030080;report['operator_compute_executed']=True
  for case in cfg['cases']:
   variant=case['variant'];report['cases'][variant]={'tokens':case['tokens'],'replaced_roles':case['replaced_roles'],'runs':{}}
   for kind in case['runs']:
    label=variant+'-'+kind;dest=out/label;dest.mkdir();binary=builds['upstream'if kind=='source-upstream'else'native'];mode=kind if kind in ['native','hybrid']else'source';run(label,[binary,mode,case['tokens'],case['input_tokens'],inputs/variant,weights,dest]);report['cases'][variant]['runs'][kind]={'control':json.loads((dest/'probe-control.json').read_text()),'stages':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in dest.glob('*.f32')}};save()
   if variant=='native16':require(report['cases'][variant]['runs']['native']['control']['actual_main_attention_bit_exact'],'actual old native16 full GDN/RAW-out byte closure')
   if variant=='zab4':
    ref=out/'zab4-source-upstream';mapping={'conv_raw.f32':'conv_output_raw.f32','conv_silu.f32':'conv_output_silu.f32','postnorm.f32':'final_output.f32','out_preferred.f32':'linear_attn_out.f32'};report['actual_live_source_closures']={stage:diff(ref/stage,live_expected/actual)for stage,actual in mapping.items()};save();require(all(v['bit_exact']for v in report['actual_live_source_closures'].values()),'whole actual live upstream GDN and preferred out must close before interpretation')
  report['comparisons']={};reference=out/'zab4-source-upstream'
  for case in cfg['cases']:
   if case['tokens']!=4:continue
   for kind in case['runs']:
    folder=out/(case['variant']+'-'+kind);report['comparisons'][folder.name]={name:diff(folder/name,reference/name)for name in ['conv_raw.f32','conv_silu.f32','q_norm.f32','k_norm.f32','decay.f32','beta.f32','delta.f32','post_rms.f32','z_silu.f32','postnorm.f32','out_raw.f32','out_preferred.f32']if(folder/name).is_file()}
  for name in ['postnorm','out_raw','out_preferred']:
   a=np.fromfile(out/f'native16-native/{name}.f32',dtype='<f4').reshape(16,-1)[:4];temp=work/(name+'-native16-prefix.f32');a.tofile(temp);report.setdefault('native4_vs_actual16_prefix',{})[name]=diff(out/f'native4-native/{name}.f32',temp)
  report['hybrid_same_input']={}
  for case in cfg['cases']:
   label=case['variant'];candidate=out/(label+'-hybrid');ref=out/(label+'-source-upstream');report['hybrid_same_input'][label]={name:diff(candidate/name,ref/name)for name in ['state.f32','conv_history.f32','postnorm.f32','out_raw.f32','out_preferred.f32']}
  # Live-source input and nonzero continuation must close exactly; RAW inputs
  # retain separate measured semantics and never imply a decoded improvement.
  require(all(v['bit_exact']for label in ['zab4','continuation1']for v in report['hybrid_same_input'][label].values()),'hybrid actual-source inputs and nonzero continuation bit closure')
  report['status']='measured_actual_source_hybrid_controls';report['limits']=['One observed table first4 prefix, same immutable quantized weights; no recipe or heldout selection.','Counterfactual projected-input replacements are operator mechanisms, not actual native free-generation outputs.','Source-native backend equivalence is measured, never assumed from algebra.','No full model, original BF16 reference, decoded output or quality promotion was executed.']
 except BaseException as e:report.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  deadline.close()
  for sig,handler in previous_signals.items():signal.signal(sig,handler)
  report['seconds']=time.monotonic()-started;owned={x+'.log'for x in report['commands']}|{'native-CMakeCache.txt','upstream-CMakeCache.txt'};report['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)};save()
if __name__=='__main__':main()
