#!/usr/bin/env python3
"""Actual postnorm/Q4 gate-up input-code attribution; no runtime/model changes."""
import hashlib,importlib.util,json,os,platform,shutil,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
spec=importlib.util.spec_from_file_location('swiglu_prior',ROOT/'tools/swiglu-source-probe/worker.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
require=prior.require;sha=prior.sha;PROFILE=prior.PROFILE
CFG_SHA='f70a0063cb81277f293d2cd41b626b9c0db4ae834b3acb4e42866043947097d9'
def admit(c):
 require(hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()==CFG_SHA,'frozen postnorm protocol')
 require(c['schema']=='lighton4b-ffn-postnorm-codes/v1'and c['tokens']=={'source':[1,4],'off':[1,4,16],'on':[1,4,16]}and c['threads']==4 and c['build_threads']==2,'closed observed geometries')
 require(c['total_timeout_seconds']==1800 and not c['model_forward_allowed']and not c['full_f32_model_allowed'],'operator only deadline')
 require([(x['role'],x['type'],x['source_shape'])for x in c['weights']]==[('gate',12,[2560,9216]),('up',12,[2560,9216])]and sum(x['bytes']for x in c['weights'])==26542080,'two whole original Q4 gate/up tensors')
 require(c['code_corners']==['source','native','source_qs_native_scale','native_qs_source_scale'],'predeclared paired code-space controls')
def code_array(np,path,n):
 require(path.stat().st_size==n*10*292,'whole declared Q8_K geometry');a=np.fromfile(path,dtype=np.dtype([('d','<f4'),('qs','i1',(256,)),('bsums','<i2',(16,))]));require(np.isfinite(a['d']).all()and np.array_equal(a['qs'].astype('i4').reshape(-1,16,16).sum(axis=2),a['bsums']),'finite code scales and coupled integer sums');return a
def code_diff(np,a,b):
 x=(a['d'].astype('f8')[:,None]*a['qs']).astype('<f4');y=(b['d'].astype('f8')[:,None]*b['qs']).astype('<f4')
 return dict(scope='Public converter/code-input interventions; not directly observed CPU_REPACK buffers',blocks=int(a.size),different_scales=int(np.count_nonzero(a['d'].view('<u4')!=b['d'].view('<u4'))),different_qs=int(np.count_nonzero(a['qs']!=b['qs'])),different_bsums=int(np.count_nonzero(a['bsums']!=b['bsums'])),scale_sign_changes=int(np.count_nonzero(np.signbit(a['d'])!=np.signbit(b['d']))),decoded_activation_difference=prior.stats(np,x.reshape(-1),y.reshape(-1)))
def corners(a,b):
 scale=a.copy();scale['d']=b['d'];integer=a.copy();integer['qs']=b['qs'];integer['bsums']=b['bsums'];return dict(source=a.copy(),native=b.copy(),source_qs_native_scale=scale,native_qs_source_scale=integer)
def main():
 import numpy as np
 require(np.__version__=='2.3.5'and sys.byteorder=='little'and platform.system()=='Linux'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'pinned CPU operator worker')
 cfg=json.loads((ROOT/'tools/ffn-postnorm-probe/protocol.json').read_text());admit(cfg);out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'postnorm-probe';work.mkdir();started=time.monotonic();deadline=Deadline(1800);deadline.__enter__();r=dict(schema=cfg['schema'],status='failed',commands={},operator_compute_executed=False,model_forward_executed=False,full_f32_model_created=False,decoded_generated=False,heldout_used=False,quality_qualified=False)
 def interrupted(sig,frame):raise SystemExit(128+sig)
 handlers={s:signal.signal(s,interrupted)for s in [signal.SIGTERM,signal.SIGINT]}
 def save():(out/'ffn-postnorm-probe.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 def remaining():n=1800-(time.monotonic()-started);require(n>0,'whole operator deadline');return n
 env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def run(label,args,expected=0,limit=600):
  r['phase']=label;r['commands'][label]={};save();print('PHASE '+label,flush=True)
  try:
   try:command(args,out/(label+'.log'),min(limit,remaining()),r['commands'][label],env=env,cwd=ROOT)
   except subprocess.CalledProcessError as error:require(expected and error.returncode==expected,'expected closed CLI exit');r['commands'][label]['status']='completed_expected_failure'
   else:require(expected==0,'missing CLI must reject')
  finally:save()
 def compare(a,b):return prior.stats(np,np.fromfile(a,dtype='<f4'),np.fromfile(b,dtype='<f4'))
 def fetch(row,target):
  start=row['source_offset'];size=row['bytes'];temp=target.with_suffix('.partial');count=0;h=hashlib.sha256()
  try:
   request=urllib.request.Request(cfg['source_url'],headers={'Range':f'bytes={start}-{start+size-1}'})
   with urllib.request.urlopen(request,timeout=min(120,remaining()))as response,temp.open('xb')as stream:
    require(response.status==206 and response.headers.get('Content-Range')==f"bytes {start}-{start+size-1}/{cfg['source_bytes']}",'exact original parameter range')
    while b:=response.read(1<<20):count+=len(b);require(count<=size,'bounded original bytes');stream.write(b);h.update(b)
   require(count==size and h.hexdigest()==row['source_packed_sha256'],'whole original packed hash');temp.rename(target)
  except BaseException:
   if temp.exists():temp.unlink()
   raise RuntimeError('bounded parameter acquisition rejected: '+row['role'])from None
 try:
  run('source',['git','rev-parse','HEAD'],limit=30);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact reference source');run('cpu',['lscpu'],limit=30);flags=next(x.split(':',1)[1].split()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('flags'));require({'avx2','fma','f16c'}<=set(flags),'actual AVX2/FMA/F16C');r['capacity']=dict(memory=prior.base.memory(),scratch=shutil.disk_usage(work).free,output=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=cfg['minimum_available_memory_bytes']and r['capacity']['scratch']>=cfg['minimum_scratch_bytes']and r['capacity']['output']>=cfg['minimum_output_bytes'],'bounded resources')
  require(sha(ROOT/'tools/swiglu-source-probe/protocol.json')==cfg['prior_protocol_sha256']and sha(ROOT/'tools/swiglu-source-probe/worker.py')==cfg['prior_worker_sha256'],'immutable prior input provenance and metric helpers')
  oldcfg=json.loads((ROOT/'tools/swiglu-source-probe/protocol.json').read_text());require(all(oldcfg[k]==cfg[k]for k in ['upstream_commit','native_ggml_commit','source_url','source_bytes','source_sha256','candidate_sha256']),'same immutable donor/backend profile')
  for name,key in [('main.cpp','adapter_cpp_sha256'),('CMakeLists.txt','adapter_cmake_sha256')]:require(sha(ROOT/'tools/ffn-postnorm-probe'/name)==cfg[key],'reviewed standalone adapter')
  matches=[p for p in Path('/kaggle/input').rglob('swiglu-source-probe.json')if p.is_file()and p.stat().st_size<4<<20 and sha(p)==cfg['prior_receipt_sha256']];require(len(matches)==1,'one exact completed operator input producer');pp=matches[0];proof=json.loads(pp.read_text());require(proof['source_sha']==cfg['prior_source_sha']and proof['status']=='measured_same_input_swiglu_operators'and not proof['model_forward_executed']and all(v['bit_exact']for v in proof['projection_closures'].values()),'actual source/native whole projection closures retained')
  inputs=out/'inputs';inputs.mkdir();cases={}
  for dataset,tokens in cfg['tokens'].items():
   extent=4 if dataset=='source'else 16;full={}
   for role,row in cfg['inputs'][dataset].items():
    path=pp.parent/row['file'];require(row['bytes']==path.stat().st_size and row['sha256']==sha(path)and proof['artifacts'][row['file']]=={k:row[k]for k in ['bytes','sha256']},'actual prior whole input hash');original=oldcfg['source_inputs'][role]if dataset=='source'else oldcfg['native_inputs'][dataset]['post_norm'if role=='postnorm'else role];require(all(original[k]==row[k]for k in ['bytes','sha256']),'original capture/full-native array identity');width=2560 if role=='postnorm'else 9216;a=np.fromfile(path,dtype='<f4');require(a.size==extent*width and np.isfinite(a).all(),'finite actual typed context');full[role]=a.reshape(extent,width)
   for n in tokens:
    label=f'{dataset}{n}';folder=inputs/label;folder.mkdir();cases[label]=(dataset,n,folder)
    for role,a in full.items():a[:n].astype('<f4').tofile(folder/(role+'.f32'))
  up=work/'upstream'
  for label,args in [('up-init',['git','init',up]),('up-remote',['git','-C',up,'remote','add','origin','https://github.com/ggml-org/llama.cpp']),('up-fetch',['git','-C',up,'fetch','--depth','1','origin',cfg['upstream_commit']]),('up-checkout',['git','-C',up,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',up,'rev-parse','HEAD'],text=True).strip()==cfg['upstream_commit'],'exact upstream');run('native-submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['native_ggml_commit'],'exact native GGML');bins={}
  for kind,source in [('native',ROOT/'ggml'),('upstream',up/'ggml')]:
   build=work/(kind+'-build');run(kind+'-configure',['cmake','-S',ROOT/'tools/ffn-postnorm-probe','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_SOURCE='+str(source),*[f'-D{k}={v}'for k,v in PROFILE.items()]]);cache=(build/'CMakeCache.txt').read_text();require(all(f'{k}:BOOL={v}'in cache for k,v in PROFILE.items()),'matched CPU profile');shutil.copyfile(build/'CMakeCache.txt',out/(kind+'-CMakeCache.txt'));run(kind+'-build',['cmake','--build',build,'--target','ffn-postnorm-probe','--parallel','2']);binary=build/'ffn-postnorm-probe';run(kind+'-noargs',[binary],1,30);require('MODE ROLE TOKENS INPUT_DIR WEIGHTS_DIR OUTPUT_FILE'in(out/(kind+'-noargs.log')).read_text(),'actual loader contract');run(kind+'-controls',[binary,'--controls'],limit=60);require('FFN_POSTNORM_CONTROLS PASS'in(out/(kind+'-controls.log')).read_text(),'actual Q4/Q8 packed-RHS backend controls');bins[kind]=binary
  weights=work/'weights';weights.mkdir();r['phase']='two-original-Q4-ranges';save()
  for row in cfg['weights']:
   require(row in oldcfg['weights']and row['sha256']==row['source_packed_sha256']and row['whole_rows_permuted']==0 and row['quant_blocks_unchanged'],'same original whole coordinate map');fetch(row,weights/row['file'])
  r['parameter_bytes']=26542080;r['operator_compute_executed']=True;outputs=out/'cases';outputs.mkdir();r['cases']={};r['actual_projection_closures']={}
  for label,(dataset,n,folder)in cases.items():
   dest=outputs/label;dest.mkdir();r['cases'][label]={}
   for kind,binary in bins.items():
    run(label+'-'+kind+'-codes',[binary,'codes','gate',n,folder,weights,dest/(kind+'-codes.bin')]);code_array(np,dest/(kind+'-codes.bin'),n)
    for role in ['gate','up']:
     for mode in ['raw','preferred','prequantized']:run(label+'-'+kind+'-'+role+'-'+mode,[binary,mode,role,n,folder,weights,dest/(kind+'-'+role+'-'+mode+'.f32')])
     prefix=kind+'-'+role;r['cases'][label][prefix]={'forced_q8_raw_output_closure':compare(dest/(prefix+'-raw.f32'),dest/(prefix+'-prequantized.f32')),'preferred_vs_raw':compare(dest/(prefix+'-preferred.f32'),dest/(prefix+'-raw.f32'))}
     if(dataset=='source'and n==4 and kind=='upstream')or(dataset in ['off','on']and n==16 and kind=='native'):
      mode='preferred'if dataset=='source'else'raw';r['actual_projection_closures'][label+'-'+role]=compare(dest/(prefix+'-'+mode+'.f32'),folder/(role+'.f32'))
   r['cases'][label]['converter_native_vs_upstream']=code_diff(np,code_array(np,dest/'native-codes.bin',n),code_array(np,dest/'upstream-codes.bin',n));save()
  require(all(v['bit_exact']for v in r['actual_projection_closures'].values()),'whole actual source4/nativeOFF16/ON16 Q4 projections close before attribution')
  require(all(row['forced_q8_raw_output_closure']['bit_exact']for case in r['cases'].values()for row in case.values()if 'forced_q8_raw_output_closure'in row),'public converter must reproduce RAW F32-input outputs before code attribution')
  r['input_effects']={};r['code_corners']={}
  for dataset in ['off','on']:
   label=dataset+'4';dest=outputs/label;reference=outputs/'source4';r['input_effects'][dataset]={'actual_postnorm':compare(inputs/label/'postnorm.f32',inputs/'source4/postnorm.f32'),'roles':{}};r['code_corners'][dataset]={}
   for role in ['gate','up']:
    A=np.fromfile(dest/('native-'+role+'-raw.f32'),dtype='<f4');B=np.fromfile(dest/('upstream-'+role+'-preferred.f32'),dtype='<f4');C=np.fromfile(reference/('upstream-'+role+'-preferred.f32'),dtype='<f4');total=A.astype('f8')-C;route=A.astype('f8')-B;drift=B.astype('f8')-C;r['input_effects'][dataset]['roles'][role]=dict(total_native_raw_vs_actual_source=prior.stats(np,A,C),same_input_route_effect=prior.stats(np,A,B),fixed_preferred_route_input_effect=prior.stats(np,B,C),decomposition_max_abs=float(abs(total-route-drift).max()),input_rms_over_total_rms=float(np.sqrt(np.mean(drift*drift))/np.sqrt(np.mean(total*total)))if np.any(total)else None)
   for kind,binary in bins.items():
    a=code_array(np,reference/(kind+'-codes.bin'),4);b=code_array(np,dest/(kind+'-codes.bin'),4);r['input_effects'][dataset][kind+'_codes']=code_diff(np,b,a);corners_root=out/'corners'/dataset/kind;corners_root.mkdir(parents=True);r['code_corners'][dataset][kind]={}
    for name,values in corners(a,b).items():
     folder=corners_root/name;folder.mkdir();values.tofile(folder/'q8_codes.bin');code_array(np,folder/'q8_codes.bin',4)
     for role in ['gate','up']:
      path=folder/(role+'.f32');run(dataset+'-'+kind+'-'+name+'-'+role,[binary,'packed',role,4,folder,weights,path])
      if name in ['source','native']:
       original=(reference if name=='source'else dest)/(kind+'-'+role+'-prequantized.f32');require(compare(path,original)['bit_exact'],'packed actual-code endpoints reproduce original prequantized outputs')
    for role in ['gate','up']:
     base_corner=corners_root/'source'/(role+'.f32');r['code_corners'][dataset][kind][role]={name:compare(corners_root/name/(role+'.f32'),base_corner)for name in cfg['code_corners']}
   for role in ['gate','up']:
    actual=np.fromfile(inputs/(dataset+'16')/(role+'.f32'),dtype='<f4').reshape(16,9216)[:4].reshape(-1);rerun=np.fromfile(dest/('native-'+role+'-raw.f32'),dtype='<f4');r['input_effects'][dataset]['roles'][role]['prefix4_vs_actual_native16']=prior.stats(np,rerun,actual)
   save()
  r['status']='measured_actual_postnorm_code_attribution';r['limits']=['Observed table layer0 first4 only; no heldout/tensor-selection/model inference.','Actual source/native inputs remain distinct; route and code interventions hold weights fixed.','Public Q8_K vectors are not directly captured CPU_REPACK buffers. RAW forced-input output closure is measured per case/role/backend.','Crossed scale/integer vectors are synthetic code-space interventions, not model states. Interpret sign changes and coupled representations explicitly.','Previous GDN decoded regression remains binding; component amplification does not establish a safe runtime fix.']
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  deadline.close()
  for sig,handler in handlers.items():signal.signal(sig,handler)
  r['seconds']=time.monotonic()-started;r['artifacts']=prior.inventory(out,r['commands']);save()
if __name__=='__main__':main()
