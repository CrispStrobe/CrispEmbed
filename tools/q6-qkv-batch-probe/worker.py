#!/usr/bin/env python3
"""Fixed actual Q6 inputs, full-vector closure, no model inference."""
import csv,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('repackbase',ROOT/'tools/repack-probe/worker.py');require=base.require;sha=base.sha
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'));from runtime_guard import Deadline,command
CFG_SHA='45504a7e9190013a27d4e12e989f0431eb06fd626ed4896f1dd0b9e25e02b935'
def admit(cfg):
 require(hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()==CFG_SHA,'immutable Q6 batch protocol')
 require(cfg['matrix']['type']==14 and cfg['matrix']['source_shape']==[2560,8192]and cfg['worker_seconds']==2400 and not cfg['model_forward_allowed'],'fixed operator-only scope')
def batch_plans(batches):
 plans={'source4':[(0,4)],'source140':[(4,144)]}
 for batch in batches:
  plans['mixed144-chunk'+str(batch)]=[(i,min(i+batch,144))for i in range(0,144,batch)]
  if batch<=140:plans['image140-chunk'+str(batch)]=[(i,min(i+batch,144))for i in range(4,144,batch)]
 return plans
def main():
 import numpy as np
 cfg=json.loads((Path(__file__).parent/'protocol.json').read_text());admit(cfg);require(np.__version__=='2.3.5'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'CPU array profile');out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'q6-batch';work.mkdir();started=time.monotonic();deadline=Deadline(2400);deadline.__enter__();r={'schema':cfg['schema'],'status':'failed','commands':{},'cases':{},'model_forward_executed':False,'quality_qualified':False,'retained_outputs':[]}
 def interrupted(sig,frame):raise SystemExit(128+sig)
 prior={sig:signal.signal(sig,interrupted)for sig in [signal.SIGTERM,signal.SIGINT]}
 def save():(out/'q6-qkv-batch.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,args,limit=600):
  r['phase']=label;r['commands'][label]={};save();print('PHASE '+label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,2400-(time.monotonic()-started)),r['commands'][label],cwd=ROOT)
  finally:save()
 def locate(name,digest):
  paths=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(paths)==1,'one exact completed '+name);return paths[0]
 def verify(path,record):require(path.stat().st_size==record['bytes']and sha(path)==record['sha256'],'immutable actual array '+path.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source');require(shutil.disk_usage(work).free>=6<<30,'bounded disk');physical=load('physicalcapacity',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');memory=physical.memory();require(memory['MemAvailable']>=4<<30,'bounded build RAM');flags=next(x.split(':',1)[1].split()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('flags'));require({'avx2','fma','f16c'}<=set(flags),'actual portable AVX2 profile');r['capacity']={'memory':memory,'scratch_free':shutil.disk_usage(work).free};tp=locate('form-text-trace.json',cfg['trace_sha256']);trace=json.loads(tp.read_text());cp=locate('combined-gdn-repack-ab.json',cfg['combined_sha256']);combined=json.loads(cp.read_text());require(trace['status']=='measured_neutral_form_text_trace'and trace['all170_scores_observer_neutral']and trace['analysis']['same_model_input_bits']and trace['analysis']['same_positions_first3_axes'],'actual neutral aligned trace');require(combined['status']=='measured_combined_gdn_repack_ab','completed native context');row=cfg['matrix'];inputs=[];expected=[]
  for graph,n in [(1,4),(2,140)]:
   values=[]
   for name,width in [('attn_norm-0',2560),('linear_attn_qkv_mixed-0',8192)]:
    key=f'on/graph-{graph}.{name}.bin';p=tp.parent/key;verify(p,trace['artifacts'][key]);a=np.fromfile(p,dtype='<f4').reshape(n,width);require(np.isfinite(a).all(),'finite actual source');values.append(a)
   inputs.append(values[0]);expected.append(values[1])
  dispatch=tp.parent/'on/source.tensor-dispatch.tsv';verify(dispatch,trace['artifacts']['on/source.tensor-dispatch.tsv']);rows=list(csv.DictReader(dispatch.open(),delimiter='\t'));selected=[x for x in rows if x['name']==row['physical_name']];require(len(selected)==1,'unique Q6 actual dispatch');d=selected[0];require([int(d['ne0']),int(d['ne1']),int(d['ne2']),int(d['ne3'])]==[2560,8192,1,1]and int(d['type'])==14 and d['buffer_type']=='CPU_Mapped'and d['extra']=='0','actual Q6 source mapped route');r['source_dispatch']=d
  nk='simple_form/on/prefill.first_chunk.layer0.qkv.f32';p=cp.parent/nk;verify(p,combined['artifacts'][nk]);native=np.fromfile(p,dtype='<f4').reshape(16,8192);permutation=base.row_map('qkv',8192);joined=np.concatenate(inputs);target=np.concatenate(expected)[:,permutation];require(joined.shape==(144,2560),'actual combined first144 source inputs');r['input_sha256']=[sha(tp.parent/f'on/graph-{g}.attn_norm-0.bin')for g in[1,2]]
  run('native-submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['native_ggml_source'],'native pin');up=work/'upstream'
  for label,args in [('up-init',['git','init',up]),('up-remote',['git','-C',up,'remote','add','origin','https://github.com/ggml-org/llama.cpp.git']),('up-fetch',['git','-C',up,'fetch','--depth','1','origin',cfg['upstream_source']]),('up-checkout',['git','-C',up,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',up,'rev-parse','HEAD'],text=True).strip()==cfg['upstream_source'],'upstream pin');programs={}
  for version,source,omp in [('native',ROOT/'ggml','OFF'),('upstream',up/'ggml','ON')]:
   for sgemm in ['ON','OFF']:
    label=version+'-'+sgemm;build=work/('build-'+label);run('configure-'+label,['cmake','-S',ROOT/'tools/q6-qkv-batch-probe','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_SOURCE='+str(source),'-DPROBE_OPENMP='+omp,'-DPROBE_LLAMAFILE='+sgemm]);cache=(build/'CMakeCache.txt').read_text().splitlines()
    for k,v in {'GGML_OPENMP':omp,'GGML_LLAMAFILE':sgemm,'GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_AVX512':'OFF','GGML_CPU_REPACK':'ON','GGML_BACKEND_DL':'OFF','GGML_BLAS':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}.items():require(f'{k}:BOOL={v}'in cache,'actual build profile '+label+'/'+k)
    shutil.copy2(build/'CMakeCache.txt',out/(label+'-CMakeCache.txt'));run('build-'+label,['cmake','--build',build,'--target','lighton4b-q6-qkv-batch-probe','--parallel','2']);programs[label]=build/'lighton4b-q6-qkv-batch-probe'
  zero=work/'zero-q6.bin';zero.write_bytes(bytes(cfg['matrix']['bytes']));zero_input=work/'zero-input.f32';np.zeros(2560,dtype='<f4').tofile(zero_input)
  for label,program in programs.items():
   for mode in ['borrowed','mapped']:
    prefix=work/'zero-output';run('synthetic-'+label+'-'+mode,[program,zero,zero_input,14,2560,8192,1,mode,prefix,4,1],30);require(not np.any(np.fromfile(str(prefix)+'.f32',dtype='<f4')),'actual zero Q6 synthetic control')
  zero.unlink();zero_input.unlink()
  weight=work/'qkv-source.bin';base.download_range(cfg,row,weight);r['whole_quantized_bytes']=row['bytes'];r['weight_sha256']=sha(weight);
  # Source4/source140 geometry and native mixed144 split16 are separate declared plans.
  plans=batch_plans(cfg['batches'])
  for label,program in programs.items():
   for threads in cfg['threads']:
    for plan,slices in plans.items():
     slices=slices;start=slices[0][0];end=slices[-1][1];chunk=max(b-a for a,b in slices);inp=work/'input.f32';joined[start:end].astype('<f4').tofile(inp);prefix=work/'projection';mode='mapped'if label.startswith('upstream')else'borrowed';tag=f'{label}-t{threads}-{plan}';run(tag,[program,weight,inp,14,2560,8192,end-start,mode,prefix,threads,chunk],120);meta=json.loads(Path(str(prefix)+'.json').read_text());require(meta['extra']==False and meta['buffer_type']==('CPU_Mapped'if mode=='mapped'else'<none>'),'actual no-repack Q6 route');value=np.fromfile(str(prefix)+'.f32',dtype='<f4').reshape(end-start,8192)[:,permutation];reference=target[:4]if plan=='source4'else target[4:]if plan=='source140'or plan.startswith('image140')else target;key=f'{label}/t{threads}/{plan}';r['cases'][key]={'metadata':meta,'vs_actual_source':base.metric(value,reference),'output_sha256':hashlib.sha256(value.tobytes()).hexdigest()}
     if threads==4 and plan in ['source140','mixed144-chunk16','image140-chunk16']:
      name=tag+'.f32';value.astype('<f4').tofile(out/name);r['retained_outputs'].append(name)
     if plan=='mixed144-chunk16':r['cases'][key]['first16_vs_actual_native']=base.metric(value[:16],native)
     save()
  require(r['cases']['upstream-ON/t4/source140']['vs_actual_source']['bit_exact']and r['cases']['upstream-ON/t4/source4']['vs_actual_source']['bit_exact'],'full actual source Q6 batch closure');require(r['cases']['native-ON/t4/mixed144-chunk16']['first16_vs_actual_native']['bit_exact'],'actual native first16 closure');r['actual_source_and_native_closures_exact']=True;r['status']='measured_source_fed_q6_batch_operators';r['limits']=['Fixed observed form inputs; operator-only, no decoded improvement claim.','Cross-version comparisons include declared OpenMP profile differences; LLAMAFILE toggles and threads are separately measured.']
 except BaseException as error:r['error_type']=type(error).__name__;r['error']=str(error);raise
 finally:
  deadline.close()
  for sig,handler in prior.items():signal.signal(sig,handler)
  r['seconds']=time.monotonic()-started;owned={n+'.log'for n in r['commands']}|{k+'-CMakeCache.txt'for k in ['native-ON','native-OFF','upstream-ON','upstream-OFF']}|set(r['retained_outputs']);r['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.iterdir()if p.is_file()and p.name in owned};save()
if __name__=='__main__':main()
