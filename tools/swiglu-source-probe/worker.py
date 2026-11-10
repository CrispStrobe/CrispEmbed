#!/usr/bin/env python3
"""Bounded actual-input SwiGLU attribution; no checkpoint/model forward."""
import hashlib,importlib.util,json,os,platform,shutil,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
spec=importlib.util.spec_from_file_location('physical',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
require=base.require;sha=base.sha
CFG_SHA='ab16a45e6c623a8ae50704121c37163b25063d3c8290dc9268de2c25a307fb13'
PROFILE={'BUILD_SHARED_LIBS':'ON','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_AVX512':'OFF','GGML_CPU_REPACK':'ON','GGML_OPENMP':'OFF','GGML_LLAMAFILE':'ON','GGML_BACKEND_DL':'OFF','GGML_BLAS':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
WIDTHS={'postnorm':2560,'gate':9216,'up':9216,'activation':9216,'down':2560}
def admit(c):
 require(hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()==CFG_SHA,'frozen operator protocol')
 require(c['schema']=='lighton4b-swiglu-source-probe/v1'and c['tokens']=={'source':[1,4],'off':[1,4,16],'on':[1,4,16]}and c['threads']==4 and c['build_threads']==2,'closed actual contexts/geometries')
 require(c['total_timeout_seconds']==1800 and not c['model_forward_allowed']and not c['full_f32_model_allowed'],'operator-only bound')
 require([(x['role'],x['type'],x['source_shape'])for x in c['weights']]==[('gate',12,[2560,9216]),('up',12,[2560,9216]),('down',14,[9216,2560])]and sum(x['bytes']for x in c['weights'])==45895680,'actual Q4/Q4/Q6 weights, never inferred from format name')
 require(all(x['sha256']==x['source_packed_sha256']and x['whole_rows_permuted']==0 and x['quant_blocks_unchanged']for x in c['weights']),'same whole original packed coordinates')
def stats(np,x,y):
 require(x.shape==y.shape and x.size and np.isfinite(x).all()and np.isfinite(y).all(),'same finite actual stage shape');d=x.astype('f8')-y.astype('f8');bad=np.flatnonzero(x.view('<u4')!=y.view('<u4'))
 return dict(elements=int(x.size),bit_exact=not len(bad),different_f32_words=int(len(bad)),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),reference_rms_magnitude=float(np.sqrt(np.mean(y.astype('f8')**2))),candidate_rms_magnitude=float(np.sqrt(np.mean(x.astype('f8')**2))),reference_max_abs=float(abs(y).max()))
def inventory(out,commands):
 owned={name+'.log'for name in commands}|{'native-CMakeCache.txt','upstream-CMakeCache.txt'}
 return {str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)}
def main():
 import numpy as np
 require(np.__version__=='2.3.5'and sys.byteorder=='little'and platform.system()=='Linux'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'pinned CPU worker')
 cfg=json.loads((ROOT/'tools/swiglu-source-probe/protocol.json').read_text());admit(cfg);out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'swiglu-probe';work.mkdir();started=time.monotonic();deadline=Deadline(1800);deadline.__enter__();r=dict(schema=cfg['schema'],status='failed',commands={},operator_compute_executed=False,model_forward_executed=False,full_f32_model_created=False,decoded_generated=False,heldout_used=False,quality_qualified=False)
 def interrupt(sig,frame):raise SystemExit(128+sig)
 handlers={s:signal.signal(s,interrupt)for s in [signal.SIGTERM,signal.SIGINT]}
 def save():(out/'swiglu-source-probe.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 def remaining():value=1800-(time.monotonic()-started);require(value>0,'whole operator deadline');return value
 env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def run(label,args,expected=0,limit=600):
  r['phase']=label;r['commands'][label]={};save();print('PHASE '+label,flush=True)
  try:
   try:command(args,out/(label+'.log'),min(limit,remaining()),r['commands'][label],env=env,cwd=ROOT)
   except subprocess.CalledProcessError as error:require(expected!=0 and error.returncode==expected,'expected closed loader exit');r['commands'][label]['status']='completed_expected_failure'
   else:require(expected==0,'loader must reject missing arguments')
  finally:save()
 def locate(name,digest):
  matches=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(matches)==1,'one exact retained '+name);return matches[0]
 def verify(p,row):require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'authenticated whole actual artifact '+p.name)
 def array(p,n,width):
  a=np.fromfile(p,dtype='<f4');require(a.size==n*width and np.isfinite(a).all(),'complete finite captured extent');return a.reshape(n,width)
 def compare(p,q):return stats(np,np.fromfile(p,dtype='<f4'),np.fromfile(q,dtype='<f4'))
 def fetch(row,target):
  start=row['source_offset'];size=row['bytes'];temp=target.with_suffix('.partial');count=0;h=hashlib.sha256()
  try:
   request=urllib.request.Request(cfg['source_url'],headers={'Range':f'bytes={start}-{start+size-1}'})
   with urllib.request.urlopen(request,timeout=min(120,remaining()))as response,temp.open('xb')as stream:
    require(response.status==206 and response.headers.get('Content-Range')==f"bytes {start}-{start+size-1}/{cfg['source_bytes']}",'exact donor whole-tensor range')
    while chunk:=response.read(1<<20):count+=len(chunk);require(count<=size,'bounded parameter acquisition');stream.write(chunk);h.update(chunk)
   require(count==size and h.hexdigest()==row['source_packed_sha256'],'original whole packed tensor hash');temp.rename(target)
  except BaseException:
   if temp.exists():temp.unlink()
   raise RuntimeError('bounded public parameter acquisition rejected: '+row['role'])from None
 try:
  run('source',['git','rev-parse','HEAD'],limit=30);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact probe source');run('cpu',['lscpu'],limit=30);flags=next(x.split(':',1)[1].split()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('flags'));require({'avx2','fma','f16c'}<=set(flags),'actual AVX2/FMA/F16C');r['capacity']=dict(memory=base.memory(),scratch=shutil.disk_usage(work).free,output=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=cfg['minimum_available_memory_bytes']and r['capacity']['scratch']>=cfg['minimum_scratch_bytes']and r['capacity']['output']>=cfg['minimum_output_bytes'],'bounded CPU operator resources')
  for name,digest in cfg['native_sources'].items():require(sha(ROOT/name)==digest,'actual native scalar activation source '+name)
  for name,key in [('main.cpp','adapter_cpp_sha256'),('CMakeLists.txt','adapter_cmake_sha256')]:require(sha(ROOT/'tools/swiglu-source-probe'/name)==cfg[key],'closed reference adapter')
  cp=locate('text-trace.json',cfg['capture_receipt_sha256']);capture=json.loads(cp.read_text());rp=locate('text-trace-analysis.json',cfg['reader_receipt_sha256']);reader=json.loads(rp.read_text());ap=locate('gdn-full-ab.json',cfg['ab_receipt_sha256']);ab=json.loads(ap.read_text());require(capture['all282_scores_observer_neutral']and reader['status']=='measured_neutral_aligned_text_trace'and reader['all_producer_artifacts_rehashed']==915 and reader['analysis']['same_model_input_bits']and reader['analysis']['same_positions_first3_axes'],'actual neutral aligned source prefix');require(ab['status']=='measured_gdn_full_ab'and ab['cases']['simple_table']['off_all_rows_stages_exact']and not ab['quality_qualified'],'completed native OFF/ON controls, decoded negative retained')
  pp=ap.parent/'physical-pack.json';verify(pp,cfg['pack_receipt']);pack=json.loads(pp.read_text());require(pack['candidate_sha256']==cfg['candidate_sha256']and not pack['requantization_executed'],'same actual quantized weights')
  for row in cfg['weights']:
   selected=[v for v in pack['tensors']if v.get('physical_name')==row['physical_name']];require(len(selected)==1 and all(selected[0][k]==row[k]for k in ['source_offset','bytes','type','source_shape','source_packed_sha256','sha256','whole_rows_permuted','quant_blocks_unchanged']),'actual whole tensor map and coordinate proof')
  full={};widths=WIDTHS
  for dataset in ['source','off','on']:
   full[dataset]={};records=cfg['source_inputs']if dataset=='source'else cfg['native_inputs'][dataset];parent=cp.parent if dataset=='source'else ap.parent;n=4 if dataset=='source'else 16
   for key,row in records.items():
    role='postnorm'if key=='post_norm'else key;path=parent/row['file'];verify(path,row);require((capture['artifacts']if dataset=='source'else ab['artifacts'])[row['file']]=={k:row[k]for k in ['bytes','sha256']},'receipt-to-protocol actual array identity');full[dataset][role]=array(path,n,widths[role])
  inputs=out/'inputs';inputs.mkdir();cases={}
  for dataset,tokens in cfg['tokens'].items():
   for n in tokens:
    label=f'{dataset}{n}';folder=inputs/label;folder.mkdir();cases[label]=(dataset,n,folder)
    for role,a in full[dataset].items():a[:n].astype('<f4').tofile(folder/(role+'.f32'))
  upstream=work/'upstream'
  for label,args in [('up-init',['git','init',upstream]),('up-remote',['git','-C',upstream,'remote','add','origin','https://github.com/ggml-org/llama.cpp']),('up-fetch',['git','-C',upstream,'fetch','--depth','1','origin',cfg['upstream_commit']]),('up-checkout',['git','-C',upstream,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()==cfg['upstream_commit'],'exact upstream revision')
  for name,digest in cfg['source_functions'].items():require(sha(upstream/name)==digest,'actual upstream FFN blueprint '+name)
  run('native-submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['native_ggml_commit'],'exact native GGML');bins={}
  for kind,source in [('native',ROOT/'ggml'),('upstream',upstream/'ggml')]:
   build=work/(kind+'-build');run(kind+'-configure',['cmake','-S',ROOT/'tools/swiglu-source-probe','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_SOURCE='+str(source),*[f'-D{k}={v}'for k,v in PROFILE.items()]]);cache=(build/'CMakeCache.txt').read_text();require(all(f'{k}:BOOL={v}'in cache for k,v in PROFILE.items()),'actual matched CPU profile');shutil.copyfile(build/'CMakeCache.txt',out/(kind+'-CMakeCache.txt'));run(kind+'-build',['cmake','--build',build,'--target','swiglu-source-probe','--parallel','2']);binary=build/'swiglu-source-probe';run(kind+'-noargs',[binary],1,30);require('MODE ROLE TOKENS INPUT_DIR WEIGHTS_DIR OUTPUT_FILE'in(out/(kind+'-noargs.log')).read_text(),'actual noargs loader contract');run(kind+'-controls',[binary,'--controls'],limit=60);require('SWIGLU_SOURCE_CONTROLS PASS'in(out/(kind+'-controls.log')).read_text(),'actual synthetic graph and backend execution');bins[kind]=binary
  r['operator_compute_executed']=True;r['activation']={};outputs=out/'cases';outputs.mkdir()
  for label,(dataset,n,folder)in cases.items():
   dest=outputs/label;dest.mkdir();r['activation'][label]={}
   for variant,kind,mode in [('scalar','native','scalar'),('native-ggml','native','ggml'),('upstream-ggml','upstream','ggml')]:
    path=dest/(variant+'.f32');run(label+'-'+variant,[bins[kind],mode,'activation',n,folder,work,path],limit=60);r['activation'][label][variant]=compare(path,folder/'activation.f32')
   r['activation'][label]['native_ggml_vs_upstream']=compare(dest/'native-ggml.f32',dest/'upstream-ggml.f32');r['activation'][label]['scalar_vs_upstream']=compare(dest/'scalar.f32',dest/'upstream-ggml.f32');save()
  require(r['activation']['source4']['upstream-ggml']['bit_exact']and all(r['activation'][case]['scalar']['bit_exact']for case in ['off16','on16']),'actual upstream fused and native scalar activation closures before interpretation')
  weights=work/'weights';weights.mkdir();r['phase']='three-whole-parameter-ranges';save()
  for row in cfg['weights']:fetch(row,weights/row['file'])
  r['parameter_bytes']=cfg['parameter_bytes'];r['projection_closures']={}
  for label,kind,route in [('source4','upstream','preferred'),('off16','native','raw'),('on16','native','raw')]:
   dataset,n,folder=cases[label];dest=outputs/label
   for role in ['gate','up']:
    path=dest/(kind+'-'+role+'.f32');run(label+'-'+kind+'-'+role,[bins[kind],route,role,n,folder,weights,path]);r['projection_closures'][label+'-'+role]=compare(path,folder/(role+'.f32'))
   path=dest/(kind+'-actual-down.f32');run(label+'-'+kind+'-actual-down',[bins[kind],'raw','down',n,folder,weights,path]);r['projection_closures'][label+'-down']=compare(path,folder/'down.f32');save()
  require(all(v['bit_exact']for v in r['projection_closures'].values()),'same original weights must reproduce whole actual source/native projections')
  r['down_interventions']={}
  for label,(dataset,n,folder)in cases.items():
   dest=outputs/label;r['down_interventions'][label]={}
   for variant in ['scalar','native-ggml','upstream-ggml']:
    inp=dest/variant;inp.mkdir();shutil.copyfile(dest/(variant+'.f32'),inp/'activation.f32');entry={};r['down_interventions'][label][variant]=entry
    for kind in ['native','upstream']:
     for mode in ['raw','prequantized','codes']:
      path=inp/(kind+'-'+mode+('.bin'if mode=='codes'else'.f32'));run(label+'-'+variant+'-'+kind+'-'+mode,[bins[kind],mode,'down',n,inp,weights,path])
     entry[kind+'_forced_q8_output_closure']=compare(inp/(kind+'-raw.f32'),inp/(kind+'-prequantized.f32'))
    entry['native_vs_upstream_same_activation']=compare(inp/'native-raw.f32',inp/'upstream-raw.f32')
   for variant in ['scalar','native-ggml']:
    inp=dest/variant;ref=dest/'upstream-ggml';entry=r['down_interventions'][label][variant]
    for kind in ['native','upstream']:
     entry[kind+'_same_weight_output_vs_source_activation']=compare(inp/(kind+'-raw.f32'),ref/(kind+'-raw.f32'))
     a=np.fromfile(inp/(kind+'-codes.bin'),dtype=np.dtype([('d','<f4'),('qs','i1',(256,)),('bsums','<i2',(16,))]));b=np.fromfile(ref/(kind+'-codes.bin'),dtype=a.dtype);require(a.shape==b.shape and a.size==n*36 and np.isfinite(a['d']).all()and np.isfinite(b['d']).all(),'whole finite public Q8_K converter vectors');entry[kind+'_public_converter_code_differences']=dict(scope='Public converter output; internal codes not directly observed. Interpret output equivalence only where forced-Q8 closure passes.',blocks=int(a.size),different_scales=int(np.count_nonzero(a['d'].view('<u4')!=b['d'].view('<u4'))),different_qs=int(np.count_nonzero(a['qs']!=b['qs'])),different_bsums=int(np.count_nonzero(a['bsums']!=b['bsums'])))
   save()
  r['status']='measured_same_input_swiglu_operators';r['limits']=['Observed table layer0 only; no tensor/layer selection, heldout or model inference.','Source4 and native16 contexts are separately authenticated; causal operator comparisons use identical gate/up arrays.','Q6 down is original quantized storage, not a Q4 repack substitution.','Q8_K vectors are public-converter interventions; actual internal buffers are not captured. Forced-Q8 output closure is measured per case/backend.','Prior hybrid decoded regression remains binding. Numerical operator equality is not decoded-quality proof.']
 except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  deadline.close()
  for sig,handler in handlers.items():signal.signal(sig,handler)
  r['seconds']=time.monotonic()-started;r['artifacts']=inventory(out,r['commands']);save()
if __name__=='__main__':main()
