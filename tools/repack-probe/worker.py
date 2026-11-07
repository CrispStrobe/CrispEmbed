#!/usr/bin/env python3
"""Actual captured inputs and authenticated whole quantized matrices, operator only."""
import hashlib,importlib.util,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command

def require(ok,why):
 if not ok:raise RuntimeError(why)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def row_map(role,rows):
 def p(h):return 16*(h%2)+h//2
 if role=='qkv':return list(range(4096))+[4096+128*p(i//128)+i%128 for i in range(4096)]
 if role=='z':return [128*p(i//128)+i%128 for i in range(rows)]
 if role in['a','b']:return [p(i)for i in range(rows)]
 if role=='gate':return list(range(rows))
 raise RuntimeError('closed matrix role')
ROWS_SHA='bacc606e79697828d923543b7cc66080540f7372548ac46fd915875ef7cefe55'
def admit(cfg):
 require(cfg['schema']=='lighton4b-source-repack-operator/v1'and cfg['tokens']==[1,4,16]and cfg['threads']==4 and cfg['worker_seconds']==1800,'closed operator scope')
 require(cfg['transport_audit_sha256']=='920b4d5db276414bdd743fcc80a7ed1033ed1c0ca3b76859ef922aa667de9dee','frozen transport audit')
 require(cfg['actual_receipt_sha256']=='36a9b14be3ace197e3a30aeaec20454eb4ab58587b2566fcc859da87b08013b3'and cfg['native_ggml_source']=='3a27b6c665b191cb230be367db5c52d81dd3e2ac'and cfg['upstream_source']=='6c01da6278ab6c69ed67d9124806b50d2b299ba7','frozen actual/source pins')
 require([r['role']for r in cfg['matrices']]==['qkv','z','a','b','gate']and [r['type']for r in cfg['matrices']]==[14,12,12,12,12],'closed actual matrix types')
 require(hashlib.sha256(json.dumps(cfg['matrices'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==ROWS_SHA,'immutable authenticated matrix manifest')
 require(cfg['source_model_sha256']=='b1dd753e6f7d7d7ee3e86134337dd1ae7b08e2ed181989ae8264aa670d8ccc26'and cfg['source_model_bytes']==2708804576 and cfg['source_model_url']=='https://huggingface.co/richardyoung/LightOnOCR-3-4B-GGUF/resolve/8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01/LightOnOCR-3-4B-Q4_K_M.gguf','immutable source file identity')
 require(sum(r['bytes']for r in cfg['matrices'])==36464640 and cfg['maximum_transfer_bytes']==40<<20,'bounded whole matrices')
 for row in cfg['matrices']:
  width,rows=row['source_shape'];require(width==2560 and sorted(row_map(row['role'],rows))==list(range(rows))and row['source_offset']>=0 and row['source_offset']+row['bytes']<=cfg['source_model_bytes'],'whole valid row mapping/range')
def selected_dispatch(rows,matrices):
 wanted={row['physical_name']for row in matrices};chosen=[row for row in rows if row['name']in wanted]
 require(len(chosen)==len(wanted)and {row['name']for row in chosen}==wanted,'unique selected actual tensor dispatch names')
 return {row['name']:row for row in chosen}
def download_range(cfg,row,path):
 start=row['source_offset'];end=start+row['bytes']-1;request=urllib.request.Request(cfg['source_model_url']+f'?download=true&repack_range={start}-{end}',headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
 with urllib.request.urlopen(request,timeout=120)as response,path.open('xb')as f:
  require(response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{cfg["source_model_bytes"]}','exact immutable whole-tensor HTTP range');total=0;h=hashlib.sha256()
  while b:=response.read(1<<20):
   total+=len(b);require(total<=row['bytes'],'bounded range response');h.update(b);f.write(b)
 require(total==row['bytes']and h.hexdigest()==row['source_packed_sha256'],'whole source matrix SHA')
def metric(candidate,reference):
 import numpy as np
 x=np.asarray(candidate,dtype='<f4');y=np.asarray(reference,dtype='<f4');require(x.shape==y.shape and x.size>0 and np.isfinite(x).all()and np.isfinite(y).all(),'finite equal projections');d=x.astype('f8')-y
 return {'bit_exact':x.tobytes()==y.tobytes(),'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(abs(d).max()),'reference_max_abs':float(abs(y).max())}
def main():
 import numpy as np
 require(Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*'))and sys.byteorder=='little','Kaggle CPU/little endian')
 cfgpath=ROOT/'docs/lighton4b-source-repack-operator-v1.json';cfg=json.loads(cfgpath.read_text());admit(cfg);out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'repack-operator';work.mkdir();started=time.monotonic();deadline=Deadline(1800);deadline.__enter__();r={'schema':cfg['schema'],'status':'failed','commands':{},'operators':{},'quality_qualified':False,'model_forward_executed':False,'source_batching_limit':'N4 matches text-prefix geometry; N1/N16 are component replays of captured rows, not actual source graph batches.'}
 def save():(out/'repack-operator.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,args,env=None,timeout=600):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(timeout,1800-(time.monotonic()-started)),r['commands'][label],env=env,cwd=ROOT)
  finally:save()
 def verify(path,row):require(path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'actual retained artifact '+path.name)
 try:
  run('source-pin',['git','rev-parse','HEAD']);r['source_sha']=(out/'source-pin.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact controller source')
  s=importlib.util.spec_from_file_location('base',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');base=importlib.util.module_from_spec(s);s.loader.exec_module(base);r['preflight']={'memory':base.memory(),'scratch_free':shutil.disk_usage(work).free,'load':os.getloadavg()};require(r['preflight']['memory']['MemAvailable']>=4<<30 and r['preflight']['scratch_free']>=8<<30,'bounded CPU build capacity')
  receipts=[p for p in Path('/kaggle/input').rglob('physical-f16-table512.json')if p.is_file()and p.stat().st_size<4<<20 and sha(p)==cfg['actual_receipt_sha256']];require(len(receipts)==1,'one exact completed actual input receipt');rp=receipts[0];actual=json.loads(rp.read_text());require(actual['status']=='measured_extended_table'and actual['source_sha']=='f789609c2f1ff1a9527c8d02545d096ed10446b0'and actual['old_cap256_prefix_anchors_exact']and actual['source_observer_full_neutral'],'actual native/source closure');r['actual_receipt_sha256']=sha(rp)
  dispatch=actual['arms']['upstream-on']['tensor_dispatch'];observed=selected_dispatch(dispatch,cfg['matrices'])
  prepared=[]
  for row in cfg['matrices']:
   x=rp.parent/row['input'];y=rp.parent/row['output'];verify(x,actual['artifacts'][row['input']]);verify(y,actual['artifacts'][row['output']]);width,rows=row['source_shape'];data=np.fromfile(x,dtype='<f4').reshape(16,width);expected=np.fromfile(y,dtype='<f4').reshape(16,rows);require(np.isfinite(data).all()and np.isfinite(expected).all(),'finite actual captured input/output');metadata=observed[row['physical_name']];require([int(metadata['ne0']),int(metadata['ne1'])]==row['source_shape']and int(metadata['ne2'])==int(metadata['ne3'])==1 and int(metadata['type'])==row['type']and metadata['buffer_type']==('CPU_Mapped'if row['type']==14 else'CPU_REPACK')and metadata['extra']==('0'if row['type']==14 else'1'),'actual loaded source buffer route');prepared.append((row,data,expected))
  run('native-submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['native_ggml_source'],'native GGML pin')
  upstream=work/'upstream'
  for label,args in [('up-init',['git','init',upstream]),('up-remote',['git','-C',upstream,'remote','add','origin','https://github.com/ggml-org/llama.cpp.git']),('up-fetch',['git','-C',upstream,'fetch','--depth','1','origin',cfg['upstream_source']]),('up-checkout',['git','-C',upstream,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()==cfg['upstream_source'],'upstream code pin');programs={}
  for leg,ggml,omp in [('native',ROOT/'ggml','OFF'),('upstream',upstream/'ggml','ON')]:
   build=work/('build-'+leg);run('configure-'+leg,['cmake','-S',ROOT/'tools/repack-probe','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_SOURCE='+str(ggml),'-DPROBE_OPENMP='+omp]);cache=(build/'CMakeCache.txt').read_text().splitlines()
   for key,value in {'BUILD_SHARED_LIBS':'ON','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_AVX512':'OFF','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_OPENMP':omp,'GGML_BACKEND_DL':'OFF','GGML_BLAS':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}.items():require(f'{key}:BOOL={value}'in cache,'actual pinned profile '+leg+'/'+key)
   shutil.copy(build/'CMakeCache.txt',out/(leg+'-CMakeCache.txt'));run('build-'+leg,['cmake','--build',build,'--target','lighton4b-repack-probe','--parallel','2']);programs[leg]=build/'lighton4b-repack-probe'
  r['whole_matrix_transfer_bytes']=0
  for row,data,expected in prepared:
   role=row['role'];width,rows=row['source_shape'];raw=work/(role+'-source.bin');download_range(cfg,row,raw);r['whole_matrix_transfer_bytes']+=row['bytes'];require(r['whole_matrix_transfer_bytes']<=cfg['maximum_transfer_bytes'],'total range transfer bound');permutation=row_map(role,rows);native_bytes=np.frombuffer(raw.read_bytes(),dtype='u1').reshape(rows,-1)[permutation].copy().tobytes();require(hashlib.sha256(native_bytes).hexdigest()==row['sha256'],'exact whole-row native transport SHA');nf=work/(role+'-native.bin');nf.write_bytes(native_bytes);values={};r['operators'][role]={'source_tensor_sha256':sha(raw),'native_tensor_sha256':sha(nf),'actual_input_sha256':sha(rp.parent/row['input']),'actual_source_buffer':observed[row['physical_name']],'cases':{}}
   for n in cfg['tokens']:
    inp=work/(role+'-'+str(n)+'.f32');data[:n].astype('<f4').tofile(inp)
    for leg,program in programs.items():
     for order,weight in [('source',raw),('native',nf)]:
      for mode in ['borrowed','mapped']+(['repack']if row['type']==12 else[]):
       label=f'{role}-{n}-{leg}-{order}-{mode}';prefix=out/label;run(label,[program,weight,inp,row['type'],width,rows,n,mode,prefix]);meta=json.loads(Path(str(prefix)+'.json').read_text());require(meta['extra']==(mode=='repack')and meta['buffer_type']==({'borrowed':'<none>','mapped':'CPU_Mapped','repack':'CPU_REPACK'}[mode]),'actual operand buffer metadata');v=np.fromfile(str(prefix)+'.f32',dtype='<f4').reshape(n,rows);require(np.isfinite(v).all(),'finite result');v=v[:,permutation]if order=='source'else v;values[(n,leg,order,mode)]=v.copy();r['operators'][role]['cases'][label]={'metadata':meta,'output_sha256':sha(Path(str(prefix)+'.f32'))}
    if n==16:require(metric(values[n,'native','native','borrowed'],expected)['bit_exact'],'actual N16 main native borrowed projection closure')
    comparisons={'raw_vs_N16_main_prefix':metric(values[n,'native','native','borrowed'],expected[:n])}
    for leg in programs:
     for order in ['source','native']:require(metric(values[n,leg,order,'mapped'],values[n,leg,order,'borrowed'])['bit_exact'],'mapped wrapper does not change raw arithmetic')
    comparisons['raw_versions']=metric(values[n,'native','native','borrowed'],values[n,'upstream','native','borrowed'])
    preferred='repack'if row['type']==12 else'mapped'
    comparisons['preferred_versions']=metric(values[n,'native','source',preferred],values[n,'upstream','source',preferred]);comparisons['native_storage_vs_source_preferred']=metric(values[n,'native','native',preferred],values[n,'upstream','source',preferred]);comparisons['source_repack_vs_raw']=metric(values[n,'upstream','source',preferred],values[n,'upstream','source','borrowed']);r['operators'][role].setdefault('comparisons',{})[str(n)]=comparisons;save()
  r['status']='measured_same_input_operator_only'
 except BaseException as error:r['error_type']=type(error).__name__;r['error']=str(error);raise
 finally:
  deadline.close();r['elapsed_seconds']=time.monotonic()-started;r['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.iterdir()if p.is_file()and p.name!='repack-operator.json'};save()
if __name__=='__main__':main()
