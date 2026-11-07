#!/usr/bin/env python3
"""Reconstructed source-prefix operators only, no graph/decode/quality claim."""
import hashlib,io,json,os,shutil,subprocess,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def main():
 require(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir(),'Kaggle CPU only');require(not list(Path('/dev').glob('nvidia[0-9]*')),'CPU only')
 import numpy as np
 require(np.__version__=='2.3.5','numpy pin')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'prefix-primitives';scratch.mkdir();report={'schema':'lighton4b-reconstructed-prefix-primitives/v1','status':'failed','actual_source_graph_captured':False,'actual_native_trace_compared':False,'model_forward_executed':False,'quality_qualified':False,'ggml_profile':{'llamafile':True,'avx2':True,'avx512':False,'blas':False,'threads':4},'scope':'reconstructed4-tokenprefix only','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 def run(label,args):
  report['phase']=label
  with(out/(label+'.log')).open('w')as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
 def fetch(url,path,start,size,expected,total):
  request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{start+size-1}'})
  h=hashlib.sha256();count=0
  with urllib.request.urlopen(request,timeout=120)as response,path.open('xb')as f:
   require(response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{start+size-1}/{total}','exact bounded range required')
   while b:=response.read(1048576):count+=len(b);require(count<=size,'range exceeded');h.update(b);f.write(b)
  require(count==size and h.hexdigest()==expected,'tensor complete range SHA')
 def diff(a,b):
  x=np.asarray(a,dtype=np.float64);y=np.asarray(b,dtype=np.float64);require(x.shape==y.shape and np.isfinite(x).all()and np.isfinite(y).all(),'same finite shape');d=x-y
  return {'bit_exact':np.asarray(a,dtype='<f4').tobytes()==np.asarray(b,dtype='<f4').tobytes(),'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.abs(d).max()),'reference_max_abs':float(np.abs(y).max())}
 try:
  require(report['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source pin')
  available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024;limit=Path('/sys/fs/cgroup/memory.max').read_text().strip();used=int(Path('/sys/fs/cgroup/memory.current').read_text());available=min(available,int(limit)-used)if limit!='max'else available
  report['preflight']={'available_bytes':available,'disk_free_bytes':shutil.disk_usage(scratch).free,'load':os.getloadavg(),'cpuinfo':Path('/proc/cpuinfo').read_text()};require(available>=2*2**30 and report['preflight']['disk_free_bytes']>=4*2**30,'bounded2GiBRAM4GiBdisk')
  cfg=json.loads((ROOT/'tools/physical-primitive-probe/weights-v1.json').read_text());report['weights_manifest_sha256']=sha(ROOT/'tools/physical-primitive-probe/weights-v1.json')
  up=scratch/'llama';run('up-init',['git','init',up]);run('up-remote',['git','-C',up,'remote','add','origin','https://github.com/ggml-org/llama.cpp']);run('up-fetch',['git','-C',up,'fetch','--depth','1','origin',cfg['upstream_commit']]);run('checkout',['git','-C',up,'checkout','--detach','FETCH_HEAD']);require(subprocess.check_output(['git','-C',up,'rev-parse','HEAD'],text=True).strip()==cfg['upstream_commit'],'upstream pin')
  run('native-submodule',['git','-C',ROOT,'submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['native_ggml_commit'],'native ggml pin')
  import sys
  sys.path.insert(0,str(up/'gguf-py'));from gguf.quants import dequantize;from gguf import GGMLQuantizationType
  prompturl='https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/9dab2f1d96ddab8cb24f0b492d7ca0aeafe4d1e8/tests/results/qwen35-ocr/lightonocr3-4b-independent-llama/Q4_K_M-simple_form/expected-prompt.ids'
  with urllib.request.urlopen(prompturl,timeout=60)as response:raw=response.read(4097)
  require(hashlib.sha256(raw).hexdigest()=='086d3459734320111e7ed988ebc483e19e4c2af0c1d7318320fd38a7013638b1','full prompt ID sourcehash');ids=list(map(int,raw.split()));require(len(ids)==154 and ids[:4]==[248045,846,198,248053],'actual prefixIDs');buf=io.BytesIO();np.save(buf,np.asarray([ids],dtype='<i8'),allow_pickle=False);require(hashlib.sha256(buf.getvalue()).hexdigest()=='6f1b0a033c6d381c877956b6fbd215d8be4515d9a4ec733ad302b47a1743203d','original fullpromptnumpyhash')
  rows={}
  for index,row in enumerate(cfg['tensors']):
   p=scratch/('weights-'+str(index));fetch(cfg['source_url'],p,row['source_offset'],row['bytes'],row['source_packed_sha256'],cfg['source_bytes']);rows[row['physical_name']]=(row,p)
  embedding=rows['token_embd.weight'][1];prefix=[]
  with embedding.open('rb')as f:
   for token in ids[:4]:f.seek(token*2100);prefix.append(dequantize(np.frombuffer(f.read(2100),dtype=np.uint8),GGMLQuantizationType.Q6_K).reshape(2560))
  x=np.asarray(prefix,dtype='<f4');x.tofile(out/'reconstructed-prefix.f32');report['reconstructed_input_sha256']=sha(out/'reconstructed-prefix.f32');embedding.unlink()
  binaries={}
  for tag,source in [('native',ROOT/'ggml'),('upstream',up/'ggml')]:
   build=scratch/(tag+'-build');run(tag+'-configure',['cmake','-S',ROOT/'tools/physical-primitive-probe','-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_SOURCE='+str(source)]);run(tag+'-build',['cmake','--build',build,'--target','physical-primitive','--parallel','2']);binaries[tag]=build/'physical-primitive';shutil.copy(build/'CMakeCache.txt',out/(tag+'-CMakeCache.txt'))
  build=scratch/'runtime-build';run('runtime-configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_AVX2=ON','-DGGML_AVX512=OFF','-DGGML_BLAS=OFF','-DGGML_LLAMAFILE=ON','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  run('runtime-build',['cmake','--build',build,'--target','test-lighton4b-physical-native-ops','test-lighton4b-physical-runtime-primitive','lighton4b-native-document','test-qwen35-stream-projection','test-lighton4b-q6-tied-head','--parallel','2'])
  for target in ('test-lighton4b-physical-native-ops','test-qwen35-stream-projection','test-lighton4b-q6-tied-head'):run(target,[build/target])
  binaries['runtime']=build/'test-lighton4b-physical-runtime-primitive';shutil.copy(build/'CMakeCache.txt',out/'runtime-CMakeCache.txt')
  for tag in ('native','upstream','runtime'):require('GGML_LLAMAFILE:BOOL=ON' in (out/(tag+'-CMakeCache.txt')).read_text(),'same explicit LLAMAFILE profile')
  gamma=rows['blk.0.attn_norm.weight'][1];report['norm']={}
  for tag,binary in binaries.items():run(tag+'-norm',[binary,'norm','f32',1,4,gamma,out/'reconstructed-prefix.f32',out/(tag+'-norm.f32')])
  require((out/'runtime-norm.f32').read_bytes()==(out/'upstream-norm.f32').read_bytes(),'runtime/upstream norm bit identity')
  report['norm']['backend_difference']=diff(np.fromfile(out/'native-norm.f32',dtype='<f4'),np.fromfile(out/'upstream-norm.f32',dtype='<f4'))
  gamma_values=np.fromfile(gamma,dtype='<f4');scalar=x.copy()
  for t in range(4):
   total=np.float32(0)
   for value in x[t]:total=np.float32(total+np.float32(value*value))
   scale=np.float32(1)/np.sqrt(np.float32(total/np.float32(2560))+np.float32(1e-6));scalar[t]=(x[t]*scale)*gamma_values
  scalar.tofile(out/'reconstructed-native-scalar-norm.f32');report['norm']['scalar_vs_native_ggml']=diff(scalar.reshape(-1),np.fromfile(out/'native-norm.f32',dtype='<f4'));report['norm']['scalar_vs_upstream']=diff(scalar.reshape(-1),np.fromfile(out/'upstream-norm.f32',dtype='<f4'));report['projections']={}
  for name in ('attn_qkv','attn_gate','ssm_alpha','ssm_beta'):
   row,p=rows['blk.0.'+name+'.weight'];kind={0:'f32',12:'q4_k',14:'q6_k'}[row['type']];n=row['source_shape'][1];key={}
   for inputname in ('reconstructed-native-scalar-norm','upstream-norm'):
    for tag,binary in binaries.items():
     for mode in ('direct','dequant'):
      label=name+'-'+inputname+'-'+tag+'-'+mode;run(label,[binary,mode,kind,n,4,p,out/(inputname+'.f32'),out/(label+'.f32')])
    load=lambda tag,mode:np.fromfile(out/(name+'-'+inputname+'-'+tag+'-'+mode+'.f32'),dtype='<f4')
    require(np.array_equal(load('runtime','direct').view('<u4'),load('upstream','direct').view('<u4')),'runtime retainedquant vsupstream exact operator')
    require(np.array_equal(load('runtime','dequant').view('<u4'),load('upstream','dequant').view('<u4')),'runtime gate-off vsupstream dequant exact operator')
    key[inputname]={'runtime_direct_exact':True,'runtime_dequant_exact':True,'backend_direct':diff(load('native','direct'),load('upstream','direct')),'backend_dequant':diff(load('native','dequant'),load('upstream','dequant')),'upstream_direct_vs_dequant':diff(load('upstream','direct'),load('upstream','dequant'))}
   report['projections'][name]=key
  report['binary_sha256']={k:sha(v)for k,v in binaries.items()};report['status']='measured_operator_only'
 except Exception as error:
  report['error_type']=type(error).__name__;report['error']=str(error);raise
 finally:(out/'prefix-primitives.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
