#!/usr/bin/env python3
"""Retained first-block rotary/GELU attribution, no model or runtime correction."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[2]
CORRECT_ROTARY=True
def main():
 dispatch={'ATEN_CPU_CAPABILITY':'avx2','ONEDNN_MAX_CPU_ISA':'AVX2','DNNL_MAX_CPU_ISA':'AVX2','MKL_ENABLE_INSTRUCTIONS':'AVX2'};os.environ.update(dispatch)
 import numpy as np
 import torch
 from torch.nn import functional as F
 from torch.utils.cpp_extension import include_paths,library_paths
 if os.environ.get('GITHUB_ACTIONS')!='true'or torch.__version__!='2.13.0+cpu'or np.__version__!='2.3.5':raise RuntimeError('hosted original pins required')
 spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'boundary-probe';scratch.mkdir();stages=out/'stages';stages.mkdir();build=scratch/'build'
 report={'schema':'lighton4b-vision-boundary-probe/v1','status':'failed','model_downloaded':False,'model_forward_executed':False,'full_model_forward_executed':False,'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cpu_dispatch':dispatch,'torch_git_version':torch.version.git_version}
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def compare(a,b):
  a=np.asarray(a,dtype='<f4').reshape(-1);b=np.asarray(b,dtype='<f4').reshape(-1);r=comparison.compare(a,b);r['bit_different_values']=int(np.count_nonzero(a.view('<u4')!=b.view('<u4')));r['bit_exact']=r['bit_different_values']==0;return r
 def run(name,args):
  report['last_phase']=name
  with(out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  os.environ['CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2']='0';os.environ['CRISPEMBED_LIGHTON4B_VISION_BF16_GELU_AVX2']='0'
  # Compile verbatim production expression blocks, preserving their operation order.
  text=(ROOT/'src/lighton4b_vision_bf16.cpp').read_text();start=text.index('    const bool source_rotary =',text.index('output.rotary_sin.resize'));end=text.index('    output.embedded = output.patch;',start);rotary=text[start:end]
  start=text.index('        const bool source_gelu =');end=text.index('        round_bf16(projected);',start);activation=text[start:end]
  generated='static void production_rotary(const vision_config &c,int tokens,int d,const std::vector<int>&rows,const std::vector<int>&columns,vision_outputs &output) {\n'+rotary+'}\nstatic void production_activation(std::vector<float>&projected) {\n'+activation+'}\n'
  (scratch/'vision_boundary_expressions.inc').write_text(generated);report['production_expressions']={'rotary_sha256':hashlib.sha256(rotary.encode()).hexdigest(),'activation_sha256':hashlib.sha256(activation.encode()).hexdigest(),'runtime_sha256':sha(ROOT/'src/lighton4b_vision_bf16.cpp'),'verbatim_extracted':True}
  args=['c++','-O2','-std=c++17','-mavx2','-mfma','-DCPU_CAPABILITY=AVX2','-DCPU_CAPABILITY_AVX2','-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),'-fno-math-errno','-fno-trapping-math','-ffp-contract=off'];args+=['-I'+p for p in include_paths()];args +=[ROOT/'tests/lighton4b_gelu_vector_source.cpp','-o',scratch/'source-gelu']
  for p in library_paths():args+=['-L'+p,'-Wl,-rpath,'+p]
  for contraction in ('off','fast'):
   variant=[v.replace('-ffp-contract=off','-ffp-contract='+contraction)if isinstance(v,str)else v for v in args];variant=[scratch/('source-gelu-'+contraction)if v==scratch/'source-gelu'else v for v in variant];run('compile-source-gelu-'+contraction,variant+['-ltorch_cpu','-lc10'])
  trig_args=[v for v in args];trig_args=[ROOT/'tests/lighton4b_trig_vector_source.cpp'if v==ROOT/'tests/lighton4b_gelu_vector_source.cpp'else scratch/'source-trig'if v==scratch/'source-gelu'else v for v in trig_args]
  run('compile-source-trig',trig_args+['-ltorch_cpu','-lc10'])
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DLIGHTON4B_BOUNDARY_GENERATED_DIR='+str(scratch)])
  run('build',['cmake','--build',build,'--target','lighton4b-vision-boundary-probe','--parallel','2'])
  item=comparison.api('actions/artifacts/13430388075')
  if item['workflow_run']['id']!=42056882067 or item['expired']or item['size_in_bytes']!=238134017:raise ValueError('retained source identity differs')
  archive=scratch/'source.zip'
  with archive.open('wb')as f:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/13430388075/zip'],stdout=f,check=True)
  if sha(archive)!='1e45293758c7c914da4d40d8c96bb4a67a3b3af4a33cd4eaab50a4db201f264b':raise ValueError('retained source ZIP differs')
  with zipfile.ZipFile(archive)as z:
   receipt=json.loads(z.read('vision-layer0.json'))
   if not receipt['original_control_bit_exact']:raise ValueError('original source control not exact')
   for filename,row in receipt['artifacts'].items():
    if not(filename.startswith('source.')or filename.startswith('weight.')):continue
    raw=z.read('stages/'+filename)
    if len(raw)!=row['bytes']or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('source member differs')
    (stages/filename).write_bytes(raw)
   raw=z.read('corrected/q4_k/vision.layer.0.f32')
   if hashlib.sha256(raw).hexdigest()!='cebff891816b57ccdcecbc72f0aca7b4bef8174821319654673d37e2c37188fb':raise ValueError('actual candidate layer0 differs')
   (stages/'actual.layer0.f32').write_bytes(raw)
  archive.unlink();torch.set_num_threads(2);torch.set_num_interop_threads(1)
  def array(name):return np.fromfile(stages/(name+'.f32'),dtype='<f4')
  def tensor(name,shape):return torch.from_numpy(array(name).copy()).reshape(shape).to(torch.bfloat16)
  def emit(name,x):x.float().contiguous().numpy().astype('<f4').tofile(stages/(name+'.f32'))
  def gelu_native(x,label):
   emit('activation.input',x);run(label,[build/'lighton4b-vision-boundary-probe',stages,'activation']);y=tensor('activation.output',x.shape);emit(label,y);return y
  qkv=tensor('source.qkv',(560,3,16,64));original_cos=torch.from_numpy(array('source.cos').copy()).reshape(560,64);original_sin=torch.from_numpy(array('source.sin').copy()).reshape(560,64)
  gates=('CRISPEMBED_LIGHTON4B_VISION_BF16_ATTENTION','CRISPEMBED_LIGHTON4B_VISION_BF16_SOFTMAX_AVX2');os.environ.update({g:'1'for g in gates});run('native-rotary',[build/'lighton4b-vision-boundary-probe',stages,'rotary'])
  native_cos=torch.from_numpy(array('native.cos').copy()).reshape(560,64);native_sin=torch.from_numpy(array('native.sin').copy()).reshape(560,64)
  coords=torch.tensor([(br*2+r,bc*2+c)for br in range(10)for bc in range(14)for r in range(2)for c in range(2)]);inv=1.0/(10000**(torch.arange(0,32,2,dtype=torch.float32)/32));freq=(coords[:,:,None].float()*inv).reshape(560,32);angle=torch.cat((freq,freq),dim=-1)
  report['rotary']={'original_cos_control':compare(array('source.cos'),angle.cos().numpy()),'original_sin_control':compare(array('source.sin'),angle.sin().numpy()),'native_cos':compare(array('source.cos'),array('native.cos')),'native_sin':compare(array('source.sin'),array('native.sin')),'source_table_actual_pool':compare(array('source.pooled'),array('source.rotary_pooled')),'native_table_actual_pool':compare(array('source.pooled'),array('native.rotary_pooled'))}
  if not all(report['rotary'][k]['bit_exact']for k in ('original_cos_control','original_sin_control','source_table_actual_pool')):raise ValueError('rotary source control differs')
  q,k,v=qkv.permute(1,0,2,3).unbind(0)
  def rotate(t,co,si):
   t=t.float();half=torch.cat((-t[...,32:],t[...,:32]),dim=-1);return(t*co[:,None,:]+half*si[:,None,:]).to(torch.bfloat16)
  def pooled(co,si):
   qr=rotate(q,co,si).transpose(0,1);kr=rotate(k,co,si).transpose(0,1);p=F.softmax((qr@kr.transpose(1,2))*0.125,dim=-1,dtype=torch.float32).to(torch.bfloat16);return(p@v.transpose(0,1)).transpose(0,1).reshape(560,1024)
  for name,t in (('q',q),('k',k)):report['rotary']['rotated_'+name]=compare(rotate(t,original_cos,original_sin).float().numpy(),rotate(t,native_cos,native_sin).float().numpy())
  up=tensor('source.up',(560,4096));expected=array('source.activation');native_activation=gelu_native(up,'native_activation')
  for contraction in ('off','fast'):run('source-gelu-'+contraction,[scratch/('source-gelu-'+contraction),stages/'source.up.f32',stages/('header.activation.'+contraction+'.f32')])
  report['activation']={'original_bf16_control':compare(expected,F.gelu(up,approximate='tanh').float().numpy()),'f32_input':compare(expected,F.gelu(up.float(),approximate='tanh').to(torch.bfloat16).float().numpy()),'native':compare(expected,native_activation.float().numpy()),'original_headers':{c:compare(expected,array('header.activation.'+c))for c in ('off','fast')}}
  if not report['activation']['original_bf16_control']['bit_exact']or not any(r['bit_exact']for r in report['activation']['original_headers'].values()):raise ValueError('original GELU control differs')
  def linear(name,x,rows,cols):return F.linear(x,tensor('weight.'+name+'.weight',(rows,cols)),tensor('weight.'+name+'.bias',(rows,)))
  def norm(name,x):return F.layer_norm(x,(1024,),tensor('weight.'+name+'.weight',(1024,)),tensor('weight.'+name+'.bias',(1024,)),1e-6)
  report['first_block_forward_executed']=True
  embedded=tensor('source.embedded',(560,1024));report['first_block']={}
  for label,native_rotary,native_gelu in (('original',False,False),('rotary_only',True,False),('activation_only',False,True),('both',True,True)):
   pool=pooled(native_cos if native_rotary else original_cos,native_sin if native_rotary else original_sin)
   residual=embedded+linear('out',pool,1024,1024);u=linear('up',norm('norm2',residual),4096,1024);a=gelu_native(u,'gelu_'+label)if native_gelu else F.gelu(u,approximate='tanh');final=residual+linear('down',a,1024,4096);emit('layer0.'+label,final)
   report['first_block'][label]={'vs_original':compare(array('source.layer0'),final.float().numpy()),'vs_actual_candidate':compare(array('actual.layer0'),final.float().numpy())}
  if not report['first_block']['original']['vs_original']['bit_exact']or not report['first_block']['both']['vs_actual_candidate']['bit_exact']:raise ValueError('composed first-block attribution control differs')
  report['disabled_corrections']={}
  for label,value in (('unset',None),('empty',''),('zero','0')):
   for gate in ('CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2','CRISPEMBED_LIGHTON4B_VISION_BF16_GELU_AVX2'):
    if value is None:os.environ.pop(gate,None)
    else:os.environ[gate]=value
   run('disabled-rotary-'+label,[build/'lighton4b-vision-boundary-probe',stages,'rotary']);act=gelu_native(up,'disabled-gelu-'+label)
   report['disabled_corrections'][label]={'cosine':compare(native_cos.numpy(),array('native.cos')),'sine':compare(native_sin.numpy(),array('native.sin')),'activation':compare(native_activation.float().numpy(),act.float().numpy())}
  if not all(v['bit_exact']for r in report['disabled_corrections'].values()for v in r.values()):raise ValueError('disabled correction bytes differ')
  grid=torch.arange(125,dtype=torch.float32)[:,None]*inv
  report['supported_trig_grid']={'inv_frequency':compare(inv.numpy(),array('native.inverse_frequency')),'cosine':compare(grid.cos().numpy(),array('helper.cos_grid')),'sine':compare(grid.sin().numpy(),array('helper.sin_grid'))}
  emit('source.angles',grid);run('source-trig',[scratch/'source-trig',stages/'source.angles.f32',stages/'header.cos_grid.f32',stages/'header.sin_grid.f32'])
  report['trig_dispatch']={label:{'cosine':compare(grid.cos().numpy(),array(prefix+'.cos_grid')),'sine':compare(grid.sin().numpy(),array(prefix+'.sin_grid'))}for label,prefix in (('original_header','header'),('nonfused_source','nonfused'))}

  os.environ['CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2']='1'if CORRECT_ROTARY else'0';os.environ['CRISPEMBED_LIGHTON4B_VISION_BF16_GELU_AVX2']='1'
  report['corrected_rotary_gate']=bool(CORRECT_ROTARY)
  run('corrected-rotary',[build/'lighton4b-vision-boundary-probe',stages,'rotary']);corrected_cos=torch.from_numpy(array('native.cos').copy()).reshape(560,64);corrected_sin=torch.from_numpy(array('native.sin').copy()).reshape(560,64)
  corrected_activation=gelu_native(up,'corrected_activation')
  pool=pooled(corrected_cos,corrected_sin);residual=embedded+linear('out',pool,1024,1024);u=linear('up',norm('norm2',residual),4096,1024);a=gelu_native(u,'corrected_composed_activation');final=residual+linear('down',a,1024,4096);emit('layer0.corrected',final)
  report['production_corrections']={'cosine':compare(array('source.cos'),corrected_cos.numpy()),'sine':compare(array('source.sin'),corrected_sin.numpy()),'activation':compare(expected,corrected_activation.float().numpy()),'pooling':compare(array('source.pooled'),array('native.rotary_pooled')),'first_block':compare(array('source.layer0'),final.float().numpy())}
  # Reference-only MKL control; never linked into the product.
  import ctypes
  report['mkl_source_control']={'mkl_available':torch.backends.mkl.is_available(),'torch_build_config':torch.__config__.show(),'direct_symbols_available':False,'product_dependency_added':False}
  lib=ctypes.CDLL(str(Path(library_paths()[0])/'libtorch_cpu.so'))
  if all(hasattr(lib,n)for n in ('vmsCos','vmsSin')):
   report['mkl_source_control']['direct_symbols_available']=True;angles=grid.numpy().astype('<f4').reshape(-1);pointer=ctypes.POINTER(ctypes.c_float);results={}
   for name,reference in (('Cos',grid.cos().numpy()),('Sin',grid.sin().numpy())):
    output=np.empty_like(angles);fn=getattr(lib,'vms'+name);fn.argtypes=[ctypes.c_int,pointer,pointer,ctypes.c_longlong];fn.restype=None;fn(len(angles),angles.ctypes.data_as(pointer),output.ctypes.data_as(pointer),2);output.tofile(stages/('mkl.'+name.lower()+'.f32'));results[name.lower()]=compare(reference,output)
   report['mkl_source_control']['HA_mode_2_finite_normal_input']=results
  headers=out/'reference-headers';headers.mkdir();report['reference_header_sha256']={}
  for name in ('ATen/cpu/vml.h','ATen/cpu/vec/vec256/vec256_float.h','ATen/native/cpu/Gelu.h','ATen/Config.h'):
   original=Path(include_paths()[0])/name;target=headers/(name.replace('/','_'));target.write_bytes(original.read_bytes());report['reference_header_sha256'][name]=sha(original)
  report['production_candidate_qualified']=all(r['bit_exact']for r in report['production_corrections'].values())and all(r['bit_exact']for r in report['supported_trig_grid'].values())
  report['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in stages.iterdir()};report['status']='attributed'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='bounded same-input boundary attribution failed; inspect phase/logs';raise RuntimeError(report['error'])from None
 finally:(out/'boundary-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='artifacts'},indent=2),flush=True)
if __name__=='__main__':main()
