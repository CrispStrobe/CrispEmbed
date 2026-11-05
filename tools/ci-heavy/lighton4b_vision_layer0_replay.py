#!/usr/bin/env python3
"""Bounded same-input BF16 vision operators; original layer0 anchor is immutable."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 import numpy as np
 import torch
 from torch.nn import functional as F
 from kaggle.api.kaggle_api_extended import KaggleApi
 if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':raise RuntimeError('hosted portable only')
 if torch.__version__!='2.13.0+cpu' or np.__version__!='2.3.5':raise RuntimeError('frozen original torch/numpy versions required')
 native=load('native',ROOT/'tools/ci-heavy/lighton4b_native_document.py')
 comparison=load('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'vision-layer0';scratch.mkdir()
 stages=out/'stages';stages.mkdir();build=scratch/'build'
 report={'schema':'lighton4b-vision-layer0-source-fed/v1','status':'failed','quality_qualified':False,
  'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  'torch_version':torch.__version__,'torch_cpu_capability':torch.backends.cpu.get_cpu_capability(),
  'decoded_scope':'No decoder execution; preserves original decoded <figure prefix and earlier negative Q4/Q8 smokes.'}
 def run(name,args,env=None):
  with (out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
 try:
  report['preflight']={'memory':native.memory(),'free_bytes':__import__('shutil').disk_usage(scratch).free}
  if report['preflight']['memory']['available_bytes']<8*1024**3 or report['preflight']['free_bytes']<8*1024**3:raise ValueError('resource bound')
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  run('build',['cmake','--build',build,'--target','lighton4b-vision-layer0-replay','--parallel','2'])
  filename,digest,size=native.CANDIDATES['q4_k'];model=scratch/filename
  native.download(f'https://huggingface.co/{native.MODEL_REPO}/resolve/{native.MODEL_REVISION}/{filename}',model)
  if model.stat().st_size!=size or native.sha(model)!=digest:raise ValueError('immutable model differs')
  report['model']={'repository':native.MODEL_REPO,'revision':native.MODEL_REVISION,'file':filename,'sha256':digest}
  provider=KaggleApi();provider.authenticate()
  provider.kernels_output(os.environ['KAGGLE_ACCOUNT']+'/lighton4b-bf16-targeted-original-cpu-prefill',str(scratch/'original'),file_pattern=r'^lighton4b-bf16-prefill/stages/(prefill\.json|vision\.(embedded|layer\.0|merged)\.f32|prefill\.logits\.f32)$',quiet=True)
  original=scratch/'original/lighton4b-bf16-prefill/stages';manifest=original/'prefill.json'
  if native.sha(manifest)!=comparison.ORIGINAL_REPORT_SHA:raise ValueError('original receipt differs')
  receipt=json.loads(manifest.read_text())
  if receipt['source_sha']!=comparison.SOURCE or not receipt['observer_first_score_bit_exact']:raise ValueError('original controls differ')
  def original_tensor(name):
   row=receipt['stages'][name];path=original/row['file']
   if path.stat().st_size!=row['bytes'] or native.sha(path)!=row['sha256']:raise ValueError('original stage differs')
   return torch.from_numpy(np.fromfile(path,dtype='<f4').copy()).reshape(-1,1024).to(torch.bfloat16)
  env=os.environ.copy();env.update({k:'1'for k in native.GATES});env['CRISPEMBED_LIGHTON4B_VISION_BF16']='1'
  binary=build/'lighton4b-vision-layer0-replay'
  run('export',[binary,model,stages,'export'],env)
  weights={}
  for p in stages.glob('weight.*.f32'):
   x=torch.from_numpy(np.fromfile(p,dtype='<f4').copy());b=x.to(torch.bfloat16)
   if not torch.equal(x,b.float()):raise ValueError('vision weights not source BF16')
   weights[p.name[7:-4]]=b
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
  def emit(name,x):x.float().contiguous().numpy().astype('<f4').tofile(stages/('source.'+name+'.f32'));return x
  def linear(name,x,rows,columns):
   emit(name+'.input',x)
   return emit(name,F.linear(x,weights[name+'.weight'].reshape(rows,columns),weights[name+'.bias']))
  def norm(name,x):return emit(name,F.layer_norm(x,(1024,),weights[name+'.weight'],weights[name+'.bias'],1e-6))
  x=emit('embedded',original_tensor('vision.embedded'));n=norm('norm1',x);qkv=linear('qkv',n,3072,1024)
  # Original merge-block traversal, Qwen3VL inv_freq and apply_rotary_pos_emb_vision.
  coords=torch.tensor([(br*2+r,bc*2+c)for br in range(10)for bc in range(14)for r in range(2)for c in range(2)])
  inv=1.0/(10000**(torch.arange(0,32,2,dtype=torch.float32)/32))
  freq=(coords[:,:,None].float()*inv).reshape(560,32);angle=torch.cat((freq,freq),dim=-1)
  co=emit('cos',angle.cos());si=emit('sin',angle.sin())
  q,k,v=qkv.reshape(560,3,16,64).permute(1,0,2,3).unbind(0)
  def rotate(t):
   t=t.float();half=torch.cat((-t[...,32:],t[...,:32]),dim=-1)
   return (t*co[:,None,:]+half*si[:,None,:]).to(torch.bfloat16)
  q=rotate(q).transpose(0,1);k=rotate(k).transpose(0,1);v=v.transpose(0,1)
  scores=(q@k.transpose(1,2))*0.125
  probabilities=F.softmax(scores,dim=-1,dtype=torch.float32).to(torch.bfloat16)
  pooled=emit('pooled',(probabilities@v).transpose(0,1).reshape(560,1024))
  mixed=linear('out',pooled,1024,1024);res=emit('residual1',x+mixed)
  n=norm('norm2',res);up=linear('up',n,4096,1024)
  activation=emit('activation',F.gelu(up,approximate='tanh'))
  down=linear('down',activation,1024,4096);final=emit('layer0',res+down)
  report['original_reconstruction']=comparison.compare(original_tensor('vision.layer.0').float().numpy(),final.float().numpy())
  # Keep diagnostics even when the composed control differs; such a result cannot establish a fix.
  report['original_control_bit_exact']=report['original_reconstruction']['different_values']==0
  if not report['original_control_bit_exact']:raise ValueError('composed original control differs')
  run('replay',[binary,model,stages,'replay'],env)
  report['operators']={}
  for name in ('norm1','qkv','pooled','out','norm2','up','activation','down'):
   a=np.fromfile(stages/('source.'+name+'.f32'),dtype='<f4');b=np.fromfile(stages/('native.'+name+'.f32'),dtype='<f4')
   report['operators'][name]=comparison.compare(a,b)
  report['first_differing_operator']=next((n for n,v in report['operators'].items()if v['different_values']),None)
  for path in stages.glob('native.*.f32'):path.rename(path.with_name(path.name.replace('native.','baseline.',1)))
  env['CRISPEMBED_LIGHTON4B_VISION_BF16_NORM']='1'
  run('corrected-replay',[binary,model,stages,'replay'],env)
  report['corrected_operators']={name:comparison.compare(np.fromfile(stages/('source.'+name+'.f32'),dtype='<f4'),np.fromfile(stages/('native.'+name+'.f32'),dtype='<f4'))for name in report['operators']}
  # Same immutable model and image, independent ordinary native document paths.
  # No source intermediate is passed into either composed inference call.
  download_original=native.download
  def cached_download(url,path):
   if url.endswith('/'+filename):
    if not path.exists():path.symlink_to(model)
   else:download_original(url,path)
  native.download=cached_download
  base_gates=tuple(native.GATES)+('CRISPEMBED_LIGHTON4B_VISION_BF16',)
  saved_args=sys.argv;report['composed']={}
  try:
   frozen=json.loads((ROOT/'tests/results/qwen35-ocr/lightonocr3-4b-vision-bf16-stages.json').read_text())['recipes']['q4_k']['stages']
   for label,corrected in (('baseline',False),('corrected',True)):
    arm=out/label;arm.mkdir();os.environ['HEAVY_OUT']=str(arm)
    native.GATES=base_gates+(('CRISPEMBED_LIGHTON4B_VISION_BF16_NORM',)if corrected else ())
    os.environ.pop('CRISPEMBED_LIGHTON4B_VISION_BF16_NORM',None)
    sys.argv=[str(ROOT/'tools/ci-heavy/lighton4b_native_document.py'),'--recipe','q4_k','--mode','smoke','--threads','4']
    native.main();result=json.loads((arm/'native-document.json').read_text())
    row={'decoded':result['decoded'],'stages':{}}
    for name in ('vision.layer.0','vision.merged','prefill.logits'):
     reference=original/receipt['stages'][name]['file']
     if native.sha(reference)!=receipt['stages'][name]['sha256']:raise ValueError('original composed hash differs')
     candidate=arm/'q4_k'/(name+'.f32')
     row['stages'][name]=comparison.compare(np.fromfile(reference,dtype='<f4'),np.fromfile(candidate,dtype='<f4'))
     row['stages'][name]['sha256']=native.sha(candidate)
     if not corrected and native.sha(candidate)!=frozen[name]['native_sha256']:raise ValueError('legacy composed byte control differs')
    report['composed'][label]=row
  finally:
   sys.argv=saved_args;os.environ['HEAVY_OUT']=str(out)

  report['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':native.sha(p)}for p in stages.iterdir()}
  report['status']='compared' if report['original_control_bit_exact'] else 'diagnostic-original-control-differs'
 except Exception as error:
  report['error_type']=type(error).__name__;report['error']='bounded hosted attribution failed; inspect retained logs';raise RuntimeError(report['error'])from None
 finally:
  (out/'vision-layer0.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items()if k!='artifacts'},indent=2),flush=True)
if __name__=='__main__':main()
