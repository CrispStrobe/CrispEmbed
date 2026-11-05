#!/usr/bin/env python3
"""Full retained source-fed vision linear operators; no model access/forward."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[2]
def main():
 dispatch={'ATEN_CPU_CAPABILITY':'avx2','ONEDNN_MAX_CPU_ISA':'AVX2','DNNL_MAX_CPU_ISA':'AVX2','MKL_ENABLE_INSTRUCTIONS':'AVX2'};os.environ.update(dispatch)
 import numpy as np
 import torch
 if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable' or torch.__version__!='2.13.0+cpu' or np.__version__!='2.3.5':raise RuntimeError('hosted portable original pins required')
 spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'linear-operators';scratch.mkdir();stages=out/'stages';stages.mkdir();build=scratch/'build'
 report={'schema':'lighton4b-vision-linear-operators/v1','status':'failed','model_downloaded':False,'model_forward_executed':False,'original_linear_calls':0,'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cpu_dispatch':dispatch,'torch_cpu_capability':torch.backends.cpu.get_cpu_capability(),'scope':'Full first-layer QKV/out/up/down at original retained inputs; patch and merger shape composition remain unverified.'}
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def compare(a,b):
  row=comparison.compare(a,b);row['bit_different_values']=int(np.count_nonzero(a.view('<u4')!=b.view('<u4')));row['bit_exact']=row['bit_different_values']==0;return row
 def run(name,args):
  with (out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  artifact=comparison.api('actions/artifacts/13414500337')
  if artifact['workflow_run']['id']!=42024288946 or artifact['workflow_run']['head_sha']!='3b194f40b9725e327bb1d0a6ebcb40ab05aeacf5' or artifact['expired']:raise ValueError('retained source differs')
  archive=scratch/'retained.zip'
  with archive.open('wb')as f:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/13414500337/zip'],stdout=f,check=True)
  if sha(archive)!='77682c5af6a8a89af78736eb70241fdb8706ee801776703ddbdf42000fb273ff':raise ValueError('retained ZIP differs')
  names=[prefix+name+suffix for name in ('qkv','out','up','down')for prefix,suffix in (('source.','.input'),('source.',''),('baseline.',''),('weight.','.weight'),('weight.','.bias'))]
  with zipfile.ZipFile(archive)as z:
   receipt=json.loads(z.read('vision-layer0.json'))
   if not receipt['original_control_bit_exact'] or receipt['artifacts']['source.layer0.f32']['sha256']!='ae4fcfbe1debb31e70dffca3be7cf36d7998228b40cebd1914758b50308c28f0':raise ValueError('original byte control differs')
   for name in names:
    filename=name+'.f32';raw=z.read('stages/'+filename);record=receipt['artifacts'][filename]
    if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('retained array differs')
    (stages/filename).write_bytes(raw)
  archive.unlink();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
  report['original_controls']={}
  for name in ('qkv','out','up','down'):
   width=4096 if name=='down'else 1024;outputs=3072 if name=='qkv'else 4096 if name=='up'else 1024
   x=np.fromfile(stages/('source.'+name+'.input.f32'),dtype='<f4').reshape(560,width);w=np.fromfile(stages/('weight.'+name+'.weight.f32'),dtype='<f4').reshape(outputs,width);b=np.fromfile(stages/('weight.'+name+'.bias.f32'),dtype='<f4');a=np.fromfile(stages/('source.'+name+'.f32'),dtype='<f4')
   reference=torch.nn.functional.linear(torch.from_numpy(x.copy()).to(torch.bfloat16),torch.from_numpy(w.copy()).to(torch.bfloat16),torch.from_numpy(b.copy()).to(torch.bfloat16)).float().numpy().reshape(-1);report['original_linear_calls']+=1
   row=compare(a,reference);row['retained_sha256']=sha(stages/('source.'+name+'.f32'));row['reference_sha256']=hashlib.sha256(reference.tobytes()).hexdigest();report['original_controls'][name]=row
   if not row['bit_exact']:raise ValueError('original linear control differs')
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  run('build',['cmake','--build',build,'--target','lighton4b-vision-linear-operators','--parallel','2']);run('operators',[build/'lighton4b-vision-linear-operators',stages])
  report['operators']={}
  for name in ('qkv','out','up','down'):
   a=np.fromfile(stages/('source.'+name+'.f32'),dtype='<f4');legacy=np.fromfile(stages/('baseline.'+name+'.f32'),dtype='<f4');old=np.fromfile(stages/('old.'+name+'.unset.f32'),dtype='<f4')
   row={'legacy':compare(a,legacy),'old_unset_matches_retained':compare(legacy,old),'candidates':{str(n):compare(a,np.fromfile(stages/('candidate.'+name+'.threads'+str(n)+'.f32'),dtype='<f4'))for n in (1,4)},'gate_controls':{g:compare(old,np.fromfile(stages/('old.'+name+'.'+g+'.f32'),dtype='<f4'))for g in ('empty','0')}}
   if not row['old_unset_matches_retained']['bit_exact'] or not all(v['bit_exact']for v in row['gate_controls'].values()):raise ValueError('old path gate control differs')
   report['operators'][name]=row
  report['candidate_full_operators_bit_exact']=all(v['bit_exact']for row in report['operators'].values()for v in row['candidates'].values())
  report['controls_stdout']=(out/'operators.log').read_text();report['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in stages.iterdir()};report['status']='compared'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='bounded full linear attribution failed; inspect logs';raise RuntimeError(report['error'])from None
 finally:
  (out/'linear-operators.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='artifacts'},indent=2),flush=True)
if __name__=='__main__':main()
