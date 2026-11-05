#!/usr/bin/env python3
"""Retained-only selected QKV dot math; no model or full native forward."""
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
 if os.environ.get('GITHUB_ACTIONS')!='true' or torch.__version__!='2.13.0+cpu' or np.__version__!='2.3.5':raise RuntimeError('hosted original pins required')
 spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'selected-dot';scratch.mkdir();stages=out/'stages';stages.mkdir()
 report={'schema':'lighton4b-vision-selected-dot/v1','status':'failed','model_downloaded':False,'model_forward_executed':False,'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cpu_dispatch':dispatch,'torch_cpu_capability':torch.backends.cpu.get_cpu_capability(),'scope':'All retained same-input QKV differing positions plus deterministic controls; no full-operator or decoded qualification.'}
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def compare(a,b):
  row=comparison.compare(a,b);row['bit_different_values']=int(np.count_nonzero(np.asarray(a,dtype='<f4').view('<u4')!=np.asarray(b,dtype='<f4').view('<u4')));row['bit_exact']=row['bit_different_values']==0;return row
 def run(name,args):
  with (out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  artifact=comparison.api('actions/artifacts/13414500337')
  if artifact['workflow_run']['id']!=42024288946 or artifact['workflow_run']['head_sha']!='3b194f40b9725e327bb1d0a6ebcb40ab05aeacf5' or artifact['expired']:raise ValueError('retained source differs')
  archive=scratch/'retained.zip'
  with archive.open('wb')as f:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/13414500337/zip'],stdout=f,check=True)
  if sha(archive)!='77682c5af6a8a89af78736eb70241fdb8706ee801776703ddbdf42000fb273ff':raise ValueError('retained ZIP differs')
  names=('source.qkv.input','source.qkv','baseline.qkv','weight.qkv.weight','weight.qkv.bias')
  with zipfile.ZipFile(archive)as z:
   receipt=json.loads(z.read('vision-layer0.json'))
   if not receipt['original_control_bit_exact'] or receipt['artifacts']['source.layer0.f32']['sha256']!='ae4fcfbe1debb31e70dffca3be7cf36d7998228b40cebd1914758b50308c28f0':raise ValueError('original byte control differs')
   for name in names:
    filename=name+'.f32';raw=z.read('stages/'+filename);record=receipt['artifacts'][filename]
    if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('retained array differs')
    (stages/filename).write_bytes(raw)
  archive.unlink()
  x=np.fromfile(stages/'source.qkv.input.f32',dtype='<f4').reshape(560,1024);w=np.fromfile(stages/'weight.qkv.weight.f32',dtype='<f4').reshape(3072,1024);b=np.fromfile(stages/'weight.qkv.bias.f32',dtype='<f4');a=np.fromfile(stages/'source.qkv.f32',dtype='<f4');legacy=np.fromfile(stages/'baseline.qkv.f32',dtype='<f4')
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
  for v in (x,w,b):
   if not np.array_equal(v,torch.from_numpy(v.copy()).to(torch.bfloat16).float().numpy()):raise ValueError('retained operand is not exact BF16')
  reference=torch.nn.functional.linear(torch.from_numpy(x.copy()).to(torch.bfloat16),torch.from_numpy(w.copy()).to(torch.bfloat16),torch.from_numpy(b.copy()).to(torch.bfloat16)).float().numpy().reshape(-1)
  report['original_operator_control']=compare(a,reference)
  report['original_operator_control']['retained_sha256']=sha(stages/'source.qkv.f32');report['original_operator_control']['reconstructed_sha256']=hashlib.sha256(reference.tobytes()).hexdigest()
  if not report['original_operator_control']['bit_exact']:raise ValueError('original QKV control differs')
  differing=np.flatnonzero(a.view('<u4')!=legacy.view('<u4'));controls=np.unique(np.concatenate((np.arange(64),np.arange(560)*3072,np.arange(560)*3072+3071,np.array([a.size-1]),np.random.default_rng(20261105).integers(0,a.size,1024))))
  selected=np.unique(np.concatenate((differing,controls))).astype(np.int64)
  if selected.size>8192:raise ValueError('selected bound')
  ids=stages/'selected.ids';ids.write_text(''.join(str(int(i))+'\n'for i in selected))
  report['selection']={'all_legacy_bit_differences':int(differing.size),'authored_control_positions':int(controls.size),'selected_positions':int(selected.size),'selected_ids_sha256':sha(ids)}
  report['legacy_selected']=compare(a[selected],legacy[selected])
  binary=scratch/'dot-probe';run('compile',['c++','-std=c++17','-O3','-ffp-contract=off','-I'+str(ROOT/'src'),'-I'+str(ROOT/'ggml/include'),ROOT/'tests/lighton4b_vision_selected_dot.cpp','-o',binary]);run('probe',[binary,stages])
  report['modes']={}
  for mode in ('register8_fma','register8_muladd','double_bias','double_then_float_bias','serial_fma'):
   values=np.fromfile(stages/(mode+'.f32'),dtype='<f4');report['modes'][mode]=compare(a[selected],values)
  report['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in stages.iterdir()}
  report['status']='compared'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='retained selected dot attribution failed; inspect logs';raise RuntimeError(report['error'])from None
 finally:
  (out/'selected-dot.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='artifacts'},indent=2),flush=True)
if __name__=='__main__':main()
