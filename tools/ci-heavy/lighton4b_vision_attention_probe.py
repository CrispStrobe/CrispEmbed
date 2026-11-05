#!/usr/bin/env python3
"""Retained-only attention primitive attribution and preservation readback."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[2]
def main():
 dispatch={'ATEN_CPU_CAPABILITY':'avx2','ONEDNN_MAX_CPU_ISA':'AVX2','DNNL_MAX_CPU_ISA':'AVX2','MKL_ENABLE_INSTRUCTIONS':'AVX2'};os.environ.update(dispatch)
 import numpy as np
 import torch
 if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable' or torch.__version__!='2.13.0+cpu' or np.__version__!='2.3.5':raise RuntimeError('hosted portable original pins required')
 spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'attention-probe';scratch.mkdir();stages=out/'stages';stages.mkdir();build=scratch/'build'
 report={'schema':'lighton4b-vision-attention-probe/v1','status':'failed','model_downloaded':False,'model_forward_executed':False,'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cpu_dispatch':dispatch,'scope':'Source-fed QKV and exact original cosine/sine; generated native rotary tables remain a separate composed boundary.'}
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def compare(a,b):
  row=comparison.compare(a,b);row['bit_different_values']=int(np.count_nonzero(a.view('<u4')!=b.view('<u4')));row['bit_exact']=row['bit_different_values']==0;return row
 def run(name,args):
  with (out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  item=comparison.api('actions/artifacts/13420770687')
  if item['workflow_run']['id']!=42038802495 or item['workflow_run']['head_sha']!='edbd2b782a2317c5cd248610f7f4f3859e4d9294'or item['expired']or item['size_in_bytes']!=403396438:raise ValueError('preserved source service identity differs')
  outer=scratch/'preserved.zip'
  with outer.open('wb')as f:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/13420770687/zip'],stdout=f,check=True)
  if sha(outer)!='6d6845c4cbe4ecfe0a9112b8ee88e8e79845a2256714713c46732c66abfcf2c5':raise ValueError('preserved outer archive differs')
  inner=scratch/'source.zip'
  with zipfile.ZipFile(outer)as z,z.open('linear/retained-source-13414500337.zip')as source,inner.open('wb')as target:shutil.copyfileobj(source,target,2**20)
  if inner.stat().st_size!=238129493 or sha(inner)!='77682c5af6a8a89af78736eb70241fdb8706ee801776703ddbdf42000fb273ff':raise ValueError('exact preserved source ZIP readback differs')
  report['preservation_readback']={'run_id':42038802495,'artifact_id':13420770687,'expires_at':item['expires_at'],'outer_sha256':sha(outer),'source_sha256':sha(inner),'source_bytes':inner.stat().st_size,'exact_remote_readback_verified':True};outer.unlink()
  names=('source.qkv','source.cos','source.sin','source.pooled','baseline.pooled')
  with zipfile.ZipFile(inner)as z:
   receipt=json.loads(z.read('vision-layer0.json'))
   if receipt['artifacts']['source.layer0.f32']['sha256']!='ae4fcfbe1debb31e70dffca3be7cf36d7998228b40cebd1914758b50308c28f0':raise ValueError('original composed byte control differs')
   for name in names:
    filename=name+'.f32';raw=z.read('stages/'+filename);row=receipt['artifacts'][filename]
    if hashlib.sha256(raw).hexdigest()!=row['sha256']or len(raw)!=row['bytes']:raise ValueError('retained attention input differs')
    (stages/filename).write_bytes(raw)
  inner.unlink();torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
  qkv=torch.from_numpy(np.fromfile(stages/'source.qkv.f32',dtype='<f4').copy()).to(torch.bfloat16).reshape(560,3,16,64)
  q,k,v=qkv.permute(1,0,2,3).unbind(0);co=torch.from_numpy(np.fromfile(stages/'source.cos.f32',dtype='<f4').copy()).reshape(560,64);si=torch.from_numpy(np.fromfile(stages/'source.sin.f32',dtype='<f4').copy()).reshape(560,64)
  def rotate(x):
   x=x.float();half=torch.cat((-x[...,32:],x[...,:32]),dim=-1);return(x*co[:,None,:]+half*si[:,None,:]).to(torch.bfloat16)
  q,k=rotate(q),rotate(k)
  scores=(q.transpose(0,1)@k.transpose(0,1).transpose(1,2))*0.125
  prob_float=torch.nn.functional.softmax(scores,dim=-1,dtype=torch.float32);prob=prob_float.to(torch.bfloat16)
  pooled=(prob@v.transpose(0,1)).transpose(0,1).reshape(560,1024).float().numpy().reshape(-1)
  original=np.fromfile(stages/'source.pooled.f32',dtype='<f4');report['original_pool_control']=compare(original,pooled)
  if not report['original_pool_control']['bit_exact']:raise ValueError('original actual pooled bytes differ')
  def emit(name,x):x.float().contiguous().numpy().astype('<f4').tofile(stages/('source.'+name+'.f32'))
  emit('qrot',q);emit('krot',k);emit('scores',scores);emit('prob',prob);emit('prob_float',prob_float);sf=scores.float();emit('exp',(sf-sf.amax(dim=-1,keepdim=True)).exp())
  inverse=prob_float.reshape(-1,560).gather(1,sf.reshape(-1,560).argmax(dim=1,keepdim=True)).reshape(-1).numpy();inverse.astype('<f4').tofile(stages/'source.inverse.f32')
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF']);run('build',['cmake','--build',build,'--target','lighton4b-vision-attention-probe','--parallel','2']);run('probe',[build/'lighton4b-vision-attention-probe',stages])
  actual=np.fromfile(stages/'actual.pooled.f32',dtype='<f4');old=np.fromfile(stages/'baseline.pooled.f32',dtype='<f4');trace=np.fromfile(stages/'trace.pooled.f32',dtype='<f4')
  report['actual_matches_retained']=compare(old,actual);report['trace_matches_actual']=compare(actual,trace);report['native_pooled_vs_original']=compare(original,actual)
  if not report['actual_matches_retained']['bit_exact']or not report['trace_matches_actual']['bit_exact']:raise ValueError('native observer control differs')
  report['operators']={}
  for source_name,targets in (('qrot',('native.qrot',)),('krot',('native.krot',)),('scores',('native.scores','sourcefed.dot8','sourcefed.dot4','sourcefed.dotdouble')),('prob',('native.prob','sourcefed.softmax_std_double','sourcefed.softmax_std_vec8','sourcefed.softmax_sourceexp_vec8')),('pooled',('sourcefed.pv4','sourcefed.pv8','sourcefed.pvdouble'))):
   a=np.fromfile(stages/('source.'+source_name+'.f32'),dtype='<f4');report['operators'][source_name]={name:compare(a,np.fromfile(stages/(name+'.f32'),dtype='<f4'))for name in targets}
  report['production_qk_pv']={'corrected':compare(original,np.fromfile(stages/'actual.corrected_pooled.f32',dtype='<f4')),'observer':compare(np.fromfile(stages/'actual.corrected_pooled.f32',dtype='<f4'),np.fromfile(stages/'trace.corrected_pooled.f32',dtype='<f4')),'disabled':{g:compare(old,np.fromfile(stages/('disabled.'+g+'.f32'),dtype='<f4'))for g in ('empty','zero')}}
  if not report['production_qk_pv']['observer']['bit_exact']or not all(v['bit_exact']for v in report['production_qk_pv']['disabled'].values()):raise ValueError('production attention gate/observer differs')
  report['torch_git_version']=torch.version.git_version
  expected=prob.float().numpy().reshape(-1);prob_ref=prob_float.numpy().reshape(-1)
  def bf16(x):
   bits=np.asarray(x,dtype='<f4').copy().view('<u4');bits+=0x7fff+((bits>>16)&1);bits&=0xffff0000;return bits.view('<f4')
  source_exp=np.fromfile(stages/'source.exp.f32',dtype='<f4').reshape(-1,560);std_exp=np.fromfile(stages/'sourcefed.std_exp.f32',dtype='<f4').reshape(-1,560)
  report['softmax_forced_original_inverse']={label:compare(expected,bf16(exp*inverse[:,None]).reshape(-1))for label,exp in (('source_exp',source_exp),('std_exp',std_exp))}
  report['softmax_reductions']={}
  for lanes in (4,8,16):
   for registers in (1,2,4,8):
    for register_order in ('tree','left'):
     accum=np.zeros((8960,registers,lanes),dtype='<f4')
     for block in range(0,560,lanes):accum[:,(block//lanes)%registers,:]+=source_exp[:,block:block+lanes]
     if register_order=='tree':
      sums=accum.copy();offset=registers//2
      while offset:sums[:,:offset,:]+=sums[:,offset:2*offset,:];offset//=2
      sums=sums[:,0,:]
     else:
      sums=accum[:,0,:].copy()
      for reg in range(1,registers):sums+=accum[:,reg,:]
     for lane_order in ('tree','left'):
      if lane_order=='tree':
       reduced=sums.copy();offset=lanes//2
       while offset:reduced[:,:offset]+=reduced[:,offset:2*offset];offset//=2
       total=reduced[:,0]
      else:
       total=sums[:,0].copy()
       for lane in range(1,lanes):total+=sums[:,lane]
      inv=np.float32(1)/total;result=bf16(source_exp*inv[:,None]).reshape(-1);label=f'lanes{lanes}-regs{registers}-{register_order}-{lane_order}'
      report['softmax_reductions'][label]={'inverse_bit_differences':int(np.count_nonzero(inv.view('<u4')!=inverse.view('<u4'))),'inverse_max_abs':float(np.max(np.abs(inv-inverse))),'probability_bit_differences':int(np.count_nonzero(result.view('<u4')!=expected.view('<u4')))}
  report['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in stages.iterdir()};report['status']='compared'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='bounded retained attention attribution failed; inspect logs';raise RuntimeError(report['error'])from None
 finally:
  (out/'attention-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='artifacts'},indent=2),flush=True)
if __name__=='__main__':main()
