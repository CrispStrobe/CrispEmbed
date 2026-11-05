#!/usr/bin/env python3
"""Bounded original vector-header/dispatch softmax attribution, no runtime fix."""
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
 from torch.utils.cpp_extension import include_paths,library_paths
 if os.environ.get('GITHUB_ACTIONS')!='true'or torch.__version__!='2.13.0+cpu'or np.__version__!='2.3.5':raise RuntimeError('hosted original pins only')
 spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'softmax-source';scratch.mkdir();stages=out/'stages';stages.mkdir()
 report={'schema':'lighton4b-softmax-vector-source/v1','status':'failed','model_forward_executed':False,'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'torch_version':torch.__version__,'torch_git_version':torch.version.git_version,'cpu_dispatch':dispatch}
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def run(name,args):
  with(out/(name+'.log')).open('w')as f:subprocess.run([str(v)for v in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 def compare(a,b):
  row=comparison.compare(a,b);row['bit_different_values']=int(np.count_nonzero(a.view('<u4')!=b.view('<u4')));row['bit_exact']=row['bit_different_values']==0;return row
 try:
  binaries={}
  for isa in ('AVX2','DEFAULT'):
   for math in ('off','fast','fastmath'):
    name=isa.lower()+'-'+math;binary=scratch/name;flags=['-mavx2','-mfma','-DCPU_CAPABILITY_AVX2']if isa=='AVX2'else['-msse2','-DCPU_CAPABILITY_DEFAULT']
    args=['c++','-O2','-std=c++17','-fno-math-errno','-fno-trapping-math','-DCPU_CAPABILITY='+isa,'-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),*flags]
    args+=['-ffp-contract='+('fast'if math=='fastmath'else math)];args+=['-ffast-math']if math=='fastmath'else[]
    args+=['-I'+p for p in include_paths()];args +=[ROOT/'tests/lighton4b_softmax_vector_source.cpp','-o',binary]
    for p in library_paths():args+=['-L'+p,'-Wl,-rpath,'+p]
    args+=['-ltorch_cpu','-lc10'];run('compile-'+name,args);binaries[name]=binary
  report['header_sha256']={str(Path(p).relative_to(include_paths()[0])):sha(Path(p))for p in [str(Path(include_paths()[0])/'ATen/cpu/vec/functional_base.h'),str(Path(include_paths()[0])/'ATen/cpu/vec/vec256/vec256_float.h')]}
  item=comparison.api('actions/artifacts/13425935293')
  if item['workflow_run']['id']!=42047075420 or item['expired']or item['size_in_bytes']!=116618388:raise ValueError('retained attention source identity differs')
  archive=scratch/'attention.zip'
  with archive.open('wb')as f:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/13425935293/zip'],stdout=f,check=True)
  if sha(archive)!='e900b4386eaebd3f9b8850bd3677c937f6e8e4da9273c318f0a0b0c39e9d0e42':raise ValueError('retained source ZIP differs')
  with zipfile.ZipFile(archive)as z:
   receipt=json.loads(z.read('attention-probe.json'))
   if not receipt['original_pool_control']['bit_exact']or not receipt['trace_matches_actual']['bit_exact']:raise ValueError('original/observer source controls differ')
   for name in ('source.scores','source.exp','source.prob_float','source.inverse','source.prob'):
    filename=name+'.f32';raw=z.read('stages/'+filename);r=receipt['artifacts'][filename]
    if len(raw)!=r['bytes']or hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('retained source array differs')
    (stages/filename).write_bytes(raw)
  archive.unlink();torch.set_num_threads(2);torch.set_num_interop_threads(1)
  x=np.fromfile(stages/'source.scores.f32',dtype='<f4').reshape(16,560,560);expected=np.fromfile(stages/'source.prob_float.f32',dtype='<f4');expected_bf16=np.fromfile(stages/'source.prob.f32',dtype='<f4');inverse=np.fromfile(stages/'source.inverse.f32',dtype='<f4');exps=np.fromfile(stages/'source.exp.f32',dtype='<f4')
  tx=torch.from_numpy(x.copy());reference=torch.nn.functional.softmax(tx.to(torch.bfloat16),dim=-1,dtype=torch.float32).numpy().reshape(-1);report['original_bf16_input_control']=compare(expected,reference)
  if not report['original_bf16_input_control']['bit_exact']:raise ValueError('original softmax bytes differ')
  f32=torch.nn.functional.softmax(tx,dim=-1,dtype=torch.float32).numpy().reshape(-1);report['f32_vs_original_bf16_input']=compare(expected,f32)
  report['variants']={}
  for name,binary in binaries.items():
   run(name,[binary,stages,name]);p=np.fromfile(stages/(name+'.prob.f32'),dtype='<f4');inv=np.fromfile(stages/(name+'.inverse.f32'),dtype='<f4');e=np.fromfile(stages/(name+'.exp.f32'),dtype='<f4');meta=json.loads((stages/(name+'.json')).read_text())
   quant=torch.from_numpy(p.copy()).to(torch.bfloat16).float().numpy()
   report['variants'][name]={'vector':meta,'probability_f32':compare(expected,p),'probability_bf16':compare(expected_bf16,quant),'reciprocal':compare(inverse,inv),'exp_vs_torch_exp':compare(exps,e)}
  report['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in stages.iterdir()};report['status']='compared'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='bounded exact vector-header attribution failed; inspect logs';raise RuntimeError(report['error'])from None
 finally:(out/'softmax-vector-source.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='artifacts'},indent=2),flush=True)
if __name__=='__main__':main()
