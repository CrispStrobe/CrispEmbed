#!/usr/bin/env python3
"""Original full Conv3D patch attribution, only3MiB immutable parameter ranges."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
def main():
 dispatch={'ATEN_CPU_CAPABILITY':'avx2','ONEDNN_MAX_CPU_ISA':'AVX2','DNNL_MAX_CPU_ISA':'AVX2','MKL_ENABLE_INSTRUCTIONS':'AVX2'};os.environ.update(dispatch)
 import numpy as np
 import torch
 from kaggle.api.kaggle_api_extended import KaggleApi
 if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable' or torch.__version__!='2.13.0+cpu' or np.__version__!='2.3.5':raise RuntimeError('hosted portable original pins required')
 spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'conv-operators';scratch.mkdir();stages=out/'stages';stages.mkdir();build=scratch/'build'
 report={'schema':'lighton4b-vision-conv-operators/v1','status':'failed','model_full_downloaded':False,'whole_model_hash_independently_verified':False,'model_forward_executed':False,'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cpu_dispatch':dispatch,'torch_cpu_capability':torch.backends.cpu.get_cpu_capability()}
 def sha(raw):return hashlib.sha256(raw).hexdigest()
 def compare(a,b):
  row=comparison.compare(a,b);row['bit_different_values']=int(np.count_nonzero(a.view('<u4')!=b.view('<u4')));row['bit_exact']=row['bit_different_values']==0;return row
 def run(name,args):
  with (out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 def request_range(start,end):
  url=f'https://huggingface.co/lightonai/LightOnOCR-3-4B/resolve/b71010f095dc3735de7ee6b9969bff8f8f50b39a/model.safetensors?range_start={start}'
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'}),timeout=120)as response:
   if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/9078620536':raise ValueError('bounded original range not honored')
   raw=response.read(end-start+2)
  if len(raw)!=end-start+1:raise ValueError('original range bytes differ')
  return raw
 try:
  prefix=request_range(0,89463)
  if int.from_bytes(prefix[:8],'little')!=89456 or sha(prefix[8:])!='a1e36638b195f2d5943e000fcb069b275cb22ccc28b1b6184ddee7bdea7ef081':raise ValueError('original header differs')
  header=json.loads(prefix[8:]);bias=header['model.visual.patch_embed.proj.bias'];weight=header['model.visual.patch_embed.proj.weight']
  if bias!={'dtype':'BF16','shape':[1024],'data_offsets':[9070664704,9070666752]}or weight!={'dtype':'BF16','shape':[1024,3,2,16,16],'data_offsets':[9070666752,9073812480]}:raise ValueError('patch source geometry differs')
  payload=request_range(89464+bias['data_offsets'][0],89464+weight['data_offsets'][1]-1);(stages/'original-patch-parameters.bf16').write_bytes(payload)
  b=(np.frombuffer(payload[:2048],dtype='<u2').astype('<u4')<<16).view('<f4').copy();w=(np.frombuffer(payload[2048:],dtype='<u2').astype('<u4')<<16).view('<f4').copy()
  b.tofile(stages/'weight.patch.bias.f32');w.tofile(stages/'weight.patch.weight.f32')
  report['source_parameters']={'repository':'lightonai/LightOnOCR-3-4B','revision':'b71010f095dc3735de7ee6b9969bff8f8f50b39a','file':'model.safetensors','header_sha256':sha(prefix[8:]),'parameter_range_bytes':len(payload),'parameter_range_sha256':sha(payload),'weight_f32_sha256':sha(w.tobytes()),'bias_f32_sha256':sha(b.tobytes()),'whole_file_hf_declared_sha256':'6b0dab74a2dab02d5b10ca56322255024ed8dca35c02746b42e9f296aaeb4e67'}
  provider=KaggleApi();provider.authenticate();provider.kernels_output(os.environ['KAGGLE_ACCOUNT']+'/lighton4b-bf16-targeted-original-cpu-prefill',str(scratch/'original'),file_pattern=r'^lighton4b-bf16-prefill/stages/(prefill\.json|processor\.patches\.f32|vision\.patch\.f32)$',quiet=True)
  original=scratch/'original/lighton4b-bf16-prefill/stages';manifest=original/'prefill.json'
  if sha(manifest.read_bytes())!=comparison.ORIGINAL_REPORT_SHA:raise ValueError('original stage receipt differs')
  receipt=json.loads(manifest.read_text())
  for name,target in (('processor.patches','source.patches'),('vision.patch','source.patch')):
   raw=(original/receipt['stages'][name]['file']).read_bytes()
   if sha(raw)!=receipt['stages'][name]['sha256']or len(raw)!=receipt['stages'][name]['bytes']:raise ValueError('original input/output differs')
   (stages/(target+'.f32')).write_bytes(raw)
  x=np.fromfile(stages/'source.patches.f32',dtype='<f4').reshape(560,1536);a=np.fromfile(stages/'source.patch.f32',dtype='<f4')
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
  tx=torch.from_numpy(x.copy()).to(torch.bfloat16);tw=torch.from_numpy(w.reshape(1024,3,2,16,16).copy()).to(torch.bfloat16);tb=torch.from_numpy(b.copy()).to(torch.bfloat16)
  reference=torch.nn.functional.conv3d(tx.reshape(560,3,2,16,16),tw,tb,stride=(2,16,16)).float().numpy().reshape(-1)
  report['original_conv_control']=compare(a,reference);report['original_conv_control']['retained_sha256']=sha(a.tobytes());report['original_conv_control']['reconstructed_sha256']=sha(reference.tobytes())
  if not report['original_conv_control']['bit_exact']:raise ValueError('original Conv3D control differs')
  linear=torch.nn.functional.linear(tx,tw.reshape(1024,1536),tb).float().numpy().reshape(-1);report['torch_bf16_linear_instead_of_conv']=compare(a,linear)
  float_conv=torch.nn.functional.conv3d(tx.float().reshape(560,3,2,16,16),tw.float(),tb.float(),stride=(2,16,16)).to(torch.bfloat16).float().numpy().reshape(-1);report['torch_f32_conv_then_bf16']=compare(a,float_conv)
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF']);run('build',['cmake','--build',build,'--target','lighton4b-vision-conv-operators','--parallel','2']);run('operators',[build/'lighton4b-vision-conv-operators',stages])
  report['modes']={mode:compare(a,np.fromfile(stages/(mode+'.f32'),dtype='<f4'))for mode in ('partial4_fma','partial4_muladd','partial4_pairwise','double','patch-linear.unset','patch-linear.empty','patch-linear.0','patch-linear.1')}
  baseline=json.loads((ROOT/'tests/results/qwen35-ocr/lightonocr3-4b-vision-fma-baseline-stages.json').read_text())['stages']['vision.patch']['sha256'];prior=json.loads((ROOT/'tests/results/qwen35-ocr/lightonocr3-4b-vision-linear-ab.json').read_text())['composed']['corrected']['stages']['vision.patch']['sha256']
  if any(sha((stages/('patch-linear.'+g+'.f32')).read_bytes())!=baseline for g in ('unset','empty','0'))or sha((stages/'patch-linear.1.f32').read_bytes())!=prior:raise ValueError('patch gate preservation differs')
  report['old_patch_and_nn_linear_alternative_bytes_preserved']=True
  report['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}for p in stages.iterdir()};report['status']='compared'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='bounded Conv3D attribution failed; inspect retained logs';raise RuntimeError(report['error'])from None
 finally:
  (out/'conv-operators.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='artifacts'},indent=2),flush=True)
if __name__=='__main__':main()
