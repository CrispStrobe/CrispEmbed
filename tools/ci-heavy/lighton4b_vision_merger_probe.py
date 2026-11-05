#!/usr/bin/env python3
"""Retained-only final-merger/GELU attribution; no full model or quality smoke."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[2]
def main():
 dispatch={'ATEN_CPU_CAPABILITY':'avx2','ONEDNN_MAX_CPU_ISA':'AVX2','DNNL_MAX_CPU_ISA':'AVX2','MKL_ENABLE_INSTRUCTIONS':'AVX2'};os.environ.update(dispatch)
 import numpy as np
 import torch
 from torch.nn import functional as F
 from torch.utils.cpp_extension import include_paths,library_paths
 from kaggle.api.kaggle_api_extended import KaggleApi
 if os.environ.get('GITHUB_ACTIONS')!='true'or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable'or torch.__version__!='2.13.0+cpu'or np.__version__!='2.3.5'or torch.version.git_version!='cf30153c4c131c8164ee7798e5022d810682e2cb':raise RuntimeError('hosted exact original pins required')
 spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'merger-probe';scratch.mkdir();stages=out/'stages';stages.mkdir();build=scratch/'build'
 report={'schema':'lighton4b-vision-merger-probe/v1','status':'failed','model_full_downloaded':False,'whole_model_hash_independently_verified':False,'full_model_forward_executed':False,'quality_qualified':False,'production_changed':True,'new_production_gate':'CRISPEMBED_LIGHTON4B_VISION_BF16_MERGER_ERF_AVX2','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cpu_dispatch':dispatch,'torch_version':torch.__version__,'torch_git_version':torch.version.git_version}
 def sha(raw):return hashlib.sha256(raw).hexdigest()
 def compare(a,b):
  a=np.asarray(a,dtype='<f4').reshape(-1);b=np.asarray(b,dtype='<f4').reshape(-1);r=comparison.compare(a,b);r['bit_different_values']=int(np.count_nonzero(a.view('<u4')!=b.view('<u4')));r['bit_exact']=r['bit_different_values']==0;return r
 def run(name,args):
  report['last_phase']=name
  with(out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 def request_range(start,end):
  url=f'https://huggingface.co/lightonai/LightOnOCR-3-4B/resolve/b71010f095dc3735de7ee6b9969bff8f8f50b39a/model.safetensors?range_start={start}'
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'}),timeout=120)as response:
   if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/9078620536':raise ValueError('bounded original range not honored')
   raw=response.read(end-start+2)
  if len(raw)!=end-start+1:raise ValueError('original range bytes differ')
  return raw
 try:
  import shutil
  mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())};report['preflight']={'load_average':os.getloadavg(),'memory_available_bytes':mem['MemAvailable'],'swap_total_bytes':mem['SwapTotal'],'swap_free_bytes':mem['SwapFree'],'scratch_free_bytes':shutil.disk_usage(scratch).free}
  if mem['MemAvailable']<8*1024**3 or report['preflight']['scratch_free_bytes']<8*1024**3:raise ValueError('bounded resource preflight failed')
  # Compile source and native observers before acquiring retained inputs/parameters.
  args=['c++','-O2','-std=c++17','-mavx2','-mfma','-DCPU_CAPABILITY=AVX2','-DCPU_CAPABILITY_AVX2','-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),'-fno-math-errno','-fno-trapping-math','-ffp-contract=off'];args+=['-I'+p for p in include_paths()];args +=[ROOT/'tests/lighton4b_merger_gelu_vector_source.cpp','-o',scratch/'source-gelu']
  for p in library_paths():args+=['-L'+p,'-Wl,-rpath,'+p]
  run('compile-source-gelu',args+['-ltorch_cpu','-lc10'])
  text=(ROOT/'src/lighton4b_vision_bf16.cpp').read_text();start=text.index('    if (!source_avx2_merger_activation(output.merger_activation))');end=text.index('    return linear(w.merger_down',start);expression=text[start:end]
  (scratch/'vision_merger_expression.inc').write_text('static void production_merger_activation(std::vector<float> &x) { vision_outputs output;output.merger_activation=x;\n'+expression+'x=std::move(output.merger_activation);}\n');report['actual_expression_sha256']=sha(expression.encode());report['runtime_sha256']=sha(text.encode())
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DLIGHTON4B_BOUNDARY_GENERATED_DIR='+str(scratch)])
  run('build',['cmake','--build',build,'--target','lighton4b-vision-merger-probe','--parallel','2'])
  prefix=request_range(0,89463)
  if int.from_bytes(prefix[:8],'little')!=89456 or sha(prefix[8:])!='a1e36638b195f2d5943e000fcb069b275cb22ccc28b1b6184ddee7bdea7ef081':raise ValueError('original header differs')
  header=json.loads(prefix[8:]);low=9016121344;high=9070664704;payload=request_range(89464+low,89464+high-1)
  mapping={'up.bias':('linear_fc1.bias',[4096]),'up.weight':('linear_fc1.weight',[4096,4096]),'down.bias':('linear_fc2.bias',[2560]),'down.weight':('linear_fc2.weight',[2560,4096]),'norm.bias':('norm.bias',[1024]),'norm.weight':('norm.weight',[1024])}
  params={}
  for name,(key,shape)in mapping.items():
   row=header['model.visual.merger.'+key];lo,hi=row['data_offsets']
   if row['dtype']!='BF16'or row['shape']!=shape or not low<=lo<hi<=high:raise ValueError('parameter identity/shape differs')
   raw=payload[lo-low:hi-low];expanded=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4').copy();expanded.tofile(stages/('weight.'+name+'.f32'));params[name]={'source':row,'bytes':len(raw),'bf16_sha256':sha(raw),'f32_sha256':sha(expanded.tobytes())}
  report['source_parameters']={'repository':'lightonai/LightOnOCR-3-4B','revision':'b71010f095dc3735de7ee6b9969bff8f8f50b39a','file':'model.safetensors','header_sha256':sha(prefix[8:]),'range_start':89464+low,'range_end':89464+high-1,'range_bytes':len(payload),'range_sha256':sha(payload),'tensors':params};del payload
  item=comparison.api('actions/artifacts/13434227834')
  if item['workflow_run']['id']!=42065352208 or item['expired']or item['size_in_bytes']!=238132631:raise ValueError('retained actual identity differs')
  archive=scratch/'actual.zip'
  with archive.open('wb')as f:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/13434227834/zip'],stdout=f,check=True)
  if sha(archive.read_bytes())!='8fe151a8ee56e07a1f81718a35cd0db507fa46c12ff54ee8801c664bb4cb293a':raise ValueError('retained actual ZIP differs')
  with zipfile.ZipFile(archive)as z:
   actual=json.loads(z.read('corrected/native-document.json'))
   if actual['source_sha']!='40c68f35f81c8403eb22b27b642037df3fd34c8d':raise ValueError('actual runtime source differs')
   for source,target in (('vision.layer.23','actual.hidden'),('vision.merged','actual.merged')):
    raw=z.read('corrected/q4_k/'+source+'.f32');row=actual['artifacts'][source+'.f32']
    if len(raw)!=row['bytes']or sha(raw)!=row['sha256']:raise ValueError('actual retained array differs')
    (stages/(target+'.f32')).write_bytes(raw)
  archive.unlink();report['actual_retained']={'run_id':42065352208,'artifact_id':13434227834,'source_sha':actual['source_sha'],'exact_zip_verified':True,'actual_gguf_weights_separately_downloaded':False,'parameter_compatibility_requires_complete_actual_merged_byte_control':True}
  provider=KaggleApi();provider.authenticate();provider.kernels_output(os.environ['KAGGLE_ACCOUNT']+'/lighton4b-bf16-targeted-original-cpu-prefill',str(scratch/'original'),file_pattern=r'^lighton4b-bf16-prefill/stages/(prefill\.json|vision\.layer\.23\.f32|vision\.merged\.f32)$',quiet=True)
  original=scratch/'original/lighton4b-bf16-prefill/stages';manifest=original/'prefill.json'
  if sha(manifest.read_bytes())!=comparison.ORIGINAL_REPORT_SHA:raise ValueError('original manifest differs')
  receipt=json.loads(manifest.read_text());report['original_manifest_sha256']=sha(manifest.read_bytes())
  for source,target in (('vision.layer.23','source.hidden'),('vision.merged','source.merged')):
   row=receipt['stages'][source];raw=(original/row['file']).read_bytes()
   if len(raw)!=row['bytes']or sha(raw)!=row['sha256']:raise ValueError('original source array differs')
   (stages/(target+'.f32')).write_bytes(raw)
  torch.set_num_threads(2);torch.set_num_interop_threads(1)
  def array(name):return np.fromfile(stages/(name+'.f32'),dtype='<f4')
  def tensor(name,shape):return torch.from_numpy(array(name).copy()).reshape(shape).to(torch.bfloat16)
  def emit(name,x):x.float().contiguous().numpy().astype('<f4').tofile(stages/(name+'.f32'))
  nw=tensor('weight.norm.weight',(1024,));nb=tensor('weight.norm.bias',(1024,));uw=tensor('weight.up.weight',(4096,4096));ub=tensor('weight.up.bias',(4096,));dw=tensor('weight.down.weight',(2560,4096));db=tensor('weight.down.bias',(2560,))
  for label in ('source','actual'):
   h=tensor(label+'.hidden',(560,1024));n=F.layer_norm(h,(1024,),nw,nb,1e-6);u=F.linear(n.reshape(140,4096),uw,ub);a=F.gelu(u,approximate='none');m=F.linear(a,dw,db)
   for suffix,x in (('norm',n),('up',u),('activation',a),('merged',m)):emit(label+'.reference_'+suffix,x)
  report['original_merger_control']=compare(array('source.merged'),array('source.reference_merged'))
  if not report['original_merger_control']['bit_exact']:raise ValueError('complete original merger control differs')
  os.environ.update({g:'1'for g in ('CRISPEMBED_QWEN35_OCR','CRISPEMBED_LIGHTON4B_VISION_BF16_FMA_NORM','CRISPEMBED_LIGHTON4B_VISION_BF16_LINEAR')});run('native-merger',[build/'lighton4b-vision-merger-probe',stages])
  report['actual_merger_control']=compare(array('actual.merged'),array('actual.native_merged'))
  if not report['actual_merger_control']['bit_exact']:raise ValueError('actual retained merger control differs')
  report['operators']={}
  for label in ('source','actual'):
   run('source-header-'+label,[scratch/'source-gelu',stages/(label+'.reference_up.f32'),stages/(label+'.header_activation.f32')])
   report['operators'][label]={'norm':compare(array(label+'.reference_norm'),array(label+'.native_norm')),'up':compare(array(label+'.reference_up'),array(label+'.native_up')),'original_header':compare(array(label+'.reference_activation'),array(label+'.header_activation')),'activation_std_erf':compare(array(label+'.reference_activation'),array(label+'.native_activation')),'source_erf_activation':compare(array(label+'.reference_activation'),array(label+'.source_erf_activation')),'native_merged_vs_source_math':compare(array(label+'.reference_merged'),array(label+'.native_merged')),'source_erf_merged_vs_source_math':compare(array(label+'.reference_merged'),array(label+'.source_erf_merged'))}
   if not all(report['operators'][label][k]['bit_exact']for k in ('norm','up','original_header')):raise ValueError('identical merger input/source header control differs')
  report['production_merger']={label:{'activation':compare(array(label+'.reference_activation'),array(label+'.production_erf_activation')),'merged':compare(array(label+'.reference_merged'),array(label+'.production_erf_merged')),'disabled':{g:compare(array(label+'.native_activation'),array(label+'.disabled_'+g))for g in ('unset','empty','0')}}for label in ('source','actual')}
  if not all(r['activation']['bit_exact']and r['merged']['bit_exact']and all(v['bit_exact']for v in r['disabled'].values())for r in report['production_merger'].values()):raise ValueError('production merger erf/gate controls differ')
  report['production_merger_guard_controls']={'nonfinite_rejected_unchanged':True,'unsupported_rounding_rejected_unchanged':True}
  report['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}for p in stages.iterdir()};report['status']='attributed'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='bounded retained merger attribution failed; inspect phase/logs';raise RuntimeError(report['error'])from None
 finally:(out/'merger-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='artifacts'},indent=2),flush=True)
if __name__=='__main__':main()
