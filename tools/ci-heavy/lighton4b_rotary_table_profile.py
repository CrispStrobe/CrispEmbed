#!/usr/bin/env python3
"""Complete model-math table generation/domain/reader proof, no model or image."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
def main():
 dispatch={'ATEN_CPU_CAPABILITY':'avx2','ONEDNN_MAX_CPU_ISA':'AVX2','DNNL_MAX_CPU_ISA':'AVX2','MKL_ENABLE_INSTRUCTIONS':'AVX2'};os.environ.update(dispatch)
 if os.environ.get('GITHUB_ACTIONS')!='true'or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':raise RuntimeError('hosted portable profile proof only')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'rotary-table';scratch.mkdir();build=scratch/'build'
 report={'schema':'lighton4b-rotary-table-profile/v1','status':'failed','production_integrated':True,'images_read':False,'activations_read':False,'checkpoint_read':False,'model_forward_executed':False,'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'cpu_dispatch':dispatch}
 def sha(raw):return hashlib.sha256(raw).hexdigest()
 def run(name,args):
  report['last_phase']=name
  with(out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())};report['preflight']={'load_average':os.getloadavg(),'memory_available_bytes':mem['MemAvailable'],'swap_total_bytes':mem['SwapTotal'],'swap_free_bytes':mem['SwapFree'],'scratch_free_bytes':shutil.disk_usage(scratch).free}
  if mem['MemAvailable']<8*1024**3 or report['preflight']['scratch_free_bytes']<8*1024**3:raise ValueError('bounded resource preflight failed')
  source_url='https://raw.githubusercontent.com/huggingface/transformers/75d3bdcd4b3ba70cee21287218d9764f33da41f0/src/transformers/models/qwen3_5/modeling_qwen3_5.py'
  with urllib.request.urlopen(source_url,timeout=60)as f:source=f.read(200001)
  if len(source)>200000 or sha(source)!='aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048':raise ValueError('original modeling source differs')
  if b'theta: float = 10000.0'not in source or b'Qwen3_5VisionRotaryEmbedding(head_dim // 2)'not in source:raise ValueError('source theta/dimension changed')
  report['transformers_default_control']={'source':source_url,'sha256':sha(source),'head_dimension':64,'embedding_dimension':32,'theta':10000.0}
  generator=ROOT/'tools/generate_lighton4b_rotary_table.py'
  run('generate',['python',generator,'--output',out/'lighton4b_rotary_table_data.inc','--manifest',out/'rotary-table-generation.json'])
  run('regenerate',['python',generator,'--output',scratch/'lighton4b_rotary_table_data.inc','--manifest',scratch/'regeneration.json'])
  if(out/'lighton4b_rotary_table_data.inc').read_bytes()!=(scratch/'lighton4b_rotary_table_data.inc').read_bytes():raise ValueError('regeneration bytes differ')
  report['generation']=json.loads((out/'rotary-table-generation.json').read_text());report['regeneration_bit_exact']=True
  if(out/'lighton4b_rotary_table_data.inc').read_bytes()!=(ROOT/'src/lighton4b_rotary_table_data.inc').read_bytes():raise ValueError('committed model constants differ from full regeneration')
  report['committed_asset_regenerated_exact']=True
  import numpy as np
  import torch
  from torch.utils.cpp_extension import library_paths
  if torch.__version__!='2.13.0+cpu'or torch.version.git_version!='cf30153c4c131c8164ee7798e5022d810682e2cb'or np.__version__!='2.3.5':raise ValueError('original reference versions differ')
  torch.set_num_threads(2);torch.set_num_interop_threads(1)
  inverse=1.0/(10000.0**(torch.arange(0,32,2,dtype=torch.float32)/32));angles=torch.outer(torch.arange(1024,dtype=torch.float32),inverse)
  # Direct original vendor route confirms all angle constants, not an inferred SLEEF substitute.
  lib=ctypes.CDLL(str(Path(library_paths()[0])/'libtorch_cpu.so'));pointer=ctypes.POINTER(ctypes.c_float);a=angles.numpy().copy().reshape(-1);report['direct_mkl_control']={}
  for name,reference in (('Cos',angles.cos().numpy()),('Sin',angles.sin().numpy())):
   output=np.empty_like(a);fn=getattr(lib,'vms'+name);fn.argtypes=[ctypes.c_int,pointer,pointer,ctypes.c_longlong];fn.restype=None;fn(len(a),a.ctypes.data_as(pointer),output.ctypes.data_as(pointer),2)
   differences=int(np.count_nonzero(reference.reshape(-1).view('<u4')!=output.view('<u4')));report['direct_mkl_control'][name.lower()]={'elements':len(a),'bit_different_values':differences}
   if differences:raise ValueError('direct original vendor route differs')
  text=(ROOT/'src/lighton4b_vision_bf16.cpp').read_text()
  if 'h > 1024 || w < 1 || w > 1024'not in text or 'tokens > 4096'not in text:raise ValueError('admitted model geometry changed; reproving bounds required')
  report['geometry_domain']={'max_axis_size':1024,'max_coordinate':1023,'merge':2,'max_patch_area':2048,'ordinary_patch_area':1024,'max_total_tokens':4096,'max_grid_rows':16,'max_frames_per_row':16,'proof':'geometry caps h,w at1024, so every emitted row/column is0..1023. Temporal frames and grid rows repeat the same two coordinate axes; no image or frame-dependent arithmetic is omitted. Fixed profile width1024/heads16 yields d64 and16 frequencies; theta10000 is original default.','runtime_source_sha256':sha(text.encode())}
  cases=[];seen=set()
  def add(grid):
   key=tuple(tuple(row)for row in grid);tokens=sum(t*h*w for t,h,w in key)
   if key in seen:return
   if not 0<tokens<=4096 or len(key)>16 or any(t<1 or t>16 or h<2 or h>1024 or w<2 or w>1024 or h%2 or w%2 or h*w>2048 for t,h,w in key):raise ValueError('invalid proof geometry')
   seen.add(key);cases.append((tokens,key))
  for h in range(2,1025,2):
   for w in range(2,min(1024,2048//h)+1,2):add([(1,h,w)])
  single_count=len(cases)
  for total in range(4,4097,4):add([(1,2,total//2)]if total<=2048 else[(1,2,1024),(1,2,(total-2048)//2)])
  for h,w in ((2,2),(2,1024),(1024,2),(8,16),(16,8),(32,64),(64,32),(4,512),(512,4)):
   for frames in range(1,min(16,4096//(h*w))+1):add([(frames,h,w)])
  add([(1,2,1024),(1,1024,2)]);add([(16,2,2)]*16)
  for prefix in range(1,8):
   for h,w in ((2,1024),(1024,2),(32,64),(64,32)):add([(1,2,2)]*prefix+[(1,h,w)])
  case_file=scratch/'cases.txt';case_file.write_text(''.join(str(n)+' '+str(len(g))+' '+' '.join(str(x)for row in g for x in row)+'\n'for n,g in cases))
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF','-DLIGHTON4B_ROTARY_TABLE_GENERATED_DIR='+str(scratch)])
  run('build',['cmake','--build',build,'--target','lighton4b-rotary-table-profile','--parallel','2'])
  binary=build/'lighton4b-rotary-table-profile';actual_file=scratch/'native-cases.f32';run('native-domain',[binary,case_file,actual_file,out/'rotary_table.native.u32'])
  controls=json.loads((out/'native-domain.log').read_text());report['native_controls']=controls
  if sha((out/'rotary_table.native.u32').read_bytes())!=report['generation']['raw_sha256']:raise ValueError('compiled complete constant bytes differ')
  run('native-dependencies',['ldd',binary]);deps=(out/'native-dependencies.log').read_text();forbidden=any(name in deps.lower()for name in ('libmkl','libtorch','libc10'))
  report['native_dependencies']={'forbidden_mkl_torch_c10':forbidden,'sonames':[line.strip().split()[0]for line in deps.splitlines()if line.strip()]}
  if forbidden:raise ValueError('native reader acquired reference runtime dependency')
  expected_hash=hashlib.sha256();actual_hash=hashlib.sha256();total_values=0
  with actual_file.open('rb')as actual:
   for index,(tokens,grid)in enumerate(cases):
    coordinates=[]
    for frames,h,w in grid:
     rr=torch.arange(h).view(h,1).expand(h,w).reshape(h//2,2,w//2,2).permute(0,2,1,3).reshape(-1)
     cc=torch.arange(w).view(1,w).expand(h,w).reshape(h//2,2,w//2,2).permute(0,2,1,3).reshape(-1)
     coordinates.append(torch.stack((rr,cc),dim=-1).repeat(frames,1))
    coord=torch.cat(coordinates,dim=0);max_grid=max(max(h,w)for t,h,w in grid);freq=torch.outer(torch.arange(max_grid,dtype=torch.float32),inverse)[coord].flatten(1);embedding=torch.cat((freq,freq),dim=-1)
    for label,reference in (('cosine',embedding.cos()),('sine',embedding.sin())):
     expected=reference.numpy().astype('<f4',copy=False).tobytes();received=actual.read(len(expected));expected_hash.update(expected);actual_hash.update(received);total_values+=reference.numel()
     if received!=expected:
      aa=np.frombuffer(expected,dtype='<u4');bb=np.frombuffer(received,dtype='<u4');report['first_failing_geometry']={'case':index,'tokens':tokens,'grid':grid,'operator':label,'bit_different_values':int(np.count_nonzero(aa!=bb))if len(aa)==len(bb)else None};raise ValueError('complete-profile source/reader geometry mismatch')
    if index%512==0:print(json.dumps({'phase':'complete-source-geometry','checked':index,'cases':len(cases)}),flush=True)
   if actual.read(1):raise ValueError('native output trailing bytes')
  report['source_vs_reader']={'bit_exact':True,'single_frame_h_w_cases':single_count,'all_total_token_counts':1024,'cases':len(cases),'float_values':total_values,'expected_sha256':expected_hash.hexdigest(),'actual_sha256':actual_hash.hexdigest(),'transient_bytes':actual_file.stat().st_size,'images_or_activations_used':False};actual_file.unlink();report['status']='proved'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='complete fixed-profile rotary arithmetic proof failed; inspect phase/logs';raise RuntimeError(report['error'])from None
 finally:(out/'rotary-table-profile.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
