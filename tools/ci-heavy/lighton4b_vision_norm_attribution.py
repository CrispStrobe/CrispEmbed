#!/usr/bin/env python3
"""Hosted model-free retained-input BF16 norm reduction/affine attribution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[2]
def main():
 import numpy as np
 import torch
 from torch.utils.cpp_extension import include_paths,library_paths
 if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':raise RuntimeError('hosted portable only')
 if torch.__version__!='2.13.0+cpu' or np.__version__!='2.3.5':raise RuntimeError('original pins differ')
 spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'norm-attribution';scratch.mkdir();stages=out/'stages';stages.mkdir();build=scratch/'build'
 report={'schema':'lighton4b-vision-norm-attribution/v1','status':'failed','model_downloaded':False,'model_forward_executed':False,'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'torch_version':torch.__version__,'torch_cpu_capability':torch.backends.cpu.get_cpu_capability(),'torch_build_config':torch.__config__.show()}
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def run(name,argv):
  with (out/(name+'.log')).open('w')as f:subprocess.run([str(v)for v in argv],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  artifact=comparison.api('actions/artifacts/13414500337')
  if artifact['workflow_run']['id']!=42024288946 or artifact['workflow_run']['head_sha']!='3b194f40b9725e327bb1d0a6ebcb40ab05aeacf5':raise ValueError('retained artifact source differs')
  if artifact['expired'] or artifact['size_in_bytes']!=238129493:raise ValueError('retained artifact identity differs')
  archive=scratch/'retained.zip'
  with archive.open('wb')as f:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/13414500337/zip'],stdout=f,check=True)
  if sha(archive)!='77682c5af6a8a89af78736eb70241fdb8706ee801776703ddbdf42000fb273ff':raise ValueError('retained ZIP differs')
  selected=['source.embedded','source.residual1','source.norm1','source.norm2']+['weight.'+n+'.'+kind for n in ('norm1','norm2')for kind in ('weight','bias')]
  with zipfile.ZipFile(archive)as z:
   receipt=json.loads(z.read('vision-layer0.json'))
   if not receipt['original_control_bit_exact'] or receipt['source_sha']!=artifact['workflow_run']['head_sha']:raise ValueError('retained original control differs')
   for name in selected:
    filename=name+'.f32';raw=z.read('stages/'+filename);record=receipt['artifacts'][filename]
    if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('retained input differs')
    (stages/filename).write_bytes(raw)
  report['retained_artifact']={'run_id':42024288946,'artifact_id':13414500337,'sha256':sha(archive),'original_control_bit_exact':True}
  archive.unlink()
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  run('build',['cmake','--build',build,'--target','lighton4b-vision-norm-variants','--parallel','2'])
  report['source_controls']={}
  for contract in ('off','fast'):
   mode='source_'+contract;binary=scratch/mode
   argv=['c++','-O3','-std=c++17','-mavx2','-mfma','-ffp-contract='+contract,'-DCPU_CAPABILITY=AVX2','-DCPU_CAPABILITY_AVX2','-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
   argv += ['-I'+p for p in include_paths()]
   argv += [str(ROOT/'tests/lighton4b_vision_norm_source.cpp'),'-o',str(binary)]
   for p in library_paths():argv+=['-L'+p,'-Wl,-rpath,'+p]
   argv+=['-ltorch_cpu','-lc10']
   run('compile-'+mode,argv);run(mode,[binary,stages,mode])
   report['source_controls'][mode]={n:comparison.compare(np.fromfile(stages/('source.'+n+'.f32'),dtype='<f4'),np.fromfile(stages/(mode+'.'+n+'.control.f32'),dtype='<f4'))for n in ('norm1','norm2')}
  run('variants',[build/'lighton4b-vision-norm-variants',stages])
  report['variants']={};report['stats']={}
  for n in ('norm1','norm2'):
   a=np.fromfile(stages/('source.'+n+'.f32'),dtype='<f4');report['variants'][n]={}
   for mode in ('old4','bf16_8','bf16_8_fma','source_off','source_fast'):
    for first in (0,1):
     for second in (0,1):
      label=mode+'.'+n+'.fma'+str(first)+str(second)
      report['variants'][n][label]=comparison.compare(a,np.fromfile(stages/(label+'.f32'),dtype='<f4'))
   report['stats'][n]={mode:comparison.compare(np.fromfile(stages/('source_off.'+n+'.stats.f32'),dtype='<f4'),np.fromfile(stages/(mode+'.'+n+'.stats.f32'),dtype='<f4'))for mode in ('old4','bf16_8','bf16_8_fma','source_fast')}
  report['production_controls']={}
  for n in ('norm1','norm2'):
   ref=np.fromfile(stages/('source.'+n+'.f32'),dtype='<f4');old=np.fromfile(stages/('production.'+n+'.unset.f32'),dtype='<f4')
   report['production_controls'][n]={'candidate':comparison.compare(ref,np.fromfile(stages/('production.'+n+'.1.f32'),dtype='<f4')),'old_gate_controls':{g:comparison.compare(old,np.fromfile(stages/('production.'+n+'.'+g+'.f32'),dtype='<f4'))for g in ('empty','0')}}
  report['candidate_operator_bit_exact']=all(v['candidate']['different_values']==0 for v in report['production_controls'].values())
  if any(c['different_values'] for v in report['production_controls'].values()for c in v['old_gate_controls'].values()):raise ValueError('old gate controls differ')
  report['corrected_stats_to_source_fast']={n:comparison.compare(np.fromfile(stages/('source_fast.'+n+'.stats.f32'),dtype='<f4'),np.fromfile(stages/('bf16_8_fma.'+n+'.stats.f32'),dtype='<f4'))for n in ('norm1','norm2')}
  report['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}for p in stages.iterdir()}
  report['status']='compared'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='bounded operator attribution failed; inspect retained logs';raise RuntimeError(report['error'])from None
 finally:
  (out/'norm-attribution.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({k:v for k,v in report.items()if k not in ('artifacts','torch_build_config')},indent=2),flush=True)
if __name__=='__main__':main()
