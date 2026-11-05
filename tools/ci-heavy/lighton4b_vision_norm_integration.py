#!/usr/bin/env python3
"""Targeted assembled PR155 norm/build/gate controls; no model inference."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
 out=Path(os.environ['HEAVY_OUT']);build=Path(os.environ['HEAVY_SCRATCH'])/'norm-attribution/build'
 report={'schema':'lighton4b-vision-norm-integration/v1','status':'failed','model_forward_executed':False,'quality_qualified':False,'default_off':True,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 def run(name,args,expected=0):
  with (out/(name+'.log')).open('w')as f:
   result=subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode!=expected:raise ValueError('targeted control failed: '+name)
 try:
  spec=importlib.util.spec_from_file_location('operators',ROOT/'tools/ci-heavy/lighton4b_vision_norm_attribution.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()
  operators=json.loads((out/'norm-attribution.json').read_text())
  if not operators['candidate_operator_bit_exact']:raise ValueError('norm operator differs')
  byte_controls={}
  for n in ('norm1','norm2'):
   records=operators['artifacts'];source=records['source.'+n+'.f32'];candidate=records['production.'+n+'.1.f32']
   if source!=candidate:raise ValueError('norm raw bytes differ')
   old=records['production.'+n+'.unset.f32']
   if any(records['production.'+n+'.'+g+'.f32']!=old for g in ('empty','0')):raise ValueError('gate bytes differ')
   if records['bf16_8_fma.'+n+'.stats.f32']!=records['source_fast.'+n+'.stats.f32']:raise ValueError('moments raw bytes differ')
   byte_controls[n]={'source':source,'candidate':candidate,'empty_zero_equal_unset':True,'moments_equal_source_fast':True}
  report['byte_controls']=byte_controls
  run('integration-build',['cmake','--build',build,'--target','test-lighton4b-bf16-round','test-lighton4b-native-context','test-lighton4b-image-compose','lighton4b-native-document','--parallel','2'])
  for name in ('test-lighton4b-bf16-round','test-lighton4b-native-context','test-lighton4b-image-compose'):run(name,[build/name])
  run('document-argument-control',[build/'lighton4b-native-document'],2)
  run('environment-doc-controls',['python',ROOT/'tools/list_env_vars.py','--check'])
  report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in ('src/lighton4b_vision_bf16.cpp','src/lighton4b_image_compose.h','src/qwen35_vision.cpp','src/qwen35_norm.cpp','src/qwen35_vision.h','src/lighton4b_vision_owner.cpp','src/qwen35_linear.cpp')}
  report['controls_stdout']={name:(out/(name+'.log')).read_text()for name in ('test-lighton4b-bf16-round','test-lighton4b-native-context','test-lighton4b-image-compose','environment-doc-controls')}
  report['status']='passed'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='targeted assembled-source controls failed; inspect logs';raise RuntimeError(report['error'])from None
 finally:
  (out/'norm-integration.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
