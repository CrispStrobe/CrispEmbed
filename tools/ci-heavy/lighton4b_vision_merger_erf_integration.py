#!/usr/bin/env python3
"""Focused measured merger-erf production/gate/build checks, no model smoke."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
 out=Path(os.environ['HEAVY_OUT']);build=Path(os.environ['HEAVY_SCRATCH'])/'merger-probe/build'
 report={'schema':'lighton4b-vision-merger-erf-integration/v1','status':'failed','quality_qualified':False,'model_full_downloaded':False,'full_model_forward_executed':False,'default_off':True,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 def run(name,args):
  with(out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  spec=importlib.util.spec_from_file_location('merger',ROOT/'tools/ci-heavy/lighton4b_vision_merger_probe.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()
  r=json.loads((out/'merger-probe.json').read_text());report['original_merger_control']=r['original_merger_control'];report['actual_merger_control']=r['actual_merger_control'];report['production_merger']=r['production_merger'];report['guard_controls']=r['production_merger_guard_controls'];report['before']=r['operators']
  if not r['original_merger_control']['bit_exact']or not r['actual_merger_control']['bit_exact']:raise ValueError('complete source/actual merger differs')
  run('native-build',['cmake','--build',build,'--target','lighton4b-native-document','test-lighton4b-native-context','test-lighton4b-bf16-linear-controls','--parallel','2'])
  for name in ('test-lighton4b-native-context','test-lighton4b-bf16-linear-controls'):run(name,[build/name])
  run('environment-docs',['python',ROOT/'tools/list_env_vars.py','--check']);report['controls_stdout']={n:(out/(n+'.log')).read_text()for n in ('test-lighton4b-native-context','test-lighton4b-bf16-linear-controls','environment-docs')};report['runtime_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in ('src/lighton4b_vision_bf16.cpp','src/lighton4b_vision_avx2_math.h')};report['status']='passed'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='focused production merger-erf controls failed; inspect logs';raise RuntimeError(report['error'])from None
 finally:(out/'merger-erf-integration.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
