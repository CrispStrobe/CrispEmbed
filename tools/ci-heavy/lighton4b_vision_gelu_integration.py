#!/usr/bin/env python3
"""Verified GELU-only integration, rejected rotary disabled; no full model."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
 out=Path(os.environ['HEAVY_OUT']);build=Path(os.environ['HEAVY_SCRATCH'])/'boundary-probe/build'
 report={'schema':'lighton4b-vision-gelu-integration/v1','status':'failed','quality_qualified':False,'rotary_candidate_enabled':False,'model_forward_executed':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 def run(name,args):
  with(out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  spec=importlib.util.spec_from_file_location('boundary',ROOT/'tools/ci-heavy/lighton4b_vision_boundary_probe.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.CORRECT_ROTARY=False;module.main()
  r=json.loads((out/'boundary-probe.json').read_text());report['operator']=r['production_corrections'];report['disabled_controls']=r['disabled_corrections'];report['mkl_source_control']=r['mkl_source_control']
  if r['corrected_rotary_gate']or not r['production_corrections']['activation']['bit_exact']:raise ValueError('GELU correction not exact or rejected rotary enabled')
  for label in ('original','rotary_only'):
   if label=='original'and not r['first_block'][label]['vs_original']['bit_exact']:raise ValueError('original control differs')
  stages=out/'stages';corrected=(stages/'layer0.corrected.f32').read_bytes();expected=(stages/'layer0.rotary_only.f32').read_bytes()
  if corrected!=expected:raise ValueError('GELU-corrected first block differs from original math with native rotary')
  report['first_block_vs_original']=r['production_corrections']['first_block'];report['first_block_expected_bytes_exact']=True
  run('native-build',['cmake','--build',build,'--target','lighton4b-native-document','test-lighton4b-native-context','test-lighton4b-bf16-linear-controls','--parallel','2'])
  for name in ('test-lighton4b-native-context','test-lighton4b-bf16-linear-controls'):run(name,[build/name])
  run('environment-docs',['python',ROOT/'tools/list_env_vars.py','--check']);report['controls_stdout']={n:(out/(n+'.log')).read_text()for n in ('test-lighton4b-native-context','test-lighton4b-bf16-linear-controls','environment-docs')};report['runtime_sha256']=hashlib.sha256((ROOT/'src/lighton4b_vision_bf16.cpp').read_bytes()).hexdigest();report['status']='passed'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='focused GELU-only integration failed; inspect logs';raise RuntimeError(report['error'])from None
 finally:(out/'gelu-integration.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
