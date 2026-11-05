#!/usr/bin/env python3
"""Complete model arithmetic plus actual production/first-block/fallback controls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
 out=Path(os.environ['HEAVY_OUT']);build=Path(os.environ['HEAVY_SCRATCH'])/'boundary-probe/build'
 report={'schema':'lighton4b-rotary-table-integration/v1','status':'failed','quality_qualified':False,'default_off':True,'model_full_downloaded':False,'full_model_forward_executed':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 def load(name,path):
  spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
 def run(name,args):
  with(out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  # Separate processes keep original Torch thread/init pins independently enforceable.
  run('complete-profile',['python',ROOT/'tools/ci-heavy/lighton4b_rotary_table_profile.py'])
  profile=json.loads((out/'rotary-table-profile.json').read_text())
  if profile['status']!='proved'or not profile['source_vs_reader']['bit_exact']or not profile['committed_asset_regenerated_exact']:raise ValueError('complete production profile not proved')
  report['complete_profile']={k:profile[k]for k in ('generation','source_vs_reader','native_controls','native_dependencies','geometry_domain')}
  boundary=load('boundary',ROOT/'tools/ci-heavy/lighton4b_vision_boundary_probe.py');boundary.CORRECT_ROTARY=False;boundary.USE_ROTARY_TABLE=True;boundary.main()
  r=json.loads((out/'boundary-probe.json').read_text())
  if not r['corrected_rotary_table_gate']or r['corrected_rotary_gate']or not all(v['bit_exact']for v in r['production_corrections'].values()):raise ValueError('actual source-fed table first-block correction differs')
  if not r['first_block']['both']['vs_actual_candidate']['bit_exact']:raise ValueError('preceding production source bytes differ')
  report['production_source_fed']=r['production_corrections'];report['preceding_first_block_control']=r['first_block']['both']['vs_actual_candidate'];report['disabled_controls']=r['disabled_corrections']
  run('native-build',['cmake','--build',build,'--target','lighton4b-native-document','test-lighton4b-native-context','test-lighton4b-bf16-linear-controls','--parallel','2'])
  for name in ('test-lighton4b-native-context','test-lighton4b-bf16-linear-controls'):run(name,[build/name])
  run('environment-docs',['python',ROOT/'tools/list_env_vars.py','--check']);report['controls_stdout']={n:(out/(n+'.log')).read_text()for n in ('test-lighton4b-native-context','test-lighton4b-bf16-linear-controls','environment-docs')};report['runtime_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in ('src/lighton4b_vision_bf16.cpp','src/lighton4b_rotary_table.h','src/lighton4b_rotary_table_data.inc')};report['status']='passed'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='focused complete rotary production integration failed; inspect phase/logs';raise RuntimeError(report['error'])from None
 finally:(out/'rotary-table-integration.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
