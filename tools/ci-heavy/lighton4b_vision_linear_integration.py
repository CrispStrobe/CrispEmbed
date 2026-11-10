#!/usr/bin/env python3
"""Assembled norm/nn.Linear/Conv3D controls and exact source ZIP preservation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
 report={'schema':'lighton4b-vision-linear-integration/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'model_full_forward_executed':False,'quality_qualified':False,'all_runtime_defaults_off':True,'artifact_retention_days_requested':90}
 def run(label,args,env=None):
  with (out/(label+'.log')).open('w')as f:
   result=subprocess.run([str(x)for x in args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:report['failed_control']=label;report['failed_returncode']=result.returncode;raise ValueError('assembled control failed')
 try:
  for phase,script,args in (('norm','lighton4b_vision_norm_integration.py',[]),('linear','lighton4b_vision_linear_operators.py',['--preserve-source-archive']),('conv','lighton4b_vision_conv_operators.py',[])):
   phase_out=out/phase;phase_out.mkdir();env=os.environ.copy();env['HEAVY_OUT']=str(phase_out)
   run(phase+'-phase',['python',ROOT/'tools/ci-heavy'/script,*args],env)
  norm=json.loads((out/'norm/norm-integration.json').read_text());linear=json.loads((out/'linear/linear-operators.json').read_text());conv=json.loads((out/'conv/conv-operators.json').read_text())
  if norm['status']!='passed' or not linear['candidate_full_operators_bit_exact'] or not conv['candidate_production_conv_bit_exact']:raise ValueError('actual operator bytes differ')
  report['norm_byte_controls']=norm['byte_controls'];report['linear_original_controls']=linear['original_controls'];report['linear_operators']=linear['operators'];report['conv_original_control']=conv['original_conv_control'];report['conv_production']=conv['production_conv'];report['conv_gate_controls']={k:conv[k]for k in ('old_patch_and_nn_linear_alternative_bytes_preserved','conv_unset_empty_zero_preserves_patch_linear_bytes')}
  archive=linear['preserved_source_archive'];source=out/'linear'/archive['file']
  if archive['bytes']!=238129493 or archive['sha256']!='77682c5af6a8a89af78736eb70241fdb8706ee801776703ddbdf42000fb273ff' or hashlib.sha256(source.read_bytes()).hexdigest()!=archive['sha256']:raise ValueError('preserved exact source ZIP differs')
  report['preserved_source_archive']={**archive,'artifact_member':'linear/'+archive['file'],'source_not_regenerated':True}
  build=scratch/'norm-attribution/build'
  run('decoder-controls-build',['cmake','--build',build,'--target','test-lighton4b-bf16-linear-controls','--parallel','2']);run('decoder-controls',[build/'test-lighton4b-bf16-linear-controls'])
  marker=(out/'decoder-controls.log').read_text()
  if 'checks=20 passed=1 model_payload=0'not in marker:raise ValueError('decoder control census differs')
  report['decoder_controls_stdout']=marker;report['native_controls_stdout']=norm['controls_stdout']
  report['runtime_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in ('src/lighton4b_vision_bf16.cpp','src/lighton4b_image_compose.h','src/qwen35_vision.cpp','src/qwen35_norm.cpp','src/lighton4b_bf16_linear.h','src/lighton4b_linear_block.h','src/lighton4b_decoder_compose.h','src/qwen35_delta.cpp','src/qwen35_delta.h')}
  report['status']='passed'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='bounded assembled operator validation failed; inspect retained logs';raise RuntimeError(report['error'])from None
 finally:
  (out/'linear-integration.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
