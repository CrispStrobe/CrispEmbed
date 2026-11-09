#!/usr/bin/env python3
"""Focused assembled exact source-fed attention source/gate/build controls, no model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
 out=Path(os.environ['HEAVY_OUT']);build=Path(os.environ['HEAVY_SCRATCH'])/'attention-probe/build'
 report={'schema':'lighton4b-vision-attention-integration/v1','status':'failed','quality_qualified':False,'source_fed_attention_bit_exact':True,'composed_attention_qualified':False,'model_forward_executed':False,'default_off':True,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 def run(name,args):
  with (out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  run('operator-probe',['python',ROOT/'tools/ci-heavy/lighton4b_vision_attention_probe.py'])
  r=json.loads((out/'attention-probe.json').read_text())
  if not all(r[k]['bit_exact']for k in ('original_pool_control','actual_matches_retained','trace_matches_actual')):raise ValueError('original/old observer bytes differ')
  if not r['operators']['scores']['sourcefed.dot8']['bit_exact']or not r['operators']['pooled']['sourcefed.pv4']['bit_exact']:raise ValueError('source QK/PV primitives differ')
  if not r['production_qk_pv']['observer']['bit_exact']or not all(v['bit_exact']for v in r['production_qk_pv']['disabled'].values()):raise ValueError('actual production gate/observer differs')
  record=r['artifacts']['actual.corrected_pooled.f32']
  if record!={'bytes':2293760,'sha256':'4de43a53811f70cc5677933a411becd38475678a22dd58f22aa44185e0f2a70b'}:raise ValueError('verified partial candidate bytes differ')
  report['pooled_before']=r['native_pooled_vs_original'];report['pooled_after']=r['production_qk_pv']['corrected'];report['verified_candidate_bytes']=record
  report['softmax_incomplete']={'best_probability_differences':min(v['probability_bit_differences']for v in r['softmax_reductions'].values()),'forced_original_inverse':r['softmax_forced_original_inverse']}
  exact=r['production_exact_softmax']
  if not all(v['bit_exact']for v in (exact['probability_f32'],exact['inverse'],exact['pooled'],*exact['disabled'].values())):raise ValueError('exact softmax integration differs')
  if r['artifacts']['actual.softmax_1.f32']!={'bytes':2293760,'sha256':'de62550a924c77b65e13093dc9d67eca5f4f051c1b243e7748dbe57fcd0bd8b9'}:raise ValueError('verified exact pooling bytes differ')
  report['exact_softmax']=exact;report['verified_exact_bytes']=r['artifacts']['actual.softmax_1.f32']
  report['preservation_readback']=r['preservation_readback']
  run('native-build',['cmake','--build',build,'--target','lighton4b-native-document','test-lighton4b-native-context','test-lighton4b-bf16-linear-controls','--parallel','2'])
  for name in ('test-lighton4b-native-context','test-lighton4b-bf16-linear-controls'):run(name,[build/name])
  run('environment-docs',['python',ROOT/'tools/list_env_vars.py','--check'])
  report['controls_stdout']={n:(out/(n+'.log')).read_text()for n in ('test-lighton4b-native-context','test-lighton4b-bf16-linear-controls','environment-docs')}
  report['runtime_sha256']={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()for n in ('src/lighton4b_vision_bf16.cpp','src/lighton4b_bf16_linear.h','src/lighton4b_linear_block.h','src/qwen35_delta.cpp','src/qwen35_delta.h')};report['status']='passed'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='focused assembled attention controls failed; inspect logs';raise RuntimeError(report['error'])from None
 finally:(out/'attention-integration.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
