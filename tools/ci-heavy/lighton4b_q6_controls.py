#!/usr/bin/env python3
"""Hosted model-free selective Q6 owner, metadata, dequant and runner builds."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_q6_recipe as recipe
def main():
 if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':raise RuntimeError('hosted portable controls required')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);build=scratch/'q6-controls'
 report={'schema':'lighton4b-q6-controls/v1','status':'failed','model_payload_bytes':0,'model_inference_executed':False,'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 def run(name,args):
  with(out/(name+'.log')).open('w')as f:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  run('recipe-controls',[sys.executable,ROOT/'tests/test_lighton4b_q6_recipe.py'])
  p=recipe.plan();rows=[]
  for row in p['all_tensors']:
   if row['name'].startswith('l.'):
    ne=list(reversed(row['source_shape']));ne += [1]*(4-len(ne));rows.append(' '.join([row['name'],str(row['type'])]+list(map(str,ne))))
  fixture=out/'q6-schema.tsv';fixture.write_text('\n'.join(rows)+'\n');report['recipe']=p['recipe'];report['q6_names_sha256']=p['q6_names_sha256']
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  targets=['test-lighton4b-q6-policy','test-lighton4b-native-context','test-lighton4b-mixed-policy','test-lighton4b-bf16-dequant-weights','lighton4b-native-document','lighton4b-native-case-preflight']
  run('build',['cmake','--build',build,'--target',*targets,'--parallel','2'])
  for name in targets[:4]:run(name,[build/name]+([fixture]if name=='test-lighton4b-q6-policy'else[]))
  for name in targets[4:]:
   result=subprocess.run([build/name],capture_output=True,text=True);report[name+'_argument_returncode']=result.returncode
   if result.returncode!=2:raise ValueError('runner argument guard changed')
  report['stdout']=(out/'test-lighton4b-q6-policy.log').read_text()
  if 'Q6_POLICY passed=1'in report['stdout']:report['status']='passed'
  else:raise ValueError('Q6 controls receipt absent')
  report['binary_sha256']={name:hashlib.sha256((build/name).read_bytes()).hexdigest()for name in targets}
 finally:(out/'q6-controls.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
