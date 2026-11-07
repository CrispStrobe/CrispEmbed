#!/usr/bin/env python3
"""Focused hosted model-free bit-exact conversion controls; no model/quality claim."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
 if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':raise RuntimeError('hosted portable controls required')
 out=Path(os.environ['HEAVY_OUT']);build=Path(os.environ['HEAVY_SCRATCH'])/'parallel-dequant-build'
 report={'schema':'lighton4b-parallel-dequant-controls/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'model_payload_bytes':0,'quality_qualified':False}
 def run(name,args):
  with (out/(name+'.log')).open('w') as log:subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 try:
  run('configure',['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  run('build',['cmake','--build',str(build),'--target','test-lighton4b-parallel-dequant','--parallel','2'])
  binary=build/'test-lighton4b-parallel-dequant';report['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
  run('controls',[str(binary)]);report['stdout']=(out/'controls.log').read_text()
  if 'PARALLEL_DEQUANT passed=1 bit_exact_cases=48' not in report['stdout']:raise RuntimeError('missing conversion proof')
  report['status']='passed'
 finally:(out/'parallel-dequant-controls.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
