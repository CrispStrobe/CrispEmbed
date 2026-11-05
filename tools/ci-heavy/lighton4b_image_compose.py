#!/usr/bin/env python3
"""Hosted model-free image splice/processor controls; no checkpoints."""
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':
        raise RuntimeError('hosted portable execution required')
    out=Path(os.environ['HEAVY_OUT'])
    scratch=Path(os.environ['HEAVY_SCRATCH'])/'lighton4b-image-compose'
    scratch.mkdir(parents=True,exist_ok=False)
    report={'schema':'lighton4b-image-compose/v1','status':'failed','model_inference':False,
            'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    def run(label,argv):
        with (out/(label+'.log')).open('w') as log:
            subprocess.run([str(x) for x in argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        run('configure',['cmake','-S',ROOT,'-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release',
             '-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        run('build',['cmake','--build',scratch/'build','--target','test-lighton4b-image-compose','--parallel','2'])
        run('controls',[scratch/'build/test-lighton4b-image-compose'])
        report['controls_stdout']=(out/'controls.log').read_text()
        if 'passed=1' not in report['controls_stdout']:raise ValueError('missing controls marker')
        report['status']='passed'
    finally:
        (out/'image-compose-receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
