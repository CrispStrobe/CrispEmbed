#!/usr/bin/env python3
"""Hosted model-free compilation and negative/type controls for the payload owner."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def main():
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':
        raise RuntimeError('hosted portable execution required')
    out=Path(os.environ['HEAVY_OUT']); build=Path(os.environ['HEAVY_SCRATCH'])/'frozen-case-build'
    report={'schema':'lighton4b-frozen-case-build/v1','status':'failed','model_payload_bytes':0,
            'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    def run(label,args):
        with (out/(label+'.log')).open('w') as log: subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        run('configure',['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
            '-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        run('build',['cmake','--build',str(build),'--target','test-lighton4b-native-context','lighton4b-native-document','test-qwen35-stream-projection','test-lighton4b-mixed-policy','lighton4b-native-case-preflight','--parallel','2'])
        binary=build/'test-lighton4b-native-context'
        report['executable_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
        run('controls',[str(binary)])
        run('stream-projection-controls',[str(build/'test-qwen35-stream-projection')])
        run('mixed-policy-controls',[str(build/'test-lighton4b-mixed-policy')])
        for name in ('lighton4b-native-document','test-qwen35-stream-projection'):
            report[name+'_sha256']=hashlib.sha256((build/name).read_bytes()).hexdigest()
        # Native binary has no model-free success mode; wrong arguments must fail.
        if subprocess.run([str(build/'lighton4b-native-document')],cwd=ROOT).returncode!=2:
            raise RuntimeError('native runner argument control failed')
        report['stdout']=(out/'controls.log').read_text()
        if 'passed=1 model_payload=0' not in report['stdout']:raise RuntimeError('missing control receipt')
        report['status']='passed'
    finally:(out/'frozen-case-build.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
