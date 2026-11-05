#!/usr/bin/env python3
"""Hosted BF16 quantized comparison controls; synthetic bytes only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':raise SystemExit('hosted portable controls required')
out=Path(os.environ['HEAVY_OUT']);build=Path(os.environ['HEAVY_SCRATCH'])/'bf16-quant-build'
r={'schema':'lighton4b-bf16-quant-controls/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'original_forward_executed':False,'native_forward_executed':False,'model_payload_bytes':0,'real_provider_write_executed':False,
    'original_execution_admitted':False,'native_execution_admitted':False,'large_retention_qualified':False,'actual_original_output_ready':False}
r['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in(
    'tools/lighton4b_bf16_quant.py','tests/test_lighton4b_bf16_quant.py','docs/lighton4b-bf16-quant-v1.json','tools/ci-heavy/lighton4b_bf16_quant_controls.py',
    'tests/lighton4b_document_diff.cpp','src/crispembed_diff.h','tools/lighton4b_original_document.py','docs/lighton4b-document-parity-v1.json')}
def run(label,args,env=None):
    with(out/(label+'.log')).open('w')as stream:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,env=env)
try:
    run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF'])
    run('build',['cmake','--build',build,'--target','crispembed-diff-lighton4b-document','--parallel','2'])
    run('controls',[sys.executable,ROOT/'tests/test_lighton4b_bf16_quant.py','-v'],dict(os.environ,LIGHTON4B_DOCUMENT_DIFF=str(build/'crispembed-diff-lighton4b-document')))
    r.update(status='passed',controls_log_sha256=hashlib.sha256((out/'controls.log').read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256((build/'crispembed-diff-lighton4b-document').read_bytes()).hexdigest())
finally:(out/'bf16-quant-controls.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
