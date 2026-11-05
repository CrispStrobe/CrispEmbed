#!/usr/bin/env python3
"""Synthetic plumbing plus actual official processor preflight; no model weights/forward."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('hosted controls required')
out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'bf16-control-preflight'
r={'schema':'lighton4b-bf16-preflight-controls/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
   'original_model_forward_executed':False,'native_forward_executed':False,'checkpoint_payload_acquired':False,'model_qualified':False}
r['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in('tools/lighton4b_bf16_freegen.py',
    'tests/test_lighton4b_bf16_freegen.py','tools/ci-heavy/lighton4b_bf16_controls.py','docs/lighton4b-bf16-freegen-v1.json')}
def run(label,args):
    with(out/(label+'.log')).open('w')as stream:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
try:
    run('dependencies',[sys.executable,'-m','pip','install','--no-cache-dir','--report',out/'pip-resolution.json','-r',ROOT/'tools/lighton4b-gdn-requirements.txt'])
    run('pip-check',[sys.executable,'-m','pip','check'])
    run('synthetic-controls',[sys.executable,ROOT/'tests/test_lighton4b_bf16_freegen.py','-v'])
    run('official-processor',[sys.executable,ROOT/'tools/lighton4b_bf16_freegen.py','--source-sha',r['source_sha'],
        '--output',out/'processor-preflight','--scratch',scratch,'--threads','2','--prepare-only'])
    prepared=json.loads((out/'processor-preflight/capture.json').read_text())
    if prepared['status']!='prepared'or prepared['original_forward_executed']or'checkpoint'in prepared:
        raise ValueError('processor-only preflight acquired/executed a model')
    r.update(status='passed',actual_grid=prepared['actual_grid'],processor_class=prepared['processor_class'],
        image_processor_class=prepared['image_processor_class'],image_processor_backend=prepared['image_processor_backend'])
finally:(out/'bf16-preflight-controls.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
