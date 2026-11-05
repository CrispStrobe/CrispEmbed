#!/usr/bin/env python3
"""Independent synthetic HTTP service transport controls, no model/provider write."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('hosted execution required')
out=Path(os.environ['HEAVY_OUT'])
r={'schema':'lighton4b-sharded-transport-controls/v1','status':'failed',
   'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
   'original_forward_executed':False,'native_forward_executed':False,'model_payload_bytes':0,
   'external_provider_write_executed':False,'external_provider_retention_verified':False,
   'original_transport_admitted':False,'capture_enabled':False,'quality_qualified':False}
r['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in(
    'tools/lighton4b_sharded_transport.py','tests/test_lighton4b_sharded_transport.py',
    'tools/ci-heavy/lighton4b_transport_controls.py','docs/lighton4b-document-parity-v1.json',
    'tools/lighton4b_original_document.py')}
try:
    with(out/'controls.log').open('w')as stream:
        subprocess.run([sys.executable,ROOT/'tests/test_lighton4b_sharded_transport.py','-v'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    r.update(status='passed',controls_log_sha256=hashlib.sha256((out/'controls.log').read_bytes()).hexdigest())
finally:(out/'transport-controls.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
