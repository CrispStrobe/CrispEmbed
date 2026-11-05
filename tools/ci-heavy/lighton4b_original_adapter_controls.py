#!/usr/bin/env python3
"""Only synthetic original-observer controls; no weights or framework forward."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('hosted controls required')
out=Path(os.environ['HEAVY_OUT'])
report={'schema':'lighton4b-original-adapter-controls/v1','status':'failed',
    'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'original_forward_executed':False,'native_forward_executed':False,'model_payload_bytes':0,
    'capture_transport_admitted':False,'document_bundle_admitted':False,'quality_qualified':False}
report['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in(
    'tools/lighton4b_original_document.py','tests/test_lighton4b_original_document.py',
    'tools/ci-heavy/lighton4b_original_adapter_controls.py','tools/dump_qwen35_ocr_reference.py',
    'docs/lighton4b-document-parity-v1.json','tools/lighton4b_document_parity.py')}
try:
    with (out/'controls.log').open('w')as stream:
        subprocess.run([sys.executable,ROOT/'tests/test_lighton4b_original_document.py','-v'],cwd=ROOT,
            stdout=stream,stderr=subprocess.STDOUT,check=True)
    report['status']='passed';report['controls_log_sha256']=hashlib.sha256((out/'controls.log').read_bytes()).hexdigest()
finally:(out/'original-adapter-controls.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
