#!/usr/bin/env python3
"""Hosted synthetic document-comparer controls. Never model execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':
    raise SystemExit('hosted portable execution required')
out=Path(os.environ['HEAVY_OUT']);build=Path(os.environ['HEAVY_SCRATCH'])/'lighton4b-document-build'
report={'schema':'lighton4b-document-controls/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'original_forward_executed':False,'native_forward_executed':False,'model_payload_bytes':0,'quality_qualified':False}
report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()for name in(
    'docs/lighton4b-document-parity-v1.json','tests/lighton4b_document_diff.cpp','tests/test_lighton4b_document_parity.py',
    'tools/lighton4b_document_parity.py','tools/ci-heavy/lighton4b_document_controls.py','src/crispembed_diff.h')}
def command(label,argv,env=None):
    with (out/(label+'.log')).open('w')as stream:subprocess.run([str(x)for x in argv],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,env=env)
try:
    command('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF'])
    command('build',['cmake','--build',build,'--target','crispembed-diff-lighton4b-document','--parallel','2'])
    binary=build/'crispembed-diff-lighton4b-document'
    command('controls',[sys.executable,ROOT/'tests/test_lighton4b_document_parity.py','-v'],dict(os.environ,LIGHTON4B_DOCUMENT_DIFF=str(binary)))
    from PIL import Image
    import PIL
    if PIL.__version__!='12.0.0':raise ValueError('pinned Pillow12 required')
    p=json.loads((ROOT/'docs/lighton4b-document-parity-v1.json').read_text());case=p['case']
    image=Image.open(ROOT/case['file']).convert('RGB')
    if list(image.size)!=case['original_rgb_size']or hashlib.sha256(image.tobytes()).hexdigest()!=case['original_rgb_sha256']:
        raise ValueError('decoded source RGB drift')
    from jinja2 import Environment
    rendered=Environment().from_string((ROOT/'tests/fixtures/lightonocr3-4b-source/chat_template.jinja').read_text()).render(
        messages=[{'role':'user','content':[{'type':'image'}]}],add_generation_prompt=True,add_vision_id=False,enable_thinking=False)
    if rendered!=case['rendered_prompt']:raise ValueError('actual frozen template render differs')
    report.update(status='passed',source_rgb_sha256=case['original_rgb_sha256'],template_render_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT/'docs/lighton4b-document-parity-v1.json').read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
finally:
    (out/'document-controls.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
