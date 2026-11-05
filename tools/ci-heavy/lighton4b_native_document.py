#!/usr/bin/env python3
"""Hosted actual Q8/Q4 native document execution; no reference arrays enter native."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from importlib.metadata import version
from pathlib import Path
import subprocess
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
MODEL_REPO='cstr/LightOnOCR-3-4B-GGUF-experimental-6b9dfe91eaa2'
MODEL_REVISION='a7ec2e54103165a96b8bb6dc3478a2887edd7459'
CANDIDATES={
 'q8_0':('lightonocr3-4b-q8_0-vision-f32.gguf','5888532207f0d98590cc028cbf92366f8a41ec925568f0c308bd69d9af6342f1',5815503232),
 'q4_k':('lightonocr3-4b-q4_k-vision-f32.gguf','f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91',4030957952)}
GATES=('CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT','CRISPEMBED_LIGHTON4B_QUANT','CRISPEMBED_QWEN35_QUANT',
 'CRISPEMBED_LIGHTON4B_GGUF_INSPECTION','CRISPEMBED_QWEN35_LIGHTON4B_MODEL_OWNER',
 'CRISPEMBED_QWEN35_LIGHTON4B_TEXT_OWNER','CRISPEMBED_LIGHTON4B_VISION_OWNER','CRISPEMBED_QWEN35_OCR',
 'CRISPEMBED_LIGHTON4B_DECODER_COMPOSE','CRISPEMBED_LIGHTON4B_LINEAR_BLOCK',
 'CRISPEMBED_LIGHTON4B_ATTENTION_BLOCK','CRISPEMBED_QWEN35_GDN_GROUPED_HEADS',
 'CRISPEMBED_QWEN35_STREAM_PROJECTION','CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE','CRISPEMBED_LIGHTON4B_FRONTEND')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def download(url,path):
 with urllib.request.urlopen(url,timeout=120) as response,path.open('wb') as f:
  while block:=response.read(8*1024*1024):f.write(block)
def memory():
 host={}
 for row in Path('/proc/meminfo').read_text().splitlines():
  key,value=row.split(':',1);host[key]=int(value.strip().split()[0])*1024
 limit=Path('/sys/fs/cgroup/memory.max');used=Path('/sys/fs/cgroup/memory.current')
 maximum=limit.read_text().strip() if limit.exists() else 'max'
 current=int(used.read_text()) if used.exists() else 0
 available=host['MemAvailable']
 if maximum!='max':available=min(available,max(0,int(maximum)-current))
 return {'host_available_bytes':host['MemAvailable'],'cgroup_limit_bytes':None if maximum=='max' else int(maximum),
  'cgroup_current_bytes':current,'available_bytes':available}
def main():
 p=argparse.ArgumentParser();p.add_argument('--recipe',choices=CANDIDATES,required=True)
 p.add_argument('--mode',choices=('smoke','full'),default='smoke');p.add_argument('--threads',type=int,default=4)
 args=p.parse_args()
 if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':
  raise RuntimeError('hosted portable execution required')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);scratch.mkdir(exist_ok=True)
 report={'schema':'lighton4b-native-document/v1','status':'failed','recipe':args.recipe,'mode':args.mode,
  'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 try:
  spec=importlib.util.spec_from_file_location('build',ROOT/'tools/ci-heavy/lighton4b_native_document_build.py')
  builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder);builder.main()
  filename,expected,size=CANDIDATES[args.recipe];model=scratch/filename
  if version('Pillow')!='11.3.0':raise RuntimeError('Pillow 11.3.0 required for source RGB decoding')
  report['source_sha256']={str(x.relative_to(ROOT)):sha(x) for x in sorted(set(
   list((ROOT/'src').glob('lighton4b*'))+list((ROOT/'src').glob('qwen35*'))+
   [ROOT/'tests/lighton4b_native_document.cpp',ROOT/'tools/ci-heavy/lighton4b_native_document.py',ROOT/'CMakeLists.txt'])) if x.is_file()}
  report['preflight']={'memory':memory(),'scratch_free_bytes':shutil.disk_usage(scratch).free,
   'output_free_bytes':shutil.disk_usage(out).free}
  required_ram=(8 if args.recipe=='q8_0' else 6)*1024**3
  if report['preflight']['memory']['available_bytes']<required_ram:raise RuntimeError('insufficient available RAM for bounded native candidate')
  if report['preflight']['scratch_free_bytes']<size+2*1024**3 or report['preflight']['output_free_bytes']<1024**3:
   raise RuntimeError('insufficient free disk for candidate and preserved arrays')
  download(f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{filename}',model)
  if model.stat().st_size!=size or sha(model)!=expected:raise RuntimeError('immutable candidate differs')
  report['model']={'repository':MODEL_REPO,'revision':MODEL_REVISION,'file':filename,'sha256':expected,'bytes':size}
  source=ROOT/'tests/regression/images/cc0/simple_form.png'
  if sha(source)!='3d986ec0d131e4e429127627929e12fd4845313f43849c26bb3d1231c3b038dc':raise RuntimeError('source PNG differs')
  from PIL import Image
  image=Image.open(source).convert('RGB');rgb=scratch/'source.rgb';rgb.write_bytes(image.tobytes())
  report['image']={'source':'tests/regression/images/cc0/simple_form.png','sha256':sha(source),
   'rgb_sha256':sha(rgb),'height':image.height,'width':image.width,'decoder':'Pillow '+Image.__version__}
  capture=scratch/'original-capture.json'
  download('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/26fe41e35/tests/results/qwen35-ocr/lightonocr3-4b-bf16-cpu-capture.json',capture)
  if sha(capture)!='170bc06c8b5e929e45d334a5eadc284024f4265b2549f2feddb46607cb49251a':raise RuntimeError('original capture differs')
  original=json.loads(capture.read_text());teacher=original['generated_ids']
  if len(teacher)!=177:raise RuntimeError('original completion differs')
  ids=scratch/'teacher.ids';ids.write_text(''.join(f'{x}\n' for x in teacher))
  native=out/args.recipe;native.mkdir();binary=scratch/'native-document-build/lighton4b-native-document'
  env=os.environ.copy();env.update({key:'1' for key in GATES})
  env.update({'CRISPEMBED_QWEN35_PRECISE_RMS':'0','CRISPEMBED_QWEN35_IMATRIX':'0','CRISPEMBED_QWEN35_WEIGHT_CACHE':'0'})
  report['gates']={key:env[key] for key in GATES};report['threads']=args.threads
  start=time.monotonic()
  with (out/'native.log').open('w') as log:
   process=subprocess.Popen([str(binary),str(model),str(rgb),str(image.height),str(image.width),str(ids),str(native),
    '2' if args.mode=='smoke' else '256',str(args.threads),args.mode],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   peak=0
   while process.poll() is None:
    try:
     for row in Path(f'/proc/{process.pid}/status').read_text().splitlines():
      if row.startswith(('VmHWM:','VmRSS:')):peak=max(peak,int(row.split()[1])*1024)
    except FileNotFoundError:pass
    time.sleep(0.5)
   report['native_peak_rss_bytes']=peak;report['memory_after_native']=memory()
   if process.returncode:raise RuntimeError(f'native process failed: exit {process.returncode}')
  report['seconds']=time.monotonic()-start;report['native']=json.loads((native/'native-summary.json').read_text())
  report['artifacts']={x.name:{'bytes':x.stat().st_size,'sha256':sha(x)} for x in native.iterdir() if x.is_file()}
  generated=[int(x) for x in (native/'generated.ids').read_text().split()]
  report['decoded']={'ids':generated,'text':(native/'generated.txt').read_text(),'exact_original_ids':generated==teacher}
  if args.mode=='full':
   protocol=ROOT/'docs/lighton4b-bf16-quant-v1.json'
   if sha(protocol)!='4d2a738a8df75947f5a8e269799c8b3a27af19bfafb82653802a7531b7c928bc':raise RuntimeError('parity protocol differs')
   eos=generated[-1]==248046
   receipt={'schema':'lighton4b-bf16-quant-native/v1','precision':args.recipe,
    'protocol_sha256':sha(protocol),'original_capture_sha256':sha(capture),
    'teacher_forced_ids':teacher,'free_generation_independent':True,
    'generated_ids':generated,'decoded':report['decoded']['text'],'stop':'eos' if eos else 'max_tokens',
    'cap_hit':not eos and len(generated)==256,'native_forward_executed':True,
    'producer_source_sha':report['source_sha'],
    'producer_run_url':'https://github.com/'+os.environ['GITHUB_REPOSITORY']+'/actions/runs/'+os.environ['GITHUB_RUN_ID'],
    'weights_sha256':expected,'scores':[]}
   for i in range(177):
    name=f'teacher.{i}.logits.f32';record=report['artifacts'][name]
    if record['bytes']!=993280:raise RuntimeError('full vocabulary row differs')
    receipt['scores'].append({'position':i,'file':name,'shape':[1,248320],**record})
   (native/'native.json').write_text(json.dumps(receipt,indent=2)+'\n')
   report['native_receipt_sha256']=sha(native/'native.json')
  report['status']='completed'
 finally:
  (out/'native-document.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
