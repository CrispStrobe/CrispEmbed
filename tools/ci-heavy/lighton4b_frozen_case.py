#!/usr/bin/env python3
"""Run one frozen development case on a hosted worker, after exact native input admission.

Transport is external: stage the independently reviewed model and original capture
on the worker and pass their exact SHA256. No original activation/logit is a native
inference input. Source-capped originals remain explicit negative diagnostics.
"""
import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
PROTOCOL_SHA='3a1738c26cf5a34680fea1468bea07c06ee26659f7ddb7e249299677746ba36d'
FROZEN_SHA='a07af76028a0d53c8bfc0ccafedac348eaeb725c8c8c55e3136906f202b8f65d'
MIXED='lighton4b-q4k-mixers-q8-v1'
GDN_OUT='lighton4b-q4k-gdn-out-q8-v1'
ORIGINAL_SOURCE='42e5ce6f9bc6bd01865ac4f21d2157c98d3475f1'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def require(condition,message):
 if not condition:raise ValueError(message)
def load_contract(case_id):
 protocol=ROOT/'docs/lighton4b-mixed-calibration-v1.json'
 frozen=ROOT/'tests/regression/lighton4b-mixed-v1/frozen-inputs.json'
 require(sha(protocol)==PROTOCOL_SHA and sha(frozen)==FROZEN_SHA,'frozen case protocol/input manifest differs')
 p=json.loads(protocol.read_text());f=json.loads(frozen.read_text())
 require(f['protocol_sha256']==PROTOCOL_SHA and f['model_forward_executed'] is False,'input freeze identity differs')
 cases=[x for x in p['cases'] if x['id']==case_id];inputs=[x for x in f['cases'] if x['id']==case_id]
 require(len(cases)==len(inputs)==1,'case outside frozen panel')
 case,record=cases[0],inputs[0]
 require(case['prompt']=='' and case['max_new_tokens']==256 and case['eos_ids']==[248046] and case['do_sample'] is False,'frozen generation differs')
 return p,case,record
def validate_original(original,case):
 require(original['schema']=='lighton4b-fourcase-bf16-capture/v1' and original['source_sha']==ORIGINAL_SOURCE and
  original['case_id']==case['id'] and original['split']==case['split'] and original['protocol_sha256']==PROTOCOL_SHA and
  original['input_manifest_sha256']==FROZEN_SHA and original['original_forward_executed'] is True,
  'reviewed four-case BF16 capture identity differs')
 require(original['source']['model']=='lightonai/LightOnOCR-3-4B' and
  original['source']['revision']=='b71010f095dc3735de7ee6b9969bff8f8f50b39a','original source revision differs')
 # The reviewed producer source fixes bfloat16 and binds its model_dtype in batch.json;
 # its actual per-case schema has no requested_dtype field. Do not invent one.
 if 'requested_dtype' in original:require(original['requested_dtype']=='bfloat16','original dtype differs')
def validate_build(build,receipt_path,expected_sha,current_source):
 require(sha(receipt_path)==expected_sha,'prebuilt receipt digest differs')
 receipt=json.loads(receipt_path.read_text())
 require(receipt['schema']=='lighton4b-frozen-case-build/v1' and receipt['status']=='passed' and
  receipt['source_sha']==current_source and receipt['model_payload_bytes']==0,'prebuilt source/control identity differs')
 for name in ('lighton4b-native-document','lighton4b-native-case-preflight','crispembed-diff-lighton4b-document'):
  require((build/name).is_file() and sha(build/name)==receipt[name+'_sha256'],'prebuilt executable differs: '+name)
 return receipt

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--case',required=True)
 parser.add_argument('--model',type=Path,required=True);parser.add_argument('--weights-sha256',required=True)
 parser.add_argument('--recipe',choices=('q8_0','q4_k',MIXED,GDN_OUT),required=True)
 parser.add_argument('--original-root',type=Path,required=True);parser.add_argument('--original-capture-sha256',required=True)
 parser.add_argument('--prebuilt-build',type=Path);parser.add_argument('--build-receipt',type=Path);parser.add_argument('--build-receipt-sha256')
 parser.add_argument('--mode',choices=('smoke','full','greedy-full'),default='full');parser.add_argument('--threads',type=int,default=4)
 args=parser.parse_args()
 require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable execution required')
 require(version('Pillow')=='11.3.0','pinned Pillow11.3.0 required')
 require(1<=args.threads<=16,'thread bound differs')
 protocol,case,frozen=load_contract(args.case)
 out=Path(os.environ['HEAVY_OUT'])/args.case;out.mkdir(exist_ok=False)
 scratch=Path(os.environ['HEAVY_SCRATCH'])/('frozen-case-'+args.case);scratch.mkdir(exist_ok=False)
 report={'schema':'lighton4b-frozen-case-native/v1','status':'failed','quality_qualified':False,'case_id':args.case,
  'split':case['split'],'mode':args.mode,'recipe':args.recipe,'protocol_sha256':PROTOCOL_SHA,'input_freeze_sha256':FROZEN_SHA,
  'producer_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
  'native_forward_executed':False,'input_admitted':False,'weights_sha256':args.weights_sha256,
  'original_capture_sha256':args.original_capture_sha256}
 def command(label,argv,env=None,check=True):
  with (out/(label+'.log')).open('w') as log:return subprocess.run([str(x) for x in argv],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=check)
 try:
  from PIL import Image
  import numpy as np
  require(len(args.weights_sha256)==len(args.original_capture_sha256)==64,'explicit reviewed SHA256 pins required')
  capture_path=args.original_root/'capture.json'
  require(sha(capture_path)==args.original_capture_sha256,'reviewed original capture differs')
  original=json.loads(capture_path.read_text())
  validate_original(original,case)
  teacher=original['generated_ids']
  require(1<=len(teacher)<=256 and all(type(x)is int and 0<=x<248320 for x in teacher),'original IDs outside frozen bounds')
  require(248046 not in teacher[:-1] and original['eos_completed']==(teacher[-1]==248046) and
   original['cap_hit']==(teacher[-1]!=248046 and len(teacher)==256),'original EOS/cap declaration differs')
  report['original_eos_completed']=original['eos_completed'];report['original_cap_hit']=original['cap_hit']
  require(args.model.is_file() and sha(args.model)==args.weights_sha256,'reviewed model payload differs')
  if args.recipe not in (MIXED,GDN_OUT):require(args.weights_sha256==protocol['candidate_inputs'][args.recipe]['sha256'],'uniform model outside fixed protocol')
  spec=importlib.util.spec_from_file_location('native_driver',ROOT/'tools/ci-heavy/lighton4b_native_document.py')
  common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
  report['resource_preflight']={'memory':common.memory(),'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free}
  require(report['resource_preflight']['memory']['available_bytes']>=protocol['resource_guards']['native_min_available_ram_bytes'],'insufficient available RAM')
  require(report['resource_preflight']['output_free_bytes']>=1024**3,'insufficient output disk')
  png=ROOT/'tests/regression/lighton4b-mixed-v1'/frozen['png_file']
  require(sha(png)==frozen['png_sha256'],'frozen rendered PNG differs')
  image=Image.open(png).convert('RGB');rgb=image.tobytes()
  require(image.size==(frozen['width'],frozen['height']) and hashlib.sha256(rgb).hexdigest()==frozen['rgb_sha256'],'frozen RGB differs')
  rgb_path=scratch/'input.rgb';rgb_path.write_bytes(rgb)
  ids_path=scratch/'teacher.ids';ids_path.write_text(''.join(f'{x}\n' for x in teacher))
  env=os.environ.copy();env.update({x:'1' for x in common.GATES})
  env.update({'CRISPEMBED_LIGHTON4B_FROZEN_CASES':'1','CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4':'1' if args.recipe==MIXED else '0',
   'CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8':'1' if args.recipe==GDN_OUT else '0',
   'CRISPEMBED_LIGHTON4B_DIRECT_QUANT':'0','CRISPEMBED_QWEN35_PRECISE_RMS':'0','CRISPEMBED_QWEN35_IMATRIX':'0','CRISPEMBED_QWEN35_WEIGHT_CACHE':'0'})
  keys=list(common.GATES)+['CRISPEMBED_LIGHTON4B_FROZEN_CASES','CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4','CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8',
   'CRISPEMBED_LIGHTON4B_DIRECT_QUANT','CRISPEMBED_QWEN35_PRECISE_RMS','CRISPEMBED_QWEN35_IMATRIX','CRISPEMBED_QWEN35_WEIGHT_CACHE']
  report['gates']={x:env[x] for x in keys}
  builder=ROOT/'tools/ci-heavy/lighton4b_frozen_case_build.py';build_report=out/'build';build_report.mkdir()
  if args.prebuilt_build:
   require(args.build_receipt is not None and args.build_receipt_sha256 is not None,'complete prebuilt admission required')
   build=args.prebuilt_build
   receipt=validate_build(build,args.build_receipt,args.build_receipt_sha256,report['producer_source_sha'])
   (build_report/'validated-build.json').write_text(json.dumps(receipt,indent=2)+'\n')
  else:
   require(args.build_receipt is None and args.build_receipt_sha256 is None,'prebuilt admission without executable directory')
   build_env=os.environ.copy();build_env['HEAVY_OUT']=str(build_report)
   command('build-helper',['python',builder],build_env)
   build=Path(os.environ['HEAVY_SCRATCH'])/'frozen-case-build'
  preflight=out/'preflight';preflight.mkdir()
  command('native-input-preflight',[build/'lighton4b-native-case-preflight',args.model,rgb_path,image.height,image.width,preflight,args.recipe],env)
  report['binary_sha256']={name:sha(build/name) for name in ('lighton4b-native-document','lighton4b-native-case-preflight')}
  arrays={}
  for field in ('pixel_values','image_grid_thw','input_ids','mm_token_type_ids','attention_mask'):
   record=original['prepared_inputs'][field];name=record['file']
   require(Path(name).name==name,'original array path must be a basename')
   path=args.original_root/name;require(sha(path)==record['sha256'],'original input payload differs: '+field)
   values=np.load(path,allow_pickle=False);require(list(values.shape)==record['shape'],'original shape differs: '+field)
   require(values.dtype==np.dtype('<f4' if field=='pixel_values' else '<i8'),'original dtype differs: '+field)
   arrays[field]=values
  require(arrays['pixel_values'].dtype==np.dtype('<f4') and arrays['pixel_values'].shape==(case['expected_patches'],1536),'official pixel shape/dtype differs')
  require((preflight/'processor.patches.f32').read_bytes()==arrays['pixel_values'].tobytes(order='C'),'native processor differs from official input')
  for field,name in [('image_grid_thw','grid'),('input_ids','prompt'),('mm_token_type_ids','modalities')]:
   values=[int(x) for x in (preflight/(name+'.ids')).read_text().split()]
   require(values==arrays[field].reshape(-1).tolist(),'native input differs: '+field)
  require(arrays['image_grid_thw'].reshape(-1).tolist()==case['expected_grid'] and arrays['input_ids'].size==case['expected_input_tokens'] and
   np.all(arrays['attention_mask']==1),'official geometry/mask differs')
  report['input_admitted']=True;report['official_prepared_inputs']=original['prepared_inputs']
  native=out/'native';native.mkdir();cap=2 if args.mode=='smoke' else case['max_new_tokens'];resource=out/'native-resources.txt'
  start=time.monotonic()
  report['native_forward_executed']=None;report['native_execution_started']=True
  process=command('native',['/usr/bin/time','-q','-f','%U %S %e %M','-o',resource,build/'lighton4b-native-document',args.model,rgb_path,
   image.height,image.width,ids_path,native,cap,args.threads,args.mode],env,check=False)
  user,system,elapsed,rss=resource.read_text().strip().split()
  report.update(native_forward_executed=True if process.returncode==0 else None,seconds=time.monotonic()-start,native_elapsed_seconds=float(elapsed),native_cpu_user_seconds=float(user),
   native_cpu_system_seconds=float(system),native_peak_rss_bytes=int(rss)*1024,independent_generation=True if process.returncode==0 else None)
  report['native_returncode']=process.returncode
  require(process.returncode==0,'native process failed: exit '+str(process.returncode))
  generated=[int(x) for x in (native/'generated.ids').read_text().split()]
  report.update(generated_ids=generated,decoded=(native/'generated.txt').read_text(),eos_completed=generated[-1]==248046,
   cap_hit=generated[-1]!=248046 and len(generated)==cap,actual_cap=cap,frozen_cap=case['max_new_tokens'],teacher_forced_ids=teacher,
   native_summary=json.loads((native/'native-summary.json').read_text()))
  count=len(teacher) if args.mode=='full' else 1
  report['scores']=[{'position':i,'file':'native/teacher.'+str(i)+'.logits.f32','shape':[1,248320],
   'bytes':(native/f'teacher.{i}.logits.f32').stat().st_size,'sha256':sha(native/f'teacher.{i}.logits.f32')} for i in range(count)]
  require(all(x['bytes']==993280 for x in report['scores']),'native vocabulary row differs')
  report['exact_original_ids']=generated==teacher;report['exact_original_decoded']=report['decoded']==original['decoded']
  report['status']='completed'
 except Exception as error:
  message=str(error)
  for path,label in [(str(args.model),'[worker-model]'),(str(args.original_root),'[original]'),(str(ROOT),'[repository]'),(str(out),'[output]'),(str(scratch),'[scratch]')]:message=message.replace(path,label)
  report['error_type']=type(error).__name__;report['error']=message
  raise
 finally:(out/'native-case.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
