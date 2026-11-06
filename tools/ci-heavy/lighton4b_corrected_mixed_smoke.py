#!/usr/bin/env python3
"""One source-vision-exact native mixed form diagnostic; no completion/heldout qualification."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
TABLE_SHA='9f52f11eff527011693a3dcae1cafe9cfbff85c7c146453f8c41b54d90ed6d0e'
PROFILE_SHA='2700720dee65b9ed85c6f7d0c270347fe72f0f21d0d692cd01449641be5ac293'
CAPTURE_SHA='170bc06c8b5e929e45d334a5eadc284024f4265b2549f2feddb46607cb49251a'

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def digest(raw):return hashlib.sha256(raw).hexdigest()
def require(ok,why):
 if not ok:raise ValueError(why)
def retrieve_prefill(scratch,comparison,mixed_control=False):
 from kaggle.api.kaggle_api_extended import KaggleApi
 provider=KaggleApi();provider.authenticate()
 Get=KaggleApi.kernels_pull.__globals__['ApiGetKernelRequest'];List=KaggleApi.kernels_output.__globals__['ApiListKernelSessionOutputRequest']
 owner=os.environ['KAGGLE_ACCOUNT'];slug=('lighton4b-bf16-quant-first-score-v1' if mixed_control=='q4' else 'lighton4b-bf16-mixed-free-generation-v1' if mixed_control=='free' else 'lighton4b-bf16-mixed-first-score-v1') if mixed_control else 'lighton4b-bf16-targeted-original-cpu-prefill';urls={};token=None;pages=0
 with provider.build_kaggle_client() as client:
  req=Get();req.user_name=owner;req.kernel_slug=slug;req.version_label='v1';client.kernels.kernels_api_client.get_kernel(req)
  while True:
   pages+=1;require(pages<=16,'original prefill listing bound')
   req=List();req.user_name=owner;req.kernel_slug=slug;req.version_label='v1';req.page_size=100
   if token:req.page_token=token
   page=client.kernels.kernels_api_client.list_kernel_session_output(req)
   for item in page.files or []:
    require(item.file_name not in urls,'duplicate prefill member');urls[item.file_name]=item.url
   token=page.next_page_token
   if not token:break
 def fetch(name,maximum):
  require(name in urls,'original prefill member missing')
  try:
   with urllib.request.urlopen(urls[name],timeout=120) as stream:raw=stream.read(maximum+1)
  except Exception:raise ValueError('bounded original prefill transfer failed') from None
  require(len(raw)<=maximum,'original prefill member byte bound');return raw
 if mixed_control=='q4':
  prefix='bf16-quant-first-score/quant-oracle/';raw=fetch(prefix+'oracle.json',8*2**20)
  require(digest(raw)=='03119ea4d0fbbf5cc86b91f80b47bbda22d90b417d89dd0c16b8fafa1a85a89b','source plainQ4 oracle differs')
  manifest=json.loads(raw);require(manifest['source_sha']=='048fd683f4ad4adf50f945f6d7c6b436bacdb9aa' and manifest['comparison_completed'],'source plainQ4 status differs')
  row=manifest['arms']['q4_k'];score=fetch(prefix+row['file'],993280)
  require(len(score)==993280 and digest(score)==row['sha256']=='e59d6054c85e81fb1be7101c9899287fa3a1cdd23a3110c8e03d874992387a0d','source plainQ4 row differs')
  return manifest,score
 if mixed_control=='free':
  prefix='bf16-mixed-freegen/quant-oracle/';raw=fetch(prefix+'oracle.json',8*2**20)
  require(digest(raw)=='ed4ef1b60c41d4d3b2a131d107c2563ee971c931873202f5f470683ebb9a1d58','source mixed full receipt differs')
  manifest=json.loads(raw);require(manifest['source_sha'].startswith('89eda20c6') and manifest['full_generation_completed'],'source mixed full status differs')
  row=manifest['arms']['mixed']['scores'][33];score=fetch(prefix+row['file'],993280)
  require(len(score)==993280 and digest(score)==row['sha256']=='50d0d71849e27db998d9014068af6d2ff27c39a49803f56c4f099476847fda9f','source mixed row33 differs')
  (scratch/'source-mixed-row33.f32').write_bytes(score)
  return manifest,score
 if mixed_control:
  prefix='bf16-mixed-first-score/quant-oracle/'
  raw=fetch(prefix+'oracle.json',8*2**20)
  require(digest(raw)=='0e3139c884a82ee9f1e0c5c3cc9943911241ab6aeb6e17b8b5602c90563c07c0','independent BF16 mixed report differs')
  manifest=json.loads(raw);require(manifest['source_sha']=='50307be7fdb5ec17f367b9b632f506f3d2307126' and manifest['comparison_completed'],'independent BF16 mixed source/status differs')
  row=manifest['arms']['mixed'];score=fetch(prefix+row['file'],993280)
  require(len(score)==993280 and digest(score)==row['sha256']=='bea1064566634615c955582808c75b0003dc8f24983cbd12cbb661a1b5c33ae9','independent BF16 mixed row differs')
  (scratch/'independent-mixed-oracle.json').write_bytes(raw)
  return manifest,score
 prefix='lighton4b-bf16-prefill/stages/';raw=fetch(prefix+'prefill.json',2**20)
 require(digest(raw)==comparison.ORIGINAL_REPORT_SHA,'original prefill manifest SHA differs')
 manifest=json.loads(raw);require(manifest['source_sha']==comparison.SOURCE and manifest['observer_first_score_bit_exact'],'original source/neutrality differs')
 row=manifest['stages']['prefill.logits'];score=fetch(prefix+row['file'],993280)
 require(len(score)==row['bytes']==993280 and digest(score)==row['sha256'],'original BF16 first score differs')
 (scratch/'original-prefill.json').write_bytes(raw);(scratch/'original-first-score.f32').write_bytes(score)
 return manifest,score

def main(full=False,parallel_ab=False,alternating=False,head_ab=False,plain_q4=False,weight_ab=False):
 if weight_ab:plain_q4=True
 generation_cap=16 if weight_ab else 256 if (full or head_ab or plain_q4) else 2
 require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable smoke required')
 import numpy as np
 from PIL import Image
 require(np.__version__=='2.3.5' and Image.__version__=='11.3.0','canonical serializer/PNG versions required')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
 native=load('native_common',ROOT/'tools/ci-heavy/lighton4b_native_document.py')
 control=load('verified_vision_gates',ROOT/'tools/ci-heavy/lighton4b_rotary_q8_smoke.py')
 mixed=load('mixed_transport',ROOT/'tools/ci-heavy/lighton4b_calibration_native.py')
 comparison=load('source_compare',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py')
 report={'schema':'lighton4b-corrected-plain-q4-full/v1' if plain_q4 else 'lighton4b-corrected-mixed-native-smoke/v1','status':'failed','quality_qualified':False,'full_completion_executed':False,
  'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'recipe':'q4_k' if plain_q4 else 'lighton4b-q4k-mixers-q8-v1',
  'scope':('observed simple_form full177 teacher scores and independent greedy cap256; excluded from calibration/heldout qualification' if full else 'observed simple_form first-score/two-token runtime diagnostic; excluded from calibration/heldout qualification'),
  'vision_gates':control.VISION_GATES,'disabled_gates':{'CRISPEMBED_LIGHTON4B_BF16_LINEAR':'0','CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2':'0','CRISPEMBED_LIGHTON4B_DIRECT_QUANT':'0','CRISPEMBED_QWEN35_WEIGHT_CACHE':'0','CRISPEMBED_QWEN35_PRECISE_RMS':'0','CRISPEMBED_QWEN35_IMATRIX':'0','CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT':'0','CRISPEMBED_LIGHTON4B_HEAD_BF16':'0','CRISPEMBED_LIGHTON4B_GREEDY_TRACE':'0','CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS':'0'},
  'rotary_table_sha256':TABLE_SHA,'complete_rotary_profile_sha256':PROFILE_SHA,'original_forward_executed':False}
 def run(label,argv,env=None):
  with (out/(label+'.log')).open('w') as log:subprocess.run([str(x)for x in argv],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 try:
  report['preflight']={'memory':native.memory(),'load_average':os.getloadavg(),'swap':{k:v for k,v in (line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines()) if k in ('SwapTotal','SwapFree')},'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free}
  require(report['preflight']['memory']['available_bytes']>=10*1024**3 and report['preflight']['scratch_free_bytes']>=12*1024**3 and report['preflight']['output_free_bytes']>=2*1024**3,'guarded mixed smoke capacity unavailable')
  require(native.sha(ROOT/'src/lighton4b_rotary_table_data.inc')==TABLE_SHA and native.sha(ROOT/'tests/results/qwen35-ocr/lightonocr3-4b-complete-rotary-profile.json')==PROFILE_SHA,'complete corrected rotary table/proof differs')
  if head_ab or plain_q4:
   report['stage']='model-free-head-controls';builder=load('native_build_controls',ROOT/'tools/ci-heavy/lighton4b_native_document_build.py');builder.main();build=scratch/'native-document-build'
   run('head-controls-build',['cmake','--build',build,'--target','test-lighton4b-head-bf16','--parallel','2'])
   run('head-controls',[build/'test-lighton4b-head-bf16'])
   require('HEAD_BF16 passed=1 source_row33_tie=1' in (out/'head-controls.log').read_text(),'head RNE/tie controls required before model acquisition')
   if weight_ab:
    run('weight-controls-build',['cmake','--build',build,'--target','test-lighton4b-bf16-dequant-weights','test-lighton4b-bf16-weight-wasm-guard','--parallel','2'])
    run('weight-controls',[build/'test-lighton4b-bf16-dequant-weights']);run('wasm-guard',[build/'test-lighton4b-bf16-weight-wasm-guard'])
    report['weight_controls_stdout']=(out/'weight-controls.log').read_text();report['wasm_preprocessor_guard_stdout']=(out/'wasm-guard.log').read_text()
    require('BF16_DEQUANT_WEIGHTS passed=1' in report['weight_controls_stdout'] and 'BF16_WEIGHT_WASM_GUARD passed=1' in report['wasm_preprocessor_guard_stdout'],'weight/WASM preprocessor controls required before model acquisition')

  report['stage']='stage-immutable-candidate'
  if plain_q4:
   filename,expected,size=native.CANDIDATES['q4_k'];model=scratch/filename
   native.download(f'https://huggingface.co/{native.MODEL_REPO}/resolve/{native.MODEL_REVISION}/{filename}',model)
   require(model.stat().st_size==size and native.sha(model)==expected,'immutable plainQ4 payload differs')
   report['model']={'repository':native.MODEL_REPO,'revision':native.MODEL_REVISION,'file':filename,'sha256':expected,'bytes':size}
  else:model=mixed.stage_model(scratch,out);report['model']=mixed.MODEL
  with urllib.request.urlopen('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/26fe41e35/tests/results/qwen35-ocr/lightonocr3-4b-bf16-cpu-capture.json',timeout=60) as stream:raw=stream.read(1000001)
  require(digest(raw)==CAPTURE_SHA,'frozen independent BF16 completion differs');reference=json.loads(raw)
  image_path=ROOT/'tests/regression/images/cc0/simple_form.png';require(native.sha(image_path)=='3d986ec0d131e4e429127627929e12fd4845313f43849c26bb3d1231c3b038dc','frozen form image differs')
  image=Image.open(image_path).convert('RGB');rgb=scratch/'input.rgb';rgb.write_bytes(image.tobytes());teacher=scratch/'teacher.ids';teacher.write_text(''.join(f'{x}\n' for x in reference['generated_ids']))
  require(len(reference['generated_ids'])==177,'original completion identity differs')
  manifest,original_logits=retrieve_prefill(scratch,comparison)
  report['original_bf16']={'capture_sha256':CAPTURE_SHA,'stage_manifest_sha256':comparison.ORIGINAL_REPORT_SHA,'first_score_sha256':digest(original_logits),'first_id':reference['generated_ids'][0]}
  mixed_reference,mixed_logits=retrieve_prefill(scratch,comparison,'q4' if plain_q4 else True)
  report['independent_bf16_q4' if plain_q4 else 'independent_bf16_mixed']={'source_sha':mixed_reference['source_sha'],'report_sha256':'03119ea4d0fbbf5cc86b91f80b47bbda22d90b417d89dd0c16b8fafa1a85a89b' if plain_q4 else '0e3139c884a82ee9f1e0c5c3cc9943911241ab6aeb6e17b8b5602c90563c07c0','first_score_sha256':digest(mixed_logits),'ids':mixed_reference['arms']['q4_k' if plain_q4 else 'mixed']['ids']}
  require({k:v['sha256'] for k,v in mixed_reference['prepared_inputs'].items()}=={k:v['sha256'] for k,v in reference['prepared_inputs'].items()},'independent mixed prepared inputs differ')
  if head_ab:
   source_free,source_row33=retrieve_prefill(scratch,comparison,'free')
   report['source_mixed_full']={'report_sha256':'ed4ef1b60c41d4d3b2a131d107c2563ee971c931873202f5f470683ebb9a1d58','source_sha':source_free['source_sha'],'row33_sha256':digest(source_row33),'ids':source_free['arms']['mixed']['ids'],'decoded':source_free['arms']['mixed']['decoded'],'eos_completed':source_free['arms']['mixed']['eos_completed']}
  report['stage']='build-corrected-native';builder=load('native_build',ROOT/'tools/ci-heavy/lighton4b_native_document_build.py')
  if not (head_ab or plain_q4):builder.main()
  build=scratch/'native-document-build'
  binary=build/'lighton4b-native-document';report['binary_sha256']=native.sha(binary)
  env=os.environ.copy();env.update({x:'1' for x in tuple(native.GATES)+tuple(control.VISION_GATES)})
  env.update(report['disabled_gates']);env['CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4']='0' if plain_q4 else '1';report['enabled_gates']={x:env[x] for x in tuple(native.GATES)+tuple(control.VISION_GATES)+('CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4',)}
  if head_ab:
   env['CRISPEMBED_LIGHTON4B_GREEDY_TRACE']='1';env['CRISPEMBED_LIGHTON4B_GREEDY_TRACE_POSITION']='33'
   report['scope']='same-worker head BF16 output dtype OFF/ON full independent greedy256, one teacher row each; observed form only, no heldout qualification'
  if plain_q4:
   env['CRISPEMBED_LIGHTON4B_HEAD_BF16']='1';report['active_overrides']={'CRISPEMBED_LIGHTON4B_HEAD_BF16':'1','CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4':'0'}
   report['scope']='observed form plainQ4 correctedvision+BF16head full independent greedy256; one teacher row; same-weight original BF16 firstscore; full same-weight source reference pending; no heldout claim'
  if weight_ab:report['scope']='plainQ4 native BF16 weight dtype OFF/ON, same image/model, one teacher score and independent16-token cap per arm; no full document/quality claim'
  candidate=out/('q4_k' if plain_q4 else 'mixed');candidate.mkdir();resource=out/'native-resources.txt';report['stage']='native-full-greedy256' if (full or head_ab or plain_q4) else 'native-first-score-two-token';start=time.monotonic()
  report['requested_teacher_positions']=177 if full else 1;report['requested_generation_cap']=generation_cap
  run('native',['/usr/bin/time','-q','-f','%U %S %e %M','-o',resource,binary,model,rgb,image.height,image.width,teacher,candidate,generation_cap,4,'full' if full else 'smoke'],env)
  user,system,elapsed,rss=resource.read_text().strip().split();report['resources']={'wall_seconds':float(elapsed),'user_seconds':float(user),'system_seconds':float(system),'peak_rss_bytes':int(rss)*1024,'measured_seconds':time.monotonic()-start}
  report['native_summary']=json.loads((candidate/'native-summary.json').read_text());require(report['native_summary']['teacher_positions']==(177 if full else 1) and report['native_summary']['cap']==generation_cap,'bounded smoke scope differs')
  ids=lambda name:np.array([int(x)for x in (candidate/name).read_text().split()],dtype='<i8')
  prompt=ids('prompt.ids');values={'input_ids':prompt.reshape(1,-1),'image_grid_thw':ids('processor.grid.ids').reshape(1,3),'mm_token_type_ids':ids('modalities.ids').reshape(1,-1),'pixel_values':np.fromfile(candidate/'processor.patches.f32',dtype='<f4').reshape(560,1536),'attention_mask':np.ones((1,len(prompt)),dtype='<i8')};report['prepared_controls']={}
  for name,value in values.items():
   data=io.BytesIO();np.save(data,value,allow_pickle=False);got=digest(data.getvalue());require(got==reference['prepared_inputs'][name]['sha256'],'actual native prepared input differs: '+name)
   report['prepared_controls'][name]={'canonical_npy_sha256':got,'source_npy_sha256':reference['prepared_inputs'][name]['sha256'],'exact':True}
  report['raw_vision_controls']={}
  for name in ['processor.patches','vision.patch','vision.position','vision.embedded']+[f'vision.layer.{i}'for i in range(24)]+['vision.merged']:
   path=candidate/(name+'.f32');row=manifest['stages'][name];actual=native.sha(path)
   require(path.stat().st_size==row['bytes'] and actual==row['sha256'],'corrected native vision not source-exact: '+name)
   report['raw_vision_controls'][name]={'bytes':path.stat().st_size,'source_sha256':row['sha256'],'native_sha256':actual,'bit_exact':True}
  actual=np.fromfile(candidate/'prefill.logits.f32',dtype='<f4');require(actual.size==248320,'complete first score required')
  report['first_score_vs_original_bf16']=comparison.compare(np.frombuffer(original_logits,dtype='<f4'),actual)
  report['first_score_vs_independent_bf16_q4' if plain_q4 else 'first_score_vs_independent_bf16_mixed']=comparison.compare(np.frombuffer(mixed_logits,dtype='<f4'),actual)
  generated=ids('generated.ids').tolist();report['decoded']={'ids':generated,'text':(candidate/'generated.txt').read_text(),'exact_original_prefix_ids':generated==reference['generated_ids'][:len(generated)],'exact_original_complete_ids':generated==reference['generated_ids'],'native_eos':report['native_summary']['eos'],'first_differing_id':next((i for i,(a,b) in enumerate(zip(generated,reference['generated_ids'])) if a!=b),min(len(generated),len(reference['generated_ids'])) if len(generated)!=len(reference['generated_ids']) else None),'original_prefix_ids':reference['generated_ids'][:len(generated)],'original_completion_tokens':177}
  if plain_q4:
   report['decoded']['stop']='eos' if report['native_summary']['eos'] else 'max_tokens'
   report['decoded']['cap_hit']=not report['native_summary']['eos'] and len(generated)==generation_cap
   report['same_weight_first_id_matches']=generated[0]==mixed_reference['arms']['q4_k']['ids'][0]
  if full:
   require(all((candidate/f'teacher.{i}.logits.f32').stat().st_size==993280 for i in range(177)),'full177 teacher rows required')
   report['full_teacher_score_comparison_executed']=False
  report['artifacts']={p.name:{'bytes':p.stat().st_size,'sha256':native.sha(p)}for p in candidate.iterdir() if p.is_file()}
  if weight_ab:
   prefix='https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/4939c9a6e70a6651fa6072aab71665a30db0c71e/tests/results/qwen35-ocr/'
   with urllib.request.urlopen(prefix+'lightonocr3-4b-corrected-plain-q4/native-report.json',timeout=60) as stream:prior_raw=stream.read(2000001)
   require(digest(prior_raw)=='c726b09da906562a7f8cdd20a696d7e7d634745f7b329400f056fc4275e45d4e','prior correctedQ4 baseline differs');prior=json.loads(prior_raw)
   with urllib.request.urlopen(prefix+'lightonocr3-4b-bf16-plain-q4-full/oracle.json',timeout=60) as stream:gold_raw=stream.read(2000001)
   require(digest(gold_raw)=='1096b5f0f6ce26f0fa26ff9cc86dd606fbb3da24a3feca06768f2f8cd66d6d1b','genuine same-Q4 full reference differs');gold=json.loads(gold_raw)
   require(gold['arms']['original']['sha256']=='a4bece2310050081b502a3910a19469e47016829d74abebb29861739f3205ffd' and gold['arms']['q4_k']['scores'][0]['sha256']=='e59d6054c85e81fb1be7101c9899287fa3a1cdd23a3110c8e03d874992387a0d','frozen Intel sameweight reference anchors differ')
   require(report['model']==prior['model'] and generated==prior['decoded']['ids'][:generation_cap],'weight-OFF model/partialIDs changed')
   shared=[name for name in report['artifacts']if name in prior['artifacts']and name not in ('native-summary.json','generated.ids','generated.txt')]
   require(all(report['artifacts'][name]==prior['artifacts'][name]for name in shared),'weight-OFF shared stage/score bytes changed')
   destination=out/'q4-bf16-weights';destination.mkdir();env['CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS']='1';resource=out/'weight-on-resources.txt';report['stage']='weight-on-firstscore-partial16'
   run('weight-on-native',['/usr/bin/time','-q','-f','%U %S %e %M','-o',resource,binary,model,rgb,image.height,image.width,teacher,destination,generation_cap,4,'smoke'],env)
   user,system,elapsed,rss=resource.read_text().strip().split();on_ids=[int(x)for x in (destination/'generated.ids').read_text().split()];on_text=(destination/'generated.txt').read_text();on_summary=json.loads((destination/'native-summary.json').read_text())
   require(on_summary['teacher_positions']==1 and on_summary['cap']==generation_cap and len(on_ids)<=generation_cap,'weight-ON smoke bounds differ')
   on_artifacts={p.name:{'bytes':p.stat().st_size,'sha256':native.sha(p)}for p in destination.iterdir()if p.is_file()}
   invariant=[name for name in report['artifacts']if name.startswith(('processor.','vision.','prefill.positions.')) or name in ('prompt.ids','modalities.ids','rendered.txt')]
   require(all(on_artifacts[name]==report['artifacts'][name]for name in invariant),'weight dtype changed prepared/vision/positions')
   before=np.fromfile(candidate/'text.embedded.f32',dtype='<f4');bits=before.view(np.uint32);rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
   require(rounded.tobytes()==(destination/'text.embedded.f32').read_bytes(),'actual ON embedding differs from independent BF16 cast')
   on_score=np.fromfile(destination/'prefill.logits.f32',dtype='<f4');source_ids=gold['arms']['q4_k']['ids']
   report['weight_dtype_ab']={'gate':'CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS','off':'0','on':'1','same_binary_model_inputs':True,'baseline_shared_artifacts_exact':shared,'unchanged_prepared_vision_position_artifacts':invariant,'on_embedding_equals_independent_bf16_cast':True,'on_ids':on_ids,'on_text':on_text,'on_summary':on_summary,'off_partial_ids_equal_sameweight_source':generated==source_ids[:len(generated)],'on_partial_ids_equal_sameweight_source':on_ids==source_ids[:len(on_ids)],'source_full_reference_sha256':digest(gold_raw),'source_full_reference_profile':'frozen Intel a4/e59 anchors, capped256 without EOS','on_first_score_vs_sameweight_bf16':comparison.compare(np.frombuffer(mixed_logits,dtype='<f4'),on_score),'on_first_score_vs_original_unquantized_bf16':comparison.compare(np.frombuffer(original_logits,dtype='<f4'),on_score),'on_artifacts':on_artifacts,'on_resources':{'wall_seconds':float(elapsed),'user_seconds':float(user),'system_seconds':float(system),'peak_rss_bytes':int(rss)*1024},'generation_cap_per_arm':generation_cap,'full_document_completion_tested':False,'quality_qualified':False,'timing_limit':'OFF first, ON second; cache warmth/order uncontrolled; not a speed experiment'}
  if head_ab:
   with urllib.request.urlopen('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/03138b363/tests/results/qwen35-ocr/lightonocr3-4b-corrected-mixed-full/native-report.json',timeout=60) as stream:previous=stream.read(2000001)
   require(digest(previous)=='2853afb5518c283f6eea356208c7f034a3c7b2e1ed56bfe61b62eab757303ee8','prior full native baseline receipt differs')
   previous=json.loads(previous);require(generated==previous['decoded']['ids'],'head-off baseline full IDs changed')
   shared=[name for name in report['artifacts'] if name in previous['artifacts'] and name!='native-summary.json']
   require(all(report['artifacts'][name]==previous['artifacts'][name] for name in shared),'head-off baseline shared stage/score artifact differs')
   destination=out/'head-bf16';destination.mkdir();env['CRISPEMBED_LIGHTON4B_HEAD_BF16']='1';resource=out/'head-bf16-resources.txt';report['stage']='head-bf16-full-independent-greedy'
   run('head-bf16-native',['/usr/bin/time','-q','-f','%U %S %e %M','-o',resource,binary,model,rgb,image.height,image.width,teacher,destination,256,4,'smoke'],env)
   user,system,elapsed,rss=resource.read_text().strip().split();head_ids=[int(x)for x in (destination/'generated.ids').read_text().split()];head_text=(destination/'generated.txt').read_text();head_summary=json.loads((destination/'native-summary.json').read_text())
   head_artifacts={p.name:{'bytes':p.stat().st_size,'sha256':native.sha(p)}for p in destination.iterdir() if p.is_file()}
   invariant=[name for name in report['artifacts'] if name.startswith(('processor.','vision.','prefill.layer.','prefill.input.','prefill.positions.','prefill.norm.','text.embedded.'))]
   require(all(head_artifacts[name]==report['artifacts'][name] for name in invariant),'head output dtype changed prefill hidden/vision/input')
   gold=source_free['arms']['mixed']['ids']
   def first_diff(a,b):return next((i for i,(x,y)in enumerate(zip(a,b))if x!=y),min(len(a),len(b))if len(a)!=len(b)else None)
   report['head_bf16_ab']={'same_binary_model_inputs':True,'baseline_first':True,'head_gate_off':'0','head_gate_on':'1','trace_gate':'1','trace_position':33,'baseline_shared_artifacts_exact':shared,'controls_stdout':(out/'head-controls.log').read_text(),'off_ids':generated,'on_ids':head_ids,'on_text':head_text,'on_summary':head_summary,'source_mixed_ids':gold,'off_first_differing_id_vs_source_mixed':first_diff(generated,gold),'on_first_differing_id_vs_source_mixed':first_diff(head_ids,gold),'on_complete_ids_equal_source_mixed':head_ids==gold,'on_text_equal_source_mixed':head_text==source_free['arms']['mixed']['decoded'],'unchanged_prefill_artifacts':invariant,'on_artifacts':head_artifacts,'on_resources':{'wall_seconds':float(elapsed),'user_seconds':float(user),'system_seconds':float(system),'peak_rss_bytes':int(rss)*1024},'full_teacher_score_comparison_executed':False,'quality_qualified':False,'trace_position':33,'trace_same_source_prefix_off':generated[:33]==gold[:33],'trace_same_source_prefix_on':head_ids[:33]==gold[:33]}
   report['full_completion_executed']=True
  if parallel_ab:
   require(not full,'bounded A/B requires smoke scope')
   parallel=out/'mixed-parallel';parallel.mkdir();env['CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT']='1';parallel_resource=out/'parallel-resources.txt'
   report['stage']='parallel-dequant-native-first-score-two-token'
   run('parallel-native',['/usr/bin/time','-q','-f','%U %S %e %M','-o',parallel_resource,binary,model,rgb,image.height,image.width,teacher,parallel,2,4,'smoke'],env)
   user,system,elapsed,rss=parallel_resource.read_text().strip().split()
   parallel_artifacts={p.name:{'bytes':p.stat().st_size,'sha256':native.sha(p)}for p in parallel.iterdir() if p.is_file()}
   same=parallel_artifacts==report['artifacts']
   report['parallel_dequant_ab']={'same_source_binary_model_inputs':True,'serial_first':True,'cache_warmth_not_controlled':True,'gate':'CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT','serial_gate':'0','parallel_gate':'1','all_emitted_artifacts_byte_exact':same,'parallel_artifacts':parallel_artifacts,'parallel_resources':{'wall_seconds':float(elapsed),'user_seconds':float(user),'system_seconds':float(system),'peak_rss_bytes':int(rss)*1024},'wall_speed_ratio':report['resources']['wall_seconds']/float(elapsed),'full_completion_executed':False,'quality_qualified':False}
   require(same,'parallel conversion changed actual emitted stages/scores/IDs/text')
   if alternating:
    repeats=[]
    for index,value in enumerate(['1','0']):
     destination=out/('repeat-'+str(index));destination.mkdir();env['CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT']=value;resources=out/('repeat-'+str(index)+'-resources.txt')
     report['stage']='balanced-order-'+str(index);before={'load_average':os.getloadavg(),'memory':native.memory()}
     run('repeat-'+str(index),['/usr/bin/time','-q','-f','%U %S %e %M','-o',resources,binary,model,rgb,image.height,image.width,teacher,destination,2,4,'smoke'],env)
     user,system,elapsed,rss=resources.read_text().strip().split()
     artifacts={p.name:{'bytes':p.stat().st_size,'sha256':native.sha(p)}for p in destination.iterdir() if p.is_file()}
     equal=artifacts==report['artifacts'];repeats.append({'gate':value,'wall_seconds':float(elapsed),'user_seconds':float(user),'system_seconds':float(system),'peak_rss_bytes':int(rss)*1024,'artifacts':artifacts,'all_emitted_artifacts_byte_exact':equal,'before':before,'after':{'load_average':os.getloadavg(),'memory':native.memory()}})
     report['parallel_dequant_ab']['balanced_repeats']=repeats
     require(equal,'balanced-order conversion changed native artifact bytes')
    report['parallel_dequant_ab']['order']=['serial','parallel','parallel','serial']
    report['parallel_dequant_ab']['cache_warmth_note']='first serial arm includes coldest model/cache state; compare warm reverse pair separately; one worker, no general performance claim'

  report['status']='measured';report['full_completion_executed']=full or head_ab or (plain_q4 and not weight_ab)
 except Exception as error:
  report['error_type']=type(error).__name__;report['error']='bounded corrected mixed smoke/control failed; inspect native/build/staging logs';raise RuntimeError(report['error'])from None
 finally:(out/('bf16-weight-native-smoke.json' if weight_ab else 'corrected-plain-q4-head-full.json' if plain_q4 else 'head-bf16-native-ab.json' if head_ab else 'parallel-dequant-balanced-ab.json' if alternating else 'parallel-dequant-native-ab.json' if parallel_ab else 'corrected-mixed-full.json' if full else 'corrected-mixed-smoke.json')).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
