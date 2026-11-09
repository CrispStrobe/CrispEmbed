#!/usr/bin/env python3
"""Predeclared Q8 donor sensitivity on two calibration cases only; no heldout/recipe admission."""
import gc,hashlib,importlib.util,io,json,os,shutil,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
ARMS=('q4','mixers','down','ffn','q8');CASES=('simple_table','arabic_printed_line')
def module(name):
 s=importlib.util.spec_from_file_location(name,ROOT/'tools/ci-heavy'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def require(x,message):
 if not x:raise ValueError(message)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def fields():
 result=[]
 for i in range(32):
  prefix=f'l.layers.{i}.'
  result.extend((prefix+'mlp.'+name+'_proj.weight',role)for name,role in [('gate',1),('up',1),('down',2)])
  suffix=['self_attn.'+x+'_proj.weight' for x in ('q','k','v','o')]if i%4==3 else ['linear_attn.'+x+'.weight' for x in ('in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj')]
  result.extend((prefix+s,0) for s in suffix)
 return result
def selected(arm,role):return arm=='q8' or arm=='mixers' and role==0 or arm=='down' and role==2 or arm=='ffn' and role!=0
def main():
 require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable only')
 import numpy as np
 from PIL import Image
 from gguf import GGUFReader
 native=module('lighton4b_native_document');original=module('lighton4b_calibration_native');worker=module('lighton4b_frozen_case');comparison=module('lighton4b_vision_bf16_compare');vision=module('lighton4b_rotary_q8_smoke')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'restoration-sensitivity';scratch.mkdir()
 report={'schema':'lighton4b-q8-restoration-sensitivity/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'arms':ARMS,'cases':{},'calibration_cases':CASES,'heldout_cases_executed':[],'quality_qualified':False,'full_completion_executed':False,'production_profile_changed':False,'limitations':['First-score/two-token sensitivity only, no full decoded/EOS qualification.','Native arithmetic fixed across arms; no per-arm independent same-weight original model oracle.','Q6_K not admitted by current LightOn owner; use only Q4_K/Q8_0.','Borrowed test-only pointer overlays do not admit a new production model recipe.']}
 try:
  require(np.__version__=='2.3.5' and Image.__version__=='11.3.0','serializer/RGB pins')
  report['preflight']={'memory':native.memory(),'load':os.getloadavg(),'scratch_free_bytes':shutil.disk_usage(scratch).free,'out_free_bytes':shutil.disk_usage(out).free,'swap':{k:v.strip() for k,v in (s.split(':',1)for s in Path('/proc/meminfo').read_text().splitlines()) if k in ('SwapTotal','SwapFree')}}
  require(report['preflight']['memory']['available_bytes']>=12*1024**3 and report['preflight']['scratch_free_bytes']>=20*1024**3 and report['preflight']['out_free_bytes']>=2*1024**3,'guarded capacity unavailable')
  def run(label,args,timeout=1200,env=None):
   report['phase']=label;print('phase:',label,flush=True)
   with (out/(label+'.log')).open('w') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout,env=env)
  protocol=json.loads((ROOT/'docs/lighton4b-restoration-sensitivity-v1.json').read_text());require(protocol['arms']=={a:[n for n,r in fields() if selected(a,r)]for a in ARMS} and protocol['calibration_cases']==list(CASES) and protocol['heldout_cases']==[],'predeclared policy changed')
  report['protocol_sha256']=native.sha(ROOT/'docs/lighton4b-restoration-sensitivity-v1.json')
  build=scratch/'build';run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  run('build',['cmake','--build',build,'--target','test-lighton4b-sensitivity-policy','lighton4b-restoration-sensitivity','--parallel','2']);run('policy-controls',[build/'test-lighton4b-sensitivity-policy'])
  require('SENSITIVITY_POLICY passed=1 unique_slots=248 counts=0,152,32,96,248 heldout=0 model_payload=0' in (out/'policy-controls.log').read_text(),'model-free policy controls')
  models={}
  for recipe in ('q4_k','q8_0'):
   name,expected,size=native.CANDIDATES[recipe];path=scratch/name;report['phase']='stage-'+recipe
   # The immutable experimental source is public; never forward credentials on redirects.
   native.download(f'https://huggingface.co/{native.MODEL_REPO}/resolve/{native.MODEL_REVISION}/{name}',path)
   require(path.stat().st_size==size and native.sha(path)==expected,'immutable model file identity')
   models[recipe]=path
  readers={k:GGUFReader(str(v),'r')for k,v in models.items()};tensors={k:{t.name:t for t in r.tensors}for k,r in readers.items()};policy=fields();require(len(policy)==248 and len(set(n for n,_ in policy))==248,'policy uniqueness')
  def tensorhash(t):
   h=hashlib.sha256();raw=t.data.view(np.uint8).reshape(-1)
   for start in range(0,len(raw),8*1024**2):h.update(memoryview(raw[start:start+8*1024**2]))
   return h.hexdigest()
  payloads={};names={n for n,_ in policy};require(set(tensors['q4_k'])==set(tensors['q8_0']) and len(tensors['q4_k'])==723,'donor inventory')
  for name,role in policy:
   q4=tensors['q4_k'][name];q8=tensors['q8_0'][name];require(str(q4.tensor_type.name)=='Q4_K' and str(q8.tensor_type.name)=='Q8_0' and np.array_equal(q4.shape,q8.shape),'donor type/shape')
   payloads[name]={'role':role,'q4_sha256':tensorhash(q4),'q8_sha256':tensorhash(q8),'q4_bytes':int(q4.n_bytes),'q8_bytes':int(q8.n_bytes)}
  protected={}
  for name in sorted(set(tensors['q4_k'])-names):
   a,b=tensors['q4_k'][name],tensors['q8_0'][name];require(a.tensor_type==b.tensor_type and np.array_equal(a.shape,b.shape),'protected tensor schema');ha,hb=tensorhash(a),tensorhash(b);require(ha==hb,'protected tensor bytes differ');protected[name]=ha
  require('l.embed_tokens.weight' in protected and len(protected)==475,'protected tied head/vision/norm inventory')
  frozen={'schema':'lighton4b-restoration-policy-freeze/v1','frozen_before_candidate_forward':True,'models':{k:{'repository':native.MODEL_REPO,'revision':native.MODEL_REVISION,'sha256':native.CANDIDATES[k][1],'bytes':native.CANDIDATES[k][2]}for k in models},'selected_payloads':payloads,'protected_sha256':protected,'arms':{arm:[n for n,role in policy if selected(arm,role)]for arm in ARMS},'calibration_cases':CASES,'heldout_cases':[],'caps':{'first_score_rows':1,'greedy_tokens':2},'production_admission_unchanged':True}
  (out/'frozen-policy.json').write_text(json.dumps(frozen,indent=2));(out/'policy-payloads.tsv').write_text(''.join(f'{n}\t{payloads[n]["q4_sha256"]}\t{payloads[n]["q8_sha256"]}\n' for n,_ in policy));report['frozen_policy_sha256']=native.sha(out/'frozen-policy.json');report['protected_tensors_exact']=475
  # Release reader mmaps before allocating actual native payloads.
  del a,b,q4,q8,readers,tensors;gc.collect()
  urls=original.original_urls();prefix='lighton4b-fourcase-bf16/'
  def get(member,path,sha,bound=8*1024**2):original.bounded_download(urls[prefix+member],path,sha,bound)
  get('batch.json',out/'original-batch.json',original.ORIGINAL['batch_sha256']);batch=json.loads((out/'original-batch.json').read_text());require(batch['source_sha']==worker.ORIGINAL_SOURCE and batch['protocol_sha256']==worker.PROTOCOL_SHA and batch['input_manifest_sha256']==worker.FROZEN_SHA and batch['model_dtype']=='bfloat16','official original batch')
  originals={};case_rows=[]
  for caseid in CASES:
   _,contract,record=worker.load_contract(caseid);folder=out/caseid;folder.mkdir();prepared=scratch/caseid;prepared.mkdir()
   get(caseid+'/capture.json',folder/'original-capture.json',batch['cases'][caseid]['capture_sha256']);capture=json.loads((folder/'original-capture.json').read_text());worker.validate_original(capture,contract);require(capture['eos_completed'] and not capture['cap_hit'],'calibration source completion')
   for name,row in capture['prepared_inputs'].items():require(Path(row['file']).name==row['file'],'prepared path');get(caseid+'/'+row['file'],prepared/row['file'],row['sha256'])
   score=capture['scores'][0];require(score['position']==0 and score['bytes']==993280 and score['shape']==[1,248320],'original score schema');get(caseid+'/'+score['file'],folder/'original-first.f32',score['sha256'],993280)
   png=ROOT/'tests/regression/lighton4b-mixed-v1'/record['png_file'];require(native.sha(png)==record['png_sha256'],'PNG source');rgb=Image.open(png).convert('RGB').tobytes();require(digest(rgb)==record['rgb_sha256'],'RGB source');rgbpath=prepared/'input.rgb';rgbpath.write_bytes(rgb);case_rows.append(f'{caseid} {rgbpath} {record["height"]} {record["width"]}\n');originals[caseid]=(capture,prepared)
  casefile=scratch/'cases.txt';casefile.write_text(''.join(case_rows))
  env=os.environ.copy();env.update({k:'1'for k in native.GATES+vision.VISION_GATES});env.update({k:'0' for k in ('CRISPEMBED_LIGHTON4B_BF16_LINEAR','CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2','CRISPEMBED_LIGHTON4B_DIRECT_QUANT','CRISPEMBED_QWEN35_WEIGHT_CACHE','CRISPEMBED_QWEN35_PRECISE_RMS','CRISPEMBED_QWEN35_IMATRIX','CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT','CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4','CRISPEMBED_LIGHTON4B_GREEDY_TRACE')});env['CRISPEMBED_LIGHTON4B_HEAD_BF16']='1';report['gates']={k:v for k,v in env.items()if k.startswith('CRISPEMBED_')}
  started=time.monotonic();run('native-calibration',[build/'lighton4b-restoration-sensitivity',models['q4_k'],models['q8_0'],casefile,out],timeout=35*60,env=env);report['native_elapsed_seconds']=time.monotonic()-started
  require((out/'policy-payloads.tsv').read_bytes()==(out/'loaded-payloads.tsv').read_bytes(),'actual loaded donor/base hashes differ from pre-forward freeze')
  for caseid in CASES:
   folder=out/caseid;capture,prepared=originals[caseid]
   def ids(name):return np.array([int(x)for x in (folder/name).read_text().split()],dtype='<i8')
   prompt=ids('prompt.ids');pixels=np.fromfile(folder/'processor.patches.f32',dtype='<f4');grid=ids('processor.grid.ids');values={'input_ids':prompt.reshape(1,-1),'pixel_values':pixels.reshape(int(np.prod(grid)),1536),'image_grid_thw':grid.reshape(1,3),'mm_token_type_ids':ids('modalities.ids').reshape(1,-1),'attention_mask':np.ones((1,len(prompt)),dtype='<i8')};controls={}
   for name,value in values.items():
    raw=io.BytesIO();np.save(raw,value,allow_pickle=False);controls[name]=digest(raw.getvalue())==capture['prepared_inputs'][name]['sha256']
   require(all(controls.values()),'actual native prepared inputs differ from original');row={'prepared_input_controls':controls,'original_first_ids':capture['generated_ids'][:2],'arms':{}};report['cases'][caseid]=row
   source=np.fromfile(folder/'original-first.f32',dtype='<f4');baseline=np.fromfile(folder/'q4.first.f32',dtype='<f4')
   for arm in ARMS:
    logits=np.fromfile(folder/(arm+'.first.f32'),dtype='<f4');prefix=ids(arm+'.prefix.ids').tolist();changed,seconds=(folder/(arm+'.txt')).read_text().split();require(int(changed)==len(frozen['arms'][arm]),'applied policy count')
    stages={}
    for i in range(32):
     a=folder/f'q4.layer.{i}.f32';b=folder/f'{arm}.layer.{i}.f32';stages[str(i)]=comparison.compare(np.fromfile(a,dtype='<f4'),np.fromfile(b,dtype='<f4'));stages[str(i)]['q4_sha256']=native.sha(a);stages[str(i)]['candidate_sha256']=native.sha(b)
    row['arms'][arm]={'selected_tensors':int(changed),'seconds':float(seconds),'first_score_sha256':native.sha(folder/(arm+'.first.f32')),'vs_original_bf16':comparison.compare(source,logits),'vs_q4':comparison.compare(baseline,logits),'prefix_ids':prefix,'prefix_decoded':(folder/(arm+'.decoded.txt')).read_text(),'prefix_exact':prefix==capture['generated_ids'][:len(prefix)],'first_argmax':int(logits.argmax()),'layer_differences_vs_q4':stages}
  report['status']='passed'
 except Exception as e:report['failure_type']=type(e).__name__;raise
 finally:
  (out/'restoration-sensitivity.json').write_text(json.dumps(report,indent=2));(out/'summary.md').write_text('Calibration Q8 restoration sensitivity: '+report['status']+'. Heldout cases untouched; first-score/two-token only.\n')
if __name__=='__main__':main()
