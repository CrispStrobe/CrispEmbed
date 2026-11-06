#!/usr/bin/env python3
"""Closed original-BF16 Q6 conversion + two calibration first-score A/Bs; no heldout."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_q6_convert as conversion
from lighton4b_q6_recipe import plan,RECIPE

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def calibration_member(case,name):
 conversion.require(case in('simple_table','arabic_printed_line')and Path(name).name==name,'only calibration basenames may be fetched')
 return 'lighton4b-fourcase-bf16/'+case+'/'+name

def main():
 require=conversion.require;require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable closed worker required')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'q6-candidate';scratch.mkdir();p=json.loads((ROOT/'docs/lighton4b-q6-conversion-protocol-v1.json').read_text())
 common=load('q6_common',ROOT/'tools/ci-heavy/lighton4b_native_document.py');cases=load('q6_cases',ROOT/'tools/ci-heavy/lighton4b_frozen_case.py');transport=load('q6_transport',ROOT/'tools/ci-heavy/lighton4b_calibration_native.py');vision=load('q6_vision',ROOT/'tools/ci-heavy/lighton4b_rotary_q8_smoke.py')
 report={'schema':'lighton4b-q6-candidate-first-score/v1','status':'blocked','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'protocol_sha256':common.sha(ROOT/'docs/lighton4b-q6-conversion-protocol-v1.json'),'recipe':RECIPE,'calibration_cases':['simple_table','arabic_printed_line'],'heldout_captures_read':False,'original_batch_json_read':False,'reference_model_forward_executed':False,'f32_model_instantiated':False,'quality_qualified':False,'cases':{}}
 active={}
 def save():(out/'q6-candidate-first-score.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 def terminate(*args):
  child=active.get('child')
  if child is not None and child.poll()is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  raise TimeoutError('bounded Q6 worker deadline')
 def run(label,args,env=None,timeout=1200):
  report['stage']=label;save()
  with(out/(label+'.log')).open('w')as log:
   child=subprocess.Popen(list(map(str,args)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active['child']=child
   try:code=child.wait(timeout=timeout)
   except subprocess.TimeoutExpired:terminate()
   active.pop('child',None);require(code==0,'bounded child failed: '+label)
 def capacity():
  memory=common.memory();a=shutil.disk_usage(scratch).free;b=shutil.disk_usage(out).free;shared=scratch.stat().st_dev==out.stat().st_dev
  require(memory['available_bytes']>=8*2**30 and (a>=27*2**30 if shared else a>=20*2**30 and b>=8*2**30),'bounded Q6 initial RAM/disk capacity unavailable')
  return {'memory':memory,'scratch_free_bytes':a,'output_free_bytes':b,'shared_filesystem':shared,'load_average':os.getloadavg()}
 def download(url,path,digest,limit):
  h=hashlib.sha256();count=0
  with urllib.request.urlopen(url,timeout=120)as src,path.open('xb')as dst:
   while block:=src.read(4*2**20):count+=len(block);require(count<=limit,'authenticated transfer exceeds bound');h.update(block);dst.write(block)
  require(h.hexdigest()==digest,'authenticated transfer digest differs');return count
 try:
  import numpy as np
  import PIL
  from PIL import Image
  require(np.__version__=='2.3.5'and PIL.__version__=='11.3.0','frozen decoder/serializer versions differ')
  require(p['calibration_only']['cases']==report['calibration_cases']and p['calibration_only']['heldout_capture_or_labels_read']is False,'closed calibration scope differs')
  report['preflight']=capacity();signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate);signal.signal(signal.SIGALRM,terminate);signal.alarm(90*60)
  build=scratch/'build';run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  targets=['lighton4b-q6-rows','test-lighton4b-q6-policy','lighton4b-native-document','lighton4b-native-case-preflight','crispembed-diff-lighton4b-document'];run('build',['cmake','--build',build,'--target',*targets,'--parallel','2']);report['binary_sha256']={x:common.sha(build/x)for x in targets}
  schema=out/'q6-schema.tsv';lines=[]
  for row in plan()['all_tensors']:
   if row['name'].startswith('l.'):
    ne=list(reversed(row['source_shape']));ne += [1]*(4-len(ne));lines.append(' '.join([row['name'],str(row['type']),*map(str,ne)]))
  schema.write_text('\n'.join(lines)+'\n');run('native-q6-controls',[build/'test-lighton4b-q6-policy',schema])
  checkpoint=scratch/'original.safetensors';base=scratch/'plain-q4.gguf';ck=p['original_checkpoint'];ba=p['base_container']
  download('https://huggingface.co/'+ck['repository']+'/resolve/'+ck['revision']+'/model.safetensors',checkpoint,ck['sha256'],ck['bytes']);require(checkpoint.stat().st_size==ck['bytes'],'checkpoint byte count differs')
  download('https://huggingface.co/'+ba['repository']+'/resolve/'+ba['revision']+'/'+ba['file'],base,ba['sha256'],ba['bytes']);require(base.stat().st_size==ba['bytes'],'plain Q4 byte count differs')
  oracle=scratch/'original-weight-oracle.json';download('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/6525e045e16906b2025cad4fa10696a2e918abac/tests/results/qwen35-ocr/lightonocr3-4b-bf16-quant-first-score-v1.json',oracle,conversion.ORACLE_SHA,2**20)
  candidate=out/(RECIPE+'.gguf');report['stage']='convert-original32-copy691';save();audit=conversion.convert(base,checkpoint,oracle,build/'lighton4b-q6-rows',candidate,out/'q6-conversion.json');report['candidate_sha256']=audit['candidate_sha256'];report['candidate_converted']=True;require(candidate.stat().st_size<=p['conversion']['candidate_byte_ceiling'],'candidate exceeds resource ceiling')
  def ref_compare(a,b,label):
   require(len(a)==len(b)==993280,'complete actual Ref vocabulary rows required')
   transport.tiny_archive(scratch/'ref-source.gguf',a);transport.tiny_archive(scratch/'ref-native.gguf',b);order=scratch/'ref-order.tsv';order.write_text('generated.scores.0 float 248320 1 1 1\n')
   result=subprocess.run([str(build/'crispembed-diff-lighton4b-document'),str(scratch/'ref-source.gguf'),str(scratch/'ref-native.gguf'),str(order)],capture_output=True,text=True,timeout=60);stages=[line for line in result.stdout.splitlines()if line.startswith('STAGE ')];require(result.returncode in(0,1)and len(stages)==1,'actual Ref measurement incomplete');(out/(label+'.ref.log')).write_text(result.stdout+result.stderr);return {'returncode':result.returncode,'stage':stages[0]}
  equal=np.ones(248320,dtype='<f4');changed=equal.copy();changed[0]+=1;require(ref_compare(equal.tobytes(),equal.tobytes(),'equal-control')['returncode']==0 and ref_compare(equal.tobytes(),changed.tobytes(),'changed-control')['returncode']==1,'actual Ref controls failed');report['actual_ref_controls_passed']=True
  urls=transport.original_urls();originals={};original=scratch/'original-calibration';original.mkdir()
  for case_id,capture_sha,_ in transport.CASES:
   require(case_id in report['calibration_cases'],'noncalibration transfer prohibited');folder=original/case_id;folder.mkdir();member=calibration_member(case_id,'capture.json');download(urls[member],folder/'capture.json',capture_sha,2**20);capture=json.loads((folder/'capture.json').read_text());_,contract,frozen=cases.load_contract(case_id);cases.validate_original(capture,contract);require(contract['split']=='calibration','case partition differs')
   for record in capture['prepared_inputs'].values():download(urls[calibration_member(case_id,record['file'])],folder/record['file'],record['sha256'],8*2**20)
   row=capture['scores'][0];download(urls[calibration_member(case_id,row['file'])],folder/row['file'],row['sha256'],993280);originals[case_id]=(capture,contract,frozen)
  env=os.environ.copy();env.update({x:'1'for x in common.GATES});env.update({x:'1'for x in vision.VISION_GATES});env.update({x:'0'for x in ('CRISPEMBED_LIGHTON4B_BF16_LINEAR','CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS','CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2','CRISPEMBED_LIGHTON4B_DIRECT_QUANT','CRISPEMBED_QWEN35_WEIGHT_CACHE','CRISPEMBED_QWEN35_PRECISE_RMS','CRISPEMBED_QWEN35_IMATRIX','CRISPEMBED_LIGHTON4B_GREEDY_TRACE','CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4','CRISPEMBED_LIGHTON4B_MIXED_GDN_OUT_Q8')});env.update(CRISPEMBED_LIGHTON4B_HEAD_BF16='1',CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT='1',CRISPEMBED_LIGHTON4B_FROZEN_CASES='1');report['fixed_gates']={k:v for k,v in env.items()if k.startswith('CRISPEMBED_')}
  for case_id in report['calibration_cases']:
   capture,contract,frozen=originals[case_id];png=ROOT/'tests/regression/lighton4b-mixed-v1'/frozen['png_file'];require(common.sha(png)==frozen['png_sha256'],'frozen calibration image differs');image=Image.open(png).convert('RGB');rgb=image.tobytes();require(image.size==(frozen['width'],frozen['height'])and hashlib.sha256(rgb).hexdigest()==frozen['rgb_sha256'],'calibration RGB differs');rgb_path=scratch/(case_id+'.rgb');rgb_path.write_bytes(rgb);teacher=scratch/(case_id+'.teacher.ids');teacher.write_text(str(capture['generated_ids'][0])+'\n');report['cases'][case_id]={}
   for arm,model,recipe in [('plain_q4',base,'q4_k'),('selective_q6',candidate,RECIPE)]:
    for path in(checkpoint,base,candidate):
     with path.open('rb')as f:os.fsync(f.fileno());os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    mem=common.memory();require(mem['available_bytes']>=8*2**30 and (mem['cgroup_limit_bytes']is None or mem['cgroup_limit_bytes']-mem['cgroup_current_bytes']>=2*2**30),'native case RAM/headroom unavailable')
    child=env.copy();child['CRISPEMBED_LIGHTON4B_Q6_K']='1'if arm=='selective_q6'else'0';folder=out/case_id/arm;folder.mkdir(parents=True);pre=folder/'preflight';pre.mkdir();native=folder/'native';native.mkdir()
    run(case_id+'-'+arm+'-preflight',[build/'lighton4b-native-case-preflight',model,rgb_path,image.height,image.width,pre,recipe],child)
    arrays={k:np.load(original/case_id/r['file'],allow_pickle=False)for k,r in capture['prepared_inputs'].items()};require(set(arrays)=={'pixel_values','image_grid_thw','input_ids','mm_token_type_ids','attention_mask'} and all(list(v.shape)==capture['prepared_inputs'][k]['shape']and v.dtype==np.dtype('<f4'if k=='pixel_values'else'<i8')for k,v in arrays.items()),'original canonical array geometry differs');require((pre/'processor.patches.f32').read_bytes()==arrays['pixel_values'].tobytes(order='C'),'actual native pixels differ')
    for field,name in [('image_grid_thw','grid'),('input_ids','prompt'),('mm_token_type_ids','modalities')]:require([int(x)for x in(pre/(name+'.ids')).read_text().split()]==arrays[field].reshape(-1).tolist(),'actual native input differs: '+field)
    require(np.all(arrays['attention_mask']==1),'original mask differs');run(case_id+'-'+arm+'-first-score',[build/'lighton4b-native-document',model,rgb_path,image.height,image.width,teacher,native,1,4,'smoke'],child,timeout=15*60)
    score=native/'prefill.logits.f32';require(score.stat().st_size==993280 and common.sha(score)==common.sha(native/'teacher.0.logits.f32')==common.sha(native/'greedy.0.logits.f32'),'same-position native paths differ')
    values=np.fromfile(score,dtype='<f4').astype(np.float64);gold=np.fromfile(original/case_id/capture['scores'][0]['file'],dtype='<f4').astype(np.float64);require(np.isfinite(values).all(),'native scores nonfinite');ids=[int(x)for x in(native/'generated.ids').read_text().split()];require(len(ids)==1 and ids[0]==int(values.argmax()),'actual independent one-token generation differs')
    if arm=='plain_q4':require(common.sha(score)==p['calibration_only']['plain_q4_first_score_sha256'][case_id],'plain baseline differs from frozen calibration before candidate')
    error=values-gold;prob=np.exp(values-values.max());prob/=prob.sum();report['cases'][case_id][arm]={'first_score_sha256':common.sha(score),'generated_ids':ids,'actual_eos':ids[-1]==248046,'cap_hit':ids[-1]!=248046,'generation_cap':1,'max_abs':float(np.abs(error).max()),'rms':float(np.sqrt(np.mean(error*error))),'cosine':float(values@gold/(np.linalg.norm(values)*np.linalg.norm(gold))),'probability_of_original_id':float(prob[capture['generated_ids'][0]]),'actual_ref':ref_compare((original/case_id/capture['scores'][0]['file']).read_bytes(),score.read_bytes(),case_id+'-'+arm),'model_sha256':conversion.BASE_SHA if arm=='plain_q4'else audit['candidate_sha256'],'input_admitted':True,'native_summary':json.loads((native/'native-summary.json').read_text()),'artifacts':{x.name:{'bytes':x.stat().st_size,'sha256':common.sha(x)}for x in native.iterdir()if x.is_file()}};save()
  report.update(status='measured',baseline_hash_controls_passed=True,candidate_converted=True,all_first_score_arms=4,full_generation_executed=False)
 except Exception as e:report['error_type']=type(e).__name__;report['error']=str(e)if isinstance(e,ValueError)else'closed bounded Q6 conversion/calibration failed';raise
 finally:signal.alarm(0);save()
if __name__=='__main__':main()
