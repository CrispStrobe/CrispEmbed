#!/usr/bin/env python3
"""Bounded real-payload BF16 weight hash and source-fed operator isolation; no generation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
ORACLE_SHA='03119ea4d0fbbf5cc86b91f80b47bbda22d90b417d89dd0c16b8fafa1a85a89b'
TRACE_SHA='5aa5a8026226f218a6e441d6e0c6d1a8a5444b7113574943f92e94d28d3c86b4'
def require(ok,why):
 if not ok:raise ValueError(why)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def provider(slug):
 from kaggle.api.kaggle_api_extended import KaggleApi
 service=KaggleApi();service.authenticate();Get=KaggleApi.kernels_pull.__globals__['ApiGetKernelRequest'];List=KaggleApi.kernels_output.__globals__['ApiListKernelSessionOutputRequest'];urls={};token=None
 with service.build_kaggle_client() as client:
  req=Get();req.user_name=os.environ['KAGGLE_ACCOUNT'];req.kernel_slug=slug;req.version_label='v1';client.kernels.kernels_api_client.get_kernel(req)
  for _ in range(16):
   req=List();req.user_name=os.environ['KAGGLE_ACCOUNT'];req.kernel_slug=slug;req.version_label='v1';req.page_size=100
   if token:req.page_token=token
   page=client.kernels.kernels_api_client.list_kernel_session_output(req)
   for item in page.files or []:
    require(item.file_name not in urls,'duplicate source member');urls[item.file_name]=item.url
   token=page.next_page_token
   if not token:return urls
 raise ValueError('source listing bound')
def fetch(urls,name,maximum,expected):
 require(name in urls,'source member missing')
 try:
  with urllib.request.urlopen(urls[name],timeout=120) as response:raw=response.read(maximum+1)
 except Exception:raise ValueError('bounded source transfer failed')from None
 require(len(raw)<=maximum and digest(raw)==expected,'source bytes/hash differ');return raw
def main():
 require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable operators required')
 import numpy as np
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);native=load('native_common',ROOT/'tools/ci-heavy/lighton4b_native_document.py');comparison=load('source_compare',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py')
 report={'schema':'lighton4b-bf16-weight-operators/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'quality_qualified':False,'generation_executed':False,'full_model_f32_storage':False,'reference_original_dtype':'bfloat16','scope':'source-fed fixed operator inputs; mixed reference FFNQ4 and sharedQ8 embedding/head; no composed inference/speed claim'}
 def run(name,args,env=None):
  with (out/(name+'.log')).open('w')as log:subprocess.run([str(x)for x in args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 try:
  report['preflight']={'memory':native.memory(),'load_average':os.getloadavg(),'scratch_free_bytes':shutil.disk_usage(scratch).free,'output_free_bytes':shutil.disk_usage(out).free}
  require(report['preflight']['memory']['available_bytes']>=10*2**30 and report['preflight']['scratch_free_bytes']>=8*2**30 and report['preflight']['output_free_bytes']>=2**30,'bounded operator capacity unavailable')
  report['stage']='authenticate-source-weights-and-inputs'
  urls=provider('lighton4b-bf16-quant-first-score-v1');raw=fetch(urls,'bf16-quant-first-score/quant-oracle/oracle.json',8*2**20,ORACLE_SHA);oracle=json.loads(raw)
  require(oracle['source_sha']=='048fd683f4ad4adf50f945f6d7c6b436bacdb9aa' and oracle['comparison_completed'],'original weight oracle status/source differs')
  weights=[row for row in oracle['q4_k-weights']['tensors']if row['quantized']];require(len(weights)==249 and all('bf16_sha256'in row for row in weights),'complete actual BF16 weight hashes required')
  expected=scratch/'expected-weights.tsv';expected.write_text(''.join(row['gguf']+'\t'+row['raw_sha256']+'\t'+row['bf16_sha256']+'\n'for row in weights))
  urls=provider('lighton4b-bf16-mixed-step33-v1');prefix='bf16-mixed-step33/quant-oracle/';raw=fetch(urls,prefix+'oracle.json',8*2**20,TRACE_SHA);trace=json.loads(raw)
  require(trace['source_sha']=='8eeb67bdeb1bf0e55b0d9ffd833a600967615765' and trace['trace_observer_all34_scores_bit_exact'] and trace['trace_observer_all34_ids_exact'] and trace['trace_input_ids']==[68100] and trace['trace_positions']==[60,60,60],'genuine source operator input controls differ')
  by_name={row['gguf']:row for row in weights}
  selected=['l.embed_tokens.weight','l.layers.0.mlp.gate_proj.weight','l.layers.0.mlp.up_proj.weight','l.layers.0.mlp.down_proj.weight']
  require(all(by_name[name]['bf16_sha256']==trace['mixed_final_bf16_parameter_sha256'][name]for name in selected),'source-fed operators must use identical effective source weights')
  report['source_operator_weight_identity']={name:by_name[name]['bf16_sha256']for name in selected}
  names=['greedy.33.input','greedy.33.head.input','greedy.33.head.output','greedy.33.layer0.post_norm','greedy.33.layer0.activation','greedy.33.layer0.gate','greedy.33.layer0.up','greedy.33.layer0.down'];inputs=scratch/'source-inputs';inputs.mkdir();report['source_arrays']={}
  for name in names:
   row=trace['trace_stages'][name];raw=fetch(urls,prefix+row['file'],row['bytes'],row['sha256']);require(len(raw)==row['bytes'],'source array size differs');(inputs/row['file']).write_bytes(raw);report['source_arrays'][name]=row
  report['reference_pins']={'weights_oracle_sha256':ORACLE_SHA,'source_trace_sha256':TRACE_SHA,'source_trace_sha':trace['source_sha'],'all249_bf16_weights_expected':True}
  report['stage']='compile-bounded-operators';build=scratch/'weight-operators-build';run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF']);run('build',['cmake','--build',build,'--target','test-lighton4b-bf16-dequant-weights','lighton4b-bf16-weight-operators','--parallel','2']);run('unit-controls',[build/'test-lighton4b-bf16-dequant-weights'])
  report['stage']='stage-immutable-plain-q4';filename,expected_sha,size=native.CANDIDATES['q4_k'];model=scratch/filename;native.download(f'https://huggingface.co/{native.MODEL_REPO}/resolve/{native.MODEL_REVISION}/{filename}',model);require(model.stat().st_size==size and native.sha(model)==expected_sha,'plainQ4 immutable payload differs');report['model']={'repository':native.MODEL_REPO,'revision':native.MODEL_REVISION,'file':filename,'sha256':expected_sha,'bytes':size}
  env=os.environ.copy();env.update({key:'1'for key in native.GATES});env.update({'CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS':'0','CRISPEMBED_LIGHTON4B_MIXED_ATTN_Q8_FFN_Q4':'0','CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT':'0','CRISPEMBED_LIGHTON4B_DIRECT_QUANT':'0','CRISPEMBED_LIGHTON4B_BF16_LINEAR':'0','CRISPEMBED_QWEN35_WEIGHT_CACHE':'0','CRISPEMBED_QWEN35_IMATRIX':'0'})
  actual=out/'operators';actual.mkdir();resources=out/'operator-resources.txt';binary=build/'lighton4b-bf16-weight-operators';report['binary_sha256']=native.sha(binary);report['stage']='actual-weight-hashes-and-source-fed-operators'
  run('operators',['/usr/bin/time','-q','-f','%U %S %e %M','-o',resources,binary,model,inputs,expected,actual],env)
  report['stdout']=(out/'operators.log').read_text();require('BF16_WEIGHT_OPERATORS passed=1 tensors=249' in report['stdout'],'complete source weight proof missing');user,system,elapsed,rss=resources.read_text().strip().split();report['resources']={'wall_seconds':float(elapsed),'user_seconds':float(user),'system_seconds':float(system),'peak_rss_bytes':int(rss)*1024}
  rows=[line.split()for line in (actual/'weight-hashes.tsv').read_text().splitlines()];require(len(rows)==249 and {row[0]for row in rows}=={row['gguf']for row in weights},'complete native weight coverage differs');report['weight_hashes_source_exact']=True
  report['operators']={};mapping={'embedding':'greedy.33.input','head':'greedy.33.head.output','gate':'greedy.33.layer0.gate','up':'greedy.33.layer0.up','down':'greedy.33.layer0.down'}
  for operator,reference_name in mapping.items():
   reference=np.fromfile(inputs/trace['trace_stages'][reference_name]['file'],dtype='<f4');row={}
   for arm in ['off','on','off-repeat']:
    filename=arm+'.'+operator+('.f32' if operator=='embedding' else '.bf16.f32');values=np.fromfile(actual/filename,dtype='<f4');row[arm]=comparison.compare(reference,values)
   require(native.sha(actual/('off.'+operator+('.f32'if operator=='embedding'else'.raw.f32')))==native.sha(actual/('off-repeat.'+operator+('.f32'if operator=='embedding'else'.raw.f32'))),'actual gate-off repeat changed bytes');report['operators'][operator]=row
  require(report['operators']['embedding']['on']['different_values']==0,'BF16-rounded native embedding must equal genuine source')
  report['artifacts']={path.name:{'bytes':path.stat().st_size,'sha256':native.sha(path)}for path in actual.iterdir()if path.is_file()};report['status']='measured'
 except Exception as problem:
  report['error_type']=type(problem).__name__;report['error']='bounded source-fed weight/operator control failed; inspect preserved logs';raise RuntimeError(report['error'])from None
 finally:(out/'bf16-weight-operators.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
