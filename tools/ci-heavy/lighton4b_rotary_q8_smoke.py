#!/usr/bin/env python3
"""One corrected Q8 native smoke with immutable source/prepared/stage controls."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
VISION_GATES=('CRISPEMBED_LIGHTON4B_VISION_BF16','CRISPEMBED_LIGHTON4B_VISION_BF16_NORM','CRISPEMBED_LIGHTON4B_VISION_BF16_FMA_NORM','CRISPEMBED_LIGHTON4B_VISION_BF16_LINEAR','CRISPEMBED_LIGHTON4B_VISION_BF16_CONV','CRISPEMBED_LIGHTON4B_VISION_BF16_ATTENTION','CRISPEMBED_LIGHTON4B_VISION_BF16_SOFTMAX_AVX2','CRISPEMBED_LIGHTON4B_VISION_BF16_GELU_AVX2','CRISPEMBED_LIGHTON4B_VISION_BF16_MERGER_ERF_AVX2','CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_TABLE')
def main():
 if os.environ.get('GITHUB_ACTIONS')!='true'or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':raise RuntimeError('hosted portable Q8 only')
 import numpy as np
 from kaggle.api.kaggle_api_extended import KaggleApi
 if np.__version__!='2.3.5':raise RuntimeError('canonical prepared-input serializer version required')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);native=load('native',ROOT/'tools/ci-heavy/lighton4b_native_document.py');comparison=load('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py')
 report={'schema':'lighton4b-rotary-q8-smoke/v1','status':'failed','full_completion_executed':False,'quality_qualified':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'recipe':'q8_0','source_vision_gates':VISION_GATES,'explicit_disabled_gates':{'CRISPEMBED_LIGHTON4B_BF16_LINEAR':'0','CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_AVX2':'0','CRISPEMBED_QWEN35_WEIGHT_CACHE':'0'}}
 def sha(raw):return hashlib.sha256(raw).hexdigest()
 try:
  mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())};report['preflight']={'memory':native.memory(),'load_average':os.getloadavg(),'swap_total_bytes':mem['SwapTotal'],'swap_free_bytes':mem['SwapFree'],'free_bytes':shutil.disk_usage(scratch).free,'required_available_bytes':10*1024**3,'required_free_bytes':12*1024**3,'estimated_native_peak_bytes':6400000000}
  if report['preflight']['memory']['available_bytes']<10*1024**3 or report['preflight']['free_bytes']<12*1024**3:raise ValueError('guarded Q8 capacity unavailable')
  expected=('lightonocr3-4b-q8_0-vision-f32.gguf','5888532207f0d98590cc028cbf92366f8a41ec925568f0c308bd69d9af6342f1',5815503232)
  if native.CANDIDATES['q8_0']!=expected or native.MODEL_REVISION!='a7ec2e54103165a96b8bb6dc3478a2887edd7459':raise ValueError('immutable Q8 identity differs')
  os.environ.update(report['explicit_disabled_gates']);native.GATES=tuple(native.GATES)+VISION_GATES
  args=sys.argv;sys.argv=[str(ROOT/'tools/ci-heavy/lighton4b_native_document.py'),'--recipe','q8_0','--mode','smoke','--threads','4']
  try:native.main()
  finally:sys.argv=args
  result=json.loads((out/'native-document.json').read_text());report['native_source_sha']=result['source_sha'];report['model']=result['model'];report['native_resources']={k:result[k]for k in ('native_elapsed_seconds','native_cpu_user_seconds','native_cpu_system_seconds','native_peak_rss_bytes')}
  # Source completion and prepared-input digests are independent of this runtime.
  source_url='https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/26fe41e35/tests/results/qwen35-ocr/lightonocr3-4b-bf16-cpu-capture.json'
  with urllib.request.urlopen(source_url,timeout=60)as f:raw=f.read(1000001)
  if len(raw)>1000000 or sha(raw)!='170bc06c8b5e929e45d334a5eadc284024f4265b2549f2feddb46607cb49251a':raise ValueError('immutable source completion differs')
  reference=json.loads(raw);teacher=reference['generated_ids'];report['original_capture_sha256']=sha(raw);prepared=reference['prepared_inputs'];candidate=out/'q8_0'
  def ids(name):return np.array([int(x)for x in (candidate/name).read_text().split()],dtype='<i8')
  prompt=ids('prompt.ids');grid=ids('processor.grid.ids');modalities=ids('modalities.ids');pixels=np.fromfile(candidate/'processor.patches.f32',dtype='<f4').reshape(560,1536)
  values={'input_ids':prompt.reshape(1,-1),'image_grid_thw':grid.reshape(1,3),'mm_token_type_ids':modalities.reshape(1,-1),'pixel_values':pixels,'attention_mask':np.ones((1,len(prompt)),dtype='<i8')};report['prepared_input_controls']={}
  for name,value in values.items():
   data=io.BytesIO();np.save(data,value,allow_pickle=False);digest=sha(data.getvalue());same=digest==prepared[name]['sha256'];report['prepared_input_controls'][name]={'shape':list(value.shape),'canonical_npy_sha256':digest,'source_npy_sha256':prepared[name]['sha256'],'exact':same,'scope':'unpadded native sequence mask contract'if name=='attention_mask'else'actual native emitted payload canonically serialized'}
   if not same:raise ValueError('source prompt/grid/modality/pixel control differs')
  generated=result['decoded']['ids'];report['decoded']={**result['decoded'],'original_prefix_ids':teacher[:len(generated)],'exact_original_prefix_ids':generated==teacher[:len(generated)],'original_completion_tokens':len(teacher)}
  if result['decoded']['exact_original_prefix_ids']!=report['decoded']['exact_original_prefix_ids']:raise ValueError('prefix observer differs from independent source')
  provider=KaggleApi();provider.authenticate();original_dir=scratch/'q8-original';provider.kernels_output(os.environ['KAGGLE_ACCOUNT']+'/lighton4b-bf16-targeted-original-cpu-prefill',str(original_dir),file_pattern=r'^lighton4b-bf16-prefill/stages/(prefill\.json|prefill\.logits\.f32)$',quiet=True)
  original=original_dir/'lighton4b-bf16-prefill/stages';raw=(original/'prefill.json').read_bytes()
  if sha(raw)!=comparison.ORIGINAL_REPORT_SHA:raise ValueError('source stage manifest differs')
  manifest=json.loads(raw);names=['processor.patches','vision.patch','vision.position','vision.embedded']+[f'vision.layer.{i}'for i in range(24)]+['vision.merged'];report['raw_vision_controls']={}
  for name in names:
   filename=name+'.f32';record=manifest['stages'][name];path=candidate/filename;digest=native.sha(path);same=path.stat().st_size==record['bytes']and digest==record['sha256'];report['raw_vision_controls'][name]={'bytes':path.stat().st_size,'source_sha256':record['sha256'],'native_sha256':digest,'bit_exact':same}
   if not same:raise ValueError('corrected Q8 vision stage not raw-source-exact')
  record=manifest['stages']['prefill.logits'];raw=(original/record['file']).read_bytes()
  if len(raw)!=record['bytes']or sha(raw)!=record['sha256']:raise ValueError('original first score differs')
  expected_logits=np.frombuffer(raw,dtype='<f4');actual=np.fromfile(candidate/'prefill.logits.f32',dtype='<f4');report['first_logits_vs_original']=comparison.compare(expected_logits,actual)
  # One historical baseline only: exact checkpoint/image but a different runtime source.
  item=comparison.api('actions/artifacts/13403459726')
  if item['workflow_run']['id']!=42005430424 or item['workflow_run']['head_sha']!='891d07da39e9c361a206f2ec4b40fe8339712e0f'or item['expired']or item['size_in_bytes']!=78635262:raise ValueError('historical Q8 identity differs')
  archive=scratch/'historical-q8.zip'
  with archive.open('wb')as f:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/13403459726/zip'],stdout=f,check=True)
  if native.sha(archive)!='0b57e702aff9a81c747498fef65c8b61e0e626a8b901ded3ef561d6e9e600c00':raise ValueError('historical Q8 ZIP differs')
  with zipfile.ZipFile(archive)as z:
   reports=[n for n in z.namelist()if n=='native-document.json'or n.endswith('/native-document.json')]
   if len(reports)!=1:raise ValueError('historical report inventory differs')
   raw=z.read(reports[0])
   if sha(raw)!='6f2a355e7bdb14515d178e4dad0d50ed0f76c93df2088586a4dbb1ffbba813e6':raise ValueError('historical report SHA differs')
   prior=json.loads(raw)
   if prior['model']!=result['model']or prior['image']!=result['image']:raise ValueError('historical checkpoint or image differs')
   matches=[n for n in z.namelist()if n.endswith('/prefill.logits.f32')]
   if len(matches)!=1:raise ValueError('historical score inventory differs')
   raw=z.read(matches[0]);r=prior['artifacts']['prefill.logits.f32']
   if len(raw)!=r['bytes']or sha(raw)!=r['sha256']:raise ValueError('historical first-score bytes differ')
   prior_logits=np.frombuffer(raw,dtype='<f4');report['historical_q8']={'run_id':42005430424,'source_sha':prior['source_sha'],'artifact_id':13403459726,'zip_sha256':'0b57e702aff9a81c747498fef65c8b61e0e626a8b901ded3ef561d6e9e600c00','report_sha256':'6f2a355e7bdb14515d178e4dad0d50ed0f76c93df2088586a4dbb1ffbba813e6','same_model_and_image':True,'same_runtime_source':False,'controlled_same_source_AB':False,'decoded':prior['decoded'],'exact_original_prefix_ids':prior['decoded']['ids']==teacher[:len(prior['decoded']['ids'])],'first_logits_vs_original':comparison.compare(expected_logits,prior_logits),'native_resources':{k:prior[k]for k in ('native_elapsed_seconds','native_cpu_user_seconds','native_cpu_system_seconds','native_peak_rss_bytes')},'enabled_gates':prior['gates'],'scope':'Historical legacy-BF16 vision runtime; new source includes all verified norm/Linear/Conv/attention/GELU/merger/rotary fixes and other gated integration work. No claim of a table-only or same-source speed/quality effect.'};report['candidate_vs_historical_logits']=comparison.compare(prior_logits,actual)
  report['status']='measured'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='bounded corrected Q8 smoke/control failed; inspect native report';raise RuntimeError(report['error'])from None
 finally:(out/'rotary-q8-smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='raw_vision_controls'},indent=2),flush=True)
if __name__=='__main__':main()
