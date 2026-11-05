#!/usr/bin/env python3
"""Independent retained ZIP/manifest stage-byte and decoded-prefix audit; no inference."""
import array
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[2]
def main():
 if os.environ.get('GITHUB_ACTIONS')!='true':raise RuntimeError('hosted retained-only audit')
 from kaggle.api.kaggle_api_extended import KaggleApi
 spec=importlib.util.spec_from_file_location('comparison',ROOT/'tools/ci-heavy/lighton4b_vision_bf16_compare.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'rotary-raw-audit';scratch.mkdir()
 report={'schema':'lighton4b-rotary-raw-audit/v1','status':'failed','model_downloaded':False,'inference_executed':False,'production_changed':False,'quality_qualified':False,'audit_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
 def sha(raw):return hashlib.sha256(raw).hexdigest()
 def member(z,path):
  value=hashlib.sha256();count=0
  with z.open(path)as stream:
   while block:=stream.read(2**20):value.update(block);count+=len(block)
  return {'bytes':count,'sha256':value.hexdigest()}
 try:
  mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())};report['preflight']={'load_average':os.getloadavg(),'memory_available_bytes':mem['MemAvailable'],'swap_total_bytes':mem['SwapTotal'],'swap_free_bytes':mem['SwapFree'],'scratch_free_bytes':shutil.disk_usage(scratch).free}
  if mem['MemAvailable']<512*2**20 or report['preflight']['scratch_free_bytes']<512*2**20:raise ValueError('report-only resource bound failed')
  item=comparison.api('actions/artifacts/13449048158')
  if item['workflow_run']['id']!=42099925931 or item['expired']or item['size_in_bytes']!=238119147:raise ValueError('final Q4 artifact identity differs')
  archive=scratch/'retained.zip'
  with archive.open('wb')as f:subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/13449048158/zip'],stdout=f,check=True)
  value=hashlib.sha256()
  with archive.open('rb')as f:
   while block:=f.read(2**20):value.update(block)
  if value.hexdigest()!='5939d0877591855bd3c494c4f57afef65f7712c40d8fdb6c99422d5c48a603ee':raise ValueError('retained complete ZIP SHA differs')
  report['retained']={'run_id':42099925931,'source_sha':'a1bb8c924607e2631435a973512aeee94ea5f0be','artifact_id':13449048158,'zip_bytes':238119147,'zip_sha256':value.hexdigest(),'expires_at':item['expires_at']}
  provider=KaggleApi();provider.authenticate();destination=scratch/'original'
  provider.kernels_output(os.environ['KAGGLE_ACCOUNT']+'/lighton4b-bf16-targeted-original-cpu-prefill',str(destination),file_pattern=r'^lighton4b-bf16-prefill/stages/prefill\.json$',quiet=True)
  original=destination/'lighton4b-bf16-prefill/stages';raw=(original/'prefill.json').read_bytes()
  if sha(raw)!=comparison.ORIGINAL_REPORT_SHA:raise ValueError('independent original manifest differs')
  source=json.loads(raw);report['original_manifest_sha256']=sha(raw);frozen=json.loads((ROOT/'tests/results/qwen35-ocr/lightonocr3-4b-vision-fma-baseline-stages.json').read_text())['stages']
  names=['processor.patches','vision.patch','vision.position','vision.embedded']+['vision.layer.'+str(i)for i in range(24)]+['vision.merged']
  downstream=['text.embedded','prefill.input','prefill.positions']+['prefill.layer.'+str(i)for i in range(32)]+['prefill.norm','prefill.logits']
  with zipfile.ZipFile(archive)as z:
   native={arm:json.loads(z.read(arm+'/native-document.json'))for arm in ('baseline','corrected')};composed=json.loads(z.read('vision-layer0.json'))
   if composed['source_sha']!=report['retained']['source_sha']or any(r['source_sha']!=report['retained']['source_sha']for r in native.values()):raise ValueError('retained executable source differs')
   common=[n for n in source['stages']if n+'.f32'in native['corrected']['artifacts']and n+'.f32'in native['baseline']['artifacts']]
   if not set(names).issubset(common):raise ValueError('required processor/vision stages missing')
   ordered=names+[n for n in downstream if n in common]+[n for n in common if n not in names and n not in downstream];report['stages']={}
   for name in ordered:
    original_row=source['stages'][name];entry={'original':{'bytes':original_row['bytes'],'sha256':original_row['sha256']}}
    for arm in ('baseline','corrected'):
     actual=member(z,arm+'/q4_k/'+name+'.f32');declared=native[arm]['artifacts'][name+'.f32']
     if actual!=declared:raise ValueError('independent archived raw bytes differ from native manifest')
     actual['equal_original_raw']=actual['bytes']==original_row['bytes']and actual['sha256']==original_row['sha256'];entry[arm]=actual
     if arm=='baseline'and name in frozen and actual['sha256']!=frozen[name]['sha256']:raise ValueError('independent fixed baseline bytes differ')
    report['stages'][name]=entry
   report['raw_identical_processor_vision_stages']=[n for n in names if report['stages'][n]['corrected']['equal_original_raw']]
   if len(report['raw_identical_processor_vision_stages'])!=29:raise ValueError('numerical vision equality failed raw-byte audit')
   report['shared_float_stages']=len(common);report['source_only_stages']=[n for n in source['stages']if n not in common];report['fixed_baseline_hashes_verified']=len(frozen)
   # Independently distinguish smoke prefix equality from full-completion equality.
   url='https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/26fe41e35/tests/results/qwen35-ocr/lightonocr3-4b-bf16-cpu-capture.json'
   with urllib.request.urlopen(url,timeout=60)as f:raw=f.read(1000001)
   if len(raw)>1000000 or sha(raw)!='170bc06c8b5e929e45d334a5eadc284024f4265b2549f2feddb46607cb49251a':raise ValueError('original completion source differs')
   teacher=json.loads(raw)['generated_ids']
   if len(teacher)!=177:raise ValueError('original completion length differs')
   report['prefix_comparison']={'original_completion_tokens':177,'original_capture_sha256':sha(raw),'original_first_ids':teacher[:3],'arms':{}}
   report['native_resources']={}
   for arm in ('baseline','corrected'):
    generated_raw=z.read(arm+'/q4_k/generated.ids');generated=[int(x)for x in generated_raw.split()]
    if sha(generated_raw)!=native[arm]['artifacts']['generated.ids']['sha256']or generated!=native[arm]['decoded']['ids']:raise ValueError('decoded ID artifact differs')
    report['prefix_comparison']['arms'][arm]={'ids':generated,'text':native[arm]['decoded']['text'],'original_prefix_ids':teacher[:len(generated)],'exact_original_prefix_ids':generated==teacher[:len(generated)],'exact_full_completion':generated==teacher}
    report['native_resources'][arm]={k:native[arm][k]for k in ('native_elapsed_seconds','native_cpu_user_seconds','native_cpu_system_seconds','native_peak_rss_bytes')}
   first=next((n for n in downstream if n in report['stages']and not report['stages'][n]['corrected']['equal_original_raw']),None);report['first_downstream_shared_raw_difference']=first
   if first:
    row=source['stages'][first]
    if row['bytes']<=8*2**20 and report['stages'][first]['corrected']['bytes']==row['bytes']:
     filename=Path(row['file']).name
     provider.kernels_output(os.environ['KAGGLE_ACCOUNT']+'/lighton4b-bf16-targeted-original-cpu-prefill',str(destination),file_pattern=r'^lighton4b-bf16-prefill/stages/'+re.escape(filename)+r'$',quiet=True)
     expected=(original/row['file']).read_bytes();actual=z.read('corrected/q4_k/'+first+'.f32')
     if sha(expected)!=row['sha256']:raise ValueError('first differing original float array differs')
     aa=array.array('f');aa.frombytes(expected);bb=array.array('f');bb.frombytes(actual);ua=array.array('I');ua.frombytes(expected);ub=array.array('I');ub.frombytes(actual)
     if aa.itemsize!=4 or ua.itemsize!=4:raise ValueError('unsupported audit float representation')
     report['first_downstream_magnitudes']={'stage':first,'elements':len(aa),'bit_different_values':sum(a!=b for a,b in zip(ua,ub)),'different_values':sum(a!=b for a,b in zip(aa,bb)),'max_abs':max(abs(a-b)for a,b in zip(aa,bb)),'rms_error':math.sqrt(math.fsum((a-b)**2 for a,b in zip(aa,bb))/len(aa)),'mean_abs':math.fsum(abs(a-b)for a,b in zip(aa,bb))/len(aa)}
   report['prefill_logits_comparison']=composed['composed']['corrected']['stages']['prefill.logits'];report['status']='verified'
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']='independent retained raw-stage/prefix audit failed; inspect report';raise RuntimeError(report['error'])from None
 finally:(out/'rotary-raw-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='stages'},indent=2),flush=True)
if __name__=='__main__':main()
