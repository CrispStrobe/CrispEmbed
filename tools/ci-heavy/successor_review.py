"""Collect completed packaged-consumer evidence; device/model limits remain explicit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

DESKTOPS=('linux-x86_64','linux-arm64','macos-arm64','windows-x86_64')
MOBILES=('android-arm64-v8a','android-armeabi-v7a','ios-arm64')

def review(root,run,audit):
 evidence={};source=None;objects=None
 def load(artifact,name):
  nonlocal source,objects
  p=root/(artifact+'-'+str(run))/name
  raw=p.read_bytes();d=json.loads(raw)
  evidence[str(p.relative_to(root))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  return d
 def binding(d):
  nonlocal source,objects
  if not re.fullmatch('[0-9a-f]{40}',d.get('source_commit','')) or d.get('consumer_source')!=audit:
   raise ValueError('Consumer source identity differs')
  if source is None:source=d['source_commit']
  if source!=d['source_commit']:raise ValueError('Mixed producer revisions')
  if 'runtime_source_closure' in d:
   actual=d['runtime_source_closure']['objects']
   if not actual:raise ValueError('Production source closure absent')
   if objects is None:objects=actual
   if actual!=objects:raise ValueError('Production source objects differ')
 def parity(v):
  return (v.get('passed') is True and type(v.get('max_abs')) in (float,int)
   and type(v.get('cosine')) in (float,int) and math.isfinite(v['max_abs'])
   and math.isfinite(v['cosine']) and 0<=v['max_abs']<=.002 and v['cosine']>=.99999)
 for platform in DESKTOPS:
  d=load('successor-consumer-'+platform,'consumer.json');binding(d)
  expected={'cli_to_cabi','offline_cli','python_to_cabi','compiled_C_to_cabi'}
  if (d.get('desktop_bounded_consumer_positive') is not True or set(d.get('comparisons',{}))!=expected
   or not all(parity(v) for v in d['comparisons'].values())
   or not all(d.get(k) is True for k in ('empty_cache_download','empty_cache_offline_rejected','offline_reuse'))
   or len(d.get('vector',[]))!=384 or d.get('compiled_hparams_size',0)<32
   or not {'crispembed_init','crispembed_encode','crispembed_free','crispembed_resolve_model','crispembed_set_offline'}<=set(d.get('abi_exports',[]))
   or not all(re.fullmatch('[0-9a-f]{64}',d.get(k,'')) for k in ('library_sha256','cli_binary_sha256','compiled_c_consumer_sha256','model_sha256'))):
   raise ValueError('Incomplete desktop checks: '+platform)
 for platform in MOBILES:
  d=load('successor-mobile-'+platform,'consumer.json');binding(d)
  if d.get('mobile_abi')!=platform or d.get('cross_link_passed') is not True or d.get('cross_link_exit_code')!=0 or d.get('device_execution_performed') is not False:
   raise ValueError('Mobile link/device scope differs: '+platform)
 d=load('successor-wasm-consumer','browser.json');binding(d)
 if (d.get('page_errors')!=[] or d.get('ocr',{}).get('authored_black_blob_cleared_rgb_exact') is not True
  or len(d.get('embed',{}).get('vector',[]))!=384 or not parity(dict(d,passed=True))):
  raise ValueError('Browser consumer failed')
 candidate=load('successor-apple-released','apple-candidate/consumer.json');binding(candidate)
 d=load('successor-apple-released','apple-released/comparison.json');binding(d)
 if (d.get('candidate_unset_one_exact') is not True or d.get('legacy_zero_preserved') is not True
  or set(d.get('comparisons',{}))!={'candidate-unset','candidate-one'}):raise ValueError('Apple compatibility incomplete')
 for v in d['comparisons'].values():
  if v.get('cases')!=50 or not all(v.get(k) is True for k in ('token_ids_exact','text_exact','stops_exact')):
   raise ValueError('Apple50 released-output mismatch')
 for name in ('released','candidate-unset','candidate-one','candidate-zero'):
  item=load('successor-apple-released','apple-released/'+name+'.json')
  if item.get('completed') is not True or len(item.get('cases',[]))!=50:raise ValueError('Apple raw case coverage missing')
 d=load('successor-prebuilt-bindings','successor-bindings.json')
 if d.get('audit_source')!=audit or d.get('passed') is not True or set(d.get('surfaces',{}))!={'rust','dart'}:
  raise ValueError('Prebuilt binding coverage failed')
 binding(d['producer'])
 if d.get('native_build_fallback') is not False or d.get('offline_cache_unchanged') is not True:
  raise ValueError('Binding fallback/cache changed')
 for v in d['surfaces'].values():
  if v.get('passed') is not True or len(v.get('comparisons',[]))!=12 or not all(parity(x) for x in v['comparisons']):
   raise ValueError('Binding vector comparison failed')
 return {'schema':'crispembed-successor-bounded-consumer-review-v1','producer_source':source,'audit_source':audit,
  'desktop_platforms':list(DESKTOPS),'mobile_cross_link_platforms':list(MOBILES),'small_reports':evidence,
  'desktop_browser_apple50_and_linux_bindings_positive':True,'mobile_cross_links_positive':True,
  'mobile_device_execution_verified':False,'model_portfolio_qualified':False,'release_published':False,
  'application_pin_changed':False,'limits':['One published MiniLM plus frozen50PosFormer compatibility; no new task-quality qualification.',
   'Mobile archives cross-link; physical-device inference remains unverified.','Optional GPU archives are outside this CPU/mobile candidate.']}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.write_text(json.dumps(review(a.root,os.environ['GITHUB_RUN_ID'],os.environ['GITHUB_SHA']),indent=2,allow_nan=False)+'\n')
