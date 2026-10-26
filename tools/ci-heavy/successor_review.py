"""Collect completed packaged-consumer evidence; device/model limits remain explicit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probe_posformer_release import compare_release_records
from successor_assets import MODEL_SHA as MINILM_MODEL_SHA, TEXT as MINILM_TEXT

DESKTOPS=('linux-x86_64','linux-arm64','macos-arm64','windows-x86_64')
MOBILES=('android-arm64-v8a','android-armeabi-v7a','ios-arm64')


def desktop_vector(record,platform):
 vector=record.get('vector')
 if (record.get('model_sha256')!=MINILM_MODEL_SHA or record.get('text')!=MINILM_TEXT
  or not isinstance(vector,list) or len(vector)!=384
  or any(type(v) not in (float,int) or not math.isfinite(v) for v in vector)):
  raise ValueError('Desktop shared model/text/finite384-vector contract differs: '+platform)
 norm=math.sqrt(math.fsum(v*v for v in vector))
 if not math.isfinite(norm) or norm<=0:
  raise ValueError('Desktop vector magnitude invalid: '+platform)
 return vector


def compare_desktop_vectors(reference,vector):
 # Recompute from retained values rather than trusting platform self-comparisons.
 reference_norm=math.sqrt(math.fsum(v*v for v in reference))
 native_norm=math.sqrt(math.fsum(v*v for v in vector))
 errors=[abs(a-b) for a,b in zip(reference,vector)]
 cosine=math.fsum(a*b for a,b in zip(reference,vector))/(reference_norm*native_norm)
 maximum=max(errors)
 return {'max_abs':maximum,'mean_abs':math.fsum(errors)/384,
  'rms':math.sqrt(math.fsum(e*e for e in errors)/384),'cosine':cosine,
  'reference_norm':reference_norm,'native_norm':native_norm,
  'passed':maximum<=.002 and math.isfinite(cosine) and cosine>=.99999}

def review(root,run,audit):
 evidence={};source=None;objects=None;desktops={}
 def load(artifact,name):
  nonlocal source,objects
  p=root/(artifact+'-'+str(run))/name
  raw=p.read_bytes();d=json.loads(raw)
  evidence[str(p.relative_to(root))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  return d
 def binding(d,required_closure=False):
  nonlocal source,objects
  if not re.fullmatch('[0-9a-f]{40}',d.get('source_commit','')) or d.get('consumer_source')!=audit:
   raise ValueError('Consumer source identity differs')
  if source is None:source=d['source_commit']
  if source!=d['source_commit']:raise ValueError('Mixed producer revisions')
  if required_closure and (d.get('schema')!='crispembed-successor-fresh-consumer-v2'
   or 'runtime_source_closure' not in d):raise ValueError('Prepared consumer source closure absent')
  if 'runtime_source_closure' in d:
   closure=d['runtime_source_closure'];actual=closure.get('objects')
   if (not isinstance(actual,list) or not actual or not all(isinstance(v,str) and v for v in actual)
    or closure.get('consumer_source')!=audit):raise ValueError('Production source closure absent or malformed')
   if objects is None:objects=actual
   if actual!=objects:raise ValueError('Production source objects differ')
 def parity(v):
  return (v.get('passed') is True and type(v.get('max_abs')) in (float,int)
   and type(v.get('cosine')) in (float,int) and math.isfinite(v['max_abs'])
   and math.isfinite(v['cosine']) and 0<=v['max_abs']<=.002 and v['cosine']>=.99999)
 for platform in DESKTOPS:
  d=load('successor-consumer-'+platform,'consumer.json');binding(d,True);desktops[platform]=d
  desktop_vector(d,platform)
  expected={'cli_to_cabi','offline_cli','python_to_cabi','compiled_C_to_cabi'}
  if (d.get('desktop_bounded_consumer_positive') is not True or set(d.get('comparisons',{}))!=expected
   or not all(parity(v) for v in d['comparisons'].values())
   or not all(d.get(k) is True for k in ('empty_cache_download','empty_cache_offline_rejected','offline_reuse'))
   or len(d.get('vector',[]))!=384 or d.get('compiled_hparams_size',0)<32
   or not {'crispembed_init','crispembed_encode','crispembed_free','crispembed_resolve_model','crispembed_set_offline'}<=set(d.get('abi_exports',[]))
   or not all(re.fullmatch('[0-9a-f]{64}',d.get(k,'')) for k in ('library_sha256','cli_binary_sha256','compiled_c_consumer_sha256','model_sha256'))):
   raise ValueError('Incomplete desktop checks: '+platform)
 desktop_comparisons={}
 reference=desktops['linux-x86_64']['vector']
 for platform in DESKTOPS:
  measured=compare_desktop_vectors(reference,desktops[platform]['vector'])
  desktop_comparisons[platform]=measured
  if not measured['passed']:
   raise ValueError('Cross-platform desktop vector parity failed: '+platform+' '+repr(measured))
 for platform in MOBILES:
  d=load('successor-mobile-'+platform,'consumer.json');binding(d,True)
  if d.get('mobile_abi')!=platform or d.get('cross_link_passed') is not True or d.get('cross_link_exit_code')!=0 or d.get('device_execution_performed') is not False:
   raise ValueError('Mobile link/device scope differs: '+platform)
 d=load('successor-wasm-consumer','browser.json');binding(d)
 if (d.get('page_errors')!=[] or d.get('ocr',{}).get('authored_black_blob_cleared_rgb_exact') is not True
  or len(d.get('embed',{}).get('vector',[]))!=384 or not parity(dict(d,passed=True))):
  raise ValueError('Browser consumer failed')
 candidate=load('successor-apple-released','apple-candidate/consumer.json');binding(candidate,True)
 mac=desktops['macos-arm64']
 if (candidate.get('payload_artifact')!=mac.get('payload_artifact') or candidate.get('archives')!=mac.get('archives')
  or not candidate.get('payload_artifact') or not candidate.get('archives')):
  raise ValueError('Apple probe and desktop used different candidate archive identities')
 d=load('successor-apple-released','apple-released/comparison.json');binding(d)
 if (d.get('candidate_unset_one_exact') is not True or d.get('legacy_zero_preserved') is not True
  or set(d.get('comparisons',{}))!={'candidate-unset','candidate-one'}):raise ValueError('Apple compatibility incomplete')
 arms={}
 for name,gate in (('released',None),('candidate-unset',None),('candidate-one','1'),('candidate-zero','0')):
  item=load('successor-apple-released','apple-released/'+name+'.json');arms[name]=item
  expected_source='78493a31fc3043f3af4ab6fd2f197a32da6f07f0' if name=='released' else candidate['source_commit']
  if (item.get('completed') is not True or len(item.get('cases',[]))!=50
   or item.get('library_source')!=expected_source or item.get('trained_forward_env')!=gate
   or not re.fullmatch('[0-9a-f]{64}',item.get('library_sha256',''))):
   raise ValueError('Apple raw case/source/gate/library coverage differs: '+name)
  if name!='released' and item['library_sha256']!=mac['library_sha256']:
   raise ValueError('Apple raw arm does not use the verified candidate library: '+name)
 for name in ('candidate-unset','candidate-one'):
  measured=compare_release_records(arms['released'],arms[name])
  if d['comparisons'][name]!=measured:raise ValueError('Apple comparison summary differs from retained raw arms')
 # Legacy0 remains a quality-negative diagnostic; require the same retained
 # model/corpus and actual case payloads without matching its decoded outputs.
 legacy=arms['candidate-zero'];released=arms['released']
 if any(legacy.get(k)!=released[k] for k in ('model_sha256','manifest_sha256','threads')):
  raise ValueError('Apple legacy diagnostic model/input profile differs')
 for old,new in zip(released['cases'],legacy['cases']):
  if (any(new.get(k)!=old[k] for k in ('id','image_sha256')) or not isinstance(new.get('text'),str)
   or not isinstance(new.get('token_ids'),list) or not new['token_ids']
   or any(type(v) is not int or v<0 for v in new['token_ids'])
   or new.get('stop')!=('eos' if new['token_ids'][-1]==2 else 'length')
   or not isinstance(new.get('confidence'),list)
   or any(type(v) not in (float,int) or not math.isfinite(v) or not 0<=v<=1 for v in new['confidence'])
   or type(new.get('mean_confidence')) not in (float,int) or not math.isfinite(new['mean_confidence'])
   or not 0<=new['mean_confidence']<=1):
   raise ValueError('Apple legacy diagnostic case payload differs')
 keys=('id','image_sha256','text','token_ids','stop','confidence','mean_confidence')
 if any(any(a[k]!=b[k] for k in keys) for a,b in zip(arms['candidate-unset']['cases'],arms['candidate-one']['cases'])):
  raise ValueError('Apple unset/literal1 raw outputs differ')
 d=load('successor-prebuilt-bindings','successor-bindings.json')
 if d.get('audit_source')!=audit or d.get('passed') is not True or set(d.get('surfaces',{}))!={'rust','dart'}:
  raise ValueError('Prebuilt binding coverage failed')
 binding(d['producer'],True)
 if d.get('native_build_fallback') is not False or d.get('offline_cache_unchanged') is not True:
  raise ValueError('Binding fallback/cache changed')
 for v in d['surfaces'].values():
  if v.get('passed') is not True or len(v.get('comparisons',[]))!=12 or not all(parity(x) for x in v['comparisons']):
   raise ValueError('Binding vector comparison failed')
 return {'schema':'crispembed-successor-bounded-consumer-review-v1','producer_source':source,'audit_source':audit,
  'desktop_platforms':list(DESKTOPS),'mobile_cross_link_platforms':list(MOBILES),'small_reports':evidence,
  'desktop_reference_platform':'linux-x86_64','desktop_model_sha256':MINILM_MODEL_SHA,'desktop_text':MINILM_TEXT,
  'desktop_vector_comparisons':desktop_comparisons,
  'desktop_browser_apple50_and_linux_bindings_positive':True,'mobile_cross_links_positive':True,
  'mobile_device_execution_verified':False,'model_portfolio_qualified':False,'release_published':False,
  'application_pin_changed':False,'limits':['One published MiniLM plus frozen50PosFormer compatibility; no new task-quality qualification.',
   'Mobile archives cross-link; physical-device inference remains unverified.','Optional GPU archives are outside this CPU/mobile candidate.']}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.write_text(json.dumps(review(a.root,os.environ['GITHUB_RUN_ID'],os.environ['GITHUB_SHA']),indent=2,allow_nan=False)+'\n')
