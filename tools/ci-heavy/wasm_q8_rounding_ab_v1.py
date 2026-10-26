#!/usr/bin/env python3
"""Fail-closed hosted v1 A/B consumer; never retarget the old890 trace guard."""
import argparse
import ctypes as C
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(Path(__file__).resolve().parent)]
from wasm_q8_rounding_ab_v1 import (BASE,CANDIDATE_GGML,OLD_GGML,TRACE_RUN,TRACE_SOURCE,
 NEUTRALITY,GATES,canonical,panel,reviewed_spec,trusted_report,operator_receipt,source_guard,overlay_ab,
 PRODUCER,MODEL_SHA,STAGES,digest,f32_bytes,floats,metrics,tokens,stage_rows,git)
from successor_wasm_trace_contract import INITIAL_RUN,INITIAL_SOURCE,NATIVE_REPORT_SHA,FIXTURE_ID,FIXTURE_BYTES,FIXTURE_SHA
from successor_assets import api,artifact,RUNTIME_PATHS
from successor_asset_manifest import sha,safe_name,archive_members
SPEC=ROOT/'docs/wasm-q8-rounding-ab-v1.json'


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def spec():return reviewed_spec(json.loads(SPEC.read_text()))

def hosted():
 if os.environ.get('GITHUB_ACTIONS')!='true':raise ValueError('Hosted execution only')
 if git(ROOT,'rev-parse','HEAD')!=os.environ.get('GITHUB_SHA'):raise ValueError('Service/checkout source differs')

def command(args,out,name,cwd=ROOT,timeout=1800):
 with (out/(name+'.log')).open('w') as stream:
  result=subprocess.run([str(x) for x in args],cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
 if result.returncode:raise ValueError(name+' failed: '+str(result.returncode))

def raw_norm(log):
 found=re.findall(r'^\[crispembed-rawnorm\] ([0-9.eE+-]+)$',log,re.M)
 if len(found)!=1 or not math.isfinite(float(found[0])) or float(found[0])<=0:raise ValueError('Expected one actual pre-L2 norm diagnostic')
 return {'six_decimal_log':float(found[0]),'provenance':'Actual runtime diagnostic rounded to six decimals; not a captured vector.'}

def zip_files(path,maximum=128*1024**2):
 with zipfile.ZipFile(path) as z:
  infos=z.infolist()
  if len(infos)>200 or sum(i.file_size for i in infos)>maximum:raise ValueError('ZIP extraction budget exceeded')
  names=[safe_name(i.filename) for i in infos]
  if len(names)!=len(set(names)) or any(i.is_dir() or i.flag_bits&1 for i in infos):raise ValueError('Duplicate/directory/encrypted ZIP member')
  return {name:z.read(i) for name,i in zip(names,infos)}

def service(run,source,path,conclusion='success'):
 item=api('actions/runs/'+str(run))
 if item['head_sha']!=source or item['path']!=path or item['status']!='completed' or item['conclusion']!=conclusion or item['event']!='workflow_dispatch':
  raise ValueError('Service identity/state differs: '+str(run))
 return {k:item[k] for k in ('id','head_sha','path','status','conclusion','event')}

def pinned_artifact(run,name,path,identity):
 actual=artifact(run,name,path,32*1024**2)
 expected=(identity['id'],identity['bytes'],'sha256:'+identity['sha256'])
 if (actual['id'],actual['size_in_bytes'],actual['digest'])!=expected:raise ValueError('Reviewed trace artifact identity differs')
 return actual

def operator_barrier(out):
 run=os.environ['GITHUB_RUN_ID'];current=os.environ['GITHUB_SHA']
 meta=api('actions/runs/'+run)
 if meta['head_sha']!=current or meta['event']!='workflow_dispatch' or meta['path'] not in ('.github/workflows/ovisocr2-metadata.yml','.github/workflows/wasm-q8-rounding-ab-v1.yml'):
  raise ValueError('Current operator parent identity differs')
 jobs=api('actions/runs/'+run+'/jobs?per_page=100')
 selected=[j for j in jobs['jobs'] if j['name']=='operator-controls / operators' or j['name'].endswith(' / operator-controls / operators')]
 if jobs['total_count']>100 or len(selected)!=1 or selected[0]['status']!='completed' or selected[0]['conclusion']!='success':
  raise ValueError('Successful same-source operator job required before model access')
 identity=artifact(run,'wasm-rounding-operator-controls-'+run,out/'operator-controls.zip',4*1024**2)
 files=zip_files(out/'operator-controls.zip',16*1024**2)
 def one(basename):
  matches=[data for name,data in files.items() if Path(name).name==basename]
  if len(matches)!=1:raise ValueError('Missing/ambiguous operator receipt: '+basename)
  return json.loads(matches[0])
 declaration=json.loads((ROOT/'docs/wasm-q8-rounding-candidate.json').read_text())
 proof=operator_receipt(one('source.json'),one('controls.json'),current,declaration)
 proof.update(run=int(run),job=selected[0]['id'],artifact=identity,
              members={name:{'bytes':len(data),'sha256':digest(data)} for name,data in files.items()})
 save(out/'operator-proof.json',proof)
 return proof

def preflight(out):
 hosted();out.mkdir(exist_ok=False);s=spec()
 check=subprocess.run(['git','-C','ggml','cat-file','-e',OLD_GGML+'^{commit}'],cwd=ROOT,stderr=subprocess.DEVNULL)
 if check.returncode:subprocess.run(['git','-C','ggml','fetch','--depth=1','origin',OLD_GGML],cwd=ROOT,check=True,timeout=180)
 guard=source_guard(ROOT,RUNTIME_PATHS)
 operators=operator_barrier(out)
 cases=panel(s);(out/'browser').mkdir();save(out/'browser/panel.json',cases)
 save(out/'ab.json',{'schema':'crispembed-wasm-q8-rounding-ab-result-v1','source':os.environ['GITHUB_SHA'],
  'protocol_sha256':sha(SPEC),'panel_sha256':digest(canonical(cases)),'source_guard':guard,'operator_controls':operators,'state':'preflight-passed','qualification':False})


def package(out,kind,proof):
 run=40339796226 if kind=='native' else 40339800364
 if proof['source_commit']!=PRODUCER or proof['producer_run']!=run or proof['ggml_gitlink']!=OLD_GGML:raise ValueError('Frozen producer proof differs')
 service(run,PRODUCER,'.github/workflows/release.yml' if kind=='native' else '.github/workflows/release-wasm.yml')
 folder=out/('old-native' if kind=='native' else 'browser');folder.mkdir(exist_ok=True)
 manifest_identity=artifact(run,'successor-'+kind+'-manifest',out/(kind+'-manifest.zip'),4*1024**2)
 payload_identity=artifact(run,'linux-x86_64' if kind=='native' else 'crispembed-wasm-release',out/(kind+'-payload.zip'),256*1024**2)
 if manifest_identity!=proof['manifest_artifact'] or payload_identity!=proof['payload_artifact']:raise ValueError('Frozen package service artifact changed')
 members=zip_files(out/(kind+'-manifest.zip'),4*1024**2)
 if list(members)!=['successor-'+kind+'-manifest.json']:raise ValueError('Manifest member differs')
 manifest=json.loads(next(iter(members.values())))
 generator=subprocess.check_output(['git','show',PRODUCER+':tools/successor_asset_manifest.py'],cwd=ROOT)
 if manifest['schema']!='crispembed-successor-archive-manifest-v1' or manifest['kind']!=kind or manifest['source_commit']!=PRODUCER or manifest['ggml_gitlink']!=OLD_GGML or manifest['generator_sha256']!=digest(generator) or manifest['version']!=git(ROOT,'show',PRODUCER+':VERSION'):
  raise ValueError('Immutable producer manifest/source differs')
 selected=[]
 for name,data in zip_files(out/(kind+'-payload.zip'),256*1024**2).items():
  rows=[r for r in manifest['files'] if r['name']==name]
  if len(rows)!=1:raise ValueError('Payload not declared by immutable manifest')
  row=rows[0];target=folder/name;target.write_bytes(data)
  if len(data)!=row['bytes'] or sha(target)!=row['sha256'] or archive_members(target)!=row['members']:raise ValueError('Package/member bytes differ')
  destination=folder if kind=='wasm' else folder/name.removesuffix('.tar.gz').removesuffix('.zip')
  destination.mkdir(exist_ok=True)
  if not name.endswith('.tar.gz'):raise ValueError('Expected immutable Linux/WASM tar package')
  with tarfile.open(target,'r:gz') as package:package.extractall(destination,filter='data')
  selected.append({'name':name,'sha256':row['sha256'],'member_count':len(row['members'])})
 if selected!=proof['archives']:raise ValueError('Frozen archive selection differs')
 return {'manifest_artifact':manifest_identity,'payload_artifact':payload_identity,'archives':selected}


def prepare(out):
 hosted();s=spec();source_guard(ROOT,RUNTIME_PATHS)
 services={'trace':service(TRACE_RUN,TRACE_SOURCE,'.github/workflows/ovisocr2-metadata.yml')}
 review=pinned_artifact(TRACE_RUN,'successor-wasm-trace-review-'+str(TRACE_RUN),out/'trace-review.zip',s['trace']['review_artifact'])
 review_files=zip_files(out/'trace-review.zip',32*1024**2)
 if 'trace.json' not in review_files or digest(review_files['trace.json'])!=s['trace']['review_artifact']['report_sha256']:raise ValueError('Reviewed first trace report bytes differ')
 trace=trusted_report(json.loads(review_files['trace.json']));(out/'trusted-trace.json').write_bytes(review_files['trace.json'])
 raw=pinned_artifact(TRACE_RUN,'successor-wasm-trace-arrays-'+str(TRACE_RUN),out/'trace-raw.zip',s['trace']['raw_artifact'])
 raw_files=zip_files(out/'trace-raw.zip');raw_dir=out/'trusted-trace';raw_dir.mkdir()
 for name,b in raw_files.items():
  identity=trace['artifacts'].get(name)
  if identity:
   if identity!={'bytes':len(b),'sha256':digest(b)}:raise ValueError('First trace raw member differs: '+name)
  elif name not in ('browser/diagnostic/crispembed_embed.js','browser/diagnostic/crispembed_embed.wasm'):
   raise ValueError('Undeclared first trace raw member: '+name)
  target=raw_dir/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 required=['packaged-1-dump.gguf','packaged-2-dump.gguf','browser/wasm-stages.gguf','browser/wasm-pool.mean.f32','browser/wasm-pool.final.f32','browser/diagnostic/crispembed_embed.js','browser/diagnostic/crispembed_embed.wasm']
 if any(name not in raw_files for name in required):raise ValueError('First trace raw coverage incomplete')
 for threads in (1,2):
  native=stage_rows(raw_files[f'packaged-{threads}-dump.gguf'],len(trace['token_ids']))
  wasm=stage_rows(raw_files['browser/wasm-stages.gguf'],len(trace['token_ids']))
  for row in trace['stages'][f'packaged-{threads}-dump']:
   if native[row['stage']]['sha256']!=row['native_sha256'] or wasm[row['stage']]['sha256']!=row['wasm_sha256']:raise ValueError('First trace stage arrays/report differ')
 packages={kind:package(out,kind,trace['packaged_'+kind]) for kind in ('native','wasm')}
 service(INITIAL_RUN,INITIAL_SOURCE,'.github/workflows/ovisocr2-metadata.yml','failure')
 jobs=api('actions/runs/'+str(INITIAL_RUN)+'/jobs?per_page=100')
 if jobs['total_count']>100 or len([j for j in jobs['jobs'] if j['name'].endswith('desktop (ubuntu-24.04, linux-x86_64)') and j['conclusion']=='success'])!=1:raise ValueError('Successful initial Linux desktop absent')
 fixture=artifact(INITIAL_RUN,'successor-minilm-fixture-'+str(INITIAL_RUN),out/'fixture.zip',20*1024**2)
 if (fixture['id'],fixture['size_in_bytes'],fixture['digest'])!=(FIXTURE_ID,FIXTURE_BYTES,'sha256:'+FIXTURE_SHA):raise ValueError('Initial fixture artifact differs')
 files=zip_files(out/'fixture.zip',32*1024**2)
 models=[n for n in files if n.endswith('/all-MiniLM-L6-v2-iq4_xs.gguf')]
 if len(files)!=2 or 'consumer.json' not in files or len(models)!=1 or digest(files['consumer.json'])!=NATIVE_REPORT_SHA or digest(files[models[0]])!=MODEL_SHA:raise ValueError('Initial model/native anchor bytes differ')
 anchor=json.loads(files['consumer.json'])
 if anchor['source_commit']!=PRODUCER or anchor['consumer_source']!=INITIAL_SOURCE or anchor['model_sha256']!=MODEL_SHA or sha(out/'old-native/crispembed-linux-x86_64/libcrispembed.so')!=anchor['library_sha256']:raise ValueError('Initial native library/model proof differs')
 (out/'native-anchor.json').write_bytes(files['consumer.json']);(out/'browser/model.gguf').write_bytes(files[models[0]])
 destination=out/'browser/old-diagnostic';destination.mkdir()
 for suffix in ('js','wasm'):shutil.copy2(raw_dir/('browser/diagnostic/crispembed_embed.'+suffix),destination)
 save(out/'prepared.json',{'trace_service':services['trace'],'review_artifact':review,'raw_artifact':raw,'packages':packages,'fixture':fixture,
  'raw_member_sha256':{n:digest(b) for n,b in raw_files.items()},'model_sha256':MODEL_SHA})


def worker(out,case_id,threads):
 cases=panel(json.loads(SPEC.read_text()));case=[c for c in cases if c['id']==case_id][0]
 hosted()
 if sha(out/'browser/model.gguf')!=MODEL_SHA:raise ValueError('Native worker model differs')
 for key in ('CRISPEMBED_WASM_Q8_0_ROUND','CRISPEMBED_DUMP_LAYERS','CRISPEMBED_DUMP_LAYERS_GGUF','CRISPEMBED_TRACE_POOL'):os.environ.pop(key,None)
 os.environ['CRISPEMBED_DEBUG_TOKENS']='1';os.environ['CRISPEMBED_EMBED_RAW_NORM']='1'
 lib=C.CDLL(str(out/'old-native/crispembed-linux-x86_64/libcrispembed.so'))
 lib.crispembed_init.argtypes=[C.c_char_p,C.c_int];lib.crispembed_init.restype=C.c_void_p
 lib.crispembed_encode.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_int)];lib.crispembed_encode.restype=C.POINTER(C.c_float)
 lib.crispembed_free.argtypes=[C.c_void_p];lib.crispembed_free.restype=None
 ctx=lib.crispembed_init(os.fsencode(out/'browser/model.gguf'),threads)
 if not ctx:raise ValueError('Native worker init failed')
 try:
  n=C.c_int();pointer=lib.crispembed_encode(ctx,case['text'].encode(),C.byref(n))
  if not pointer or n.value!=384:raise ValueError('Native worker shape differs')
  raw=C.string_at(pointer,1536);vector=floats(raw)
  name=case_id+'-'+str(threads);(out/'native-vectors'/(name+'.f32')).write_bytes(raw)
  save(out/'native-vectors'/(name+'.json'),{'vector':vector,'sha256':digest(raw)})
 finally:lib.crispembed_free(ctx)
 sys.stdout.flush();sys.stderr.flush();os._exit(0)


def native_reference(out):
 hosted();spec();(out/'native-vectors').mkdir()
 for case in panel(json.loads(SPEC.read_text())):
  for threads in (1,2):
   command([sys.executable,__file__,'worker','--output',out,'--case',case['id'],'--threads',threads],out,'native-'+case['id']+'-'+str(threads),timeout=180)


def freeze_old(out):
 hosted();s=spec();cases=panel(s);browser=json.loads((out/'browser/reference-browser.json').read_text())
 if browser['state']!='complete' or browser['page_errors'] or set(browser['cases'])!={c['id'] for c in cases}:raise ValueError('Old browser panel incomplete')
 trace=trusted_report(json.loads((out/'trusted-trace.json').read_text()))
 initial=json.loads((ROOT/'tests/results/packaging/successor-wasm-initial.json').read_text())
 anchor=json.loads((out/'native-anchor.json').read_text())
 freeze={'schema':'crispembed-wasm-q8-old-panel-freeze-v1','state':'old-panel-frozen','created_utc':datetime.now(timezone.utc).isoformat(),
  'panel_sha256':digest(canonical(cases)),'trace_report_sha256':sha(out/'trusted-trace.json'),'model_sha256':MODEL_SHA,'cases':{},'source':os.environ['GITHUB_SHA']}
 for case in cases:
  row=browser['cases'][case['id']];old=f32_bytes(row['packaged']['vector']);diagnostic=f32_bytes(row['diagnostic']['vector'])
  if old!=diagnostic:raise ValueError('Old packaged/diagnostic neutrality failed: '+case['id'])
  ids=tokens('\n'.join(row['diagnostic']['logs']));native={}
  if not 2<=len(ids)<=512 or ids[0]!=101 or ids[-1]!=102 or case['expected_token_count'] is not None and len(ids)!=case['expected_token_count']:raise ValueError('Old panel token contract failed')
  for threads in (1,2):
   name=case['id']+'-'+str(threads);value=json.loads((out/'native-vectors'/(name+'.json')).read_text());log=(out/('native-'+name+'.log')).read_text()
   if tokens(log)!=ids or digest(f32_bytes(value['vector']))!=value['sha256']:raise ValueError('Old native token/vector identity failed')
   native[str(threads)]={**value,'raw_norm':raw_norm(log)}
  if case['id']=='original':
   if ids!=trace['token_ids'] or old!=f32_bytes(initial['reports']['browser.json']['report']['embed']['vector']) or any(f32_bytes(v['vector'])!=f32_bytes(anchor['vector']) for v in native.values()):raise ValueError('Original anchors changed while freezing supplement')
   if f32_bytes(row['dump']['vector'])!=old or tokens('\n'.join(row['dump']['logs']))!=ids:raise ValueError('Old full-dump neutrality failed')
   observed=stage_rows((out/'browser/arrays/old-dump-original-stages.gguf').read_bytes(),len(ids))
   reference=stage_rows((out/'trusted-trace/browser/wasm-stages.gguf').read_bytes(),len(ids))
   if any(observed[n]['sha256']!=reference[n]['sha256'] for n in STAGES):raise ValueError('Old full-stage anchors changed')
   for suffix in ('mean','final'):
    if (out/('browser/arrays/old-dump-original-pool.'+suffix+'.f32')).read_bytes()!=(out/('trusted-trace/browser/wasm-pool.'+suffix+'.f32')).read_bytes():raise ValueError('Old actual pooling anchors changed')
  freeze['cases'][case['id']]={'text_sha256':case['utf8_sha256'],'token_ids':ids,'native':native,
   'old_packaged_browser_vector':row['packaged']['vector'],'old_browser_sha256':digest(old),'diagnostic_raw_norm':raw_norm('\n'.join(row['diagnostic']['logs'])),
   'old_packaged_diagnostic_neutral':True,'native_thread_comparison':metrics(native['1']['vector'],native['2']['vector']),
   'old_browser_to_native':{t:metrics(v['vector'],row['packaged']['vector']) for t,v in native.items()}}
 body=canonical(freeze)+b'\n'
 for path in (out/'old-panel-freeze.json',out/'browser/old-panel-freeze.json'):
  with path.open('xb') as stream:stream.write(body)
 (out/'browser/old-panel-freeze.sha256').write_text(digest(body)+'\n')


def build(out):
 hosted();spec();source_guard(ROOT,RUNTIME_PATHS)
 # The reference panel must already exist before even preparing candidate execution.
 freeze=json.loads((out/'old-panel-freeze.json').read_text())
 if freeze['state']!='old-panel-frozen' or sha(out/'old-panel-freeze.json')!=(out/'browser/old-panel-freeze.sha256').read_text().strip():raise ValueError('Frozen old panel absent/changed')
 source=out/'candidate-overlay';subprocess.run(['git','worktree','add','--detach',str(source),os.environ['GITHUB_SHA']],cwd=ROOT,check=True,timeout=180)
 subprocess.run(['git','submodule','update','--init','--depth','1','ggml'],cwd=source,check=True,timeout=180)
 changes=overlay_ab(source)
 save(out/'candidate-overlay.json',{'source':os.environ['GITHUB_SHA'],'ggml':CANDIDATE_GGML,'changes':changes,
  'diff':git(source,'diff','--','src/crispembed.cpp','wasm/embed_wrapper.c','build-embed-wasm.sh'),
  'limits':['Test-only environment and actual pooling capture; global trait quantizer probes are not used as evidence for selected per-backend conversion.']})
 command(['bash','build-embed-wasm.sh','--clean','--no-simd','--','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],out,'build-candidate',source,timeout=2400)
 destination=out/'browser/candidate';destination.mkdir()
 for suffix in ('js','wasm'):shutil.copy2(source/('build-embed-wasm/crispembed_embed.'+suffix),destination)
 shutil.copy2(source/'build-embed-wasm/compile_commands.json',out/'candidate-compile-commands.json')
 command(['c++','-std=c++17','-O2','-I'+str(ROOT/'src'),ROOT/'tests/test_successor_wasm_trace_diff.cpp','-o',out/'stage-diff'],out,'build-stage-diff',timeout=120)


def compare(out):
 hosted();s=spec();freeze_bytes=(out/'old-panel-freeze.json').read_bytes();freeze=json.loads(freeze_bytes)
 candidate=json.loads((out/'browser/candidate-browser.json').read_text());cases=panel(s)
 if candidate['state']!='complete' or candidate['page_errors'] or candidate['old_panel_freeze_sha256']!=digest(freeze_bytes) or set(candidate['cases'])!=set(freeze['cases']):raise ValueError('Candidate matrix incomplete/freeze differs')
 report=json.loads((out/'ab.json').read_text());report.update(state='comparing',old_panel_freeze_sha256=digest(freeze_bytes),cases={},failures=[],qualification=False)
 def control(ok,label):
  if not ok:report['failures'].append(label)
 for case in cases:
  key=case['id'];old=freeze['cases'][key];rows=candidate['cases'][key]
  if set(rows)!=set(GATES):raise ValueError('Candidate gate coverage differs')
  report['cases'][key]={}
  for gate in GATES:
   if set(rows[gate])!={'ordinary','dump'}:raise ValueError('Candidate observer coverage differs')
   ordinary,dump=rows[gate]['ordinary'],rows[gate]['dump']
   for mode,value in rows[gate].items():
    control(tokens('\n'.join(value['logs']))==old['token_ids'],key+'/'+gate+'/'+mode+' tokens')
   vector=ordinary['vector'];raw=f32_bytes(vector)
   control(raw==f32_bytes(dump['vector']),key+'/'+gate+' observer neutrality')
   if gate!='1':control(digest(raw)==old['old_browser_sha256'],key+'/'+gate+' legacy anchor')
   pool=out/('browser/arrays/candidate-'+gate+'-'+key+'-pool.final.f32')
   control(pool.read_bytes()==raw,key+'/'+gate+' actual pool final')
   mean=floats((out/('browser/arrays/candidate-'+gate+'-'+key+'-pool.mean.f32')).read_bytes())
   if len(mean)!=384:raise ValueError('Actual candidate mean shape differs')
   comparisons={t:metrics(n['vector'],vector) for t,n in old['native'].items()}
   passed=all(m['max_abs']<=.002 and m['cosine']>=.99999 for m in comparisons.values())
   if gate=='1':control(passed,key+'/1 final gates')
   report['cases'][key][gate]={'vector':vector,'sha256':digest(raw),'native_comparison':comparisons,'final_gate_pass':passed,
    'legacy_bit_exact':digest(raw)==old['old_browser_sha256'],'ordinary_dump_bit_exact':raw==f32_bytes(dump['vector']),
    'ordinary_raw_norm':raw_norm('\n'.join(ordinary['logs'])),'dump_raw_norm':raw_norm('\n'.join(dump['logs'])),
    'actual_wasm_pre_L2_mean_l2':math.sqrt(sum(x*x for x in mean)),
    'pre_L2_provenance':'Actual test-overlay WASM buffer. Native panel exposes only six-decimal raw norm diagnostics; no captured native pre-L2 vector claim.'}
  save(out/'ab.json',report)
 report['first_sentence_stages']={};report['first_sentence_pooling']={};count=len(freeze['cases']['original']['token_ids'])
 old_wasm=stage_rows((out/'trusted-trace/browser/wasm-stages.gguf').read_bytes(),count)
 for gate in GATES:
  path=out/('browser/arrays/candidate-'+gate+'-original-stages.gguf');candidate_stages=stage_rows(path.read_bytes(),count)
  report['first_sentence_stages'][gate]={};report['first_sentence_pooling'][gate]={}
  for threads in (1,2):
   native_path=out/f'trusted-trace/packaged-{threads}-dump.gguf';native=stage_rows(native_path.read_bytes(),count)
   name='stage-'+gate+'-'+str(threads);command([out/'stage-diff',native_path,path],out,name,timeout=60)
   cpp=[json.loads(line) for line in (out/(name+'.log')).read_text().splitlines() if line.startswith('{')]
   if [r['stage'] for r in cpp]!=list(STAGES):raise ValueError('crispembed-diff coverage differs')
   result=[]
   for stage,cr in zip(STAGES,cpp):
    mine=candidate_stages[stage];ref=native[stage];m=metrics(ref['values'],mine['values'])
    if abs(cr['max_abs']-m['max_abs'])>max(1e-8,m['max_abs']*1e-5) or abs(cr['cos_global']-m['cosine'])>2e-6:raise ValueError('Independent full-stage metrics disagree')
    passed=m['max_abs']<=.002 and cr['cos_min']>=.99999
    if gate=='1':control(passed,'original/1/stage/'+str(threads)+'/'+stage)
    else:control(mine['sha256']==old_wasm[stage]['sha256'],'original/'+gate+'/old-stage/'+stage)
    result.append({'stage':stage,'shape':ref['shape'],'native_sha256':ref['sha256'],'candidate_sha256':mine['sha256'],
     'old_wasm_bit_exact':mine['sha256']==old_wasm[stage]['sha256'],'independent':m,'crispembed_diff':cr,'gate_pass':passed})
   report['first_sentence_stages'][gate][str(threads)]=result
   import struct
   def roundf(x):return struct.unpack('<f',struct.pack('<f',x))[0]
   native_mean=[0.]*384
   for t in range(count):
    for h in range(384):native_mean[h]=roundf(native_mean[h]+native['encoder_out']['values'][t*384+h])
   native_mean=[roundf(x/count) for x in native_mean]
   actual_wasm_mean=floats((out/('browser/arrays/candidate-'+gate+'-original-pool.mean.f32')).read_bytes())
   m=metrics(native_mean,actual_wasm_mean)
   report['first_sentence_pooling'][gate][str(threads)]={'native_provenance':'Sequential-F32 recomputation from frozen actual native encoder_out and source-derived all-one mask; not captured.',
    'wasm_provenance':'Actual test-overlay buffer','comparison':m}
   if gate=='1':control(m['max_abs']<=.002 and m['cosine']>=.99999,'original/1/recomputed-mean/'+str(threads))
  save(out/'ab.json',report)
 interleaving=candidate.get('interleaving',[])
 if [r['label'] for r in interleaving]!=['A0','B1','C0','A0-repeat']:raise ValueError('Interleaving coverage differs')
 for row in interleaving:
  gate='1' if row['label']=='B1' else '0'
  control(f32_bytes(row['vector'])==f32_bytes(report['cases']['original'][gate]['vector']),'interleaving/'+row['label'])
  control(tokens('\n'.join(row['logs']))==freeze['cases']['original']['token_ids'],'interleaving/'+row['label']+' tokens')
 report['interleaving']=interleaving;report['passed']=not report['failures'];report['state']='completed-supported' if report['passed'] else 'completed-rejected'
 report['limits']=['Owned ten-case numerical/tokenization/length panel only; no representative task accuracy, default promotion or asset qualification.',
  'The first trace and original first-sentence anchors stay immutable; this old-native/browser supplementary panel was frozen before candidate inference.',
  'Global Q8_0 trait conversion stays legacy. This lane tests the selected backend through actual graphs, not a trait-probe surrogate.',
  'Full eight-stage comparison covers the original sentence only. Other dump modes test output neutrality and actual pooling buffers; native pre-L2 vectors are not captured.']
 report['artifact_files']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in (out/'browser/arrays').glob('*') if p.is_file()}
 save(out/'ab.json',report)
 if not report['passed']:raise ValueError('Unchanged A/B gates rejected '+str(len(report['failures']))+' controls/stages')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('preflight','prepare','native-reference','freeze-old','build','compare','worker'));p.add_argument('--output',type=Path,required=True)
 p.add_argument('--case');p.add_argument('--threads',type=int,choices=(1,2));a=p.parse_args()
 try:
  if a.phase=='worker':worker(a.output,a.case,a.threads)
  else:{'preflight':preflight,'prepare':prepare,'native-reference':native_reference,'freeze-old':freeze_old,'build':build,'compare':compare}[a.phase](a.output)
 except Exception as e:
  if a.phase!='worker':
   a.output.mkdir(exist_ok=True,parents=True);path=a.output/'ab.json';receipt=json.loads(path.read_text()) if path.exists() else {'schema':'crispembed-wasm-q8-rounding-ab-result-v1','qualification':False}
   receipt.update(failed_phase=a.phase,error=str(e))
   if receipt.get('state')!='completed-rejected':receipt['state']='failed'
   save(path,receipt)
  raise
