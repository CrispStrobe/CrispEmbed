#!/usr/bin/env python3
"""Hosted-only frozen packaged MiniLM WASM/native source attribution."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from successor_wasm_trace_contract import (PRODUCER,INITIAL_RUN,INITIAL_SOURCE,MODEL_SHA,
 NATIVE_REPORT_SHA,FIXTURE_ID,FIXTURE_BYTES,FIXTURE_SHA,STAGES,TEXT,digest,f32_bytes,
 floats,metrics,tokens,gguf,stage_rows,overlay)
from successor_assets import prepare,artifact,api,sha,runtime_closure


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def command(args,out,name,cwd=ROOT,env=None,timeout=1800):
 with (out/(name+'.log')).open('w',encoding='utf-8') as log:
  result=subprocess.run([str(x) for x in args],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 if result.returncode:raise ValueError(name+' failed with '+str(result.returncode))

def hosted():
 if os.environ.get('GITHUB_ACTIONS')!='true':raise ValueError('Hosted execution only')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=os.environ['GITHUB_SHA']:
  raise ValueError('Checkout/service source mismatch')
 runtime_closure(PRODUCER)

def setup(out):
 hosted();out.mkdir(exist_ok=False)
 service=api('actions/runs/'+str(INITIAL_RUN))
 if service['head_sha']!=INITIAL_SOURCE or service['path']!='.github/workflows/ovisocr2-metadata.yml' or service['status']!='completed' or service['conclusion']!='failure':
  raise ValueError('Frozen initial consumer service identity differs')
 jobs=api('actions/runs/'+str(INITIAL_RUN)+'/jobs?per_page=100')
 if jobs['total_count']>100 or len([j for j in jobs['jobs'] if j['name'].endswith('desktop (ubuntu-24.04, linux-x86_64)') and j['conclusion']=='success'])!=1:
  raise ValueError('Successful initial Linux consumer job absent')
 native=prepare(40339796226,'native','linux-x86_64',out/'native')
 wasm=prepare(40339800364,'wasm','crispembed-wasm-release',out/'browser')
 if native['source_commit']!=PRODUCER or wasm['source_commit']!=PRODUCER:raise ValueError('Producer source changed')
 initial=json.loads((ROOT/'tests/results/packaging/successor-wasm-initial.json').read_text())
 expected=initial['reports']['consumer.json']['report']
 for key in ('manifest_artifact','payload_artifact','archives'):
  if wasm[key]!=expected[key]:raise ValueError('Frozen WASM artifact identity changed')
 identity=artifact(INITIAL_RUN,'successor-minilm-fixture-'+str(INITIAL_RUN),out/'fixture.zip',20*1024**2)
 if (identity['id'],identity['size_in_bytes'],identity['digest'])!=(FIXTURE_ID,FIXTURE_BYTES,'sha256:'+FIXTURE_SHA):raise ValueError('Fixture artifact identity changed')
 with zipfile.ZipFile(out/'fixture.zip') as z:
  if len(z.namelist())!=2 or len(set(z.namelist()))!=2:raise ValueError('Fixture members differ')
  reports=[n for n in z.namelist() if n=='consumer.json'];models=[n for n in z.namelist() if n.endswith('/all-MiniLM-L6-v2-iq4_xs.gguf')]
  if len(reports)!=1 or len(models)!=1:raise ValueError('Fixture model/report paths differ')
  b=z.read(reports[0]);model=z.read(models[0])
  if digest(b)!=NATIVE_REPORT_SHA or digest(model)!=MODEL_SHA:raise ValueError('Fixture report/model content differs')
 anchor=json.loads(b)
 if anchor['source_commit']!=PRODUCER or anchor['consumer_source']!=INITIAL_SOURCE or anchor['text']!=TEXT or anchor['model_sha256']!=MODEL_SHA:raise ValueError('Native anchor source/model/text differs')
 library=out/'native/crispembed-linux-x86_64/libcrispembed.so'
 if sha(library)!=anchor['library_sha256']:raise ValueError('Initial packaged library differs')
 (out/'browser/model.gguf').write_bytes(model);(out/'native-anchor.json').write_bytes(b)
 meta,inventory=gguf(model)
 pool=meta.get('bert.pooling_method')
 if pool is None:pool=max(0,meta.get('bert.pooling_type',0)-1)
 if pool!=0:raise ValueError('Frozen mean-pooling blueprint differs')
 qweights=[r for r in inventory if r['name'] in ('enc.0.attn.q.weight','blk.0.attn_q.weight')]
 if len(qweights)!=1:raise ValueError('Actual first query weight mapping ambiguous')
 q=qweights[0];operator={'weight_name':q['name'],'stored_type':q['type'],'shape':q['shape'],
  'scope':'First output-channel dot with actual first-token emb_ln_out input; no end-to-end fix/cause claim.'}
 if q['type']==20 and q['shape']==[384,384]:
  row=model[q['data_offset']:q['data_offset']+(384//32)*18]
  operator.update(enabled=True,row_sha256=digest(row),row_bytes=len(row))
  (out/'browser/operator-weight.bin').write_bytes(row)
 else:operator.update(enabled=False,reason='Actual first query weight is not the specifically mapped IQ4_NL[384,384]/Q8_0 control')
 save(out/'browser/operator-weight.json',operator)
 save(out/'model-inventory.json',{'model_sha256':MODEL_SHA,'architecture':meta.get('general.architecture'),
  'pooling_type':meta.get('bert.pooling_type'),'pooling_method':meta.get('bert.pooling_method'),'effective_pool_method':pool,'tensors':inventory,
  'limits':['Stored types are read from each tensor, not inferred from the model filename.']})
 overlay_source=out/'overlay'
 # Source copy is a disposable test workspace; production checkout and archives stay unchanged.
 subprocess.run(['git','worktree','add','--detach',str(overlay_source),PRODUCER],cwd=ROOT,check=True,timeout=180)
 subprocess.run(['git','submodule','update','--init','--depth','1','ggml'],cwd=overlay_source,check=True,timeout=180)
 changes=overlay(overlay_source)
 save(out/'overlay.json',{'schema':'crispembed-test-only-source-overlay-v1','producer_source':PRODUCER,'changes':changes,
  'diff':subprocess.check_output(['git','diff','--','src/crispembed.cpp','wasm/embed_wrapper.c','build-embed-wasm.sh'],cwd=overlay_source,text=True),
  'purpose':'Read actual mean/final pooling buffers and expose a test-only environment setter; numerical statements unchanged.'})
 save(out/'trace.json',{'schema':'crispembed-successor-wasm-source-trace-v1','source':os.environ['GITHUB_SHA'],
  'producer_source':PRODUCER,'initial_run':INITIAL_RUN,'initial_source':INITIAL_SOURCE,'fixture':identity,
  'native_anchor_sha256':NATIVE_REPORT_SHA,'model_sha256':MODEL_SHA,'state':'prepared','qualification':False,
  'sources':{str(p):sha(ROOT/p) for p in ('tools/ci-heavy/successor_wasm_trace.py','tools/successor_wasm_trace_contract.py','tests/test_successor_wasm_trace.cjs','tests/test_successor_wasm_trace_diff.cpp','.github/workflows/successor-wasm-trace.yml')},
  'packaged_native':native,'packaged_wasm':wasm})


def build(out):
 hosted();source=out/'overlay'
 command(['bash','build-embed-wasm.sh','--clean','--no-simd','--','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],out,'build-wasm',source,timeout=2400)
 shutil.copy2(source/'build-embed-wasm/compile_commands.json',out/'compile-commands.json')
 destination=out/'browser/diagnostic';destination.mkdir()
 for suffix in ('js','wasm'):shutil.copy2(source/('build-embed-wasm/crispembed_embed.'+suffix),destination)
 command(['c++','-std=c++17','-O2','-I'+str(ROOT/'src'),ROOT/'tests/test_successor_wasm_trace_diff.cpp','-o',out/'trace-diff'],out,'build-diff',timeout=120)


def worker(out,library,threads,mode,name):
 # Separate processes prevent environment/context reuse from making neutrality circular.
 model=out/'browser/model.gguf'
 if sha(model)!=MODEL_SHA:raise ValueError('Worker model SHA differs')
 for key in ('CRISPEMBED_DUMP_LAYERS','CRISPEMBED_DUMP_LAYERS_GGUF','CRISPEMBED_TRACE_POOL'):os.environ.pop(key,None)
 os.environ['CRISPEMBED_DEBUG_TOKENS']='1';os.environ['CRISPEMBED_EMBED_RAW_NORM']='1'
 if mode=='dump':
  os.environ['CRISPEMBED_DUMP_LAYERS']='1';os.environ['CRISPEMBED_DUMP_LAYERS_GGUF']=str(out/(name+'.gguf'))
  os.environ['CRISPEMBED_TRACE_POOL']=str(out/name)
 lib=C.CDLL(str(library));lib.crispembed_init.argtypes=[C.c_char_p,C.c_int];lib.crispembed_init.restype=C.c_void_p
 lib.crispembed_encode.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_int)];lib.crispembed_encode.restype=C.POINTER(C.c_float)
 lib.crispembed_free.argtypes=[C.c_void_p];lib.crispembed_free.restype=None
 ctx=lib.crispembed_init(os.fsencode(model),threads)
 if not ctx:raise ValueError('Native init failed')
 try:
  n=C.c_int();p=lib.crispembed_encode(ctx,TEXT.encode(),C.byref(n))
  if not p or n.value!=384:raise ValueError('Native output shape differs')
  raw=C.string_at(p,n.value*4);floats(raw);(out/(name+'.f32')).write_bytes(raw)
 finally:lib.crispembed_free(ctx)
 sys.stdout.flush();sys.stderr.flush();os._exit(0)


def native_runs(out):
 hosted()
 for kind,library in [('packaged',out/'native/crispembed-linux-x86_64/libcrispembed.so')]:
  for threads in (1,2):
   for mode in ('ordinary','dump'):
    name=f'{kind}-{threads}-{mode}'
    command([sys.executable,__file__,'worker','--output',out,'--library',library,'--threads',threads,'--mode',mode,'--name',name],out,name,timeout=180)


def analyze(out):
 hosted();report=json.loads((out/'trace.json').read_text());report['state']='analyzing';save(out/'trace.json',report)
 anchor=json.loads((out/'native-anchor.json').read_text());native_anchor=f32_bytes(anchor['vector'])
 initial=json.loads((ROOT/'tests/results/packaging/successor-wasm-initial.json').read_text())
 browser_anchor=f32_bytes(initial['reports']['browser.json']['report']['embed']['vector'])
 browser=json.loads((out/'browser/browser-trace.json').read_text())
 if browser['page_errors']:raise ValueError('Browser page error')
 report['neutrality']={};all_tokens=[]
 def neutral(name,actual,expected):
  result={'sha256':digest(actual),'expected_sha256':digest(expected),'bit_exact':actual==expected}
  report['neutrality'][name]=result;save(out/'trace.json',report)
  if not result['bit_exact']:raise ValueError('Ordinary/observer anchor changed: '+name)
 for kind in ('packaged',):
  for threads in (1,2):
   for mode in ('ordinary','dump'):
    name=f'{kind}-{threads}-{mode}';neutral(name,(out/(name+'.f32')).read_bytes(),native_anchor)
    all_tokens.append(tokens((out/(name+'.log')).read_text()))
 for mode in ('packaged','diagnostic-ordinary','diagnostic-dump'):
  neutral('wasm-'+mode,f32_bytes(browser['modes'][mode]['vector']),browser_anchor)
  if mode!='packaged':all_tokens.append(tokens('\n'.join(browser['modes'][mode]['logs'])))
 if any(ids!=all_tokens[0] for ids in all_tokens):raise ValueError('Actual native/browser token IDs differ')
 report['token_ids']=all_tokens[0];count=len(all_tokens[0]);report['stages']={}
 if count>=512 or all_tokens[0][0]!=101 or all_tokens[0][-1]!=102:raise ValueError('Unpadded WordPiece CLS/SEP blueprint differs')
 wasm=stage_rows((out/'browser/wasm-stages.gguf').read_bytes(),count)
 for kind in ('packaged',):
  for threads in (1,2):
   name=f'{kind}-{threads}-dump';native=stage_rows((out/(name+'.gguf')).read_bytes(),count)
   command([out/'trace-diff',out/(name+'.gguf'),out/'browser/wasm-stages.gguf'],out,name+'-diff',timeout=60)
   cpp=[json.loads(line) for line in (out/(name+'-diff.log')).read_text().splitlines() if line.startswith('{')]
   if [row['stage'] for row in cpp]!=list(STAGES):raise ValueError('crispembed-diff stage coverage differs')
   rows=[]
   for stage,cr in zip(STAGES,cpp):
    nr,wr=native[stage],wasm[stage];m=metrics(nr['values'],wr['values'])
    if abs(cr['max_abs']-m['max_abs'])>max(1e-8,m['max_abs']*1e-5) or abs(cr['cos_global']-m['cosine'])>2e-6:raise ValueError('Independent stage metric disagreement')
    rows.append({'stage':stage,'shape':nr['shape'],'native_sha256':nr['sha256'],'wasm_sha256':wr['sha256'],'independent':m,'crispembed_diff':cr,
                 'unchanged_gate_pass':m['max_abs']<=.002 and cr['cos_min']>=.99999})
   report['stages'][name]=rows
  save(out/'trace.json',report)
 report['pooling']={}
 neutral('wasm-pool-final',(out/'browser/wasm-pool.final.f32').read_bytes(),browser_anchor)
 # The frozen single-sentence public C API path uses WordPieceTokenizer::encode,
 # with no padding/truncation for this <512-token sentence. The tokenizer initializes
 # attn_mask to one for every returned ID (source-bound tokenizer.cpp).
 # Native mean is recomputed, not captured; both actual final outputs remain anchors.
 def roundf(x):return struct.unpack('<f',struct.pack('<f',x))[0]
 def mean_and_l2(encoder):
  import math
  derived=[0.]*384
  for t in range(count):
   for h in range(384):derived[h]=roundf(derived[h]+encoder[t*384+h])
  derived=[roundf(x/count) for x in derived]
  norm=0.
  for x in derived:norm=roundf(norm+roundf(x*x))
  norm=roundf(math.sqrt(max(norm,roundf(1e-12))))
  return derived,[roundf(x/norm) for x in derived],norm
 wm=floats((out/'browser/wasm-pool.mean.f32').read_bytes())
 if len(wm)!=384:raise ValueError('Actual WASM pooling capture shape differs')
 wasm_derived,wasm_l2,wasm_norm=mean_and_l2(wasm['encoder_out']['values'])
 if f32_bytes(wasm_derived)!=f32_bytes(wm):raise ValueError('Actual WASM mean differs from source-order reconstruction')
 report['pooling']['mask']={'values':[1]*count,'provenance':'Source-derived from the frozen unpadded single-sentence tokenizer path; not a runtime mask capture.'}
 report['pooling']['wasm']={'mean_provenance':'Actual test-overlay pre-L2 buffer; exact sequential-F32 reconstruction verified',
  'recomputed_raw_norm':wasm_norm,'recomputed_L2_to_actual':metrics(wasm_l2,floats(browser_anchor))}
 for threads in (1,2):
  name=f'packaged-{threads}-dump'
  encoder=stage_rows((out/(name+'.gguf')).read_bytes(),count)['encoder_out']['values']
  nm,nl2,norm=mean_and_l2(encoder)
  (out/(name+'.recomputed-mean.f32')).write_bytes(f32_bytes(nm))
  import re
  raw_norms=re.findall(r'^\[crispembed-rawnorm\] ([0-9.eE+-]+)$',(out/(name+'.log')).read_text(),re.M)
  if len(raw_norms)!=1:raise ValueError('Expected one actual packaged pre-L2 norm diagnostic')
  report['pooling'][str(threads)]={'actual_packaged_raw_norm_six_decimal_log':float(raw_norms[0]),'recomputed_to_logged_raw_norm_abs':abs(norm-float(raw_norms[0])),'native_mean_provenance':'Recomputed sequential-F32 from actual packaged encoder_out and source-derived all-one mask; not captured',
   'native_recomputed_raw_norm':norm,'mean_native_recomputed_to_wasm_captured':metrics(nm,wm),
   'native_recomputed_L2_to_actual':metrics(nl2,floats(native_anchor)),
   'final_actual_to_wasm_actual':metrics(floats(native_anchor),floats(browser_anchor))}
  # Scalar reconstruction may round differently if a native compiler contracts
  # a multiply/add. Record the measured error; anchor equality above is exact.
 save(out/'trace.json',report)
 report['operators']=browser['modes']['diagnostic-dump'].get('operators')
 if not report['operators'] or report['operators'].get('error'):raise ValueError('Test-only operator controls failed: '+str(report['operators']))
 report['final']=metrics(floats(native_anchor),floats(browser_anchor))
 rows=report['stages']['packaged-2-dump']
 report['earliest_nonidentical_stage']=next((r['stage'] for r in rows if r['native_sha256']!=r['wasm_sha256']),None)
 report['earliest_gate_failure']=next((r['stage'] for r in rows if not r['unchanged_gate_pass']),None)
 report['state']='attribution-complete';report['final_gate_pass']=report['final']['max_abs']<=.002 and report['final']['cosine']>=.99999
 report['qualification']=False;report['limits']=['One frozen published MiniLM IQ4_XS-labelled artifact, one owned sentence, CPU native1/2 and published --no-simd WASM build; actual backend compile flags and per-tensor types recorded separately (ggml forces SIMD128).',
  'Internal stages are trusted only after exact initial-output and observer neutrality controls. A completed attribution job does not qualify the failing production output or identify an operator cause.',
  'No production source/archive, precision, output gates, registry or app changes. WASM pooling is captured in a test-only overlay; native pre-L2 pooling is explicitly source-recomputed, not captured. Raw final outputs are actual anchored outputs.']
 report['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in out.rglob('*') if p.is_file() and (p.suffix in ('.log','.f32','.gguf') and 'model.gguf'!=p.name) and 'overlay' not in p.parts and 'native-build' not in p.parts and 'native' not in p.parts}
 save(out/'trace.json',report)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','build','native','analyze','worker'));p.add_argument('--output',type=Path,required=True)
 p.add_argument('--library',type=Path);p.add_argument('--threads',type=int,choices=(1,2));p.add_argument('--mode',choices=('ordinary','dump'));p.add_argument('--name');a=p.parse_args()
 try:
  if a.phase=='worker':worker(a.output,a.library,a.threads,a.mode,a.name)
  else:dict(prepare=setup,build=build,native=native_runs,analyze=analyze)[a.phase](a.output)
 except Exception as e:
  if a.phase!='worker' and a.output.exists():
   path=a.output/'trace.json';receipt=json.loads(path.read_text()) if path.exists() else {'schema':'crispembed-successor-wasm-source-trace-v1','qualification':False}
   receipt.update(state='failed',failed_phase=a.phase,error=str(e));save(path,receipt)
  raise
