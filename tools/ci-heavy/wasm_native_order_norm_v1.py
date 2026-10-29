#!/usr/bin/env python3
"""Closed live-input norm candidate and unchanged ten-output/eight-stage A/B."""
import argparse
import base64
import hashlib
import ctypes as C
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/wasm-native-order-norm-v1'),str(ROOT/'tools'),str(ROOT/'tools/ci-heavy')]
import contract as K
from successor_assets import api,artifact
from successor_wasm_trace_contract import MODEL_SHA,STAGES,stage_rows,tokens,f32_bytes,floats
from wasm_q8_rounding_ab_v1 import panel,overlay_ab,canonical
# Pure bounded archive helpers; no historical guard or preflight is called/retargeted.
spec=importlib.util.spec_from_file_location('old_ab_archive_helpers',ROOT/'tools/ci-heavy/wasm_q8_rounding_ab_v1.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from ci_process import run


def source():
 cfg=K.source(os.environ.get('GITHUB_SHA',''))
 K.require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Genuine GHA execution required')
 meta=api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
 K.require(meta['head_sha']==os.environ['GITHUB_SHA']and meta['status']=='in_progress'and meta['event']=='workflow_dispatch'
           and meta['path']in ['.github/workflows/wasm-native-order-norm-candidate-v1.yml','.github/workflows/ovisocr2-metadata.yml'],
           'Actual service/source/workflow differs')
 return cfg

def command(out,name,argv,cwd=ROOT,timeout=2400,env=None,expected=(0,)):
 log=out/'review'/(name+'.log');result=run(argv,log,cwd=cwd,env=env,timeout=timeout)
 K.write(out/'review'/(name+'-command.json'),dict(result,argv=list(map(str,argv)),log=log.name,**K.identity(log)))
 K.require(not result['timed_out']and result['exit']in expected,'Command rejected: '+name)
 return result['exit']

def fetch(out,key,pin):
 old.service(pin['run'],pin['source'],pin['workflow'],pin['conclusion'])
 path=out/(key+'.zip');actual=artifact(pin['run'],pin['name'],path,pin['bytes'])
 K.require(actual['id']==pin['id']and actual['size_in_bytes']==pin['bytes']and actual['digest']=='sha256:'+pin['sha256']
           and K.sha(path)==pin['sha256'],'Frozen artifact differs: '+key)
 K.write(out/'review'/(key+'-artifact.json'),actual)
 return old.zip_files(path,128*1024**2)

def one(files,name):
 found=[(n,b)for n,b in files.items()if Path(n).name==name]
 K.require(len(found)==1,'Missing/ambiguous authenticated member: '+name)
 return found[0][1]

def preflight(out):
 cfg=source();out.mkdir(exist_ok=False)
 for name in ['review','raw','browser','browser/component-inputs','browser/arrays']:(out/name).mkdir(parents=True,exist_ok=True)
 K.write(out/'review/source.json',dict(source=os.environ['GITHUB_SHA'],base=K.BASE,ggml=K.GGML,
         protocol_sha256=K.sha(K.PROFILE),runtime_delta=cfg['runtime_delta'],qualification=False))
 ram=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
 K.require(ram>=6*2**30 and shutil.disk_usage(out).free>=16*2**30,'Hosted6GiB RAM/16GiB disk guard')
 K.write(out/'review/resources.json',dict(available_ram_bytes=ram,free_disk_bytes=shutil.disk_usage(out).free,workers=2))

def component_prepare(out):
 cfg=source();review=fetch(out,'norm-risk-review',cfg['risk']['review']);raw=fetch(out,'norm-risk-raw',cfg['risk']['raw'])
 body=one(review,'norm-risk.json');K.require(__import__('hashlib').sha256(body).hexdigest()==cfg['risk']['report_sha256'],'Actual norm risk producer differs')
 risk=json.loads(body);K.require(risk['completed']and risk['local_risk_supported']and risk['baseline_byte_anchors']==30
             and risk['native_order_norm_bit_controls']==12 and risk['comparisons']==56 and not risk['qualification'],'Reviewed local risk absent')
 manifest=json.loads(one(review,'raw-manifest.json'))
 prepared=json.loads(one(review,'norm-prepared.json'));builds=json.loads(one(review,'build-artifacts.json'))
 expected=dict(manifest)
 expected.update(builds)
 expected.update({'control.gguf':{k:risk['adapters']['original'][k]for k in ['bytes','sha256']},
                  'replay.gguf':{k:risk['adapters']['native'][k]for k in ['bytes','sha256']}})
 # norm-packets is an auxiliary JSON written by the authenticated old preparer;
 # the complete immutable ZIP SHA authenticates its bytes too.
 K.require(set(manifest)<=set(raw) and set(raw)<=set(expected)|{'browser/norm-packets.json'},'Retained norm inventory/auxiliary coverage differs')
 for n,b in raw.items():
  if n in expected:K.require(expected[n]==dict(bytes=len(b),sha256=__import__('hashlib').sha256(b).hexdigest()),'Retained norm raw differs: '+n)
  else:K.require(__import__('hashlib').sha256(b).hexdigest()==prepared['packets_sha256'],'Old packet auxiliary SHA differs')
 prepared=json.loads(one(review,'norm-prepared.json'));rows=[]
 for name in ['attention_native','ffn_native','attention_candidate','ffn_candidate']:
  slug=name.replace('_','-');input_name='browser/norm-inputs/'+slug+'.f32';target_name='browser/norm-targets/native1-'+slug+'.f32'
  K.require(input_name in raw and target_name in raw and len(raw[input_name])==len(raw[target_name])==18432,'Exact12x384 component coverage required')
  input_path=out/'browser/component-inputs'/(name+'.f32');input_path.write_bytes(raw[input_name])
  target=out/'raw'/(name+'.target.f32');target.write_bytes(raw[target_name])
  rows.append(dict(name=name,input=K.identity(input_path),target=K.identity(target),target_path=str(target.relative_to(out))))
 K.write(out/'browser/component-inputs.json',rows)
 K.write(out/'review/component-prepared.json',dict(rows=rows,old_simd_risk_preserved=True,new_script_profile_recovery_required=True))

def component_build(out):
 source();native=out/'component-native';wasm=out/'component-wasm'
 command(out,'component-native-configure',['cmake','-S',ROOT/'tests/wasm-native-order-norm','-B',native,'-DCMAKE_BUILD_TYPE=Release'])
 command(out,'component-native-build',['cmake','--build',native,'--target','norm-candidate-controls','norm-candidate-ref','--parallel','2'])
 command(out,'component-authored',[native/'norm-candidate-controls'])
 command(out,'component-wasm-configure',['emcmake','cmake','-S',ROOT/'tests/wasm-native-order-norm','-B',wasm,
         '-DCMAKE_BUILD_TYPE=Release','-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm','-DCMAKE_C_FLAGS=-pthread','-DCMAKE_CXX_FLAGS=-pthread'])
 command(out,'component-wasm-build',['cmake','--build',wasm,'--target','norm-candidate-controls','--parallel','2'])
 destination=out/'browser/component';destination.mkdir()
 for suffix in ['js','wasm']:shutil.copyfile(wasm/('norm-candidate-controls.'+suffix),destination/('norm-candidate-controls.'+suffix))
 for row in K.read(out/'browser/component-inputs.json'):
  for threads in [1,2]:command(out,'component-'+row['name']+'-'+str(threads),[native/'norm-candidate-controls',
      out/'browser/component-inputs'/(row['name']+'.f32'),out/'raw'/(row['name']+'.native'+str(threads)+'.f32'),str(threads)])
 for kind,build in [('native',native),('wasm',wasm)]:
  commands=K.read(build/'compile_commands.json')
  own=[r for r in commands if Path(r['file']).name=='wasm_native_order_norm.cpp']
  K.require(len(own)==1 and '-ffp-contract=off'in own[0]['command']and '-ffast-math'not in own[0]['command'], 'Actual component rounding flags differ')
  K.write(out/'review'/('component-'+kind+'-compile.json'),commands)

def component_check(out):
 source();browser=K.read(out/'browser/component-browser.json');K.require(browser['completed']and browser['controls']and not browser['page_errors'],'Browser component incomplete')
 checks=[]
 for row in K.read(out/'browser/component-inputs.json'):
  K.require(K.identity(out/'browser/component-inputs'/(row['name']+'.f32'))==row['input'],'Component input changed')
  for backend in ['native1','native2','wasm']:
   path=(out/'raw'/(row['name']+'.'+backend+'.f32'))if backend!='wasm'else out/'browser/component-inputs'/(row['name']+'.wasm.f32')
   actual=K.identity(path);checks.append(dict(name=row['name'],backend=backend,actual=actual,expected=row['target'],passed=actual==row['target']))
 K.write(out/'browser/component-proof.json',dict(passed=all(r['passed']for r in checks),checks=checks,comparisons=12,qualification=False))
 K.require(all(r['passed']for r in checks),'New no-SIMD component must recover all12 native targets')

def prepare_panel(out):
 cfg=source();K.require(K.read(out/'browser/component-proof.json')['passed'],'Component byte recovery required BEFOREmodel access')
 review=fetch(out,'old-ab-review',cfg['old_ab']['review']);raw=fetch(out,'old-ab-raw',cfg['old_ab']['raw'])
 report_bytes=one(review,'ab.json');K.require(__import__('hashlib').sha256(report_bytes).hexdigest()==cfg['old_ab']['report_sha256'],'Old completed negative producer differs')
 old_report=json.loads(report_bytes);K.require(old_report['state']=='completed-rejected'and old_report['source']==cfg['old_ab']['review']['source'],'Preserve old AB negative')
 freeze_bytes=one(review,'old-panel-freeze.json');K.require(__import__('hashlib').sha256(freeze_bytes).hexdigest()==old_report['old_panel_freeze_sha256'],'Old native freeze differs')
 frozen=json.loads(freeze_bytes);cases=panel(K.read(ROOT/'docs/wasm-q8-rounding-ab-v1.json'))
 K.require(set(frozen['cases'])=={c['id']for c in cases}and len(cases)==10,'All10 frozen cases required')
 for name,pin in old_report['artifact_files'].items():
  K.require(name in raw and dict(bytes=len(raw[name]),sha256=__import__('hashlib').sha256(raw[name]).hexdigest())==pin,'Old AB array differs: '+name)
 targets={};stage_targets=out/'raw/old-stage-targets';stage_targets.mkdir()
 for item in cases:
  key=item['id'];original=frozen['cases'][key]
  K.require(item['utf8_sha256']==original['text_sha256'],'Frozen text differs')
  row=dict(text=item['text'],id=key,text_sha256=item['utf8_sha256'],token_ids=original['token_ids'],native=original['native'],wasm={})
  for threads in ['1','2']:
   native=original['native'][threads];native_bytes=f32_bytes(native['vector']);path='native-vectors/'+key+'-'+threads+'.f32'
   K.require(len(native_bytes)==1536 and path in raw and raw[path]==native_bytes
       and __import__('hashlib').sha256(native_bytes).hexdigest()==native['sha256'],'Old raw/native freeze vector differs')
  for gate in ['0','1']:
   old_case=old_report['cases'][key][gate]
   value=old_case['vector'];K.require(__import__('hashlib').sha256(f32_bytes(value)).hexdigest()==old_case['sha256'],'Old WASM vector bits differ')
   # The old producer's JSON hash remains an immutable historical observation;
   # new byte anchors use the authenticated old actual pooling-result buffer.
   pool_name='browser/arrays/candidate-'+gate+'-'+key+'-pool.final.f32'
   K.require(pool_name in raw and len(raw[pool_name])==1536 and floats(raw[pool_name])==value,'Old actual final buffer/display values differ')
   row['wasm'][gate]=dict(vector=value,sha256=hashlib.sha256(raw[pool_name]).hexdigest(),raw_hex=raw[pool_name].hex(),
        historical_json_sha256=old_case['sha256'],actual_buffer_source=pool_name)
  targets[key]=row
 for threads in [1,2]:
  name='trusted-trace/packaged-'+str(threads)+'-dump.gguf';K.require(name in raw,'Old native full stages absent')
  values=stage_rows(raw[name],len(targets['original']['token_ids']))
  for observed in old_report['first_sentence_stages']['1'][str(threads)]:
   K.require(values[observed['stage']]['sha256']==observed['native_sha256'],'Frozen native stage differs')
  (stage_targets/('native'+str(threads)+'.gguf')).write_bytes(raw[name])
 for gate in ['0','1']:
  name='browser/arrays/candidate-'+gate+'-original-stages.gguf';K.require(name in raw,'Old WASM full stages absent')
  (stage_targets/('wasm'+gate+'.gguf')).write_bytes(raw[name])
 # Actual old compiler commands authenticate the no-SIMD profile, not the newer SIMD operator profile.
 (out/'review/old-wasm-compile.json').write_bytes(one(review,'candidate-compile-commands.json'))
 K.write(out/'browser/frozen-panel.json',dict(source=os.environ['GITHUB_SHA'],old_report_sha256=cfg['old_ab']['report_sha256'],
         held_q8_rounding='1',additional_legacy_q8_rounding='0',norm_default=False,cases=[targets[c['id']]for c in cases]))
 fixture=fetch(out,'model-fixture',cfg['fixture']);names=[n for n in fixture if n.endswith('/all-MiniLM-L6-v2-iq4_xs.gguf')]
 K.require(len(names)==1 and __import__('hashlib').sha256(fixture[names[0]]).hexdigest()==MODEL_SHA,'Frozen model identity differs')
 (out/'browser/model.gguf').write_bytes(fixture[names[0]])
 K.write(out/'review/panel-prepared.json',dict(model_sha256=MODEL_SHA,cases=10,stages=list(STAGES),original_rerun=False,
         frozen_panel=K.identity(out/'browser/frozen-panel.json'),old_negative_preserved=True,qualification=False))

def arithmetic_flags(command_text):
 import shlex
 return sorted(x for x in shlex.split(command_text)if x.startswith(('-m','-O','-ffast','-fno-fast','-ffp-contract','-DGGML','-DNDEBUG')))

def build_full(out):
 source();K.require(K.read(out/'browser/component-proof.json')['passed'],'Component barrier')
 overlay=out/'candidate-source';command(out,'worktree',['git','worktree','add','--detach',overlay,os.environ['GITHUB_SHA']],timeout=180)
 command(out,'submodule',['git','submodule','update','--init','ggml'],cwd=overlay,timeout=300)
 changes=overlay_ab(overlay);K.write(out/'review/observer-overlay.json',dict(changes=changes,test_only=True,production_math_changed=False))
 command(out,'wasm-full-build',['bash','build-embed-wasm.sh','--clean','--no-simd','--','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],cwd=overlay,timeout=3600)
 destination=out/'browser/candidate';destination.mkdir()
 for suffix in ['js','wasm']:shutil.copyfile(overlay/('build-embed-wasm/crispembed_embed.'+suffix),destination/('crispembed_embed.'+suffix))
 actual=K.read(overlay/'build-embed-wasm/compile_commands.json');previous=K.read(out/'review/old-wasm-compile.json');profile=[]
 for name in ['crispembed.cpp','ggml-cpu.c','ggml-cpu.cpp','vec.cpp','ops.cpp']:
  a=[r for r in actual if Path(r['file']).name==name];b=[r for r in previous if Path(r['file']).name==name]
  K.require(len(a)==len(b)==1,'Exact compiler coverage: '+name)
  K.require(arithmetic_flags(a[0]['command'])==arithmetic_flags(b[0]['command']),'Changed old no-SIMD arithmetic flags: '+name)
  K.require(('-msimd128'in a[0]['command'])==(name!='crispembed.cpp'),'Actual old script profile: frontend SIMDoff/backend SIMDon')
  profile.append(dict(file=name,old=b[0],current=a[0],arithmetic_flags=arithmetic_flags(a[0]['command'])))
 own=[r for r in actual if Path(r['file']).name=='wasm_native_order_norm.cpp']
 K.require(len(own)==1 and '-ffp-contract=off'in own[0]['command']and '-msimd128'not in own[0]['command']and '-ffast-math'not in own[0]['command']and '-Ofast'not in own[0]['command'],'Production helper scalar/contraction profile differs')
 K.write(out/'review/full-wasm-profile.json',dict(comparisons=profile,new_helper=own[0],script_no_simd=True,frontend_simd=False,ggml_backend_simd=True,pthreads=True))
 K.write(out/'review/full-wasm-compile.json',actual)
 native=out/'native-build'
 command(out,'native-full-configure',['cmake','-S',ROOT,'-B',native,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
         '-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF',
         '-DGGML_LLAMAFILE=ON','-DCRISPEMBED_BUILD_SHARED=ON'])
 command(out,'native-full-build',['cmake','--build',native,'--target','crispembed-shared','--parallel','2'],timeout=3600)
 libs=list(native.glob('libcrispembed.so'));K.require(len(libs)==1,'Exact native C ABI library required')
 dependencies={str(p.resolve()):K.identity(p)for p in native.rglob('libggml*.so*')if p.is_file()}
 K.require(dependencies,'Actual native GGML shared dependencies required')
 commands=K.read(native/'compile_commands.json');K.write(out/'review/full-native-compile.json',commands)
 helper=[r for r in commands if Path(r['file']).name=='wasm_native_order_norm.cpp']
 K.require(helper and all('-ffp-contract=off'in r['command']for r in helper),'Native helper source flags differ')
 command(out,'native-symbols',['nm','-C',libs[0]],timeout=180)
 symbols=(out/'review/native-symbols.log').read_text()
 K.require(not any(name in symbols for name in ['norm_candidate_controls','norm_candidate_replay','wasm_trace_setenv']),'New test observer leaked into production library')
 K.write(out/'review/build-artifacts.json',dict(native_library=K.identity(libs[0]),native_dependencies=dependencies,
     production_observer_exclusion=True,wasm={s:K.identity(destination/('crispembed_embed.'+s))for s in ['js','wasm']}))

def independent(a,b,shape):
 K.require(len(a)==len(b)==math.prod(shape)and len(shape)==2 and all(math.isfinite(v)for v in a+b),'Exact finite F32 comparison required')
 d=[y-x for x,y in zip(a,b)];width=shape[-1];cosines=[]
 for at in range(0,len(a),width):
  x,y=a[at:at+width],b[at:at+width];aa=sum(v*v for v in x);bb=sum(v*v for v in y)
  cosines.append(sum(v*w for v,w in zip(x,y))/math.sqrt(aa*bb)if aa>1e-18 and bb>1e-18 else 1. if aa<=1e-18 and bb<=1e-18 else 0.)
 row=dict(shape=shape,elements=len(a),max_absolute=max(abs(x)for x in d),mean_absolute=sum(abs(x)for x in d)/len(d),
          mean_error=sum(d)/len(d),rms=math.sqrt(sum(x*x for x in d)/len(d)),min_row_cosine=min(cosines),
          original_l2=math.sqrt(sum(x*x for x in a)),native_l2=math.sqrt(sum(x*x for x in b)))
 for label,values in [('original',a),('native',b)]:row.update({label+'_min':min(values),label+'_max':max(values),label+'_abs_min':min(abs(v)for v in values),label+'_abs_max':max(abs(v)for v in values)})
 row['passed']=row['max_absolute']<=.002 and row['min_row_cosine']>=.99999
 return row

def actual_ref(out,label,values):
 sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'));import diff_adapter
 folder=out/'raw/adapters';folder.mkdir(exist_ok=True);pairs=[];measurements={}
 for name,(a,b,shape)in values.items():
  measurements[name]=independent(a,b,shape);pair=dict(name=name)
  for side,v in [('original',a),('native',b)]:
   path=folder/(label+'-'+name+'-'+side+'.f32');path.write_bytes(f32_bytes(v))
   pair[side]=dict(path=str(path),shape=shape,sha256=K.sha(path))
  pairs.append(pair)
 left,right=folder/(label+'-original.gguf'),folder/(label+'-native.gguf')
 adapters=diff_adapter.write_pair(left,right,pairs);target=out/'review'/(label+'-ref.json')
 code=command(out,label+'-ref',[out/'component-native/norm-candidate-ref',left,right,target],timeout=180,expected=(0,1))
 cpp=diff_adapter.collect(code,target,measurements)
 return dict(cpp=cpp,independent=measurements,adapters=adapters,passed=all(v['passed']for v in measurements.values()))

def reference_controls(out):
 source();x=[float(i)/16 for i in range(16)];positive=actual_ref(out,'ref-control-positive',{'control':(x,x,[2,8])})
 y=list(x);y[0]+=.01;negative=actual_ref(out,'ref-control-negative',{'control':(x,y,[2,8])})
 sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'));import diff_adapter
 path=out/'raw/adapters/wrong-shape.gguf';data=out/'raw/adapters/wrong-shape.f32';data.write_bytes(f32_bytes(x))
 diff_adapter.write_gguf(path,[dict(name='control',path=str(data),shape=[4,4],sha256=K.sha(data))])
 shape=command(out,'ref-control-shape',[out/'component-native/norm-candidate-ref',out/'raw/adapters/ref-control-positive-original.gguf',path,
      out/'review/ref-control-shape.json'],expected=(2,))
 K.require(positive['passed']and not negative['passed']and shape==2,'Actual Ref control outcomes differ')
 K.write(out/'review/ref-controls.json',dict(positive=0,numeric_negative=1,wrong_shape=2,passed=True))

def native_worker(out,key,threads):
 source();frozen=K.read(out/'browser/frozen-panel.json');row=next(x for x in frozen['cases']if x['id']==key)
 libpath=out/'native-build/libcrispembed.so'
 K.require(K.identity(libpath)==K.read(out/'review/build-artifacts.json')['native_library']and K.sha(out/'browser/model.gguf')==MODEL_SHA,'Authenticated library/model required')
 os.environ.update(CRISPEMBED_FORCE_CPU='1',CRISPEMBED_DEBUG_TOKENS='1',CRISPEMBED_WASM_NATIVE_ORDER_NORM='1',CRISPEMBED_WASM_Q8_0_ROUND='1')
 lib=C.CDLL(str(libpath));lib.crispembed_init.argtypes=[C.c_char_p,C.c_int];lib.crispembed_init.restype=C.c_void_p
 lib.crispembed_encode.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_int)];lib.crispembed_encode.restype=C.POINTER(C.c_float)
 lib.crispembed_free.argtypes=[C.c_void_p];lib.crispembed_free.restype=None
 libraries={line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if '/'in line and any(x in line for x in ['libcrispembed','libggml'])}
 K.require(libraries and all(Path(p).resolve().is_relative_to((out/'native-build').resolve())for p in libraries),'Exact new-build GGML library closure required')
 expected=K.read(out/'review/build-artifacts.json')['native_dependencies']
 for path in libraries:
  if 'libggml'in path:K.require(expected.get(str(Path(path).resolve()))==K.identity(path),'Native GGML dependency SHA differs')
 results=[]
 for observed in [False,True]:
  os.environ.pop('CRISPEMBED_DUMP_LAYERS',None);os.environ.pop('CRISPEMBED_DUMP_LAYERS_GGUF',None)
  if observed:
   os.environ['CRISPEMBED_DUMP_LAYERS']='1'
   if key=='original':os.environ['CRISPEMBED_DUMP_LAYERS_GGUF']=str(out/'raw'/('native-current-'+str(threads)+'.gguf'))
  ctx=lib.crispembed_init(os.fsencode(out/'browser/model.gguf'),threads);K.require(ctx,'Current native init failed')
  try:
   for repeat in [0,1]:
    size=C.c_int();value=lib.crispembed_encode(ctx,row['text'].encode(),C.byref(size));K.require(value and size.value==384,'Native shape differs')
    data=C.string_at(value,1536);results.append(dict(observed=observed,repeat=repeat,vector=floats(data),sha256=__import__('hashlib').sha256(data).hexdigest()))
  finally:lib.crispembed_free(ctx)
 K.write(out/'raw'/('native-'+key+'-'+str(threads)+'.json'),dict(key=key,threads=threads,results=results,libraries=sorted(libraries)))
 sys.stdout.flush();sys.stderr.flush();os._exit(0)

def native_check(out):
 source();rows=[];env={k:v for k,v in os.environ.items()if k not in ['LD_PRELOAD','PYTHONPATH']and not k.startswith('CRISPEMBED_')}
 dirs=[out/'native-build/ggml/src',out/'native-build/ggml/src/ggml-cpu',Path(sys.base_prefix)/'lib']
 env['LD_LIBRARY_PATH']=':'.join(str(p)for p in dirs if p.is_dir())
 for case in K.read(out/'browser/frozen-panel.json')['cases']:
  for threads in [1,2]:
   label='native-'+case['id']+'-'+str(threads)
   command(out,label,[sys.executable,'-I',__file__,'native-worker','--output',out,'--case',case['id'],'--threads',str(threads)],env=env,timeout=300)
   result=K.read(out/'raw'/(label+'.json'));expected=case['native'][str(threads)]['sha256']
   K.require(len(result['results'])==4 and all(v['sha256']==expected for v in result['results']),'Native byte neutrality differs: '+label)
   log=(out/'review'/(label+'.log')).read_text();matches=re.findall(r'^crispembed: token_ids.*$',log,re.M)
   K.require(len(matches)==8 and all(tokens('\n'.join(matches[i:i+2]))==case['token_ids']for i in range(0,8,2)),'Native public/graph ID mismatch')
   rows.append(dict(case=case['id'],threads=threads,sha256=expected,ordinary_observer_repeat_exact=True,gate_one_native_ignored=True))
 for threads in [1,2]:
  old_stages=stage_rows((out/'raw/old-stage-targets'/('native'+str(threads)+'.gguf')).read_bytes(),len(K.read(out/'browser/frozen-panel.json')['cases'][0]['token_ids']))
  current=stage_rows((out/'raw'/('native-current-'+str(threads)+'.gguf')).read_bytes(),len(K.read(out/'browser/frozen-panel.json')['cases'][0]['token_ids']))
  K.require(all(old_stages[n]['sha256']==current[n]['sha256']for n in STAGES),'Native observed stage bytes changed')
 K.write(out/'review/native-proof.json',dict(passed=True,anchors=20,stage_anchors=16,rows=rows))

def actual_vector(row,browser=None):
 # JSON numbers are display-only: JSON.stringify erases the sign of zero.
 try:data=base64.b64decode(row['raw_base64'],validate=True)
 except Exception:raise ValueError('Invalid captured vector base64')from None
 K.require(len(data)==1536 and hashlib.sha256(data).hexdigest()==row['raw_sha256'],'Actual WASM byte extent/SHA differs')
 values=floats(data);display=row['vector']
 K.require(len(display)==384 and all(type(v)in [int,float]and math.isfinite(v)for v in display)
           and values==display,'Captured bytes/display logical values differ')
 if browser is not None:
  name=row['vector_file'];K.require(re.fullmatch('[a-z0-9_.-]+',name)is not None,'Unsafe captured vector filename')
  K.require((browser/'arrays'/name).read_bytes()==data,'Retained vector file differs from captured bytes')
 return data

def check_browser(out,phase):
 source();frozen_bytes=(out/'browser/frozen-panel.json').read_bytes();frozen=json.loads(frozen_bytes)
 result=K.read(out/'browser'/(phase+'-browser.json'));K.require(result['completed']and not result['page_errors']and result['source']==os.environ['GITHUB_SHA']
         and result['frozen_sha256']==__import__('hashlib').sha256(frozen_bytes).hexdigest(),'Complete source-bound browser required')
 K.require(set(result['cases'])=={c['id']for c in frozen['cases']},'All10 browser cases required')
 failures=[];pairs={};controls=[]
 def control(ok,name):
  controls.append(dict(name=name,passed=bool(ok)))
  if not ok:failures.append(name)
 arms=['legacy','unset','empty','0']if phase=='baseline'else['1']
 for case in frozen['cases']:
  key=case['id'];observed=result['cases'][key];K.require(set(observed)==set(arms),'Exact norm-gate coverage required')
  for arm in arms:
   rows=observed[arm];K.require(set(rows)=={'ordinary','observed'}and all(len(v)==2 for v in rows.values()),'Complete observer/repeat coverage required')
   raw=actual_vector(rows['ordinary'][0],out/'browser')
   for mode,values in rows.items():
    for repeat,v in enumerate(values):
     control(tokens('\n'.join(v['logs']))==case['token_ids'],key+'/'+arm+'/'+mode+'/'+str(repeat)+'/IDs')
     control(actual_vector(v,out/'browser')==raw,key+'/'+arm+'/'+mode+'/'+str(repeat)+'/neutral')
     if mode=='observed':
      prefix=out/'browser/arrays'/('-'.join([arm,key,mode,str(repeat)])+'-')
      control(Path(str(prefix)+'pool.final.f32').read_bytes()==raw,key+'/'+arm+'/'+str(repeat)+'/actualpool')
      mean=floats(Path(str(prefix)+'pool.mean.f32').read_bytes());K.require(len(mean)==384,'Actual pool mean axes differ')
      if key=='original':
       encoder=stage_rows(Path(str(prefix)+'stages.gguf').read_bytes(),len(case['token_ids']))['encoder_out']['values']
       roundf=lambda v:struct.unpack('<f',struct.pack('<f',v))[0]
       recomputed=[0.]*384
       for t in range(len(case['token_ids'])):
        for h in range(384):recomputed[h]=roundf(recomputed[h]+encoder[t*384+h])
       recomputed=[roundf(v/len(case['token_ids']))for v in recomputed]
       control(f32_bytes(mean)==f32_bytes(recomputed),key+'/'+arm+'/'+str(repeat)+'/source-mean-pool')
   if phase=='baseline':
    gate='0'if arm=='legacy'else'1';control(__import__('hashlib').sha256(raw).hexdigest()==case['wasm'][gate]['sha256'],key+'/'+arm+'/oldWASMbytes')
   else:
    for threads in [1,2]:pairs['final_'+key.replace('-','_')+'_t'+str(threads)]=(case['native'][str(threads)]['vector'],floats(raw),[1,384])
   if key=='original':
    for repeat in [0,1]:
     path=out/'browser/arrays'/('-'.join([arm,key,'observed',str(repeat)])+'-stages.gguf')
     stages=stage_rows(path.read_bytes(),len(case['token_ids']))
     if phase=='baseline':
      old_stages=stage_rows((out/'raw/old-stage-targets'/('wasm'+('0'if arm=='legacy'else'1')+'.gguf')).read_bytes(),len(case['token_ids']))
      for name in STAGES:control(stages[name]['sha256']==old_stages[name]['sha256'],'original/'+arm+'/'+str(repeat)+'/'+name+'/oldbytes')
     elif repeat==0:
      for threads in [1,2]:
       native=stage_rows((out/'raw/old-stage-targets'/('native'+str(threads)+'.gguf')).read_bytes(),len(case['token_ids']))
       for name in STAGES:pairs['stage_'+name+'_t'+str(threads)]=(native[name]['values'],stages[name]['values'],[len(case['token_ids']),384])
     else:
      first=stage_rows((out/'browser/arrays'/('1-original-observed-0-stages.gguf')).read_bytes(),len(case['token_ids']))
      for name in STAGES:control(first[name]['sha256']==stages[name]['sha256'],'original/1/'+name+'/stage-repeat')
 if phase=='baseline':
  native=K.read(out/'review/native-proof.json');K.require(native['passed']and native['anchors']==20,'All native byte anchors required before ON')
  proof=dict(passed=not failures,failures=failures,controls=controls,frozen_sha256=result['frozen_sha256'],
             q8_held_one=True,legacy_q8_zero_also_verified=True,norm_on_executed=False,qualification=False)
  K.write(out/'browser/off-proof.json',proof);K.write(out/'review/off-proof.json',proof)
  K.require(not failures,'OFF source/profile bytes changed; ON blocked')
  return
 K.require(K.read(out/'browser/off-proof.json')['passed'],'OFF barrier required')
 comparison=actual_ref(out,'whole-panel',pairs)
 first_difference=next((name for name in STAGES if any(f32_bytes(pairs['stage_'+name+'_t'+str(t)][0])!=f32_bytes(pairs['stage_'+name+'_t'+str(t)][1])for t in [1,2])),None)
 first_failure=next((name for name in STAGES if any(not comparison['independent']['stage_'+name+'_t'+str(t)]['passed']for t in [1,2])),None)
 worst=max(comparison['independent'],key=lambda n:comparison['independent'][n]['max_absolute'])
 for name,m in comparison['independent'].items():control(m['passed'],name+'/unchanged-budgets')
 interleaving=result.get('interleaving',[]);K.require([x['index']for x in interleaving]==[0,1,2,0],'Context0/1/0/repeat coverage required')
 first=next(c for c in frozen['cases']if c['id']=='original')
 for i,row in enumerate(interleaving):
  expected=actual_vector(result['cases']['original']['1']['ordinary'][0],out/'browser')if row['index']==1 else bytes.fromhex(first['wasm']['1']['raw_hex'])
  control(actual_vector(row,out/'browser')==expected,'context-'+str(i))
 report=dict(schema='crispembed-wasm-native-order-norm-full-ab-v1',source=os.environ['GITHUB_SHA'],completed=True,passed=not failures,
             failures=failures,controls=controls,comparison=comparison,final_comparisons=20,stage_comparisons=16,
             first_bit_differing_stage=first_difference,first_budget_failing_stage=first_failure,worst_comparison=worst,
             off_proof=K.identity(out/'browser/off-proof.json'),native_proof=K.identity(out/'review/native-proof.json'),
             source_profile=K.sha(K.PROFILE),norm_default=False,q8_default_changed=False,q8_rounding_held='1',frontend_simd=False,ggml_backend_simd=True,
             old_negative_preserved=True,qualification=False,release_qualified=False,original_rerun=False)
 K.write(out/'review/ab.json',report)
 if failures:raise ArithmeticError('Unchanged full-panel gates rejected; all numeric records retained')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['preflight','component-prepare','component-build','component-check',
  'prepare-panel','build-full','native-check','off-check','compare','native-worker']);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--case');p.add_argument('--threads',type=int,choices=[1,2]);a=p.parse_args()
 try:
  actions={'preflight':preflight,'component-prepare':component_prepare,'component-build':component_build,
   'component-check':lambda o:(component_check(o),reference_controls(o)),'prepare-panel':prepare_panel,'build-full':build_full,
   'native-check':native_check,'off-check':lambda o:check_browser(o,'baseline'),'compare':lambda o:check_browser(o,'candidate')}
  if a.phase=='native-worker':native_worker(a.output,a.case,a.threads)
  else:actions[a.phase](a.output)
 except ArithmeticError as e:
  K.write(a.output/'review'/('numeric-rejection-'+a.phase+'.json'),dict(phase=a.phase,error=str(e),numeric_rejection=True,qualification=False));return 1
 except Exception as e:
  K.write(a.output/'review'/('structural-rejection-'+a.phase+'.json'),dict(phase=a.phase,error=type(e).__name__+': '+str(e),qualification=False));return 2
 finally:
  if a.phase not in ['preflight','native-worker']:
   K.write(a.output/'review/file-manifest.json',{str(p.relative_to(a.output)):K.identity(p)for folder in [a.output/'review',a.output/'raw',a.output/'browser']
      for p in folder.rglob('*')if p.is_file()and p.name!='file-manifest.json'})
 return 0
if __name__=='__main__':raise SystemExit(main())
