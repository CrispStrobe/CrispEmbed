#!/usr/bin/env python3
"""Hosted-only one-sentence live norm operands and source-faithful replay."""
import argparse,base64,ctypes as C,hashlib,importlib.util,json,math,os,re,shutil,struct,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/wasm-live-norm-v1'),str(ROOT/'tools/wasm-native-order-norm-v1'),str(ROOT/'tools'),str(ROOT/'tools/ci-heavy')]
import live_contract as K
from overlay import apply
from successor_assets import api,artifact
from successor_asset_manifest import safe_name
from successor_wasm_trace_contract import MODEL_SHA,STAGES,f32_bytes,floats,tokens,stage_rows
from wasm_layer3_attribution_v1 import NODES
spec=importlib.util.spec_from_file_location('frozen_norm_helpers',ROOT/'tools/ci-heavy/wasm_native_order_norm_v1.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
# Only pure archive/comparison helpers are reused. No old guard/preflight is called or retargeted.
command,one,independent,frozen_actual_ref,actual_vector=old.command,old.one,old.independent,old.actual_ref,old.actual_vector

def actual_ref(out,label,values):
 result=frozen_actual_ref(out,label,values)
 for row in result['independent'].values():row['independent_passed']=row.pop('numpy_passed')
 result['independent_implementation']='CPython F64 arithmetic over exact F32 source values; independent from GGML'
 return result

def zip_files(path,member_limit=1024,byte_limit=128*2**20):
 with zipfile.ZipFile(path)as z:
  rows=z.infolist();names=[safe_name(r.filename)for r in rows]
  K.require(0<len(rows)<=member_limit and sum(r.file_size for r in rows)<=byte_limit,'Bounded ZIP coverage exceeded')
  K.require(len(names)==len(set(names))and not any(r.is_dir()or r.flag_bits&1 for r in rows),'Unsafe duplicate/directory/encrypted ZIP')
  return {name:z.read(row)for name,row in zip(names,rows)}

def fetch(out,key,pin):
 service=api('actions/runs/'+str(pin['run']))
 K.require(service['head_sha']==pin['source']and service['path']==pin['workflow']and service['status']=='completed'
           and service['conclusion']==pin['conclusion']and service['event']=='workflow_dispatch'and service['run_attempt']==1,'Frozen service source/attempt/state differs')
 K.write(out/'review'/(key+'-service.json'),{k:service[k]for k in ['id','head_sha','path','status','conclusion','event','run_attempt']})
 path=out/(key+'.zip');meta=artifact(pin['run'],pin['name'],path,pin['bytes'])
 K.require(meta['id']==pin['id']and meta['size_in_bytes']==pin['bytes']and meta['digest']=='sha256:'+pin['sha256']and K.sha(path)==pin['sha256'],'Frozen artifact identity differs')
 K.write(out/'review'/(key+'-artifact.json'),meta)
 return zip_files(path)

def source():
 cfg=K.source(os.environ.get('GITHUB_SHA',''))
 K.require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','Genuine GHA only')
 meta=api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
 K.require(meta['head_sha']==os.environ['GITHUB_SHA']and meta['event']=='workflow_dispatch'and meta['status']=='in_progress'
           and meta['path']in ['.github/workflows/wasm-live-norm-operands-v1.yml','.github/workflows/ovisocr2-metadata.yml'],'Service/source mismatch')
 return cfg

def preflight(out):
 cfg=source();out.mkdir(exist_ok=False)
 for p in ['review','raw','browser/arrays','browser/component-inputs']:(out/p).mkdir(parents=True,exist_ok=True)
 ram=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
 K.require(ram>=6*2**30 and shutil.disk_usage(out).free>=16*2**30,'Hosted6GiB RAM/16GiB disk')
 K.write(out/'review/source.json',dict(source=os.environ['GITHUB_SHA'],base=K.BASE,candidate=K.CANDIDATE,ggml=K.GGML,protocol_sha256=K.sha(K.PROFILE),
         protected=cfg['protected_git_objects'],qualification=False))
 K.write(out/'review/resources.json',dict(available_ram_bytes=ram,free_disk_bytes=shutil.disk_usage(out).free,workers=2))

def prepare(out):
 cfg=source();review=fetch(out,'candidate-review',cfg['previous']['review']);raw=fetch(out,'candidate-raw',cfg['previous']['raw'])
 body=one(review,'ab.json');K.require(hashlib.sha256(body).hexdigest()==cfg['previous']['report_sha256'],'Previous negative producer differs')
 report=json.loads(body);K.require(report['completed']and not report['passed']and not report['qualification']and report['source']==K.CANDIDATE
         and len(report['failures'])==22 and report['final_comparisons']==20 and report['stage_comparisons']==16,'Retain actual64 negative')
 manifest=json.loads(one(review,'file-manifest.json'))
 for name,data in raw.items():K.require(manifest.get(name)==dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),'Raw inventory/member differs: '+name)
 frozen=json.loads(one(review,'frozen-panel.json'));case=next(x for x in frozen['cases']if x['id']=='original')
 K.require(len(case['token_ids'])==12 and hashlib.sha256(case['text'].encode()).hexdigest()==case['text_sha256'],'Frozen first sentence differs')
 baseline=json.loads(one(review,'baseline-browser.json'));candidate=json.loads(one(review,'candidate-browser.json'))
 targets={}
 for gate,producer in [('0',baseline),('1',candidate)]:
  r=producer['cases']['original'][gate]['ordinary'][0]
  data=base64.b64decode(r['raw_base64'],validate=True)
  K.require(len(data)==1536 and hashlib.sha256(data).hexdigest()==r['raw_sha256'],'Old actual returned vector differs')
  targets[gate]=dict(bytes=1536,sha256=r['raw_sha256'],raw_base64=r['raw_base64'])
  name='browser/arrays/'+r['vector_file'];K.require(raw.get(name)==data,'Old captured buffer/file differ')
 (out/'review/old-wasm-compile.json').write_bytes(one(review,'full-wasm-compile.json'))
 for t in [1,2]:
  name='raw/native-current-'+str(t)+'.gguf';K.require(name in raw,'Native frozen full stages absent')
  (out/'raw'/('native-target-'+str(t)+'.gguf')).write_bytes(raw[name])
 rows=json.loads(one(review,'component-prepared.json'))['rows']
 K.require([r['name']for r in rows]==['attention_native','ffn_native','attention_candidate','ffn_candidate'],'Four component inputs')
 for r in rows:
  ip='browser/component-inputs/'+r['name']+'.f32';tp=r['target_path']
  for n,pin in [(ip,r['input']),(tp,r['target'])]:
   K.require(n in raw and dict(bytes=len(raw[n]),sha256=hashlib.sha256(raw[n]).hexdigest())==pin,'Component raw differs')
  (out/ip).write_bytes(raw[ip]);(out/tp).parent.mkdir(parents=True,exist_ok=True);(out/tp).write_bytes(raw[tp])
 K.write(out/'browser/component-inputs.json',rows)
 K.write(out/'browser/frozen.json',dict(source=os.environ['GITHUB_SHA'],case=case,previous_vectors=targets,previous_report_sha256=cfg['previous']['report_sha256']))
 K.write(out/'review/prepared.json',dict(previous_negative_preserved=True,retained_raw_members=len(raw),case='original',tokens=12,original_forward=False,qualification=False))

def component_build(out):
 source();native=out/'component-native';wasm=out/'component-wasm'
 command(out,'component-native-configure',['cmake','-S',ROOT/'tests/wasm-live-norm','-B',native,'-DCMAKE_BUILD_TYPE=Release'])
 command(out,'component-native-build',['cmake','--build',native,'--target','norm-candidate-controls','norm-candidate-ref','live-norm-replay','--parallel','2'])
 command(out,'component-authored',[native/'norm-candidate-controls'])
 command(out,'component-wasm-configure',['emcmake','cmake','-S',ROOT/'tests/wasm-live-norm','-B',wasm,
         '-DCMAKE_BUILD_TYPE=Release','-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm','-DCMAKE_C_FLAGS=-pthread','-DCMAKE_CXX_FLAGS=-pthread'])
 command(out,'component-wasm-build',['cmake','--build',wasm,'--target','norm-candidate-controls','live-norm-replay','--parallel','2'])
 destination=out/'browser/component';destination.mkdir()
 for name in ['norm-candidate-controls','live-norm-replay']:
  for suffix in ['js','wasm']:shutil.copyfile(wasm/(name+'.'+suffix),destination/(name+'.'+suffix))
 command(out,'live-observer-controls',[native/'live-norm-replay'])
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


def model(out):
 cfg=source();K.require(K.read(out/'browser/component-proof.json')['passed'],'Twelve new-profile component recoveries BEFOREmodel access')
 data=fetch(out,'model-fixture',cfg['fixture']);matches=[v for k,v in data.items()if k.endswith('/all-MiniLM-L6-v2-iq4_xs.gguf')]
 K.require(len(matches)==1 and hashlib.sha256(matches[0]).hexdigest()==MODEL_SHA,'Exact frozen model required')
 (out/'browser/model.gguf').write_bytes(matches[0])

def build_full(out):
 source();K.require(K.read(out/'browser/component-proof.json')['passed']and K.sha(out/'browser/model.gguf')==MODEL_SHA,'Model/component barrier')
 overlay=out/'observer-source';command(out,'observer-worktree',['git','worktree','add','--detach',overlay,os.environ['GITHUB_SHA']],timeout=180)
 command(out,'observer-submodule',['git','submodule','update','--init','ggml'],cwd=overlay,timeout=300)
 changes=apply(overlay);K.write(out/'review/overlay.json',dict(changes=changes,test_only=True,numeric_nodes_added=0,runtime_source_unchanged=True))
 command(out,'wasm-observer-build',['bash','build-embed-wasm.sh','--clean','--no-simd','--','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],cwd=overlay,timeout=3600)
 destination=out/'browser/candidate';destination.mkdir()
 for suffix in ['js','wasm']:shutil.copyfile(overlay/('build-embed-wasm/crispembed_embed.'+suffix),destination/('crispembed_embed.'+suffix))
 actual=K.read(overlay/'build-embed-wasm/compile_commands.json');previous=K.read(out/'review/old-wasm-compile.json');profiles=[]
 for name in ['crispembed.cpp','ggml-cpu.c','ggml-cpu.cpp','vec.cpp','ops.cpp','wasm_native_order_norm.cpp']:
  a=[r for r in actual if Path(r['file']).name==name];b=[r for r in previous if Path(r['file']).name==name]
  K.require(len(a)==len(b)==1 and old.arithmetic_flags(a[0]['command'])==old.arithmetic_flags(b[0]['command']),'Frozen64 compile profile differs: '+name)
  profiles.append(dict(file=name,old=b[0],current=a[0]))
 K.write(out/'review/wasm-profile.json',profiles)
 flags=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF',
        '-DGGML_BLAS=OFF','-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF','-DGGML_LLAMAFILE=ON','-DCRISPEMBED_BUILD_SHARED=ON']
 for label,src in [('plain',ROOT),('observed',overlay)]:
  build=out/('native-'+label)
  command(out,label+'-configure',['cmake','-S',src,'-B',build,*flags])
  command(out,label+'-build',['cmake','--build',build,'--target','crispembed-shared','--parallel','2'],timeout=3600)
  library=build/'libcrispembed.so';K.require(library.is_file(),'Root C ABI library absent')
  command(out,label+'-symbols',['nm','-D',library],timeout=180)
  symbols=(out/'review'/(label+'-symbols.log')).read_text()
  K.require(('crispembed_live_norm_finish'in symbols)==(label=='observed'),'Observer export exclusion differs')
  command(out,label+'-all-symbols',['nm','-C',library],timeout=180)
  all_symbols=(out/'review'/(label+'-all-symbols.log')).read_text()
  if label=='plain':K.require('live_norm_observer'not in all_symbols and 'crispembed_live_norm_finish'not in all_symbols,'Observer leaked into plain production')
  K.write(out/'review'/(label+'-binaries.json'),dict(library=K.identity(library),dependencies={str(p.relative_to(build)):K.identity(p)for p in build.rglob('libggml*.so*')if p.is_file()}))
 K.write(out/'review/wasm-binaries.json',{s:K.identity(destination/('crispembed_embed.'+s))for s in ['js','wasm']})

def native_context(lib,model,threads,mode):
 K.require(mode in ['ordinary','observed'],'Native observer mode')
 for name in ['CRISPEMBED_LIVE_NORM_CAPTURE','CRISPEMBED_DUMP_LAYERS','CRISPEMBED_DUMP_LAYERS_GGUF']:os.environ.pop(name,None)
 # ctx->dump_layers is snapshotted by crispembed_init, not by encode.
 if mode=='observed':os.environ['CRISPEMBED_DUMP_LAYERS']='1'
 return lib.crispembed_init(os.fsencode(model),threads)

def native_worker(out,threads):
 source();frozen=K.read(out/'browser/frozen.json');case=frozen['case'];K.require(K.sha(out/'browser/model.gguf')==MODEL_SHA,'Frozen model changed')
 results={}
 os.environ.update(CRISPEMBED_FORCE_CPU='1',CRISPEMBED_DEBUG_TOKENS='1',CRISPEMBED_WASM_NATIVE_ORDER_NORM='1',CRISPEMBED_WASM_Q8_0_ROUND='1')
 # Fresh child process per plain/observed library prevents DSO namespace mixing.
 label=os.environ['LIVE_NORM_LIBRARY'];K.require(label in ['plain','observed'],'Library arm')
 build=out/('native-'+label);pin=K.read(out/'review'/(label+'-binaries.json'))
 K.require(K.identity(build/'libcrispembed.so')==pin['library'],'Library SHA changed')
 lib=C.CDLL(str(build/'libcrispembed.so'));lib.crispembed_init.argtypes=[C.c_char_p,C.c_int];lib.crispembed_init.restype=C.c_void_p
 lib.crispembed_encode.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_int)];lib.crispembed_encode.restype=C.POINTER(C.c_float)
 lib.crispembed_free.argtypes=[C.c_void_p];lib.crispembed_free.restype=None
 if label=='observed':lib.crispembed_live_norm_finish.argtypes=[C.c_void_p];lib.crispembed_live_norm_finish.restype=C.c_int
 mapped={line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if '/'in line and any(x in line for x in ['libcrispembed','libggml'])}
 K.require(mapped and all(Path(p).resolve().is_relative_to(build.resolve())for p in mapped),'Mapped GGML closure differs')
 for p in mapped:
  if 'libggml'in p:K.require(K.identity(p)==pin['dependencies'].get(str(Path(p).resolve().relative_to(build.resolve()))),'Mapped dependency SHA differs')
 for mode in (['ordinary']if label=='plain'else['ordinary','observed']):
  ctx=native_context(lib,out/'browser/model.gguf',threads,mode);K.require(ctx,'Native init failed')
  rows=[]
  try:
   for repeat in [0,1]:
    prefix=out/'raw'/('native'+str(threads)+'-'+label+'-'+mode+'-'+str(repeat));prefix.mkdir()
    if mode=='observed':
     os.environ['CRISPEMBED_LIVE_NORM_CAPTURE']=str(prefix);os.environ['CRISPEMBED_DUMP_LAYERS']='1';os.environ['CRISPEMBED_DUMP_LAYERS_GGUF']=str(prefix/'stages.gguf')
    n=C.c_int();p=lib.crispembed_encode(ctx,case['text'].encode(),C.byref(n));K.require(p and n.value==384,'Native output shape')
    raw=C.string_at(p,1536);K.require(all(math.isfinite(x)for x in floats(raw)),'Native finite')
    (prefix/'vector.f32').write_bytes(raw)
    if mode=='observed':K.require(lib.crispembed_live_norm_finish(ctx)==1,'All13 connected norms/49 observations required')
    rows.append(dict(path=str(prefix.relative_to(out)),vector=K.identity(prefix/'vector.f32')))
  finally:lib.crispembed_free(ctx)
  results[mode]=rows
 K.write(out/'review'/('native'+str(threads)+'-'+label+'.json'),results)

def native(out):
 source()
 for label in ['plain','observed']:
  build=out/('native-'+label);dirs=sorted({str(p.parent.resolve())for p in build.rglob('libggml*.so*')if p.is_file()}|{str(build.resolve())})
  for threads in [1,2]:
   env=dict(os.environ,LIVE_NORM_LIBRARY=label,LD_LIBRARY_PATH=':'.join(dirs))
   command(out,'native'+str(threads)+'-'+label,[sys.executable,'-I',__file__,'native-worker','--output',out,'--threads',str(threads)],env=env,timeout=600)
 for t in [1,2]:
  target=K.read(out/'browser/frozen.json')['case']['native'][str(t)]['sha256']
  for label in ['plain','observed']:
   for rows in K.read(out/'review'/('native'+str(t)+'-'+label+'.json')).values():
    for row in rows:K.require(row['vector']['sha256']==target,'Native first-sentence ordinary/observed/repeat frozen bytes differ')
 K.write(out/'browser/native-proof.json',dict(passed=True,threads=[1,2],native_anchors=12,case='original',qualification=False))

def native_tokens(log,expected,forwards):
 rows=re.findall(r'^crispembed: token_ids \(n=(\d+)\):([ 0-9]+)$',log,re.M)
 return len(rows)==forwards*2 and len(re.findall(r'^crispembed: token_ids',log,re.M))==len(rows) and all(int(n)==len(expected)and list(map(int,ids.split()))==expected for n,ids in rows)

def capture_rows(folder,enabled):
 proof=K.read(folder/'capture.json');expected={f'norm{i:02d}_{kind}'for i in range(13)for kind in ['input','output']}|{'layer3_'+n for n in NODES}
 K.require(proof['schema']=='crispembed-wasm-live-norm-capture-v1'and proof['architecture']=='bert'and proof['width']==384
           and proof['layers']==6 and proof['heads']==12 and proof['intermediate']==1536 and proof['pre_ln']is False
           and proof['cpu']is True and proof['enabled']is enabled and proof['epsilon_bits']==struct.unpack('<I',struct.pack('<f',1e-12))[0]
           and proof['connected_norm_nodes']==13 and proof['custom_norm_nodes']==(13 if enabled else 0),'Actual live model/gate/connected norms differ')
 K.require(len(proof['files'])==49 and {r['name']for r in proof['files']}==expected,'All13 pairs+23 node labels required')
 result={}
 for row in proof['files']:
  name=row['name'];shape=[12,1536 if name in ['layer3_fc1_raw','layer3_fc1','layer3_gelu']else 384]
  K.require(row['shape']==shape and row['bytes']==math.prod(shape)*4 and row['hits']==1,'Actual live contiguous shape/hits differ')
  path=folder/(name+'.f32');K.require(path.stat().st_size==row['bytes'],'Captured extent differs');floats(path.read_bytes())
  result[name]=dict(path=path,shape=shape,**K.identity(path))
 return result

def prepare_replay(out):
 source();frozen=K.read(out/'browser/frozen.json');browser=K.read(out/'browser/live-capture-browser.json')
 K.require(browser['completed']and not browser['page_errors']and browser['frozen_sha256']==K.sha(out/'browser/frozen.json'),'Actual browser capture incomplete')
 captures={};controls=[]
 def control(v,name):controls.append(dict(name=name,passed=bool(v)))
 for gate in ['0','1']:
  arms=browser['cases']['original'][gate];target=frozen['previous_vectors'][gate]['sha256']
  for mode in ['ordinary','observed']:
   for i,row in enumerate(arms[mode]):
    data=actual_vector(row,out/'browser');control(hashlib.sha256(data).hexdigest()==target,'wasm'+gate+'/'+mode+'/'+str(i)+'/previous64bytes')
    control(tokens('\n'.join(row['logs']))==frozen['case']['token_ids'],'wasm'+gate+'/'+mode+'/'+str(i)+'/IDs')
    if mode=='observed':
     folder=out/'browser/arrays'/row['capture_path'];rows=capture_rows(folder,gate=='1')
     if i==0:captures['wasm'+gate]=rows
     else:control(all(rows[n]['sha256']==captures['wasm'+gate][n]['sha256']for n in rows),'wasm'+gate+'/all49-repeat')
 for t in [1,2]:
  ordinary=K.read(out/'review'/('native'+str(t)+'-plain.json'))['ordinary']
  observed=K.read(out/'review'/('native'+str(t)+'-observed.json'))['observed']
  for i,row in enumerate(observed):
   rows=capture_rows(out/row['path'],False)
   control(row['vector']==ordinary[i]['vector'],'native'+str(t)+'/observer-neutral')
   actual=stage_rows((out/row['path']/'stages.gguf').read_bytes(),12);target=stage_rows((out/'raw'/('native-target-'+str(t)+'.gguf')).read_bytes(),12)
   control(all(actual[n]['sha256']==target[n]['sha256']for n in STAGES),'native'+str(t)+'/old8stages')
   if i==0:captures['native'+str(t)]=rows
   else:control(all(rows[n]['sha256']==captures['native'+str(t)][n]['sha256']for n in rows),'native'+str(t)+'/all49-repeat')
 control(browser['cases']['original']['0']['ordinary'][0]['raw_sha256']!=browser['cases']['original']['1']['ordinary'][0]['raw_sha256'],'gate-sensitive-first-sentence')
 for label in ['plain','observed']:
  for t in [1,2]:
   log=(out/'review'/('native'+str(t)+'-'+label+'.log')).read_text()
   control(native_tokens(log,frozen['case']['token_ids'],2 if label=='plain'else 4),'native'+str(t)+'/'+label+'/IDs')
 public={label:{name:{**{k:v for k,v in r.items()if k!='path'},'path':str(r['path'].relative_to(out))}for name,r in rows.items()}for label,rows in captures.items()}
 K.write(out/'review/capture-proof.json',dict(passed=all(c['passed']for c in controls),controls=controls,connected_norms_per_arm=13,arms=public,qualification=False))
 K.require(all(c['passed']for c in controls),'Activation/ordinary/observer/repeat/source-byte anchors differ; replay blocked')
 folder=out/'browser/replay-inputs';folder.mkdir();packets=[]
 for label,rows in captures.items():
  for i in range(13):
   name=label+'_norm'+str(i).zfill(2);r=rows['norm'+str(i).zfill(2)+'_input'];p=folder/(name+'.f32');p.write_bytes(r['path'].read_bytes())
   packets.append(dict(name=name,source_arm=label,norm=i,**K.identity(p)))
 K.write(out/'browser/replay-inputs.json',packets)

def replay(out):
 source();proof=K.read(out/'review/capture-proof.json');K.require(proof['passed'],'Capture proof required')
 browser=K.read(out/'browser/live-replay-browser.json');K.require(browser['completed']and not browser['page_errors'],'Actual browser replay incomplete')
 packets=K.read(out/'browser/replay-inputs.json');K.require(len(packets)==52,'Four natural sources by13 norm inputs')
 K.require(len(browser['rows'])==208 and len({(r['name'],r['mode'],r['threads'])for r in browser['rows']})==208,'Exact WASM replay coverage')
 for r in browser['rows']:
  K.require(r['repeat_exact']and r['input_exact']and K.identity(out/'browser/arrays'/r['file'])==dict(bytes=r['bytes'],sha256=r['sha256']),'WASM replay buffer/hash/repeat differs')
 controls=[];local_pairs={};natural_pairs={};own=[]
 def values(p):return floats(Path(p).read_bytes())
 for packet in packets:
  name=packet['name'];inp=out/'browser/replay-inputs'/(name+'.f32');K.require(K.identity(inp)=={k:packet[k]for k in ['bytes','sha256']},'Replay input mutated')
  outputs={}
  for mode in [0,1]:
   for t in [1,2]:
    target=out/'raw'/(name+'-native-'+str(mode)+'-'+str(t)+'.f32')
    command(out,'replay-'+name+'-'+str(mode)+'-'+str(t),[out/'component-native/live-norm-replay',inp,target,str(mode),str(t)],timeout=60)
    outputs['native'+str(mode)+str(t)]=target
    b=next(r for r in browser['rows']if(r['name'],r['mode'],r['threads'])==(name,mode,t));outputs['wasm'+str(mode)+str(t)]=out/'browser/arrays'/b['file']
  label=packet['source_arm'];index=packet['norm'];observed=out/proof['arms'][label][f'norm{index:02d}_output']['path']
  actual=outputs['wasm11']if label=='wasm1'else outputs['wasm01']if label=='wasm0'else outputs['native0'+label[-1]]
  own.append(dict(name=name,observed=K.identity(observed),replayed=K.identity(actual),passed=observed.read_bytes()==actual.read_bytes()))
  # Exact SAME input: actual native GGML norm vs both unchanged helper builds.
  for t in [1,2]:
   for backend,path in [('native',outputs['native1'+str(t)]),('wasm',outputs['wasm1'+str(t)])]:
    key=name+'_'+backend+str(t);local_pairs[key]=(values(outputs['native0'+str(t)]),values(path),[12,384])
  for mode in [0,1]:
   controls.append(dict(name=name+'/mode'+str(mode)+'/native-threads',passed=outputs['native'+str(mode)+'1'].read_bytes()==outputs['native'+str(mode)+'2'].read_bytes()))
 # Keep each Ref adapter <=128 rows, preserving complete outcomes including numeric failures.
 comparisons=[]
 for at in range(0,len(local_pairs),104):comparisons.append(actual_ref(out,'same-input-'+str(at//104),dict(list(local_pairs.items())[at:at+104])))
 for label in ['wasm0','wasm1']:
  for name,r in proof['arms'][label].items():
   for t in [1,2]:
    native=proof['arms']['native'+str(t)][name]
    natural_pairs[label+'_'+name+'_t'+str(t)]=(values(out/native['path']),values(out/r['path']),r['shape'])
 natural=[]
 for at in range(0,len(natural_pairs),98):natural.append(actual_ref(out,'natural-'+str(at//98),dict(list(natural_pairs.items())[at:at+98])))
 own_pass=all(x['passed']for x in own);local_pass=all(r['passed']for r in comparisons)
 measured={name:row for batch in natural for name,row in batch['independent'].items()}
 first={}
 for label in ['wasm0','wasm1']:
  for t in [1,2]:
   native=proof['arms']['native'+str(t)];wasm=proof['arms'][label]
   first[label+'_t'+str(t)]=dict(
    first_bit_differing_norm_input=next((f'norm{i:02d}'for i in range(13)if native[f'norm{i:02d}_input']['sha256']!=wasm[f'norm{i:02d}_input']['sha256']),None),
    first_bit_differing_layer3_node=next((n for n in NODES if native['layer3_'+n]['sha256']!=wasm['layer3_'+n]['sha256']),None),
    first_budget_failing_layer3_node=next((n for n in NODES if not measured[label+'_layer3_'+n+'_t'+str(t)]['passed']),None))
 worst=max(measured,key=lambda n:measured[n]['max_absolute'])
 report=dict(schema='crispembed-wasm-live-norm-operands-result-v1',source=os.environ['GITHUB_SHA'],completed=True,
  passed=own_pass and local_pass and all(r['passed']for r in controls),case='original',held_q8_rounding='1',norm_nodes=13,
  captured_tensors_per_arm=49,arms=['native1','native2','wasm0','wasm1'],same_input_comparisons=len(local_pairs),
  same_input=comparisons,natural_comparisons=len(natural_pairs),natural=natural,first_divergences=first,worst_natural_comparison=worst,own_replays=own,controls=controls,
  own_replays_bit_exact=own_pass,local_norm_replays_passed=local_pass,natural_strict_passed=all(r['passed']for r in natural),
  old64_negative_preserved=True,qualification=False,release_qualified=False,production_change=False,original_rerun=False,
  limits='One frozen12-token sentence; natural differences remain negatives. No full10 rerun or precision/math change. No FC1 workspace causality claim without separately captured actual conversion barriers.')
 K.write(out/'review/live-norm.json',report)
 K.require(report['passed'],'Local replay/identity rejection; all numerical outcomes retained')

def ref_controls(out):
 source();x=[float(i)/16 for i in range(16)]
 pos=actual_ref(out,'ref-control-positive',{'control':(x,x,[2,8])});y=list(x);y[0]+=.01
 neg=actual_ref(out,'ref-control-negative',{'control':(x,y,[2,8])})
 sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'));import diff_adapter
 path=out/'raw/adapters/wrong-shape.gguf';f=out/'raw/adapters/wrong-shape.f32';f.write_bytes(f32_bytes(x))
 diff_adapter.write_gguf(path,[dict(name='control',path=str(f),shape=[4,4],sha256=K.sha(f))])
 wrong=command(out,'ref-control-shape',[out/'component-native/norm-candidate-ref',out/'raw/adapters/ref-control-positive-original.gguf',path,out/'review/ref-control-shape.json'],expected=(2,))
 K.require(pos['passed']and not neg['passed']and wrong==2,'Actual Ref positive/numeric/shape controls')
 K.write(out/'review/ref-controls.json',dict(positive=0,numeric_negative=1,wrong_shape=2,passed=True))

def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('phase',choices=['preflight','prepare','component-build','component-check','model','build-full','native','native-worker','prepare-replay','replay','ref-controls'])
 p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,choices=[1,2]);a=p.parse_args();out=a.output.resolve()
 phases={'preflight':preflight,'prepare':prepare,'component-build':component_build,'component-check':component_check,'model':model,'build-full':build_full,'native':native,
 'native-worker':lambda o:native_worker(o,a.threads),'prepare-replay':prepare_replay,'replay':replay,'ref-controls':ref_controls}
 try:phases[a.phase](out)
 except Exception as e:
  out.joinpath('review').mkdir(parents=True,exist_ok=True)
  K.write(out/'review'/('rejection-'+a.phase+'.json'),dict(source=os.environ.get('GITHUB_SHA'),phase=a.phase,completed=False,error=str(e),qualification=False));raise
 finally:
  if out.exists():
   files={}
   for folder in ['raw','browser','review']:
    for path in (out/folder).rglob('*'):
     if path.is_file()and path.name!='file-manifest.json':files[str(path.relative_to(out))]=K.identity(path)
   K.write(out/'review/file-manifest.json',files)
if __name__=='__main__':main()
