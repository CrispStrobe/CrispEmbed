#!/usr/bin/env python3
"""Honest Kaggle CPU successor of6e327: unchanged numerical/typed/C-quantizer body."""
import argparse,hashlib,json,math,os,platform,re,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/gemma-decision-q4-cpu-v2'),str(ROOT/'tools/gemma-decision-q4-v1'),str(ROOT/'tools')]
import cpu_contract as K
SOURCE=None
import gemma_q4_recipe as R
from ci_process import run
STAGES=['post_embed','ple']+['layer_'+str(i)for i in range(24)]+['final_norm','tokens','embedding']

def source():
 return K.source(SOURCE)

def command(out,name,args,expected=(0,),timeout=2400):
 log=out/'review'/(name+'.log');env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 for key in list(env):
  if key.startswith('CRISPEMBED_GEMMA_DECISION'):env.pop(key)
 result=run(args,log,cwd=ROOT,env=env,timeout=timeout)
 K.write(out/'review'/(name+'-command.json'),dict(result,argv=list(map(str,args)),log=log.name,**K.identity(log)))
 K.require(not result['timed_out']and result['exit']in expected,'Command rejected: '+name)
 return result['exit']

def fetch(out,repo,revision,record):
 path=out/'frozen'/record['path'];path.parent.mkdir(parents=True,exist_ok=True);K.require(not path.exists(),'No cached latest/fallback payload')
 url='https://huggingface.co/'+repo+'/resolve/'+revision+'/'+record['path']
 if record.get('dataset'):url=url.replace('huggingface.co/','huggingface.co/datasets/',1)
 partial=path.with_name(path.name+'.partial');total=0;h=hashlib.sha256();git=hashlib.sha1(('blob '+str(record['bytes'])+'\0').encode())
 with urllib.request.urlopen(url,timeout=120)as response,partial.open('xb')as f:
  while b:=response.read(4*2**20):
   total+=len(b);K.require(total<=record['bytes'],'Pinned response exceeds exact length');f.write(b);h.update(b);git.update(b)
 K.require(total==record['bytes'],'Pinned response truncated')
 if record.get('sha256'):K.require(h.hexdigest()==record['sha256'],'Pinned full-file SHA differs')
 else:K.require(git.hexdigest()==record['git_blob_oid'],'Pinned Git blob differs')
 partial.replace(path);return path

def preflight(out):
 cfg=source();K.require(sys.version_info[:2]==(3,12)and platform.machine()=='x86_64','Frozen Python3.12/x86_64 renderer required');out.mkdir(exist_ok=False)
 for name in ['review','raw','frozen','models']:(out/name).mkdir()
 ram=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
 K.require(ram>=8*2**30 and shutil.disk_usage(out).free>=16*2**30 and (os.cpu_count()or 0)>=2,'Hosted8GiB RAM/16GiB disk/2CPU admission')
 K.write(out/'review/source.json',dict(source=SOURCE,provider='Kaggle CPU',external_exact_version_service_proof_required=True,base=K.BASE,ggml=K.GGML,protocol_sha256=K.sha(K.PROFILE),protected=cfg['protected_git_objects'],successor_protected=K.declaration()['protected_git_objects'],scope='engine-only',publication_qualified=False))
 K.write(out/'review/resources.json',dict(available_ram_bytes=ram,free_disk_bytes=shutil.disk_usage(out).free,workers=2,host=dict(system=platform.system(),machine=platform.machine(),processor=platform.processor())))

def build(out):
 source();cmake=(ROOT/'CMakeLists.txt').read_text();start=cmake.index('function(crispembed_apply_arch_flags target)');end=cmake.index('endfunction()',start)+len('endfunction()')
 helper=out/'root-arch.cmake';helper.write_text(cmake[start:end]+'\n')
 K.write(out/'review/root-arch.json',dict(source='CMakeLists.txt',**K.identity(helper)))
 b=out/'build'
 command(out,'configure',['cmake','-S',ROOT/'tests/gemma-decision-q4','-B',b,'-DCMAKE_BUILD_TYPE=Release','-DROOT_ARCH_HELPER='+str(helper)])
 command(out,'build',['cmake','--build',b,'--target','gemma-q4-probe','gemma-q4-schema','gemma-q4-activation','crispembed-quantize','--parallel','2'],timeout=3600)
 cache_text=(b/'CMakeCache.txt').read_text();compiler=re.search(r'^CMAKE_CXX_COMPILER:FILEPATH=(.+)$',cache_text,re.M)
 K.require(compiler,'Actual configured compiler path');command(out,'compiler-version',[compiler[1],'--version'],timeout=60)
 K.write(out/'review/compiler.json',dict(path=compiler[1],**K.identity(compiler[1])))
 command(out,'activation',[b/'gemma-q4-activation']);command(out,'renderer',[sys.executable,ROOT/'tests/gemma_decision_rendering.py',b/'gemma-q4-schema'])
 command(out,'engine-archive-symbols',['nm','-C',b/'libgemma-audit-engine.a'],timeout=180)
 symbols=(out/'review/engine-archive-symbols.log').read_text();K.require('crispembed_diff::Ref'not in symbols,'Reference comparator leaked into engine archive')
 cache=(b/'CMakeCache.txt').read_text();K.require('GGML_LLAMAFILE:BOOL=ON'in cache and 'GGML_NATIVE:BOOL=OFF'in cache,'Exact GGML profile required')
 commands=K.read(b/'compile_commands.json');K.write(out/'review/compile_commands.json',commands)
 required=['gemma_decision.cpp','tokenizer_bpe.cpp','gguf_loader.cpp','shortest_double.c','quantize.cpp','probe.cpp']
 for name in required:K.require(sum(Path(r['file']).name==name for r in commands)==1,'Exact source TU coverage: '+name)
 K.write(out/'review/binaries.json',{name:K.identity(b/name)for name in ['gemma-q4-probe','gemma-q4-schema','gemma-q4-activation','crispembed-quantize','libgemma-audit-engine.a']})
 reference_controls(out)


def validate_fixture(fixtures):
 K.require(len(fixtures)==3 and [len(f['texts'])for f in fixtures]==[13,11,6]and sum(len(f['answers'])for f in fixtures)==8,'Complete original bank/questions')
 for f in fixtures:
  texts=f['texts'];K.require(len(set(texts))==len(texts)and all(isinstance(t,str)and t for t in texts),'Original unique bank coverage')
  K.require(len(f['text_embeds'])==len(texts)and all(len(v)==768 and all(math.isfinite(x)for x in v)for v in f['text_embeds']),'Original embeddings shape/finite')
  n=len(f['answers']);ind=f['indices'];K.require(len(f['logits'])==len(f['probabilities'])==n,'All original head observations')
  K.require(all(len(ind[k])==n for k in ['state_index','question_index','option_index','question_type']),'All original head mappings')
  for i,answer in enumerate(f['answers'].values()):
   options=list(answer['probabilities']);count=len(options)
   K.require(count>0 and len(f['logits'][i])>=count and len(f['probabilities'][i])>=count and all(math.isfinite(x)for x in [*f['logits'][i][:count],*f['probabilities'][i][:count],*answer['probabilities'].values()]),'Finite complete original head gold')
   K.require(answer['kind']in ['choice','score','noul']and len(ind['option_index'][i])>=count and ind['question_type'][i]==['choice','score','noul'].index(answer['kind']),'Original type/option mapping')
   K.require(all(type(x)is int and 0<=x<len(texts)for x in [ind['state_index'][i],ind['question_index'][i],*ind['option_index'][i][:count]]),'Original index out of range')
 return fixtures

def prepare(out):
 cfg=source();pins=cfg['inputs'];records={}
 for group in ['fixtures','notices']:
  g=pins[group]
  for record in g['files']:
   path=fetch(out,g['repo'],g['revision'],record);records[record['path']]=K.identity(path)
 refs=out/'frozen/gemma-decision/original-fp32';fixtures=validate_fixture(K.read(refs/'decisions.json'))
 import numpy as np
 K.require(np.__version__=='2.2.6','Exact locked NumPy')
 K.require(not any(name in sys.modules for name in ['torch','transformers','safetensors','gguf','sentence_transformers']),'No original framework imports')
 K.write(out/'review/environment.json',dict(python=sys.version,numpy=np.__version__,original_frameworks_imported=False,requirements=K.identity(ROOT/'tools/gemma-decision-q4-v1/requirements.lock')))
 plan=[]
 for i,fixture in enumerate(fixtures):
  folder=out/'raw/inputs'/str(i);folder.mkdir(parents=True);request=folder/'request.json';K.write(request,fixture['request'])
  rendered=out/'review'/('rendered-'+str(i)+'.json')
  command(out,'render-scenario-'+str(i),[out/'build/gemma-q4-schema',request])
  K.require(json.loads((out/'review'/('render-scenario-'+str(i)+'.log')).read_text())==fixture['texts'],'Exact original bank renderer')
  bank=[]
  for j,text in enumerate(fixture['texts']):
   path=folder/('bank-'+str(j)+'.txt');path.write_text(text,encoding='utf-8');bank.append(str(path))
  heads=[];emb=np.asarray(fixture['text_embeds'],dtype='<f4');ind=fixture['indices'];(folder/'bank-original.f32').write_bytes(emb.tobytes())
  for j,answer in enumerate(fixture['answers'].values()):
   n=len(answer['probabilities']);head=dict(options=n,kind=ind['question_type'][j])
   for label,data in [('state',emb[ind['state_index'][j]]),('question',emb[ind['question_index'][j]]),('option_embeddings',emb[ind['option_index'][j][:n]])]:
    path=folder/('head-'+str(j)+'-'+label+'.f32');path.write_bytes(np.ascontiguousarray(data,dtype='<f4').tobytes());head[label]=str(path)
   heads.append(head)
  plan.append(dict(request=str(request),bank=bank,heads=heads))
 K.write(out/'raw/inputs/typed-manifest.json',plan)
 record=pins['model']['file'];model=fetch(out,pins['model']['repo'],pins['model']['revision'],record)
 parsed=R.inventory(model);census={name:dict(type=r['type'],shape=r['stored_shape'])for name,r in parsed['tensors'].items()}
 K.write(out/'review/input-freeze.json',dict(source=SOURCE,files=records,model=K.identity(model),model_path=str(model.relative_to(out)),
  fixtures=str(refs.relative_to(out)),typed_manifest=K.identity(out/'raw/inputs/typed-manifest.json'),prepared_input_files={str(p.relative_to(out)):K.identity(p)for p in (out/'raw/inputs').rglob('*')if p.is_file()},tensor_census=census,recipes={k:R.plan(parsed['tensors'],k)for k in R.RECIPES},
  bank_sizes=[13,11,6],original_forwards=0,encoder_forwards_per_completed_model=88,decision_internal_forward_counts_source_derived=[13,11,6,13]))


def verify_inputs(out):
 source();freeze=K.read(out/'review/input-freeze.json')
 K.require(freeze['source']==SOURCE,'Immutable input/source freeze required')
 for name,pin in freeze['files'].items():K.require(K.identity(out/'frozen'/name)==pin,'Frozen reference/notice changed: '+name)
 K.require(K.identity(out/freeze['model_path'])==freeze['model']and freeze['model']==dict(bytes=R.F32_BYTES,sha256=R.F32_SHA),'Published full F32 changed')
 K.require(K.identity(out/'raw/inputs/typed-manifest.json')==freeze['typed_manifest'],'Typed request/head mapping changed')
 for name,pin in freeze['prepared_input_files'].items():K.require(K.identity(out/name)==pin,'Prepared frozen input differs: '+name)
 bins=K.read(out/'review/binaries.json')
 for name,pin in bins.items():K.require(K.identity(out/'build'/name)==pin,'Exact compiled source/binary required')
 return freeze

def independent(original,native,shape):
 import numpy as np
 a=np.asarray(original,dtype=np.float64);b=np.asarray(native,dtype=np.float64)
 K.require(a.size==b.size==math.prod(shape)and a.size>0 and np.isfinite(a).all()and np.isfinite(b).all(),'Strict finite stage extent')
 a=a.reshape(-1,shape[0]);b=b.reshape(a.shape);d=b-a;an=np.linalg.norm(a,axis=1);bn=np.linalg.norm(b,axis=1)
 na=np.sum(a*a,axis=1);nb=np.sum(b*b,axis=1);positive=(na>1e-18)&(nb>1e-18)
 cos=np.where((na<=1e-18)&(nb<=1e-18),1.,0.);cos[positive]=np.sum(a[positive]*b[positive],axis=1)/(an[positive]*bn[positive])
 cosine=float(np.min(cos));norm=float(np.linalg.norm(a));relative=float(np.linalg.norm(d)/max(norm,1e-12))
 return dict(cosine_min=cosine,relative_l2=relative,max_absolute=float(np.max(np.abs(d))),mean_absolute=float(np.mean(np.abs(d))),rms=float(np.sqrt(np.mean(d*d))),mean_error=float(np.mean(d)),
  original_l2=norm,native_l2=float(np.linalg.norm(b)),original_min=float(a.min()),original_max=float(a.max()),native_min=float(b.min()),native_max=float(b.max()),passed=cosine>=.99999 and relative<=.003)

def tensor(path,row):
 import numpy as np
 with Path(path).open('rb')as f:f.seek(row['data_offset']);data=f.read(row['bytes'])
 K.require(len(data)==row['bytes']and row['type']==0,'Exact F32 reference/stage span')
 return np.frombuffer(data,dtype='<f4')

def stage_report(out,folder,ref,code):
 K.require(code in [0,1]and(folder/'stage.json').is_file(),'Completed stage probe required')
 report=K.read(folder/'stage.json');K.require(report['completed']and report['encoder_forwards']==3 and report['observer_repeat_bit_exact']and [r['name']for r in report['stages']]==STAGES,'All29 actual Ref stages/neutrality required')
 a=R.parse(ref);b=R.parse(folder/'stages.gguf');K.require(set(a['tensors'])==set(STAGES)|{'input_ids'}and set(b['tensors'])==set(STAGES)-{'embedding'},'Complete original/native stage inventory')
 rows=[]
 for r in report['stages']:
  name=r['name'];original=a['tensors'][name];shape=original['stored_shape']
  if name=='embedding':
   import numpy as np
   K.require(shape==[768,1],'Original batch1 embedding shape');native=np.frombuffer((folder/'embedding-1.f32').read_bytes(),dtype='<f4')
  else:
   K.require(b['tensors'][name]['stored_shape']==shape,'Exact stage axis order, no reshape');native=tensor(folder/'stages.gguf',b['tensors'][name])
  m=independent(tensor(ref,original),native,shape);K.require(r['ggml_shape']==shape and r['elements']==math.prod(shape)and r['row_axis']==0,'Actual Ref exact shape/row axis')
  agreement={k:abs(m[k]-r[k])<=2e-7*max(1.,abs(m[k]),abs(r[k]))for k in m if k!='passed'}
  rows.append(dict(name=name,cpp=r,independent=m,metrics_agree=all(agreement.values()),agreement=agreement,strict_f32_passed=r['passed']and m['passed']and all(agreement.values())))
 K.require((code==0)==report['strict_f32_passed'],'Actual comparator exit matches strict old gates')
 return dict(completed=True,ids=report['ids'],stages=rows,strict_f32_passed=all(r['strict_f32_passed']for r in rows),first_failure=next((r['name']for r in rows if not r['strict_f32_passed']),None),worst_absolute=max(rows,key=lambda r:r['independent']['max_absolute'])['name'])

def typed_report(out,folder,fixtures,code):
 import numpy as np
 K.require(code==0 and(folder/'typed.json').is_file(),'Completed native typed engine panel required')
 report=K.read(folder/'typed.json');K.require(report['completed']and report['explicit_bank_forwards']==30 and report['head_on_original_calls']==16
  and report['decision_requests']==4 and report['invalid_requests']==3 and report['decision_internal_forwards_source_derived']==43,'Actual explicit/source-derived NN call scope differs')
 rows=[];head_rows=[];banks=[]
 for i,f in enumerate(fixtures):
  native=np.frombuffer((folder/('bank-'+str(i)+'.f32')).read_bytes(),dtype='<f4');reference=np.asarray(f['text_embeds'],dtype='<f4')
  bank=independent(reference,native,[768,len(f['texts'])]);banks.append(dict(scenario=i,**bank))
  for q,answer in enumerate(f['answers'].values()):
   n=len(answer['probabilities']);actual=np.frombuffer((folder/('head-'+str(i)+'-'+str(q)+'-0.f32')).read_bytes(),dtype='<f4')
   repeat=(folder/('head-'+str(i)+'-'+str(q)+'-1.f32')).read_bytes()
   K.require(actual.size==2*n and np.isfinite(actual).all()and repeat==(folder/('head-'+str(i)+'-'+str(q)+'-0.f32')).read_bytes(),'Head control extent/finite/repeat')
   logits=np.asarray(f['logits'][q][:n]);prob=np.asarray(f['probabilities'][q][:n])
   head_rows.append(dict(scenario=i,question=list(f['answers'])[q],max_logit_error=float(np.max(np.abs(actual[:n]-logits))),
    max_probability_error=float(np.max(np.abs(actual[n:]-prob))),passed=bool(np.allclose(actual[:n],logits,atol=3e-5,rtol=3e-5)and np.allclose(actual[n:],prob,atol=3e-6,rtol=3e-5))))
  try:actual=K.read(folder/('answer-'+str(i)+'.json'))
  except Exception as e:rows.append(dict(scenario=i,passed=False,error=type(e).__name__+': '+str(e)));continue
  K.require(list(actual)==list(f['answers']),'Complete ordered question IDs, no dropped decisions')
  for name,answer in f['answers'].items():
   r=actual[name];p=np.asarray(list(r['probabilities'].values()));target=np.asarray(list(answer['probabilities'].values()))
   schema=(list(r)==list(answer)and r['kind']==answer['kind']and type(r['value'])is type(answer['value'])
           and list(r['probabilities'])==list(answer['probabilities']))
   finite=np.isfinite(p).all()and abs(p.sum()-1)<2e-6 and abs(r['confidence']-p.max())<2e-6
   label=(r['label']==answer['label']and r['value']==answer['value'])
   probs=bool(np.allclose(p,target,atol=3e-4,rtol=3e-4))
   ordinal=(r['expected_level']is None)if answer['expected_level']is None else bool(np.allclose(r['expected_level'],answer['expected_level'],atol=3e-3,rtol=1e-7))
   rows.append(dict(scenario=i,question=name,native=r,original=answer,schema_exact=schema,finite_normalized=bool(finite),label_value_match=label,
    max_probability_error=float(np.max(np.abs(p-target))),expected_level_absolute_error=None if answer['expected_level']is None else abs(r['expected_level']-answer['expected_level']),
    probability_strict_f32_passed=probs,ordinal_strict_f32_passed=ordinal,passed=bool(schema and finite and label and probs and ordinal)))
 repeat=(folder/'answer-repeat.json').read_bytes()==(folder/'answer-0.json').read_bytes()
 bank_cpp=bank_reference(out,folder,fixtures,banks)
 complete=len(rows)==8 and all('question'in r for r in rows)and len(report['scenarios'])==3 and all(r['answer_nonempty']for r in report['scenarios'])
 return dict(completed=complete,scope='engine-only',bank_actual_ref=bank_cpp,questions=rows,question_count=sum('question'in r for r in rows),banks=banks,head_on_original=head_rows,
  repeat_bytes_exact=repeat,strict_f32_passed=len(rows)==8 and all(r['passed']for r in rows)and all(r['passed']for r in banks+head_rows)and bank_cpp['strict_f32_passed']and repeat,
  calls=dict(explicit_encoder_forwards=30,decision_requests=4,internal_encoder_forwards_source_derived=43,head_on_original=16),
  original_bank_token_ids_available=False,bank_native_ids_captured=True,successful_internal_forward_count_source_derived=43 if complete else None)

def model_panel(out,model,label,expected_model=None):
 freeze=verify_inputs(out);expected_model=freeze['model']if expected_model is None else expected_model;K.require(K.identity(model)==expected_model,'Selected model differs from verified conversion/publication');refs=out/freeze['fixtures'];fixtures=validate_fixture(K.read(refs/'decisions.json'));folder=out/'raw'/label;folder.mkdir()
 results=[]
 for i in range(5):
  dest=folder/('probe-'+str(i));dest.mkdir();code=command(out,label+'-probe-'+str(i),[out/'build/gemma-q4-probe','stages',model,refs/(str(i).zfill(3)+'.gguf'),dest],expected=(0,1,2),timeout=2400)
  K.require(K.identity(model)==expected_model,'Selected model changed during stage probe')
  try:
   row=stage_report(out,dest,refs/(str(i).zfill(3)+'.gguf'),code)
   K.require(row['ids']==[18,20,25,19,549][i],'Exact frozen probe ID counts')
  except Exception as e:row=dict(completed=False,strict_f32_passed=False,error=type(e).__name__+': '+str(e),exit_code=code)
  results.append(dict(probe=i,**row));K.write(out/'review'/(label+'-stage-'+str(i)+'.json'),row)
 dest=folder/'typed';dest.mkdir();code=command(out,label+'-typed',[out/'build/gemma-q4-probe','typed',model,out/'raw/inputs/typed-manifest.json',dest],expected=(0,1,2),timeout=7200)
 K.require(K.identity(model)==expected_model,'Selected model changed during typed panel')
 try:typed=typed_report(out,dest,fixtures,code)
 except Exception as e:typed=dict(completed=False,strict_f32_passed=False,error=type(e).__name__+': '+str(e),exit_code=code)
 K.write(out/'review'/(label+'-typed.json'),typed)
 complete=all(r['completed']for r in results)and typed['completed']
 passed=complete and all(r['strict_f32_passed']for r in results)and typed['strict_f32_passed']
 combined=dict(schema='crispembed-gemma-decision-q4-panel-v1',source=SOURCE,completed=complete,model=K.identity(model),label=label,
  actual_ref_stages=sum(len(r.get('stages',[]))for r in results),probes=results,typed=typed,strict_f32_passed=passed,
  completed_encoder_forwards=88 if complete else None,encoder_forward_maximum=88,original_forwards=0,
  public_capi_qualified=False,publication_qualified=False,independent_task_quality_qualified=False)
 K.write(out/'review'/(label+'-panel.json'),combined);return combined

def baseline(out):
 freeze=verify_inputs(out);model=out/freeze['model_path'];report=model_panel(out,model,'f32')
 K.write(out/'review/f32-barrier.json',dict(passed=report['strict_f32_passed'],source=SOURCE,model=K.identity(model),
  panel=K.identity(out/'review/f32-panel.json'),actual_ref_stages=report['actual_ref_stages'],typed_questions=report['typed'].get('question_count',0),
  source_inputs=K.identity(out/'review/input-freeze.json'),binaries=K.identity(out/'review/binaries.json')))
 K.require(report['completed']and report['actual_ref_stages']==145 and report['typed']['question_count']==8 and report['strict_f32_passed'],
  'Current-source full F32145/8 barrier rejected; no production-model quantization authorized')

def synthetic_quantizer(out):
 from gemma_q4_quantizer_fixture import run
 proof=run(out,command);K.write(out/'review/quantizer-fixture.json',proof)

def admit_candidates(barrier,source,panel,binaries,inputs,record):
 K.require(barrier['passed']is True and barrier['source']==source and barrier['actual_ref_stages']==145 and barrier['typed_questions']==8
  and barrier['panel']==panel and barrier['binaries']==binaries and barrier['source_inputs']==inputs,'Exact positive current-source F32 barrier required')
 K.require(record['completed']is True and record['strict_f32_passed']is True and record['actual_ref_stages']==145
  and len(record['probes'])==5 and all(r['completed']and r['strict_f32_passed']and len(r['stages'])==29 and all(s['strict_f32_passed']for s in r['stages'])for r in record['probes'])
  and record['typed']['completed']and record['typed']['strict_f32_passed']and record['typed']['question_count']==8
  and len(record['typed']['questions'])==8 and all(r['passed']for r in record['typed']['questions']),'Complete independently positive F32 panel required')

def candidates(out):
 freeze=verify_inputs(out);barrier=K.read(out/'review/f32-barrier.json')
 admit_candidates(barrier,SOURCE,K.identity(out/'review/f32-panel.json'),K.identity(out/'review/binaries.json'),K.identity(out/'review/input-freeze.json'),K.read(out/'review/f32-panel.json'))
 model=out/freeze['model_path'];schema=R.inventory(model);synthetic_quantizer(out);panels=[]
 for name in ['broad-mixed','protected-input']:
  try:
   rules=R.plan(schema['tensors'],name);target=out/'models'/(name+'.gguf')
   args=[out/'build/crispembed-quantize',model,target,'q4_k']
   for tensor,kind in rules.items():
    if kind==0 and schema['tensors'][tensor]['storage_rank']>=2:args+=['--tensor-type',tensor+'=f32']
   command(out,'quantize-'+name,args,timeout=1800)
   conversion=R.verify(model,target,name);K.write(out/'review'/(name+'-conversion.json'),conversion)
   panel=model_panel(out,target,name,dict(bytes=conversion['model_bytes'],sha256=conversion['model_sha256']));panels.append(dict(recipe=name,panel=panel,conversion=K.identity(out/'review'/(name+'-conversion.json'))))
  except Exception as e:
   failure=dict(recipe=name,completed=False,strict_f32_passed=False,error=type(e).__name__+': '+str(e),publication_qualified=False)
   K.write(out/'review'/(name+'-rejection.json'),failure);panels.append(failure)
 result=dict(schema='crispembed-gemma-decision-first-q4-result-v1',source=SOURCE,completed=all('panel'in r and r['panel']['completed']for r in panels),
  baseline=K.identity(out/'review/f32-panel.json'),recipes=panels,selected_recipe=None,publication_qualified=False,public_capi_qualified=False,
  independent_task_quality_qualified=False,original_forwards=0,maximum_encoder_forwards=264,quantized_strict_f32_flags_only=True,
  limits='Five old development probes and eight original typed answers; no new Q4 tolerance, task holdout, alias, Rust/Dart, GPU or performance qualification.')
 K.write(out/'review/result.json',result)
 K.require(result['completed'],'All recipe outcomes retained; structural rejection remains visible')
 # A completed numerical rejection remains a failed quality experiment.
 K.require(all(r['panel']['strict_f32_passed']for r in panels),'Q4 numeric/typed strict F32 flags rejected; no relaxed Q4 acceptance')

def pair_adapters(out,label,pairs):
 sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'));import diff_adapter
 folder=out/'raw/adapters';folder.mkdir(exist_ok=True)
 left,right=folder/(label+'-original.gguf'),folder/(label+'-native.gguf')
 receipt=diff_adapter.write_pair(left,right,pairs)
 target=folder/(label+'-ref');target.mkdir()
 code=command(out,label+'-ref',[out/'build/gemma-q4-probe','compare',left,right,target],expected=(0,1,2),timeout=180)
 return code,target,receipt

def reference_controls(out):
 import struct
 folder=out/'raw/ref-controls';folder.mkdir();a=folder/'original.f32';b=folder/'negative.f32'
 values=[i/16 for i in range(16)];a.write_bytes(struct.pack('<16f',*values));bad=list(values);bad[0]+=.1;b.write_bytes(struct.pack('<16f',*bad))
 codes=[]
 for label,path in [('positive',a),('negative',b)]:
  pairs=[dict(name='control',original=dict(path=str(a),shape=[2,8],sha256=K.sha(a)),native=dict(path=str(path),shape=[2,8],sha256=K.sha(path)))]
  code,target,receipt=pair_adapters(out,'control-'+label,pairs);codes.append(code)
 sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'));import diff_adapter
 wrong=folder/'wrong-shape.gguf';diff_adapter.write_gguf(wrong,[dict(name='control',path=str(a),shape=[4,4],sha256=K.sha(a))])
 dest=folder/'wrong-shape';dest.mkdir()
 codes.append(command(out,'control-shape-ref',[out/'build/gemma-q4-probe','compare',out/'raw/adapters/control-positive-original.gguf',wrong,dest],expected=(2,),timeout=180))
 K.require(codes==[0,1,2],'Actual old-budget Ref positive/numeric-negative/wrong-shape controls')
 K.write(out/'review/ref-controls.json',dict(passed=True,exit_codes=codes,minimum_row_cosine=.99999,relative_l2_budget=.003,absolute_budget=None))

def bank_reference(out,folder,fixtures,banks):
 pairs=[]
 for i,f in enumerate(fixtures):
  a=out/'raw/inputs'/str(i)/'bank-original.f32';b=folder/('bank-'+str(i)+'.f32');shape=[len(f['texts']),768]
  pairs.append(dict(name='bank_'+str(i),original=dict(path=str(a),shape=shape,sha256=K.sha(a)),native=dict(path=str(b),shape=shape,sha256=K.sha(b))))
 label=folder.parent.name+'-bank';code,target,receipt=pair_adapters(out,label,pairs)
 K.require(code in [0,1],'Bank actual Ref structural rejection')
 report=K.read(target/'comparison.json');K.require([r['name']for r in report['stages']]==['bank_0','bank_1','bank_2'],'All30 bank vectors through actual Ref')
 for i,r in enumerate(report['stages']):
  m=banks[i];K.require(r['ggml_shape']==[768,len(fixtures[i]['texts'])]and r['elements']==768*len(fixtures[i]['texts']),'Exact bank Ref shape')
  agreement={k:abs(m[k]-r[k])<=2e-7*max(1.,abs(m[k]),abs(r[k]))for k in r if k in m and isinstance(r[k],(int,float))and not isinstance(r[k],bool)}
  m.update(cpp=r,metrics_agree=all(agreement.values()),agreement=agreement,passed=m['passed']and r['passed']and all(agreement.values()))
 return dict(adapters=receipt,exit_code=code,strict_f32_passed=all(r['passed']for r in banks),rows=3,bank_vectors=30)

def main():
 global SOURCE
 p=argparse.ArgumentParser(__doc__);p.add_argument('phase',choices=['preflight','build','prepare','baseline','candidates']);p.add_argument('--output',required=True,type=Path);p.add_argument('--reviewed-source',required=True);a=p.parse_args();SOURCE=a.reviewed_source;out=a.output.resolve()
 try:globals()[a.phase](out)
 except Exception as e:
  K.write(out/'review'/('rejection-'+a.phase+'.json'),dict(source=SOURCE,phase=a.phase,completed=False,error=type(e).__name__+': '+str(e),publication_qualified=False));raise
 finally:
  if out.exists():
   files={}
   for folder in ['review','raw','models','frozen']:
    for path in (out/folder).rglob('*'):
     if path.is_file()and path.name!='file-manifest.json':files[str(path.relative_to(out))]=K.identity(path)
   K.write(out/'review/file-manifest.json',files)
if __name__=='__main__':main()
