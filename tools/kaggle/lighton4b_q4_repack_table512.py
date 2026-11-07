#!/usr/bin/env python3
"""Frozen source-feature Q4 repack OFF/ON table; no reference/model precision change."""
import hashlib,importlib.util,io,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
f=module('fullform',ROOT/'tools/kaggle/lighton4b_physical_f16_full_form.py');require=f.require;sha=f.sha
TABLE='36a9b14be3ace197e3a30aeaec20454eb4ab58587b2566fcc859da87b08013b3';PHYSICAL='712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2';PROOF_SOURCE='5f6dfb9f09b49002201c7702ff71a5e345294af2'
def admit(cfg):
 require(cfg['schema']=='lighton4b-q4-repack-table512/v1'and cfg['case']=='simple_table'and cfg['arms']==['off','on'],'closed paired case/arms')
 require(cfg['cap']==512 and cfg['threads']==4 and cfg['context']==2048 and cfg['total_timeout_seconds']==2700 and cfg['per_arm_timeout_seconds']==900,'bounded fixed scope')
 require(cfg['table_receipt_sha256']==TABLE and cfg['controls_source_sha']==PROOF_SOURCE,'frozen reference/proof sources')
 proof=cfg.get('controls_receipt_sha256');require(isinstance(proof,str)and len(proof)==64 and all(c in '0123456789abcdef'for c in proof),'reviewed completed controls receipt required')
 require(proof=='c7130c6d7b54e6859439bbf87c8e54880c30bd5d2db33c3571ded179e6711cf5','immutable completed control digest')
 require(cfg['table_source_sha']=='f789609c2f1ff1a9527c8d02545d096ed10446b0','immutable table source')
 require(cfg['minimum_available_memory_bytes']==10<<30 and cfg['minimum_scratch_free_bytes']==25<<30 and cfg['minimum_output_free_bytes']==4<<30,'closed resource floor')
 baseline=json.loads((ROOT/'docs/lighton4b-physical-f16-full-form-v1.json').read_text())
 for key in ['model_repo','model_revision','models','base_model']:require(cfg[key]==baseline[key],'immutable source/precision '+key)
 require(cfg['driver_source_sha256']==sha(ROOT/'tests/lighton4b_native_document.cpp'),'reviewed reference-only CLI')
 require(cfg['expected_candidate_sha256']=='843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43','unchanged physical bytes')
def score_row(path,np):
 row=np.fromfile(path,dtype='<f4');require(row.size==248320 and np.isfinite(row).all(),'finite full first-score row');return row
def first_score(dest,np):
 row=score_row(dest/'prefill.logits.f32',np)
 require(sha(dest/'prefill.logits.f32')==sha(dest/'greedy.0.logits.f32')==sha(dest/'teacher.0.logits.f32'),'same actual prefill/teacher/greedy first row')
 return row
def cli_controls(native,work,out,run,env):
 base=[native,work/'absent.gguf',work/'absent.rgb',1,1,work/'absent.ids',out,256,4,'greedy-full','--upstream-vision',work/'absent.feature'];rows=[]
 for label,cap,threads,mode,flag,expected in [('legacy256',256,4,'greedy-full',None,4),('legacy257',257,4,'greedy-full',None,2),('legacy512',512,4,'greedy-full',None,2),('extended512',512,4,'greedy-full','--extended-table-cap512',4),('extended513',513,4,'greedy-full','--extended-table-cap512',2),('threads',512,2,'greedy-full','--extended-table-cap512',2),('mode',512,4,'full','--extended-table-cap512',2),('flag',512,4,'greedy-full','--unknown',2),('extended256',256,4,'greedy-full','--extended-table-cap512',2)]:
  args=list(base);args[7:10]=[cap,threads,mode]
  if flag:args.append(flag)
  try:run('scope-'+label,args,env,10)
  except subprocess.CalledProcessError as e:code=e.returncode
  else:code=0
  require(code==expected,'actual reference CLI '+label);rows.append({'case':label,'returncode':code})
 return rows

def interrupted(signum,frame):
 raise SystemExit(128+signum)

def main():
 import numpy as np
 from PIL import Image
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'Kaggle CPU only');require(np.__version__=='2.3.5'and Image.__version__=='11.3.0','serializer pins')
 cfgpath=ROOT/'docs/lighton4b-q4-repack-table512-v1.json';cfg=json.loads(cfgpath.read_text());admit(cfg)
 out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'q4-repack-table';work.mkdir();start=time.monotonic();deadline=Deadline(2700);deadline.__enter__()
 handlers={sig:signal.getsignal(sig)for sig in [signal.SIGTERM,signal.SIGINT]}
 for sig in handlers:signal.signal(sig,interrupted)
 base=module('physical',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
 r={'schema':cfg['schema'],'status':'failed','commands':{},'arms':{},'quality_qualified':False,'upstream_forward_executed':False,'original_bf16_forward_executed':False,'cap256_evidence_replaced':False,'f32_model_instantiated':False}
 def save():(out/'q4-repack-table512.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def run(label,args,env=None,limit=600,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,2700-(time.monotonic()-start)),r['commands'][label],env=env,cwd=ROOT,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  found=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(found)==1,'one exact '+name);return found[0]
 def verify(p,row):require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'immutable artifact '+p.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source');run('cpu',['lscpu'])
  r['preflight']={'memory':base.memory(),'scratch_free':shutil.disk_usage(work).free,'output_free':shutil.disk_usage(out).free};require(r['preflight']['memory']['MemAvailable']>=cfg['minimum_available_memory_bytes']and r['preflight']['scratch_free']>=cfg['minimum_scratch_free_bytes']and r['preflight']['output_free']>=cfg['minimum_output_free_bytes'],'resource admission before weights')
  proofp=locate('runtime-repack-controls.json',cfg['controls_receipt_sha256']);proof=json.loads(proofp.read_text());require(proof['status']=='passed_operator_cache_controls'and proof['source_sha']==PROOF_SOURCE and len(proof['sourcefed'])==30 and all(v['metric']['bit_exact']for v in proof['sourcefed'].values())and not proof['model_forward_executed'],'completed supported/cache/sourcefed controls');r['controls_receipt_sha256']=sha(proofp)
  # The outer bootstrap finalizes after its child: never trust its pre-final artifact row.
  proof_members=['CMakeCache.txt']+['run-'+target+'.log'for target in ['test-lighton4b-physical-q4-repack','test-lighton4b-physical-q4-repack-main-stub','test-lighton4b-physical-f16-attention','test-lighton4b-physical-native-ops','test-lighton4b-hf-tiled-out-operator','test-lighton4b-q6-tied-head','test-lighton4b-main-f16-link']]
  for name in proof_members:verify(proofp.parent/name,proof['artifacts'][name])
  require('PHYSICAL_Q4_REPACK PASS 57'in(proofp.parent/'run-test-lighton4b-physical-q4-repack.log').read_text(),'closed actual supported controls')
  for label,row in proof['sourcefed'].items():
   name=label+'.f32';verify(proofp.parent/name,proof['artifacts'][name]);require(sha(proofp.parent/name)==row['output_sha256'],'all30 actual runtime output bytes')
  r['controls_closed_members_rehashed']=38
  run('fetch-tested',['git','fetch','--depth','1','origin',PROOF_SOURCE]);run('tested-runtime',['git','diff','--exit-code',PROOF_SOURCE,'HEAD','--','src','ggml','CMakeLists.txt'])
  tp=locate('physical-f16-table512.json',TABLE);old=json.loads(tp.read_text());require(old['status']=='measured_extended_table'and old['source_sha']==cfg['table_source_sha']and old['source_observer_full_neutral']and old['old_cap256_prefix_anchors_exact'],'frozen complete table');r['table_receipt_sha256']=sha(tp)
  pp=locate('physical-decode.json',PHYSICAL);physical=json.loads(pp.read_text());require(physical['status']=='measured'and physical['source_sha']=='ac34f72ca89823a177c538ad598dfa00fc69f124','frozen physical input provenance');r['physical_receipt_sha256']=sha(pp)
  protocol=json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text());case=next(x for x in protocol['cases']if x['id']=='simple_table');folder=pp.parent/'simple_table';art=physical['cases']['simple_table']['artifacts'];gold=old['arms']['native'];require(len(gold['generated_ids'])==282 and gold['summary']['eos'],'old282EOS complete')
  for rel in ['input.rgb','original.json','upstream_features/prompt.ids']:verify(folder/rel,art[rel])
  for rel in ['upstream-off/response.json','upstream-on/source.vision.0.f32','upstream-on/source.logits.0.f32']+['native/'+x for x in f.STAGES]+[f'native/greedy.{j}.logits.f32'for j in range(282)]:verify(tp.parent/rel,old['artifacts'][rel])
  upstream=json.loads((tp.parent/'upstream-off/response.json').read_text());require(upstream['stop_type']=='eos'and len(upstream['tokens'])==282,'complete actual upstream reference');original=json.loads((folder/'original.json').read_text());require(len(original['generated_ids'])==222 and original['generated_ids'][-1]in[248044,248046],'frozen original teacher reference')
  reference_score=score_row(tp.parent/'upstream-on/source.logits.0.f32',np)
  image=Image.open(ROOT/case['image']).convert('RGB');require(sha(ROOT/case['image'])==case['image_sha256']and image.tobytes()==(folder/'input.rgb').read_bytes(),'frozen PNG/RGB');ids=list(map(int,(folder/'upstream_features/prompt.ids').read_text().split()));b=io.BytesIO();np.save(b,np.array([ids],dtype='<i8'),allow_pickle=False);require(hashlib.sha256(b.getvalue()).hexdigest()==case['expected_input_ids_npy_sha256'],'frozen actual prompt IDs')
  run('submodule',['git','submodule','update','--init','ggml']);build=work/'build';flags={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]])
  cache=(build/'CMakeCache.txt').read_text().splitlines()
  for k,v in flags.items():require(f'{k}:BOOL={v}'in cache,'exact build profile '+k)
  shutil.copyfile(build/'CMakeCache.txt',out/'CMakeCache.txt');run('build',['cmake','--build',build,'--target','lighton4b-native-document','test-lighton4b-physical-q4-repack','--parallel','2'],limit=900);run('supported',[build/'test-lighton4b-physical-q4-repack']);require('PHYSICAL_Q4_REPACK PASS 'in(out/'supported.log').read_text(),'actual supported profile required')
  native=build/'lighton4b-native-document';libraries=sorted(p for p in build.rglob('libggml*.so*')if p.is_file());require(libraries,'actual backend libraries');loader=':'.join(sorted({str(p.parent)for p in libraries}));env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(old['gates']);env['LD_LIBRARY_PATH']=loader;r['scope_controls']=cli_controls(native,work,out,run,env)
  bundle=out/'runtime';(bundle/'bin').mkdir(parents=True);(bundle/'lib').mkdir();shutil.copy2(native,bundle/'bin'/native.name)
  for library in libraries:shutil.copy2(library,bundle/'lib'/library.name)
  r['runtime_bundle']={str(p.relative_to(bundle)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in bundle.rglob('*')if p.is_file()}
  row=cfg['models']['Q4_K_M'];source=work/row['file'];r['phase']='acquire-source';save();base.download('https://huggingface.co/'+cfg['model_repo']+'/resolve/'+cfg['model_revision']+'/'+row['file'],source,row['bytes'],row['sha256']);row=cfg['base_model'];protected=work/row['file'];r['phase']='acquire-protected';save();base.download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256']);candidate=work/'candidate.gguf';audit=pack.convert(protected,source,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==cfg['expected_candidate_sha256'],'immutable transport');r['candidate_sha256']=audit['candidate_sha256'];protected.unlink()
  teacher=work/'one.ids';teacher.write_text(str(original['generated_ids'][0])+'\n');feature=tp.parent/'upstream-on/source.vision.0.f32'
  for arm in cfg['arms']:
   require(base.memory()['MemAvailable']>=cfg['minimum_available_memory_bytes'],'per-arm memory');dest=out/arm;dest.mkdir();child=dict(env);child['CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK']='1'if arm=='on'else'0';checked=[0];last=[0.]
   def tick():
    try:progress=json.loads((dest/'generated-progress.json').read_text());count=int(progress['generated_tokens'])
    except(OSError,ValueError,KeyError):return
    if arm=='off':
     while checked[0]<min(282,count-1):
      j=checked[0];require(sha(dest/f'greedy.{j}.logits.f32')==old['artifacts'][f'native/greedy.{j}.logits.f32']['sha256'],'early closed OFF row');checked[0]+=1
    if time.monotonic()-last[0]>=30:last[0]=time.monotonic();r['progress']={'arm':arm,'generated_tokens':count,'last_id':progress.get('last_id'),'off_rows_checked':checked[0]};save();print('REPACK_HEARTBEAT '+json.dumps(r['progress']),flush=True)
   run(arm,[native,candidate,folder/'input.rgb',image.height,image.width,teacher,dest,512,4,'greedy-full','--upstream-vision',feature,'--extended-table-cap512'],child,900,tick)
   summary=json.loads((dest/'native-summary.json').read_text());generated=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode('utf8');require(generated and len(generated)<=512 and summary['eos']==(generated[-1]in[248044,248046])and(summary['eos']or len(generated)==512),'independent EOS/cap')
   require(summary['teacher_positions']==1 and summary['physical_q4_repack']==(arm=='on')and summary['converted_f32_weight_bytes']==summary['converted_f32_weight_entries']==0,'actual mode/no F32 weights');require(summary['physical_repack_entries']==(216 if arm=='on'else 0)and summary['physical_repack_bytes']<=2<<30 and(summary['physical_repack_bytes']>0)==(arm=='on'),'actual bounded216Q4 cache')
   for j,token in enumerate(generated):
    row=np.fromfile(dest/f'greedy.{j}.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all()and int(row.argmax())==token,'every finite greedy row/argmax')
   for rel in ['prompt.ids','modalities.ids','vision.merged.f32','prefill.input.f32','prefill.positions.f32']:require(sha(dest/rel)==old['artifacts']['native/'+rel]['sha256'],'same actual input '+rel)
   if arm=='off':
    require(generated==gold['generated_ids']and text==gold['decoded']and summary['eos'],'OFF exact full282 IDs/text/EOS')
    for j in range(282):require(sha(dest/f'greedy.{j}.logits.f32')==old['artifacts'][f'native/greedy.{j}.logits.f32']['sha256'],'OFF all282 raw rows')
    for rel in f.STAGES:require(sha(dest/rel)==old['artifacts']['native/'+rel]['sha256'],'OFF all38 stages')
    r['off_282_rows_38_stages_ids_exact']=True
   row=first_score(dest,np);d=row.astype('f8')-reference_score;r['arms'][arm]={'summary':summary,'gates':{k:v for k,v in child.items()if k.startswith('CRISPEMBED_')},'generated_ids':generated,'decoded':text,'cap_hit':not summary['eos'],'vs_original_bf16':f.metric(base,generated,text,original['generated_ids'],original['decoded']),'vs_actual_upstream':f.metric(base,generated,text,upstream['tokens'],upstream['content']),'firstscore_vs_actual_upstream':{'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(abs(d).max()),'reference_max_abs':float(abs(reference_score).max())},'resources':r['commands'][arm]};save()
  stages=[]
  for layer in range(32):
   x=np.fromfile(out/f'on/prefill.layer.{layer}.f32',dtype='<f4');y=np.fromfile(out/f'off/prefill.layer.{layer}.f32',dtype='<f4');require(x.shape==y.shape and np.isfinite(x).all()and np.isfinite(y).all(),'finite paired layers');d=x.astype('f8')-y;stages.append({'layer':layer,'bit_exact':x.tobytes()==y.tobytes(),'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(abs(d).max()),'off_max_abs':float(abs(y).max())})
  r['paired_prefill_layers']={'scope':'Native OFF versus ON, not actual upstream hidden','first_different_layer':next((x['layer']for x in stages if not x['bit_exact']),None),'layers':stages};require(sha(candidate)==cfg['expected_candidate_sha256']and sha(source)==cfg['models']['Q4_K_M']['sha256'],'unchanged weights');candidate.unlink();r['status']='measured_same_feature_table_repack'
 except BaseException as e:r['error_type']=type(e).__name__;r['error']=str(e);raise
 finally:
  deadline.close()
  for sig,handler in handlers.items():signal.signal(sig,handler)
  r['elapsed_seconds']=time.monotonic()-start;r['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for arm in cfg['arms']for p in(out/arm).rglob('*')if p.is_file()};save()
if __name__=='__main__':main()
