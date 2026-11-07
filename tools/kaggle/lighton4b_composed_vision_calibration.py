#!/usr/bin/env python3
"""Observed table/Arabic corrected-vision diagnostic, same actual form binary."""
import hashlib,importlib.util,io,json,os,platform,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
f=load('form_tools',ROOT/'tools/kaggle/lighton4b_physical_f16_full_form.py');require=f.require;sha=f.sha
FORM='aec0d0b10befb5b3498b55a87f9cda785998242c0d9f49a4a1b3f72a92ed160b';SOURCE='b98567cfd46aaf594f33076aaab4ff8e4610edbb'
CASES=['simple_table','arabic_printed_line'];ARMS=['source-features','composed']
def admit(cfg):
 require(cfg['schema']=='lighton4b-composed-vision-calibration/v1'and cfg['cases']==CASES and cfg['arms']==ARMS,'closed observed case order')
 require(cfg['cap']==512 and cfg['context']==2048 and cfg['threads']==4 and cfg['worker_seconds']==3600 and cfg['arm_seconds']==900,'fixed bounded execution')
 require(cfg['form_source_sha']==SOURCE and cfg['form_receipt_sha256']==FORM and cfg['physical_receipt_sha256']=='712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2'and cfg['extended_table_receipt_sha256']=='36a9b14be3ace197e3a30aeaec20454eb4ab58587b2566fcc859da87b08013b3'and cfg['calibration_receipt_sha256']=='7639d6559da7cc58443ccfbe1dac518af0d70f3e7a83718d037a97bb147eb2b7','immutable completed references')
 require(cfg['frozen_source_feature_ids']==dict(simple_table=282,arabic_printed_line=6)and cfg['quality_qualified']is False,'complete frozen scope')
 baseline=json.loads((ROOT/'docs/lighton4b-composed-form-vision-v1.json').read_text())
 for key in ['model_repo','model_revision','models','base_model','candidate_sha256']:require(cfg[key]==baseline[key],'unchanged model '+key)
def admit_form(form):
 require(form['status']=='measured_composed_form_vision'and form['source_sha']==SOURCE and form['all_component_owned_members_rehashed']==73,'actual completed form')
 for arm,count in [('old',194),('fixed',167)]:require(form['arms'][arm]['expected_text_and_rows_exact']and form['arms'][arm]['all_vision_processor_anchors_exact']and form['arms'][arm]['summary']['eos']and not form['arms'][arm]['summary']['source_fed_vision']and len(form['arms'][arm]['generated_ids'])==count,'complete pixel-only form anchors')
 gates=form['arms']['fixed']['gates'];require(all(gates[k]=='1'for k in ['CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION'])and all(gates[k]=='0'for k in ['CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK','CRISPEMBED_LIGHTON4B_HEAD_BF16','CRISPEMBED_LIGHTON4B_BF16_LINEAR','CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS']),'frozen corrected vision/text gates')
def cpu_admit(system,machine,flags):require(system=='Linux'and machine.lower()in['x86_64','amd64']and {'avx2','fma','f16c'}<=set(flags),'supported private ABI host')
def args(native,candidate,rgb,h,w,teacher,dest,arm,feature):
 require(arm in ARMS,'closed arm');a=[native,candidate,rgb,h,w,teacher,dest,512,4,'greedy-full']
 if arm=='source-features':a+=['--upstream-vision',feature]
 return a+['--observed-composed-cap512']
def vector(a,b,np):
 x=np.fromfile(a,dtype='<f4');y=np.fromfile(b,dtype='<f4');require(x.size>0 and x.shape==y.shape and np.isfinite(x).all()and np.isfinite(y).all(),'same finite vectors');d=x.astype('f8')-y;return dict(bit_exact=x.tobytes()==y.tobytes(),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),reference_max_abs=float(abs(y).max()),candidate_sha256=sha(a),reference_sha256=sha(b))
def canceled(sig,frame):raise SystemExit(128+sig)
def main():
 import numpy as np
 from PIL import Image
 require(Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*'))and np.__version__=='2.3.5'and Image.__version__=='11.3.0','Kaggle CPU/serializer pins')
 flags=next(x.split(':',1)[1].split()for x in Path('/proc/cpuinfo').read_text().splitlines()if x.startswith('flags'));cpu_admit(platform.system(),platform.machine(),flags)
 cfgpath=ROOT/'docs/lighton4b-composed-vision-calibration-v1.json';cfg=json.loads(cfgpath.read_text());admit(cfg);out=Path(os.environ['HEAVY_OUT']);out.mkdir(exist_ok=True);work=Path(os.environ['HEAVY_SCRATCH'])/'composed-calibration';work.mkdir();started=time.monotonic();deadline=Deadline(3600);deadline.__enter__();prior={s:signal.signal(s,canceled)for s in [signal.SIGTERM,signal.SIGINT]}
 r=dict(schema=cfg['schema'],status='failed',commands={},cases={},quality_qualified=False,heldout=False,original_forward_executed=False,upstream_forward_executed=False,runtime_rebuilt=False,f32_model_instantiated=False)
 def save():(out/'composed-vision-calibration.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def run(label,argv,env=None,limit=600,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(argv,out/(label+'.log'),min(limit,3600-(time.monotonic()-started)),r['commands'][label],env=env,cwd=ROOT,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  found=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(found)==1,'one exact '+name);return found[0]
 def verify(p,meta):require(p.stat().st_size==meta['bytes']and sha(p)==meta['sha256'],'immutable '+p.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source');run('cpu',['lscpu'])
  base=load('physical_base',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');r['capacity']=dict(memory=base.memory(),scratch_free=shutil.disk_usage(work).free,output_free=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=10<<30 and r['capacity']['scratch_free']>=25<<30 and r['capacity']['output_free']>=4<<30,'resource admission before weights')
  fp=locate('composed-form-vision.json',FORM);form=json.loads(fp.read_text());admit_form(form);r['actual_form_receipt_sha256']=FORM;r['runtime_producer_source_sha']=SOURCE
  verify(fp.parent/'CMakeCache.txt',form['artifacts']['CMakeCache.txt']);cache=(fp.parent/'CMakeCache.txt').read_text().splitlines()
  for key,value in {'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_OPENMP':'OFF','GGML_BLAS':'OFF','GGML_CPU_REPACK':'ON','GGML_AVX512':'OFF'}.items():require(f'{key}:BOOL={value}'in cache,'actual producer profile '+key)
  bundle=out/'runtime'
  for name,meta in form['runtime_bundle'].items():
   source=fp.parent/'runtime'/name;verify(source,meta);dest=bundle/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);verify(dest,meta)
  r['runtime_bundle']=form['runtime_bundle'];native=bundle/'bin/lighton4b-native-document';require(native.is_file(),'actual producer native executable')
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(form['arms']['fixed']['gates']);env['LD_LIBRARY_PATH']=str(bundle/'lib');r['gates']=form['arms']['fixed']['gates']
  for arm in ARMS:
   try:run('cli-'+arm,args(native,work/'missing.gguf',work/'missing.rgb',1,1,work/'missing.ids',out,arm,work/'missing.feature'),env,30)
   except subprocess.CalledProcessError as e:require(e.returncode==4,'actual same-binary CLI '+arm)
   else:raise RuntimeError('missing-model CLI unexpectedly passed')
  pp=locate('physical-decode.json',cfg['physical_receipt_sha256']);physical=json.loads(pp.read_text());require(physical['status']=='measured','complete input receipt');ep=locate('physical-f16-table512.json',cfg['extended_table_receipt_sha256']);extended=json.loads(ep.read_text());require(extended['status']=='measured_extended_table'and extended['old_cap256_prefix_anchors_exact']and extended['source_observer_full_neutral'],'complete table512 reference');cp=locate('physical-f16-calibration.json',cfg['calibration_receipt_sha256']);calib=json.loads(cp.read_text());require(calib['status']=='measured_two_case_calibration','complete Arabic reference');protocol=json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text());prepared=[]
  for name in CASES:
   case=next(x for x in protocol['cases']if x['id']==name);folder=pp.parent/name;art=physical['cases'][name]['artifacts']
   for rel in ['input.rgb','original.json','retained-upstream-response.json','upstream_features/prompt.ids','upstream_features/modalities.ids','upstream_features/prefill.positions.f32','observer-on/source.vision.0.f32','observer-on/source.logits.0.f32']:verify(folder/rel,art[rel])
   image=Image.open(ROOT/case['image']).convert('RGB');require(sha(ROOT/case['image'])==case['image_sha256']and image.tobytes()==(folder/'input.rgb').read_bytes(),'same RGB '+name);prompt=list(map(int,(folder/'upstream_features/prompt.ids').read_text().split()));b=io.BytesIO();np.save(b,np.array([prompt],dtype='<i8'),allow_pickle=False);require(hashlib.sha256(b.getvalue()).hexdigest()==case['expected_input_ids_npy_sha256'],'same prompt '+name)
   original=json.loads((folder/'original.json').read_text());require(sha(folder/'original.json')==case['reference_sha256']and original['generated_ids'][-1]in[248044,248046],'original output reference');upstream=json.loads((folder/'retained-upstream-response.json').read_text());require(sha(folder/'retained-upstream-response.json')==case['upstream_response_sha256'],'retained source reference')
   if name=='simple_table':
    gp=ep.parent/'native';ga=extended['artifacts'];prefix='native/';gold=extended['arms']['native'];verify(ep.parent/'upstream-off/response.json',extended['artifacts']['upstream-off/response.json']);upstream=json.loads((ep.parent/'upstream-off/response.json').read_text())
   else:gp=cp.parent/name;ga=calib['artifacts'];prefix=name+'/';gold=calib['cases'][name]
   count=cfg['frozen_source_feature_ids'][name];require(len(gold['generated_ids'])==count and gold['summary']['eos']and upstream['tokens'][-1]in[248044,248046],'complete golden/source EOS')
   for rel in list(f.STAGES)+[f'greedy.{i}.logits.f32'for i in range(count)]:verify(gp/rel,ga[prefix+rel])
   source_row=np.fromfile(folder/'observer-on/source.logits.0.f32',dtype='<f4');require(source_row.size==248320 and np.isfinite(source_row).all(),'finite actual source row');prepared.append(dict(name=name,case=case,folder=folder,art=art,image=image,original=original,upstream=upstream,golden=gold,golden_path=gp,golden_art=ga,prefix=prefix,count=count,source_row=source_row))
  r['all_cases_authenticated_before_weights']=True;save()
  sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
  row=cfg['models']['Q4_K_M'];up=work/row['file'];r['phase']='acquire-upstream';save();base.download('https://huggingface.co/'+cfg['model_repo']+'/resolve/'+cfg['model_revision']+'/'+row['file'],up,row['bytes'],row['sha256']);row=cfg['base_model'];protected=work/row['file'];r['phase']='acquire-protected';save();base.download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256']);candidate=work/'candidate.gguf';audit=pack.convert(protected,up,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==cfg['candidate_sha256'],'same immutable candidate');r['candidate_sha256']=audit['candidate_sha256'];protected.unlink()
  for item in prepared:
   name=item['name'];folder=item['folder'];feature=folder/'observer-on/source.vision.0.f32';teacher=work/(name+'.ids');teacher.write_text(str(item['original']['generated_ids'][0])+'\n');r['cases'][name]=dict(arms={},reference_scope='Original BF16 output reference, not human GT; upstream table complete282EOS retained.',upstream_vision_stage_limit='Actual merged feature reference only for these cases; internal stages not captured.',actual_upstream_vs_original=f.metric(base,item['upstream']['tokens'],item['upstream']['content'],item['original']['generated_ids'],item['original']['decoded']))
   for arm in ARMS:
    require(base.memory()['MemAvailable']>=10<<30 and shutil.disk_usage(out).free>=2<<30,'per-arm capacity');dest=out/name/arm;dest.mkdir(parents=True);checked=[0];last=[0.]
    def tick():
     try:count_now=int(json.loads((dest/'generated-progress.json').read_text())['generated_tokens'])
     except(OSError,ValueError,KeyError):return
     while arm=='source-features'and checked[0]<min(item['count'],count_now-1):
      j=checked[0];require(sha(dest/f'greedy.{j}.logits.f32')==item['golden_art'][item['prefix']+f'greedy.{j}.logits.f32']['sha256'],'early closed source-feature row');checked[0]+=1
     if time.monotonic()-last[0]>=30:last[0]=time.monotonic();r['progress']=dict(case=name,arm=arm,generated_tokens=count_now,closed_rows=checked[0]);save();print('heartbeat',json.dumps(r['progress']),flush=True)
    label=name+'-'+arm;run(label,args(native,candidate,folder/'input.rgb',item['image'].height,item['image'].width,teacher,dest,arm,feature),env,900,tick)
    summary=json.loads((dest/'native-summary.json').read_text());ids=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode('utf8');require(ids and len(ids)<=512 and summary['eos']==(ids[-1]in[248044,248046])and(summary['eos']or len(ids)==512)and summary['teacher_positions']==1 and summary['source_fed_vision']==(arm=='source-features')and not summary['physical_q4_repack'],'actual independent mode/EOS/cap')
    for j,token in enumerate(ids):
     row=np.fromfile(dest/f'greedy.{j}.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all()and int(row.argmax())==token,'finite greedy argmax')
    row=np.fromfile(dest/'prefill.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all()and sha(dest/'prefill.logits.f32')==sha(dest/'teacher.0.logits.f32')==sha(dest/'greedy.0.logits.f32'),'finite same actual first row')
    for rel in ['prompt.ids','modalities.ids','prefill.positions.f32']:require(sha(dest/rel)==item['art']['upstream_features/'+rel]['sha256'],'same actual prompt/positions')
    merged=vector(dest/'vision.merged.f32',feature,np);delta=row.astype('f8')-item['source_row'];exact={rel:(dest/rel).is_file()and sha(dest/rel)==item['golden_art'][item['prefix']+rel]['sha256']for rel in list(f.STAGES)+[f'greedy.{i}.logits.f32'for i in range(item['count'])]};decoded_exact=ids==item['golden']['generated_ids']and text==item['golden']['decoded']and summary['eos'];vsoriginal=f.metric(base,ids,text,item['original']['generated_ids'],item['original']['decoded']);floor=r['cases'][name]['actual_upstream_vs_original']
    result=dict(summary=summary,generated_ids=ids,decoded=text,cap_hit=not summary['eos'],vs_original_reference=vsoriginal,vs_actual_upstream=f.metric(base,ids,text,item['upstream']['tokens'],item['upstream']['content']),merged_vs_actual_source=merged,expected_source_feature_text_rows_exact=bool(decoded_exact and all(exact.values())),source_feature_parity_interpretable=merged['bit_exact'],first_different_expected_text_artifact=next((x for x,v in exact.items()if not v),None),first_score_vs_actual_upstream=dict(rms=float(np.sqrt(np.mean(delta*delta))),max_abs=float(abs(delta).max()),reference_max_abs=float(abs(item['source_row']).max())),meets_observed_output_floor=bool(summary['eos']and vsoriginal['character_edits']<=floor['character_edits']and vsoriginal['word_edits']<=floor['word_edits']),resources=r['commands'][label]);r['cases'][name]['arms'][arm]=result;save()
    if arm=='source-features':require(decoded_exact and all(exact.values())and merged['bit_exact'],'frozen complete source-feature baseline before composed')
    else:
     for rel in ['processor.patches.f32','processor.grid.ids']:require(sha(dest/rel)==sha(out/name/'source-features'/rel),'same corrected preprocessing both arms')
   r['cases'][name]['composed_matches_source_feature_parity']=bool(r['cases'][name]['arms']['composed']['source_feature_parity_interpretable']and r['cases'][name]['arms']['composed']['expected_source_feature_text_rows_exact']);save()
  r['all_observed_output_floors_met']=all(r['cases'][name]['arms']['composed']['meets_observed_output_floor']for name in CASES)
  r['all_composed_merged_byte_exact']=all(r['cases'][name]['arms']['composed']['merged_vs_actual_source']['bit_exact']for name in CASES)
  require(sha(candidate)==cfg['candidate_sha256']and sha(up)==cfg['models']['Q4_K_M']['sha256'],'unchanged weights');candidate.unlink();r['status']='measured_composed_vision_calibration';r['limits']='Observed cases only; one teacher row plus full independent greedy. Actual merged byte identity required for source-feature text parity interpretation. No humanGT/heldout/general shipping claim.'
 except BaseException as e:r['error_type']=type(e).__name__;r['error']=str(e);raise
 finally:
  deadline.close()
  for s,h in prior.items():signal.signal(s,h)
  r['elapsed_seconds']=time.monotonic()-started;owned={x+'.log'for x in r['commands']}|{'physical-pack.json'};r['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)};save()
if __name__=='__main__':main()
