#!/usr/bin/env python3
"""Observed composed form; fixed vision only after exact source component admission."""
import hashlib,importlib.util,io,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
f=load('form',ROOT/'tools/kaggle/lighton4b_physical_f16_full_form.py');require=f.require;sha=f.sha
component=load('component',ROOT/'tools/vision-profile-probe/component.py')
VISION=['vision.patch.f32','vision.position.f32','vision.embedded.f32']+[f'vision.layer.{i}.f32'for i in range(24)]+['vision.merged.f32']
GATES=['CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION']
COMPONENT='039b024f41b499cfd8344cb4e89835bbfe559a1f298225866c691000d4e6c8c0'
PRODUCTION='b59e5d3a9d9e333f67f9646c1a43a1edc70fccfa'
def admit(cfg):
 require(cfg['schema']=='lighton4b-composed-form-vision/v1'and cfg['case']=='simple_form'and cfg['arms']==['old','fixed'],'closed observed form')
 require(cfg['cap']==512 and cfg['context']==2048 and cfg['threads']==4 and cfg['worker_seconds']==3600 and cfg['arm_seconds']==1200,'closed budgets')
 require(cfg['component_source_sha']==PRODUCTION and cfg['component_receipt_sha256']==COMPONENT,'completed component required')
 require(cfg['composed_receipt_sha256']=='1458d07e6098b4b1b8fa078c18fe0b636f832a0d1219f789b035e1a033ef4ba0'and cfg['physical_receipt_sha256']=='712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2'and cfg['projector_receipt_sha256']=='427b38fd03a9d4aec47c9292f15f8771e99e263b2258f415149f0f20dd612f0c','immutable receipts')
 require(cfg['changed_gates']==GATES and cfg['old_ids']==194 and cfg['fixed_source_feature_ids']==167 and cfg['original_reference_ids']==177 and cfg['quality_qualified']is False,'fixed comparisons')
 require(cfg['source_feature_sha256']=='b9a171d18d5698e5c897662ddedf6cf399448838e55350c3b206e7a24c240d1b'and cfg['candidate_sha256']=='843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43','same features/weights')
 require(sha(ROOT/'tests/lighton4b_native_document.cpp')==cfg['driver_sha256'],'reviewed CLI bytes')
 baseline=json.loads((ROOT/'docs/lighton4b-physical-f16-full-form-v1.json').read_text())
 for key in ['model_repo','model_revision','models','base_model']:require(cfg[key]==baseline[key],'unchanged model '+key)
def gate_pair(base):
 env=dict(base);env.update({g:'0'for g in GATES});env.update(CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK='0',CRISPEMBED_LIGHTON4B_HEAD_BF16='0')
 require(env['CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION']=='1','fixed attention')
 fixed=dict(env);fixed.update({g:'1'for g in GATES});require({k for k in env if env[k]!=fixed[k]}==set(GATES),'only two gates change');return env,fixed
def admit_component(r):
 require(r['status']=='measured_same_input_projector_component'and r['source_sha']==PRODUCTION and r['legacy_anchor_exact']and r['metadata_controls_before_weights']and r['all_source_stages_byte_exact']and r['first_differing_stage']is None,'actual component exact before model')
 require(set(r['source_metrics'])==set(VISION)and all(v['bit_exact']and v['candidate_sha256']==v['reference_sha256']for v in r['source_metrics'].values()),'all28 source stage bytes required')
def first_score(dest,np):
 row=np.fromfile(dest/'prefill.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all(),'finite first row');require(sha(dest/'prefill.logits.f32')==sha(dest/'greedy.0.logits.f32')==sha(dest/'teacher.0.logits.f32'),'same actual prefill/teacher/greedy row');return row
def canceled(sig,frame):raise SystemExit(128+sig)
def main():
 import numpy as np
 from PIL import Image
 require(Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*'))and np.__version__=='2.3.5'and Image.__version__=='11.3.0','CPU/serializer profile')
 cfgpath=ROOT/'docs/lighton4b-composed-form-vision-v1.json';cfg=json.loads(cfgpath.read_text());admit(cfg);out=Path(os.environ['HEAVY_OUT']);out.mkdir(exist_ok=True);work=Path(os.environ['HEAVY_SCRATCH'])/'composed-form-vision';work.mkdir();start=time.monotonic();deadline=Deadline(3600);deadline.__enter__();prior={s:signal.signal(s,canceled)for s in [signal.SIGTERM,signal.SIGINT]}
 r=dict(schema=cfg['schema'],status='failed',commands={},arms={},quality_qualified=False,heldout=False,original_forward_executed=False,upstream_forward_executed=False,f32_model_instantiated=False)
 def save():(out/'composed-form-vision.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def run(label,args,env=None,limit=900,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,3600-(time.monotonic()-start)),r['commands'][label],env=env,cwd=ROOT,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  rows=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(rows)==1,'one exact '+name);return rows[0]
 def verify(path,meta):require(path.stat().st_size==meta['bytes']and sha(path)==meta['sha256'],'closed artifact '+path.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
  cp=locate('vision-component.json',COMPONENT);proof=json.loads(cp.read_text());admit_component(proof)
  for name,meta in proof['artifacts'].items():verify(cp.parent/name,meta)
  r['all_component_owned_members_rehashed']=len(proof['artifacts']);r['component_receipt_sha256']=COMPONENT
  pp=locate('projector-audit.json',cfg['projector_receipt_sha256']);source=json.loads(pp.read_text());require(source['live_arithmetic_closures_exact']and source['source_observer_byte_neutral'],'live source exact')
  for name,metric in proof['source_metrics'].items():
   source_path={'vision.patch.f32':'on/patch_bias.f32','vision.position.f32':'on/learned_pos.f32','vision.embedded.f32':'on/inp_pos_emb.f32','vision.merged.f32':'on/source.merged.f32'}.get(name,'on/layer_out-'+name.split('.')[2]+'.f32')
   verify(pp.parent/source_path,source['artifacts'][source_path]);verify(cp.parent/'f16'/name,proof['artifacts']['f16/'+name]);require(sha(pp.parent/source_path)==metric['reference_sha256']==sha(cp.parent/'f16'/name),'actual source component byte closure')
  fp=locate('composed-vision.json',cfg['composed_receipt_sha256']);old=json.loads(fp.read_text());require(old['status']=='measured_observed_projector_pair'and old['case']=='simple_form'and old['candidate_sha256']==cfg['candidate_sha256'],'completed composed reference')
  physicalp=locate('physical-decode.json',cfg['physical_receipt_sha256']);physical=json.loads(physicalp.read_text());require(physical['status']=='measured','complete physical inputs');folder=physicalp.parent/'simple_form';art=physical['cases']['simple_form']['artifacts']
  for name in ['input.rgb','original.json','retained-upstream-response.json','upstream_features/prompt.ids','upstream_features/modalities.ids','upstream_features/prefill.positions.f32','observer-on/source.vision.0.f32','observer-on/source.logits.0.f32']:verify(folder/name,art[name])
  require(sha(folder/'observer-on/source.vision.0.f32')==sha(pp.parent/'on/source.merged.f32')==cfg['source_feature_sha256'],'same actual source features')
  for name in ['on/source.input.rgb','on/resized.rgb','source.patches.f32']:verify(pp.parent/name,source['artifacts'][name])
  require(sha(folder/'input.rgb')==sha(pp.parent/'on/source.input.rgb'),'same actual source RGB')
  require(sha(cp.parent/'f16/processor.patches.f32')==sha(pp.parent/'source.patches.f32'),'same actual normalized patches')
  original=json.loads((folder/'original.json').read_text());upstream=json.loads((folder/'retained-upstream-response.json').read_text());require(len(original['generated_ids'])==177 and original['generated_ids'][-1]in[248044,248046]and upstream['tokens'][-1]in[248044,248046],'complete reference EOS')
  protocol=json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text());case=next(x for x in protocol['cases']if x['id']=='simple_form');require(sha(folder/'original.json')==case['reference_sha256']and sha(folder/'retained-upstream-response.json')==case['upstream_response_sha256'],'reference pins');image=Image.open(ROOT/case['image']).convert('RGB');require(sha(ROOT/case['image'])==case['image_sha256']and image.tobytes()==(folder/'input.rgb').read_bytes(),'same image RGB')
  prompt=list(map(int,(folder/'upstream_features/prompt.ids').read_text().split()));buf=io.BytesIO();np.save(buf,np.array([prompt],dtype='<i8'),allow_pickle=False);require(hashlib.sha256(buf.getvalue()).hexdigest()==case['expected_input_ids_npy_sha256'],'same prompt IDs')
  for arm,count in [('composed',194),('source-features',167)]:
   require(len(old['arms'][arm]['generated_ids'])==count and old['arms'][arm]['eos'],'frozen full decode')
   for name in list(f.STAGES)+[f'greedy.{i}.logits.f32'for i in range(count)]+(['processor.patches.f32']+VISION if arm=='composed'else[]):verify(fp.parent/arm/name,old['artifacts'][arm+'/'+name])
  base=load('capacity',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');r['capacity']=dict(memory=base.memory(),scratch_free=shutil.disk_usage(work).free,output_free=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=10<<30 and r['capacity']['scratch_free']>=25<<30 and r['capacity']['output_free']>=4<<30,'resource admission before weights')
  run('fetch-production',['git','fetch','--depth','1','origin',PRODUCTION]);run('production-bytes',['git','diff','--exit-code',PRODUCTION,'HEAD','--','src','ggml','CMakeLists.txt','tests/lighton4b_native_document.cpp']);run('submodule',['git','submodule','update','--init','ggml']);build=work/'build';flags=dict(component.controls.profile.BUILD_FLAGS);flags['GGML_CPU_REPACK']='ON'
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]]);component.controls.profile.validate_profile((build/'CMakeCache.txt').read_text().splitlines());shutil.copyfile(build/'CMakeCache.txt',out/'CMakeCache.txt');run('build',['cmake','--build',build,'--target','lighton4b-native-document','test-lighton4b-physical-f16-vision','test-lighton4b-physical-f16-attention','--parallel','2']);run('supported-vision',[build/'test-lighton4b-physical-f16-vision']);run('supported-attention',[build/'test-lighton4b-physical-f16-attention']);require('PHYSICAL_F16_VISION_SYNTHETIC PASS 'in(out/'supported-vision.log').read_text()and'PHYSICAL_F16_ATTENTION PASS 'in(out/'supported-attention.log').read_text(),'real supported graph/profile')
  libs=sorted(p for p in build.rglob('libggml*.so*')if p.is_file());require(libs,'CPU libraries');native=build/'lighton4b-native-document';env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(old['gates']);env['LD_LIBRARY_PATH']=':'.join(sorted({str(p.parent)for p in libs}));old_env,fixed_env=gate_pair(env)
  for label,args,expected in [('composed',[native,work/'missing.gguf',work/'missing.rgb',1,1,work/'missing.ids',out,512,4,'greedy-full','--observed-composed-cap512'],4),('unflagged',[native,work/'missing.gguf',work/'missing.rgb',1,1,work/'missing.ids',out,512,4,'greedy-full'],2)]:
   try:run('cli-'+label,args,old_env,30)
   except subprocess.CalledProcessError as e:require(e.returncode==expected,'actual CLI '+label)
   else:raise RuntimeError('unexpected accepted missing model')
  bundle=out/'runtime';(bundle/'bin').mkdir(parents=True);(bundle/'lib').mkdir();shutil.copy2(native,bundle/'bin'/native.name)
  for lib in libs:shutil.copy2(lib,bundle/'lib'/lib.name)
  r['runtime_bundle']={str(p.relative_to(bundle)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in bundle.rglob('*')if p.is_file()}
  sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
  row=cfg['models']['Q4_K_M'];up=work/row['file'];r['phase']='acquire-upstream';save();base.download('https://huggingface.co/'+cfg['model_repo']+'/resolve/'+cfg['model_revision']+'/'+row['file'],up,row['bytes'],row['sha256']);row=cfg['base_model'];protected=work/row['file'];r['phase']='acquire-protected';save();base.download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256']);candidate=work/'candidate.gguf';audit=pack.convert(protected,up,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==cfg['candidate_sha256'],'same physical weights');r['candidate_sha256']=audit['candidate_sha256'];protected.unlink();teacher=work/'one.ids';teacher.write_text(str(original['generated_ids'][0])+'\n');source_score=np.fromfile(folder/'observer-on/source.logits.0.f32',dtype='<f4');require(source_score.size==248320 and np.isfinite(source_score).all(),'actual source row')
  for arm,child,golden,count in [('old',old_env,'composed',194),('fixed',fixed_env,'source-features',167)]:
   require(base.memory()['MemAvailable']>=10<<30,'per-arm memory');dest=out/arm;dest.mkdir();checked=[0];last=[0.]
   def tick():
    try:count_now=int(json.loads((dest/'generated-progress.json').read_text())['generated_tokens'])
    except(OSError,ValueError,KeyError):return
    while arm=='old'and checked[0]<min(count,count_now-1):
     j=checked[0];require(sha(dest/f'greedy.{j}.logits.f32')==old['artifacts'][f'{golden}/greedy.{j}.logits.f32']['sha256'],'early frozen raw row '+arm);checked[0]+=1
    if time.monotonic()-last[0]>=30:last[0]=time.monotonic();r['progress']=dict(arm=arm,generated_tokens=count_now,closed_rows_checked=checked[0]);save();print('heartbeat',json.dumps(r['progress']),flush=True)
   run(arm,[native,candidate,folder/'input.rgb',image.height,image.width,teacher,dest,512,4,'greedy-full','--observed-composed-cap512'],child,1200,tick)
   summary=json.loads((dest/'native-summary.json').read_text());ids=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode('utf8');require(ids and len(ids)<=512 and summary['eos']==(ids[-1]in[248044,248046])and(summary['eos']or len(ids)==512)and summary['teacher_positions']==1 and not summary['source_fed_vision'],'actual composed independent decode/EOS/cap')
   for j,token in enumerate(ids):
    row=np.fromfile(dest/f'greedy.{j}.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all()and int(row.argmax())==token,'finite greedy argmax')
   row=first_score(dest,np);d=row.astype('f8')-source_score
   r['arms'][arm]=dict(summary=summary,generated_ids=ids,decoded=text,cap_hit=not summary['eos'],gates={k:v for k,v in child.items()if k.startswith('CRISPEMBED_')},vs_original_reference=f.metric(base,ids,text,original['generated_ids'],original['decoded']),vs_actual_upstream=f.metric(base,ids,text,upstream['tokens'],upstream['content']),first_score_vs_actual_upstream=dict(rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),reference_max_abs=float(abs(source_score).max())),resources=r['commands'][arm]);save()
   expected_names=list(f.STAGES)+[f'greedy.{i}.logits.f32'for i in range(count)]
   exact={name:(dest/name).is_file()and sha(dest/name)==old['artifacts'][golden+'/'+name]['sha256']for name in expected_names}
   decoded_exact=ids==old['arms'][golden]['generated_ids']and text==old['arms'][golden]['decoded']and summary['eos']
   for name in ['prompt.ids','modalities.ids','prefill.positions.f32']:require(sha(dest/name)==old['artifacts'][golden+'/'+name]['sha256'],'same text inputs '+arm)
   vision_exact={}
   for name in ['processor.patches.f32']+VISION:
    expected=old['artifacts']['composed/'+name]['sha256']if arm=='old'else proof['artifacts']['f16/'+name]['sha256'];vision_exact[name]=(dest/name).is_file()and sha(dest/name)==expected
   r['arms'][arm]['expected_text_and_rows_exact']=bool(decoded_exact and all(exact.values()));r['arms'][arm]['first_different_expected_text_artifact']=next((n for n,v in exact.items()if not v),None)
   r['arms'][arm]['all_vision_processor_anchors_exact']=all(vision_exact.values());r['arms'][arm]['first_different_vision_artifact']=next((n for n,v in vision_exact.items()if not v),None)
   r['arms'][arm]['vs_retained_source_feature_output']=f.metric(base,ids,text,old['arms']['source-features']['generated_ids'],old['arms']['source-features']['decoded']);save()
   if arm=='old':require(decoded_exact and all(exact.values())and all(vision_exact.values()),'old194/38text/29vision baseline exact before fixed')
  require(sha(candidate)==cfg['candidate_sha256']and sha(up)==cfg['models']['Q4_K_M']['sha256'],'weights unchanged');candidate.unlink();r['status']='measured_composed_form_vision';r['limits']='Observed form only; BF16 output reference, not human GT. One teacher row plus full independent greedy. No upstream/reference forward or heldout/general qualification.'
 except BaseException as e:r['error_type']=type(e).__name__;r['error']=str(e);raise
 finally:
  deadline.close()
  for s,h in prior.items():signal.signal(s,h)
  r['elapsed_seconds']=time.monotonic()-start;owned={x+'.log'for x in r['commands']}|{'CMakeCache.txt','physical-pack.json'};r['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)};save()
if __name__=='__main__':main()
