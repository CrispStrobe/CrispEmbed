#!/usr/bin/env python3
"""Frozen Arabic old/corrected native comparison using the actual tested binary."""
import hashlib,importlib.util,json,os,platform,shutil,signal,stat,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
shared=load('q6_full',ROOT/'tools/q6-tiled-full-ab/worker.py');base=shared.base;f=shared.f;require=shared.require;sha=shared.sha
Deadline=shared.Deadline;WorkerDeadline=shared.WorkerDeadline;command=shared.command
PRODUCER='f5d0077052b907e2d674caef9e7c4ae09b08456ff48a79f327cbdfd030b2dcc9';SOURCE='8fcdf1f739fedf9ef3a068ad8028bc108ad5bc8a'
PHYSICAL=shared.PHYSICAL;CALIB=shared.CALIB
CHANGED={'CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS','CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK','CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED'}
PROFILE={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
CFG_SHA='4c2f8a65b30fe4087adfe68eeb161ae8425570fb48fb5afaf86f3fb68d90d4c3'
def admit(path):
 require(sha(path)==CFG_SHA,'immutable Arabic protocol');return json.loads(Path(path).read_text())
def gates(env):
 off=dict(env);off.update({k:'0'for k in CHANGED});on=dict(off);on.update({k:'1'for k in CHANGED});require({k for k in off if off[k]!=on[k]}==CHANGED,'exact declared three-mechanism change');return off,on
def stage(path):require(path.is_file(),'owned binary');path.chmod(path.stat().st_mode|stat.S_IXUSR)
def main():
 import numpy as np
 from PIL import Image
 cfg=admit(ROOT/'docs/lighton4b-q6-arabic-v1.json');require(np.__version__=='2.3.5'and Image.__version__=='11.3.0'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'CPU dependency profile')
 flags=next(v.split(':',1)[1].split()for v in Path('/proc/cpuinfo').read_text().splitlines()if v.startswith('flags'));require(platform.system()=='Linux'and platform.machine().lower()in['x86_64','amd64']and {'avx2','fma','f16c'}<=set(flags),'supported actual CPU')
 out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'q6-arabic';work.mkdir();started=time.monotonic();deadline=Deadline(3600);deadline.__enter__();prior={s:signal.signal(s,lambda sig,frame:(_ for _ in()).throw(SystemExit(128+sig)))for s in[signal.SIGTERM,signal.SIGINT]}
 r=dict(schema=cfg['schema'],status='failed',commands={},arms={},quality_qualified=False,heldout=False,runtime_rebuilt=False,upstream_forward_executed=False)
 def save():(out/'q6-arabic.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def run(label,args,env=None,limit=900,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,3600-(time.monotonic()-started)),r['commands'][label],env=env,cwd=ROOT,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  rows=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(rows)==1,'one immutable '+name);return rows[0],json.loads(rows[0].read_text())
 def verify(path,row):require(path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'immutable '+path.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact controller source')
  r['capacity']=dict(memory=base.memory(),scratch=shutil.disk_usage(work).free,output=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=10<<30 and r['capacity']['scratch']>=25<<30 and r['capacity']['output']>=4<<30,'resources before weights')
  pp,producer=locate('q6-tiled-full-ab.json',PRODUCER);require(producer['status']=='measured_q6_tiled_full_ab'and producer['source_sha']==SOURCE and all(v['off_all_rows_stages_exact']for v in producer['cases'].values()),'completed tested binary producer');verify(pp.parent/'test-lighton4b-physical-q6-tiled.log',producer['artifacts']['test-lighton4b-physical-q6-tiled.log']);require('PHYSICAL_Q6_TILED_CONTROLS PASS checks=38' in(pp.parent/'test-lighton4b-physical-q6-tiled.log').read_text(),'actual supported unit')
  verify(pp.parent/'CMakeCache.txt',producer['artifacts']['CMakeCache.txt']);cache=(pp.parent/'CMakeCache.txt').read_text().splitlines()
  for k,v in PROFILE.items():require(f'{k}:BOOL={v}'in cache,'actual binary profile '+k)
  bundle=out/'runtime'
  for name,row in producer['runtime_bundle'].items():
   src=pp.parent/'runtime'/name;verify(src,row);dest=bundle/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);verify(dest,row)
  native=bundle/'bin/lighton4b-native-document';stage(native);r['runtime_producer_source']=SOURCE;r['runtime_bundle']=producer['runtime_bundle'];env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(producer['cases']['simple_form']['arms']['on']['gates']);env['LD_LIBRARY_PATH']=str(bundle/'lib');off,on=gates(env)
  for k in ['CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION','CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE']:require(off[k]=='1','unchanged corrected pipeline')
  for k in ['CRISPEMBED_LIGHTON4B_HEAD_BF16','CRISPEMBED_LIGHTON4B_BF16_LINEAR','CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS']:require(off[k]=='0','unchanged quantized precision')
  try:run('cli',[native,work/'missing',work/'missing.rgb',1,1,work/'missing.ids',out,512,4,'greedy-full','--observed-composed-cap512'],off,30)
  except subprocess.CalledProcessError as e:require(e.returncode==4,'actual closed cap512 CLI')
  else:raise RuntimeError('missing model admitted')
  ph,physical=locate('physical-decode.json',PHYSICAL);cp,calib=locate('composed-vision-calibration.json',CALIB);require(physical['status']=='measured'and calib['status']=='measured_composed_vision_calibration'and calib['all_composed_merged_byte_exact'],'completed source and old native references')
  case='arabic_printed_line';folder=ph.parent/case;art=physical['cases'][case]['artifacts'];prefix=case+'/composed/';gold=calib['cases'][case]['arms']['composed'];require(len(gold['generated_ids'])==6 and gold['summary']['eos'],'old six-token EOS baseline')
  for name in ['input.rgb','original.json','retained-upstream-response.json','observer-on/source.vision.0.f32','observer-on/source.logits.0.f32','upstream_features/prompt.ids','upstream_features/modalities.ids','upstream_features/prefill.positions.f32']:verify(folder/name,art[name])
  anchors=list(shared.STAGES)+[f'greedy.{i}.logits.f32'for i in range(6)];visual=[k[len(prefix):]for k in calib['artifacts']if k.startswith(prefix+'vision.')or k.startswith(prefix+'processor.')]
  for name in anchors+visual+['prompt.ids','modalities.ids']:verify(cp.parent/prefix/name,calib['artifacts'][prefix+name])
  original=json.loads((folder/'original.json').read_text());up=json.loads((folder/'retained-upstream-response.json').read_text());reference=np.fromfile(folder/'observer-on/source.logits.0.f32',dtype='<f4');require(reference.size==248320 and np.isfinite(reference).all(),'finite actual source vocabulary');protocol=json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text());c=next(v for v in protocol['cases']if v['id']==case);image=Image.open(ROOT/c['image']).convert('RGB');require(sha(ROOT/c['image'])==c['image_sha256']and image.tobytes()==(folder/'input.rgb').read_bytes(),'same frozen RGB')
  r['actual_upstream_vs_original']=f.metric(base,up['tokens'],up['content'],original['generated_ids'],original['decoded']);r['all_inputs_admitted_before_weights']=True;save()
  sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
  def download(url,path,size,digest):
   r['phase']='download-'+Path(path).name;records=r.setdefault('downloads',{}).setdefault(Path(path).name,[]);save()
   try:shared.transfer(base.download,url,path,size,digest,records)
   finally:save()
  model=json.loads((ROOT/'docs/lighton4b-composed-form-vision-v1.json').read_text());v=model['models']['Q4_K_M'];upmodel=work/v['file'];download('https://huggingface.co/'+model['model_repo']+'/resolve/'+model['model_revision']+'/'+v['file'],upmodel,v['bytes'],v['sha256']);v=model['base_model'];protected=work/v['file'];download('https://huggingface.co/'+v['repository']+'/resolve/'+v['revision']+'/'+v['file'],protected,v['bytes'],v['sha256']);candidate=work/'candidate.gguf';audit=pack.convert(protected,upmodel,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==cfg['candidate_sha256'],'same immutable model');protected.unlink();teacher=work/'teacher.ids';teacher.write_text(str(original['generated_ids'][0])+'\n')
  for arm,child in [('off',off),('on',on)]:
   require(base.memory()['MemAvailable']>=10<<30 and shutil.disk_usage(out).free>=2<<30,'per-arm resources');dest=out/arm;dest.mkdir();checked=[0]
   def tick():
    if arm!='off':return
    n=shared.progress_count(dest)
    if n is None:return
    while checked[0]<min(6,n-1):
     j=checked[0];require(sha(dest/f'greedy.{j}.logits.f32')==calib['artifacts'][prefix+f'greedy.{j}.logits.f32']['sha256'],'early closed OFF score');checked[0]+=1
   run(arm,[native,candidate,folder/'input.rgb',image.height,image.width,teacher,dest,512,4,'greedy-full','--observed-composed-cap512'],child,900,tick)
   summary=json.loads((dest/'native-summary.json').read_text());ids=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode();require(summary['generated_tokens']==len(ids)and ids and len(ids)<=512 and summary['eos']==(ids[-1]in[248044,248046])and(summary['eos']or len(ids)==512)and not summary['source_fed_vision']and summary['teacher_positions']==1,'independent mode/count/EOS');shared.rows(dest,ids,np)
   require(summary['physical_q6_tiled']==(arm=='on')and summary['physical_q4_repack']==(arm=='on')and summary['physical_repack_entries']==(216 if arm=='on'else 0)and summary['physical_repack_bytes']<=2<<30 and bool(summary['physical_repack_bytes'])==(arm=='on')and summary['converted_f32_weight_entries']==summary['converted_f32_weight_bytes']==0,'mode/cache receipt')
   if arm=='off':
    require(ids==gold['generated_ids']and text==gold['decoded']and summary['eos'],'old complete baseline')
    for name in anchors+visual:require(sha(dest/name)==calib['artifacts'][prefix+name]['sha256'],'OFF whole stage/row closure')
    r['off_all_rows_stages_exact']=True
   else:
    for name in visual+['prompt.ids','modalities.ids','prefill.input.f32','text.embedded.f32','prefill.positions.f32']:require(sha(dest/name)==sha(out/'off'/name),'same vision/decoder inputs')
   merged=np.fromfile(dest/'vision.merged.f32',dtype='<f4');feature=np.fromfile(folder/'observer-on/source.vision.0.f32',dtype='<f4');require(merged.shape==feature.shape and merged.tobytes()==feature.tobytes(),'actual source merged identity');x=np.fromfile(dest/'prefill.logits.f32',dtype='<f4').astype('f8');d=x-reference;vs=f.metric(base,ids,text,original['generated_ids'],original['decoded']);floor=r['actual_upstream_vs_original'];r['arms'][arm]=dict(summary=summary,generated_ids=ids,decoded=text,gates={k:v for k,v in child.items()if k.startswith('CRISPEMBED_')},vs_original=vs,vs_upstream=f.metric(base,ids,text,up['tokens'],up['content']),meets_observed_upstream_floor=bool(summary['eos']and vs['character_edits']<=floor['character_edits']and vs['word_edits']<=floor['word_edits']),firstscore_vs_upstream=dict(rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),candidate_rms_magnitude=float(np.sqrt(np.mean(x*x))),reference_rms_magnitude=float(np.sqrt(np.mean(reference.astype('f8')**2)))),resources=r['commands'][arm]);save()
  paired=[]
  for layer in range(32):
   x=np.fromfile(out/'on'/f'prefill.layer.{layer}.f32',dtype='<f4');y=np.fromfile(out/'off'/f'prefill.layer.{layer}.f32',dtype='<f4');require(x.shape==y.shape and x.size and np.isfinite(x).all()and np.isfinite(y).all(),'paired finite native layers');d=x.astype('f8')-y;paired.append(dict(layer=layer,bit_exact=x.tobytes()==y.tobytes(),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),off_rms_magnitude=float(np.sqrt(np.mean(y.astype('f8')**2))),on_rms_magnitude=float(np.sqrt(np.mean(x.astype('f8')**2)))))
  r['paired_native_layers']=dict(scope='Native old/corrected; not actual upstream hidden',first_different_layer=next((v['layer']for v in paired if not v['bit_exact']),None),layers=paired)
  r['status']='measured_q6_arabic';r['limits']=['Frozen observed Arabic only; explicit three-mechanism change from old baseline.','One teacher row and full independent greedy, not full teacher parity.','No heldout or shipping qualification.']
 except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  deadline.close()
  for sig,handler in prior.items():signal.signal(sig,handler)
  r['elapsed_seconds']=time.monotonic()-started;r['artifacts']=shared.artifacts(out,r['commands']);save()
if __name__=='__main__':main()
