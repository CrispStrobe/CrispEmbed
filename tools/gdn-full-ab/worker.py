#!/usr/bin/env python3
"""Frozen observed form/table GDN OFF/ON; no reference forward or qualification."""
import hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,WorkerDeadline,command
GATE='CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS'
PROOF='6b1f4f9c3150d06df0a195c1f21c42136874e69934b9586205712114354e798d'
SOURCE='cd50e9e691530525fb998edda8a449cc0725be91'
PHYSICAL='712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2'
TABLE='36a9b14be3ace197e3a30aeaec20454eb4ab58587b2566fcc859da87b08013b3'
FORM='aec0d0b10befb5b3498b55a87f9cda785998242c0d9f49a4a1b3f72a92ed160b'
CALIB='f69d23ffe6d65d9abf6c7c5c660723613ce84cc901383c5151e17d8f8dc5fe64'
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=module('physical',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py')
f=module('fullform',ROOT/'tools/kaggle/lighton4b_physical_f16_full_form.py')
require=base.require;sha=base.sha
STAGES=list(f.STAGES)+['teacher.0.logits.f32']
CFG_SHA='37bb2ff725ffcfd8c5a1a3acc58e09136856a710d80436456e7ff67b7c9cae63'
def admit_protocol(path):
 require(sha(path)==CFG_SHA,'immutable frozen protocol');return json.loads(Path(path).read_text())
def gates(env):
 off=dict(env);off[GATE]='0';on=dict(off);on[GATE]='1'
 require({k for k in off if off[k]!=on[k]}=={GATE},'one gate difference');return off,on
def rows(dest,ids,np):
 for i,token in enumerate(ids):
  p=dest/f'greedy.{i}.logits.f32';x=np.fromfile(p,dtype='<f4');require(x.size==248320 and np.isfinite(x).all()and int(x.argmax())==token,'complete finite greedy argmax');
 require(sha(dest/'prefill.logits.f32')==sha(dest/'teacher.0.logits.f32')==sha(dest/'greedy.0.logits.f32'),'same first vocabulary row')
def artifacts(out,commands):
 owned={label+'.log'for label in commands}|{'CMakeCache.txt','physical-pack.json'}
 return {str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)}
def transfer(fetch,url,path,size,digest,records):
 require(not Path(path).exists(),'fresh owned download destination')
 for attempt in range(1,4):
  row=dict(attempt=attempt,expected_bytes=size);records.append(row)
  try:fetch(url,path,size,digest);row['status']='authenticated';return
  except WorkerDeadline:row['status']='worker_deadline';raise
  except (TimeoutError,ConnectionError,urllib.error.URLError)as e:
   row.update(status='transport_failure',error_type=type(e).__name__,partial_bytes=Path(path).stat().st_size if Path(path).exists()else 0)
   if Path(path).exists():Path(path).unlink()
   if attempt==3:raise RuntimeError('bounded authenticated transfer failed: '+Path(path).name)from None
def main():
 import numpy as np
 require(np.__version__=='2.3.5'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'pinned Kaggle CPU')
 cfg=admit_protocol(ROOT/'docs/lighton4b-gdn-full-ab-v1.json');require(cfg['proof_receipt']==PROOF and cfg['production_source']==SOURCE and cfg['cases']==['simple_table','simple_form']and cfg['cap']==512 and cfg['threads']==4,'closed scope')
 out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'gdn-full-ab';work.mkdir();start=time.monotonic();deadline=Deadline(6000);deadline.__enter__();prior={s:signal.signal(s,lambda sig,frame:(_ for _ in()).throw(SystemExit(128+sig)))for s in[signal.SIGTERM,signal.SIGINT]}
 r=dict(schema='lighton4b-gdn-full-ab/v1',status='failed',cases={},commands={},quality_qualified=False,heldout=False,reference_forward_executed=False)
 def save():(out/'gdn-full-ab.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def run(label,args,env=None,limit=900,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,6000-(time.monotonic()-start)),r['commands'][label],env=env,cwd=ROOT,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  found=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(found)==1,'one frozen '+name);return found[0],json.loads(found[0].read_text())
 def verify(path,row):require(path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'frozen artifact '+path.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
  r['capacity']=dict(memory=base.memory(),scratch=shutil.disk_usage(work).free,output=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=10<<30 and r['capacity']['scratch']>=25<<30 and r['capacity']['output']>=6<<30,'pre-model resource admission')
  pp,proof=locate('gdn-hybrid-probe.json',PROOF);require(proof['status']=='measured_actual_source_hybrid_controls'and proof['source_sha']==SOURCE and not proof['model_forward_executed'],'actual hybrid controls')
  for name,row in proof['artifacts'].items():verify(pp.parent/name,row)
  require(all(x['bit_exact']for x in proof['actual_live_source_closures'].values())and all(x['bit_exact']for role in['zab4','continuation1']for x in proof['hybrid_same_input'][role].values()),'source and nonzero state closure')
  physicalp,physical=locate('physical-decode.json',PHYSICAL);tablep,table=locate('physical-f16-table512.json',TABLE);formp,form=locate('composed-form-vision.json',FORM);calibp,calib=locate('composed-vision-calibration.json',CALIB)
  require(table['status']=='measured_extended_table'and table['source_observer_full_neutral']and form['arms']['fixed']['expected_text_and_rows_exact']and calib['all_composed_merged_byte_exact'],'completed baselines')
  prepared=[]
  for case,count,receiptp,receipt,prefix,gold in [('simple_table',282,calibp,calib,'simple_table/composed',calib['cases']['simple_table']['arms']['composed']),('simple_form',167,formp,form,'fixed',form['arms']['fixed'])]:
   inp=physicalp.parent/case;art=physical['cases'][case]['artifacts']
   for name in['input.rgb','original.json']:verify(inp/name,art[name])
   original=json.loads((inp/'original.json').read_text());upath=tablep.parent/'upstream-off/response.json'if case=='simple_table'else inp/'retained-upstream-response.json';meta=table['artifacts']['upstream-off/response.json']if case=='simple_table'else art['retained-upstream-response.json'];verify(upath,meta);up=json.loads(upath.read_text())
   require(len(gold['generated_ids'])==count and gold['summary']['eos'],'complete frozen native EOS')
   anchors=STAGES+[f'greedy.{j}.logits.f32'for j in range(count)];visual=[k[len(prefix)+1:]for k in receipt['artifacts']if k.startswith(prefix+'/vision.')or k.startswith(prefix+'/processor.')];anchors+=visual
   for name in anchors+['prompt.ids','modalities.ids']:verify(receiptp.parent/prefix/name,receipt['artifacts'][prefix+'/'+name])
   scorepath=tablep.parent/'upstream-on/source.logits.0.f32'if case=='simple_table'else inp/'observer-on/source.logits.0.f32';scoremeta=table['artifacts']['upstream-on/source.logits.0.f32']if case=='simple_table'else art['observer-on/source.logits.0.f32'];verify(scorepath,scoremeta);reference=np.fromfile(scorepath,dtype='<f4');require(reference.size==248320 and np.isfinite(reference).all(),'finite full upstream reference row')
   prepared.append((case,count,receiptp,receipt,prefix,gold,inp,original,up,anchors,visual,reference))
  run('fetch-tested',['git','fetch','--depth','1','origin',SOURCE]);run('tested-bytes',['git','diff','--exit-code',SOURCE,'HEAD','--','src','ggml','CMakeLists.txt','tests/lighton4b_native_document.cpp']);run('submodule',['git','submodule','update','--init','ggml'])
  build=work/'build';flags={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]]);cache=(build/'CMakeCache.txt').read_text().splitlines()
  for k,v in flags.items():require(f'{k}:BOOL={v}'in cache,'matched native profile '+k)
  shutil.copyfile(build/'CMakeCache.txt',out/'CMakeCache.txt');run('build',['cmake','--build',build,'--target','lighton4b-native-document','test-lighton4b-physical-gdn-source-ops','test-lighton4b-physical-f16-attention','--parallel','2'],limit=1200)
  for target,marker in [('test-lighton4b-physical-gdn-source-ops','PHYSICAL_GDN_SOURCE_OPS_CONTROLS PASS'),('test-lighton4b-physical-f16-attention','PHYSICAL_F16_ATTENTION PASS')]:run(target,[build/target]);require(marker in(out/(target+'.log')).read_text(),'actual supported controls')
  native=build/'lighton4b-native-document';libs=sorted(p for p in build.rglob('libggml*.so*')if p.is_file());require(libs,'backend libraries');env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(form['arms']['fixed']['gates']);env['LD_LIBRARY_PATH']=':'.join(sorted({str(p.parent)for p in libs}));off,on=gates(env)
  for k in['CRISPEMBED_LIGHTON4B_HEAD_BF16','CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK','CRISPEMBED_LIGHTON4B_BF16_LINEAR','CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS']:require(off[k]=='0','unchanged text quant precision')
  require(all(off[k]=='1'for k in['CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION','CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION']),'unchanged corrected pipeline')
  for flag,expected in[('--observed-composed-cap512',4),('--unknown',2)]:
   try:run('cli-'+flag,[native,work/'missing',work/'missing.rgb',1,1,work/'missing.ids',out,512,4,'greedy-full',flag],off,30)
   except subprocess.CalledProcessError as e:require(e.returncode==expected,'closed actual CLI')
   else:raise ValueError('missing model admitted')
  bundle=out/'runtime';(bundle/'bin').mkdir(parents=True);(bundle/'lib').mkdir();shutil.copy2(native,bundle/'bin'/native.name)
  for lib in libs:shutil.copy2(lib,bundle/'lib'/lib.name)
  r['runtime_bundle']={str(p.relative_to(bundle)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in bundle.rglob('*')if p.is_file()};save()
  sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
  def download(url,path,size,digest):
   r['phase']='download-'+Path(path).name;records=r.setdefault('downloads',{}).setdefault(Path(path).name,[]);save()
   try:transfer(base.download,url,path,size,digest,records)
   finally:save()
  modelcfg=json.loads((ROOT/'docs/lighton4b-composed-form-vision-v1.json').read_text());row=modelcfg['models']['Q4_K_M'];upmodel=work/row['file'];download('https://huggingface.co/'+modelcfg['model_repo']+'/resolve/'+modelcfg['model_revision']+'/'+row['file'],upmodel,row['bytes'],row['sha256']);row=modelcfg['base_model'];protected=work/row['file'];download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256']);candidate=work/'candidate.gguf';audit=pack.convert(protected,upmodel,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==cfg['candidate_sha256'],'unchanged immutable model');protected.unlink()
  for case,count,receiptp,receipt,prefix,gold,inp,original,up,anchors,visual,reference in prepared:
   # Dimensions authenticated by the frozen source protocol, never inferred from outputs.
   sourcecfg=json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text());c=next(c for c in sourcecfg['cases']if c['id']==case);from PIL import Image
   image=Image.open(ROOT/c['image']).convert('RGB');require(sha(ROOT/c['image'])==c['image_sha256']and image.tobytes()==(inp/'input.rgb').read_bytes(),'same frozen RGB')
   teacher=work/(case+'.ids');teacher.write_text(str(original['generated_ids'][0])+'\n');r['cases'][case]={'arms':{}}
   for arm,child in [('off',off),('on',on)]:
    require(base.memory()['MemAvailable']>=10<<30 and shutil.disk_usage(out).free>=2<<30,'per-arm capacity');dest=out/case/arm;dest.mkdir(parents=True);checked=[0]
    def tick():
     if arm!='off':return
     try:n=int(json.loads((dest/'generated-progress.json').read_text())['generated_tokens'])
     except(OSError,ValueError,KeyError):return
     while checked[0]<min(count,n-1):
      j=checked[0];require(sha(dest/f'greedy.{j}.logits.f32')==receipt['artifacts'][prefix+f'/greedy.{j}.logits.f32']['sha256'],'early closed OFF score');checked[0]+=1
    label=case+'-'+arm;run(label,[native,candidate,inp/'input.rgb',image.height,image.width,teacher,dest,512,4,'greedy-full','--observed-composed-cap512'],child,900,tick)
    summary=json.loads((dest/'native-summary.json').read_text());ids=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode();require(summary['generated_tokens']==len(ids)and ids and len(ids)<=512 and summary['eos']==(ids[-1]in[248044,248046])and(summary['eos']or len(ids)==512)and not summary['source_fed_vision']and summary['teacher_positions']==1,'independent EOS/cap/mode');require(summary['converted_f32_weight_bytes']==0 and not summary['physical_q4_repack'],'no alternate weight cache');rows(dest,ids,np)
    if arm=='off':
     require(ids==gold['generated_ids']and text==gold['decoded']and summary['eos'],'OFF complete baseline IDs/text/EOS')
     for name in anchors:require(sha(dest/name)==receipt['artifacts'][prefix+'/'+name]['sha256'],'OFF full row/stage closure')
     r['cases'][case]['off_all_rows_stages_exact']=True
    else:
     for name in visual+['prompt.ids','modalities.ids','text.embedded.f32','prefill.input.f32','prefill.positions.f32']:require(sha(dest/name)==sha(out/case/'off'/name),'unchanged vision/decoder input')
    d=np.fromfile(dest/'prefill.logits.f32',dtype='<f4').astype('f8')-reference
    r['cases'][case]['arms'][arm]=dict(firstscore_vs_actual_upstream=dict(rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),reference_max_abs=float(abs(reference).max()),candidate_rms_magnitude=float(np.sqrt(np.mean(np.fromfile(dest/'prefill.logits.f32',dtype='<f4').astype('f8')**2))),reference_rms_magnitude=float(np.sqrt(np.mean(reference.astype('f8')**2)))),summary=summary,gates={k:v for k,v in child.items()if k.startswith('CRISPEMBED_')},generated_ids=ids,decoded=text,vs_original=f.metric(base,ids,text,original['generated_ids'],original['decoded']),vs_upstream=f.metric(base,ids,text,up['tokens'],up['content']),resources=r['commands'][label]);save()
   paired=[]
   for layer in range(32):
    x=np.fromfile(out/case/'on'/f'prefill.layer.{layer}.f32',dtype='<f4');y=np.fromfile(out/case/'off'/f'prefill.layer.{layer}.f32',dtype='<f4');require(x.shape==y.shape and np.isfinite(x).all()and np.isfinite(y).all(),'finite paired layer');d=x.astype('f8')-y;paired.append(dict(layer=layer,bit_exact=x.tobytes()==y.tobytes(),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),off_max_abs=float(abs(y).max()),off_rms_magnitude=float(np.sqrt(np.mean(y.astype('f8')**2))),on_rms_magnitude=float(np.sqrt(np.mean(x.astype('f8')**2)))))
   r['cases'][case]['paired_native_layers']=dict(scope='Native OFF/ON; not actual upstream hidden',layers=paired,first_different_layer=next((v['layer']for v in paired if not v['bit_exact']),None));save()
  require(sha(candidate)==cfg['candidate_sha256'],'weights unchanged');r['status']='measured_gdn_full_ab'
 except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  deadline.close()
  for s,h in prior.items():signal.signal(s,h)
  r['elapsed_seconds']=time.monotonic()-start;r['artifacts']=artifacts(out,r['commands']);save()
if __name__=='__main__':main()
