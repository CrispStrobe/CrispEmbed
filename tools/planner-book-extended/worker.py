#!/usr/bin/env python3
"""Resource-only book replay from an authenticated tested native bundle."""
import hashlib,importlib.util,json,os,platform,shutil,signal,stat,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sys.path.insert(0,str(ROOT/'tools/planner-heldout'));import contracts as contracts
ex=contracts.ex;require=contracts.require;sha=contracts.sha
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'));from runtime_guard import Deadline,WorkerDeadline,command
sys.path.insert(0,str(ROOT/'tools/heldout'));from transfer import download as authenticated_transfer
PROFILE={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
CFG_SHA='e8e783b3526f56208e4d5f6cbf496afd2f286853f76e2d975c41092a0142d2e6'
def admit(cfg):
 require(hashlib.sha256((json.dumps(cfg,indent=2)+'\n').encode()).hexdigest()==CFG_SHA,'immutable extended protocol')
 require(cfg['case']=='two-column-page'and cfg['compiled_source']=='a6164205620c71427a39c97ab386591406c4ca73'and cfg['prefix_rows']==1054,'fixed source and retained prefix')
 require(cfg['cap']==1536 and cfg['context']==2048 and cfg['threads']==4 and cfg['worker_seconds']==4200 and cfg['arm_seconds']==2400,'resource-only extension')
 require(all(cfg[k]is False for k in['runtime_rebuilt','old_baseline_decode_allowed','reference_forward_allowed','input_or_label_change_allowed','qualified']),'no source/baseline rerun or promotion')
 selection=ROOT/'docs/lighton4b-heldout-v1.json';require(sha(selection)==cfg['selection_protocol_sha256'],'original selection unchanged');require(sha(ROOT/'docs/lighton4b-heldout-executor-v1.json')==cfg['frozen_executor_sha256'],'original reference/model protocol unchanged');selected=json.loads(selection.read_text());ex.prepare.admit(selected)
 for c in selected['cases']:
  for k in['source','gold']:require(sha(ROOT/c[k])==c[k+'_sha256'],'unchanged frozen source/gold')
 for p,d in selected['reference']['scorer'].items():require(sha(ROOT/p)==d,'unchanged scorer')
 require(sha(ROOT/'tests/regression/images/cc0/ground_truth.json')==selected['reference']['gold_registry_sha256'],'unchanged registry')
 return selected

def bind_complete(cfg):
 for k in['timeout_audit_sha256','runtime_receipt_sha256','runtime_manifest_sha256']:
  s=cfg[k];require(isinstance(s,str)and len(s)==64 and all(c in'0123456789abcdef'for c in s),'actual completed '+k+' required')
def prefix_keys(receipt):
 rows=[]
 for i in range(1054):
  key=f'corrected-composed/greedy.{i}.logits.f32';row=receipt['artifacts'][key];require(row['bytes']==248320*4,'retained full prefix row');rows.append(row['sha256'])
 return rows

def main():
 import numpy as np
 from PIL import Image
 cfg=json.loads((ROOT/'docs/lighton4b-planner-book-extended-v1.json').read_text());selection=admit(cfg);bind_complete(cfg);require(np.__version__=='2.3.5'and Image.__version__=='11.3.0'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'fixed CPU serializers');require(platform.system()=='Linux'and platform.machine().lower()in['x86_64','amd64'],'native CPU scope');flags=next(v.split(':',1)[1].split()for v in Path('/proc/cpuinfo').read_text().splitlines()if v.startswith('flags'));require({'avx2','fma','f16c'}<=set(flags),'actual CPU supports tested bundle')
 out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'book-extended';work.mkdir();started=time.monotonic();deadline=Deadline(4200);deadline.__enter__();r=dict(schema=cfg['schema'],status='failed',commands={},qualified=False,runtime_rebuilt=False,reference_forward_executed=False,old_baseline_decode_executed=False,source_feature_decode_executed=False,model_forward_executed=False)
 def save():(out/'planner-book-extended.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def canceled(sig,frame):raise SystemExit(128+sig)
 prior={s:signal.signal(s,canceled)for s in[signal.SIGTERM,signal.SIGINT]}
 def run(label,args,env=None,limit=60,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,4200-(time.monotonic()-started)),r['commands'][label],cwd=ROOT,env=env,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  found=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(found)==1,'one exact completed '+name);return found[0],json.loads(found[0].read_text())
 def verify(p,row):require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'actual closed member '+p.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact controller');base=load('physical_base',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');r['capacity']=dict(memory=base.memory(),scratch=shutil.disk_usage(work).free,output=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=10<<30 and r['capacity']['scratch']>=25<<30 and r['capacity']['output']>=6<<30,'bounded replay capacity')
  fp,failed=locate('planner-heldout.json',cfg['failed_receipt_sha256']);require(failed['status']=='failed'and failed['error_type']=='TimeoutExpired'and failed['phase']=='corrected-composed'and failed['case']==cfg['case']and failed['source_sha']==cfg['compiled_source']and failed['old_baseline_all_rows_stages_exact'],'exact original timeout');prefix=prefix_keys(failed)
  ap,audit=locate('raw-audit.json',cfg['timeout_audit_sha256']);require(audit['status']=='complete-timeout-integrity'and audit['receipt_sha256']==sha(fp)and audit['partial_rows']==1054 and audit['all_rows_finite_argmax_audited']and not audit['completed_planner_output'],'independent actual prefix integrity');r['timeout_receipt_sha256']=sha(fp);r['timeout_audit_sha256']=sha(ap)
  ip,inventory=locate('planner-heldout.json',cfg['runtime_receipt_sha256']);mp,manifest=locate('runtime-bundle.json',cfg['runtime_manifest_sha256']);require(inventory['source_sha']==manifest['source_sha']==cfg['compiled_source']and manifest['producer_receipt_sha256']==sha(ip)and inventory['status']=='passed_model_free_planner_heldout_controls'and inventory['model_acquisition']is False and inventory['heldout_forward_executed']is False and inventory['actual_cli_controls']==10 and inventory['position_export_control']and manifest['status']=='tested-model-free-runtime-bundle','completed tested binary producer');require(manifest['native_binary_sha256']==inventory['native_binary_sha256'],'actual tested native binary')
  for name,row in inventory['artifacts'].items():verify(ip.parent/name,row)
  r['all_actual_model_free_control_members_authenticated']=True
  runtime=work/'runtime';total=0;require(2<=len(manifest['files'])<=20,'bounded bundle file count')
  for name,row in manifest['files'].items():
   n=Path(name);require(not n.is_absolute()and '..'not in n.parts and n.parts[0]in['bin','lib']and 0<row['bytes']<=64<<20,'closed bundle member');total+=row['bytes'];require(total<=256<<20,'bundle total bound');src=mp.parent/'runtime'/n;verify(src,row);dest=runtime/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);verify(dest,row)
  native=runtime/'bin/lighton4b-native-document';require(sha(native)==manifest['native_binary_sha256'],'owned tested binary');native.chmod(native.stat().st_mode|stat.S_IXUSR);cache=ip.parent/'CMakeCache.txt';verify(cache,inventory['artifacts']['CMakeCache.txt']);require(sha(cache)==manifest['cmake_cache_sha256'],'tested profile identity');lines=cache.read_text().splitlines();require(all(f'{k}:BOOL={v}'in lines for k,v in PROFILE.items()),'actual supported compiled profile');shutil.copyfile(cache,out/'CMakeCache.txt');r['runtime_manifest_sha256']=sha(mp);r['native_binary_sha256']=sha(native)
  # Authenticate the entire earlier baseline and immutable inputs; never rerun it.
  priorarm=failed['arms']['baseline-old'];require(priorarm['complete']and len(priorarm['ids'])==660,'previous complete baseline');contracts.archive_closed(fp.parent/'baseline-old',priorarm)
  for name,row in failed['artifacts'].items():
   if name.startswith(('baseline-old/prefill.','baseline-old/vision.','baseline-old/processor.'))or name in['baseline-old/generated.txt','baseline-old/generated.ids','baseline-old/native-summary.json']:verify(fp.parent/name,row)
  r['prior_baseline_authenticated_without_decode']=True
  op,original=locate('heldout-result.json',cfg['original_receipt_sha256']);require(original['source_observer_full_neutral']and original['qualified']is False and sha(op)==failed['historical_reference_sha256'],'unchanged actual source reference');case=next(c for c in selection['cases']if c['id']==cfg['case']);png=op.parent/'inputs/input.png';rgb=op.parent/'inputs/input.rgb';verify(png,original['artifacts']['inputs/input.png']);verify(rgb,original['artifacts']['inputs/input.rgb']);image=Image.open(png).convert('RGB');require(sha(png)==case['prepared']['png_sha256']and sha(rgb)==case['prepared']['rgb_sha256']and image.tobytes()==rgb.read_bytes(),'same prepared image bytes');seal=fp.parent/'pre-forward-inputs.json';verify(seal,failed['artifacts']['pre-forward-inputs.json']);require(sha(seal)==failed['pre_forward_input_seal_sha256']==original['pre_forward_input_seal_sha256'],'unchanged original seal');shutil.copyfile(seal,out/'pre-forward-inputs.json')
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(failed['native_gates']['on']);env['LD_LIBRARY_PATH']=str(runtime/'lib');r['native_gates']={k:v for k,v in env.items()if k.startswith('CRISPEMBED_')};require(r['native_gates']==failed['native_gates']['on'],'no gate change');teacher=work/'one.ids';teacher.write_text('248046\n');dummy=work/'missing';args=ex.native_args(native,dummy,rgb,image.height,image.width,teacher,out,'native-composed')
  try:run('cli',args,env,30)
  except subprocess.CalledProcessError as e:require(e.returncode==4,'tested 1536 CLI before weights')
  else:raise ValueError('missing model admitted')
  sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack;original_cfg=json.loads((ROOT/'docs/lighton4b-heldout-executor-v1.json').read_text())
  def download(url,path,size,digest):
   r['phase']='download-'+Path(path).name;records=r.setdefault('downloads',{}).setdefault(Path(path).name,[]);save()
   try:authenticated_transfer(base.download,url,path,size,digest,records)
   finally:save()
  row=original_cfg['models']['Q4_K_M'];upmodel=work/row['file'];download('https://huggingface.co/'+original_cfg['model_repo']+'/resolve/'+original_cfg['model_revision']+'/'+row['file'],upmodel,row['bytes'],row['sha256']);row=original_cfg['base_model'];protected=work/row['file'];download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256']);candidate=work/'candidate.gguf';packed=pack.convert(protected,upmodel,candidate,out/'physical-pack.json');require(packed['candidate_sha256']==cfg['model_sha256'],'same physical weights');upmodel.unlink();protected.unlink()
  dest=out/'planner-on';dest.mkdir();checked=[0];stage_checked=[False];stage_keys=[n for n in failed['artifacts']if n.startswith(('corrected-composed/prefill.','corrected-composed/vision.','corrected-composed/processor.'))or n in['corrected-composed/prompt.ids','corrected-composed/modalities.ids','corrected-composed/text.embedded.f32']]
  def check_prefix():
   try:n=int(json.loads((dest/'generated-progress.json').read_text())['generated_tokens'])
   except WorkerDeadline:raise
   except(OSError,ValueError,KeyError):return
   if n>=2 and not stage_checked[0]:
    for key in stage_keys:require(sha(dest/key.split('/',1)[1])==failed['artifacts'][key]['sha256'],'all earlier actual planner prefill/vision bytes');stage_checked[0]=True
   while checked[0]<min(1054,n-1):
    i=checked[0];require(sha(dest/f'greedy.{i}.logits.f32')==prefix[i],'strict earlier1054 raw-row prefix');checked[0]+=1
  r['model_forward_executed']=True;run('planner-on-extended',ex.native_args(native,candidate,rgb,image.height,image.width,teacher,dest,'native-composed'),env,2400,check_prefix)
  for i,digest in enumerate(prefix):require(sha(dest/f'greedy.{i}.logits.f32')==digest,'final complete1054 prefix SHA')
  for key in stage_keys:require(sha(dest/key.split('/',1)[1])==failed['artifacts'][key]['sha256'],'final all earlier stages')
  r['strict_1054_prefix_and_stages_reproduced']=True;summary=json.loads((dest/'native-summary.json').read_text());ids=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode();contracts.native_complete(summary,ids,False);require(summary['physical_source_prefill_chunks']and summary['physical_q6_tiled']and summary['physical_q4_repack']and summary['physical_repack_entries']==216 and 0<summary['physical_repack_bytes']<=2<<30,'actual same modes/cache');row=np.fromfile(dest/'prefill.logits.f32',dtype='<f4');sourcefile=op.parent/'upstream-on/first-score.f32';verify(sourcefile,original['artifacts']['upstream-on/first-score.f32']);source=np.fromfile(sourcefile,dtype='<f4');require(row.size==source.size==248320 and np.isfinite(row).all()and np.isfinite(source).all(),'finite source/native vocabulary');d=row.astype('f8')-source;gold=json.loads((ROOT/case['gold']).read_text());score=ex.score(gold,text,summary['eos']);reference_score=ex.score(gold,original['arms']['upstream-off']['decoded'],True);r['arm']=dict(summary=summary,ids=ids,decoded=text,score=score,raw_rows=ex.audit_rows(dest,ids),firstscore=dict(rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max())));r['reference_score']=reference_score;r['meets_provisional_source_floor']=ex.floor(score,reference_score,False);r['status']='measured_extended_planner_book';require(sha(candidate)==cfg['model_sha256'],'weights unchanged')
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  deadline.close()
  for sig,handler in prior.items():signal.signal(sig,handler)
  r['elapsed_seconds']=time.monotonic()-started;owned={k+'.log'for k in r['commands']}|{'CMakeCache.txt','physical-pack.json','pre-forward-inputs.json'};r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)};save()
if __name__=='__main__':main()
