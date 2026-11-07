#!/usr/bin/env python3
"""One frozen provisional case: retained upstream reference, old native replay, corrected pixels/features."""
import argparse,hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));import contracts as c
ex=c.ex;require=c.require;sha=c.sha
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'));from runtime_guard import Deadline,WorkerDeadline,command
sys.path.insert(0,str(ROOT/'tools/heldout'));from transfer import download as authenticated_transfer

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 import numpy as np
 from PIL import Image
 parser=argparse.ArgumentParser();parser.add_argument('--case',choices=ex.prepare.CASE_IDS);parser.add_argument('--controls-only',action='store_true');parser.add_argument('--controls-receipt-sha');opt=parser.parse_args();require(opt.controls_only or (opt.case and opt.controls_receipt_sha and len(opt.controls_receipt_sha)==64),'fixed case and completed controls required')
 cfg=json.loads((ROOT/'docs/lighton4b-q6-heldout-v1.json').read_text());original_cfg=c.admit(cfg);selection=json.loads(ex.prepare.PROTOCOL.read_text())
 require(np.__version__=='2.3.5'and Image.__version__=='11.3.0'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'frozen CPU serialization')
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(exist_ok=True);work=Path(os.environ['HEAVY_SCRATCH'])/'q6-heldout';work.mkdir();start=time.monotonic();budget=1800 if opt.controls_only else 6000;deadline=Deadline(budget);deadline.__enter__()
 def canceled(sig,frame):raise SystemExit(128+sig)
 prior={s:signal.signal(s,canceled)for s in[signal.SIGTERM,signal.SIGINT]}
 r=dict(schema=cfg['schema'],status='failed',case=opt.case,arms={},commands={},qualified=False,references_expert_adjudicated=False,reference_forward_executed=False,heldout_forward_executed=False,model_acquisition=False,reference_limits=selection['reference'],comparison_scope=cfg['scope'])
 def save():(out/'q6-heldout.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def run(label,args,env=None,limit=900,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,budget-(time.monotonic()-start)),r['commands'][label],cwd=ROOT,env=env,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  found=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(found)==1,'one exact completed '+name);return found[0],json.loads(found[0].read_text())
 def verify(path,row):require(path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'immutable member '+path.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
  base=module('physical_base',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');profile=module('vision_profile',ROOT/'tools/physical-f16-attention-probe/worker.py')
  r['capacity']=dict(memory=base.memory(),scratch=shutil.disk_usage(work).free,output=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=10<<30 and r['capacity']['scratch']>=25<<30 and r['capacity']['output']>=8<<30,'frozen capacity')
  run('fetch-production',['git','fetch','--depth','1','origin',cfg['production_source']]);run('production-bytes',['git','diff','--exit-code',cfg['production_source'],'HEAD','--','src','ggml']);run('ggml',['git','submodule','update','--init','ggml'])
  build=work/'build';flags=dict(profile.BUILD_FLAGS);flags['GGML_CPU_REPACK']='ON'
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]]);cache=(build/'CMakeCache.txt').read_text().splitlines();require(all(f'{k}:BOOL={v}'in cache for k,v in flags.items()),'actual full native profile');shutil.copyfile(build/'CMakeCache.txt',out/'CMakeCache.txt')
  targets=['lighton4b-native-document','lighton4b-native-case-preflight','test-lighton4b-physical-q6-tiled','test-lighton4b-physical-gdn-source-ops','test-lighton4b-physical-q4-repack','test-lighton4b-physical-f16-attention','test-lighton4b-physical-f16-vision'];run('build',['cmake','--build',build,'--target',*targets,'--parallel','2'],limit=1200)
  markers=['PHYSICAL_Q6_TILED_CONTROLS PASS','PHYSICAL_GDN_SOURCE_OPS_CONTROLS PASS','PHYSICAL_Q4_REPACK PASS 57','PHYSICAL_F16_ATTENTION PASS','PHYSICAL_F16_VISION_SYNTHETIC PASS']
  for target,marker in zip(targets[2:],markers):run(target,[build/target]);require(marker in(out/(target+'.log')).read_text(),'actual supported controls')
  native=build/'lighton4b-native-document';teacher=work/'one.ids';teacher.write_text('248046\n');dummy=work/'missing';env=dict(os.environ);env['CRISPEMBED_LIGHTON4B_PROVISIONAL_HELDOUT']='1';checks=[]
  for arm in ['native-composed','native-source-features']:checks.append((arm,ex.native_args(native,dummy,dummy,1,1,teacher,out,arm,dummy),4,env))
  canonical=ex.native_args(native,dummy,dummy,1,1,teacher,out,'native-composed')
  for name,index,value in [('cap-low',7,1535),('cap-high',7,1537),('threads',8,2),('mode',9,'full')]:args=list(canonical);args[index]=value;checks.append((name,args,2,env))
  disabled=dict(env);disabled['CRISPEMBED_LIGHTON4B_PROVISIONAL_HELDOUT']='0';checks += [('disabled',canonical,2,disabled),('unflagged',canonical[:-1],2,env),('old256',[native,dummy,dummy,1,1,teacher,out,256,4,'greedy-full'],4,disabled),('old512',[native,dummy,dummy,1,1,teacher,out,512,4,'greedy-full','--observed-composed-cap512'],4,disabled)]
  for label,args,expected,child in checks:
   try:run('cli-'+label,args,child,30)
   except subprocess.CalledProcessError as e:require(e.returncode==expected,'actual CLI '+label)
   else:raise ValueError('missing model admitted')
  position_dir=out/'position-export-control';position_dir.mkdir();position_env=dict(env);position_env['CRISPEMBED_LIGHTON4B_FROZEN_CASES']='1';position_env['CRISPEMBED_QWEN35_OCR']='1'
  run('position-export-control',[build/'lighton4b-native-case-preflight','--positions-control',position_dir],position_env,30);require('NATIVE_CASE_PREFLIGHT_POSITIONS PASS'in(out/'position-export-control.log').read_text()and(position_dir/'prefill.positions.f32').stat().st_size==96,'actual shared three-axis exporter before models')
  for name,flag,code in [('ocr-off','CRISPEMBED_QWEN35_OCR',8),('owner-off','CRISPEMBED_LIGHTON4B_FROZEN_CASES',2)]:
   disabled_positions=dict(position_env);disabled_positions[flag]='0'
   try:run('position-export-'+name,[build/'lighton4b-native-case-preflight','--positions-control',position_dir],disabled_positions,30)
   except subprocess.CalledProcessError as error:require(error.returncode==code,'position exporter gate remains closed')
   else:raise ValueError('disabled position exporter admitted')
  r['position_export_control']=True
  r['actual_cli_controls']=len(checks);r['native_binary_sha256']=sha(native)
  if opt.controls_only:r['status']='passed_model_free_q6_heldout_controls';return
  controlp,control=locate('q6-heldout.json',opt.controls_receipt_sha);require(control['status']=='passed_model_free_q6_heldout_controls'and control['source_sha']==r['source_sha']and control['model_acquisition']is False and control['heldout_forward_executed']is False and control['actual_cli_controls']==10 and control['position_export_control'],'same-source hosted controls required')
  for name,row in control['artifacts'].items():verify(controlp.parent/name,row)
  r['controls_receipt_sha256']=sha(controlp)
  # Original references are authenticated, never rerun or relabelled.
  oldp,old=locate('heldout-result.json',cfg['old_case_receipts'][opt.case]);require(old['status']=='measured_provisional_heldout'and old['case']==opt.case and old['source_sha']=='d378c6cc055e3a31b0d4cc9dad54a5b41f10d792'and old['source_observer_full_neutral']and old['observed167_runtime_byte_control']and old['qualified']is False and old['reference_limits']==selection['reference'],'complete unchanged previous panel')
  olddir=oldp.parent;case=next(x for x in selection['cases']if x['id']==opt.case)
  needed=['inputs/input.png','inputs/input.rgb','pre-forward-inputs.json','source-preflight.json','source-preflight.prompt.ids','source-preflight.modalities.ids','source-preflight.positions.f32','native-preflight/prompt.ids','native-preflight/modalities.ids','native-preflight/grid.ids','native-preflight/prefill.positions.f32','upstream-off/response.json','upstream-on/response.json','upstream-on/source.vision.0.f32','upstream-on/first-score.f32']
  needed += [name for name in old['artifacts']if name.startswith('native-composed/prefill.')or name.startswith('native-composed/vision.')or name.startswith('native-composed/processor.')or name in['native-composed/prompt.ids','native-composed/modalities.ids','native-composed/text.embedded.f32']]
  for name in needed:verify(olddir/name,old['artifacts'][name])
  off_response=json.loads((olddir/'upstream-off/response.json').read_text());on_response=json.loads((olddir/'upstream-on/response.json').read_text());ex.neutral(off_response,on_response);require(ex.source_response(off_response,off_response['tokens_evaluated']),'historical source complete')
  for arm,response in [('upstream-off',off_response),('upstream-on',on_response)]:require(old['arms'][arm]['ids']==response['tokens']and old['arms'][arm]['decoded']==response['content']and old['arms'][arm]['complete'],'historical source response identity')
  r['historical_raw_identity']={'upstream-on':c.archive_closed(olddir/'upstream-on',{'raw_rows':old['source_raw_rows']}),'native-composed':c.archive_closed(olddir/'native-composed',old['arms']['native-composed'])};r['historical_reference_sha256']=sha(oldp);r['historical_source_host_scope']='Historical independently audited source OFF/ON execution; no new source forward or same-host arithmetic assertion.';r['source_observer_full_neutral']=True;r['source_raw_scope']=original_cfg['source_raw_scope']
  png=olddir/'inputs/input.png';rgb=olddir/'inputs/input.rgb';image=Image.open(png).convert('RGB');require(sha(png)==case['prepared']['png_sha256']and sha(rgb)==case['prepared']['rgb_sha256']and image.tobytes()==rgb.read_bytes(),'unchanged frozen input pixels')
  inputcopy=out/'inputs';inputcopy.mkdir();shutil.copy2(png,inputcopy/'input.png');shutil.copy2(rgb,inputcopy/'input.rgb')
  fp,form=locate('composed-form-vision.json',original_cfg['observed_form_receipt_sha256']);qp,q6=locate('q6-tiled-full-ab.json',cfg['observed_receipt_sha256']);require(q6['status']=='measured_q6_tiled_full_ab'and q6['cases']['simple_form']['off_all_rows_stages_exact'],'measured corrected observed prerequisite')
  gates_off,gates_on=c.gates(old['native_gates'],q6['cases']['simple_form']['arms']['on']['gates']);libs=sorted(p for p in build.rglob('libggml*.so*')if p.is_file());require(libs,'native backend libraries')
  def environment(gates):
   result=dict(os.environ);result.update({k:'0'for k in result if k.startswith('CRISPEMBED_')});result.update(gates);result['LD_LIBRARY_PATH']=':'.join(sorted({str(p.parent)for p in libs}));return result
  offenv=environment(gates_off);onenv=environment(gates_on);r['changed_gates']=sorted(c.CHANGED);r['native_gates']={'off':gates_off,'on':gates_on}
  # Closed observed old and corrected167 runs precede every heldout forward.
  physicalp,physical=locate('physical-decode.json','712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2');observed_rgb=physicalp.parent/'simple_form/input.rgb';verify(observed_rgb,physical['cases']['simple_form']['artifacts']['input.rgb'])
  r['model_acquisition']=True;save();sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
  def download(url,path,size,digest):
   attempts=r.setdefault('downloads',{}).setdefault(Path(path).name,[])
   try:authenticated_transfer(base.download,url,path,size,digest,attempts)
   finally:save()
  row=original_cfg['models']['Q4_K_M'];upmodel=work/row['file'];download('https://huggingface.co/'+original_cfg['model_repo']+'/resolve/'+original_cfg['model_revision']+'/'+row['file'],upmodel,row['bytes'],row['sha256']);row=original_cfg['base_model'];protected=work/row['file'];download('https://huggingface.co/'+row['repository']+'/resolve/'+row['revision']+'/'+row['file'],protected,row['bytes'],row['sha256']);candidate=work/'candidate.gguf';audit=pack.convert(protected,upmodel,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==cfg['model_sha256'],'same immutable physical weights');upmodel.unlink();protected.unlink()
  for tag,child,gold,folder,prefix,receipt in [('observed-old',offenv,form['arms']['fixed'],fp.parent,'fixed',form),('observed-corrected',onenv,q6['cases']['simple_form']['arms']['on'],qp.parent,'simple_form/on',q6)]:
   dest=out/tag;dest.mkdir();run(tag,[native,candidate,observed_rgb,317,452,teacher,dest,512,4,'greedy-full','--observed-composed-cap512'],child,1200);require(list(map(int,(dest/'generated.ids').read_text().split()))==gold['generated_ids']and(dest/'generated.txt').read_bytes().decode()==gold['decoded'],'observed full IDs/text')
   anchors=[name[len(prefix)+1:]for name in receipt['artifacts']if name.startswith(prefix+'/prefill.')or name.startswith(prefix+'/vision.')or name.startswith(prefix+'/processor.')or name.startswith(prefix+'/greedy.')and name.endswith('.logits.f32')]
   require(len([x for x in anchors if x.startswith('greedy.')])==167,'observed full167 raw census')
   for name in anchors:require(sha(dest/name)==receipt['artifacts'][prefix+'/'+name]['sha256'],'observed full runtime byte closure')
   r[tag]=dict(all_rows_stages_exact=True,raw_rows=ex.audit_rows(dest,gold['generated_ids']));save()
  # Metadata-only native preflight closes the original source/input seal again.
  pre=out/'native-preflight';pre.mkdir();run('native-preflight',[build/'lighton4b-native-case-preflight',candidate,rgb,image.height,image.width,pre,'lighton4b-upstream-q4km-physical-v1'],onenv)
  ids=list(map(int,(pre/'prompt.ids').read_text().split()));mods=list(map(int,(pre/'modalities.ids').read_text().split()));grid=list(map(int,(pre/'grid.ids').read_text().split()));source_ids=list(map(int,(olddir/'source-preflight.prompt.ids').read_text().split()));source_mods=list(map(int,(olddir/'source-preflight.modalities.ids').read_text().split()));require(len(ids)==off_response['tokens_evaluated'],'historical source prompt count')
  seal=out/'pre-forward-inputs.json';seal_sha=ex.seal_inputs(seal,case,ids,mods,grid,(pre/'prefill.positions.f32').read_bytes(),source_ids,source_mods,(olddir/'source-preflight.positions.f32').read_bytes(),original_cfg['expected_media_prompt'],original_cfg);require(seal_sha==old['pre_forward_input_seal_sha256'],'exact original pre-forward seal');r['pre_forward_input_seal_sha256']=seal_sha;save()
  feature=olddir/'upstream-on/source.vision.0.f32';source_row=np.fromfile(olddir/'upstream-on/first-score.f32',dtype='<f4');require(source_row.size==248320 and np.isfinite(source_row).all(),'finite historical source vocabulary');gold=json.loads((ROOT/case['gold']).read_text());r['reference_scores']={arm:ex.score(gold,old['arms'][arm]['decoded'],True)for arm in['upstream-off','upstream-on']}
  for arm,child,source_fed in [('baseline-old',offenv,False),('corrected-composed',onenv,False),('corrected-source-features',onenv,True)]:
   require(base.memory()['MemAvailable']>=10<<30 and shutil.disk_usage(out).free>=2<<30,'per-arm resources');require(sha(seal)==seal_sha,'unchanged input seal');dest=out/arm;dest.mkdir();role='native-source-features'if source_fed else'native-composed';r['heldout_forward_attempted']=True;save();run(arm,ex.native_args(native,candidate,rgb,image.height,image.width,teacher,dest,role,feature),child,1200);r['heldout_forward_executed']=True
   summary=json.loads((dest/'native-summary.json').read_text());generated=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode();c.native_complete(summary,generated,source_fed)
   require(summary['physical_q6_tiled']==(arm!='baseline-old')and summary['physical_q4_repack']==(arm!='baseline-old'),'actual correction cache flags');require(summary['physical_repack_entries']==(0 if arm=='baseline-old'else 216)and summary['physical_repack_bytes']<=2<<30 and (summary['physical_repack_bytes']==0 if arm=='baseline-old'else summary['physical_repack_bytes']>0),'bounded quantized cache')
   require((dest/'prefill.positions.f32').read_bytes()==(pre/'prefill.positions.f32').read_bytes()and list(map(int,(dest/'prompt.ids').read_text().split()))==ids,'actual sealed prompt/positions')
   if arm=='baseline-old':
    oldarm=old['arms']['native-composed'];require(generated==oldarm['ids']and text==oldarm['decoded']and summary['eos'],'old complete baseline closure');c.raw_equal(dest,oldarm)
    for name in needed:
     if name.startswith('native-composed/'):require(sha(dest/name.split('/',1)[1])==old['artifacts'][name]['sha256'],'old full stage/vision closure')
    r['old_baseline_all_rows_stages_exact']=True
   else:
    for name in ['prompt.ids','modalities.ids','text.embedded.f32','prefill.input.f32','prefill.positions.f32','vision.merged.f32']:require(sha(dest/name)==sha(out/'baseline-old'/name),'unchanged native input/merged across correction')
    if arm=='corrected-composed':
     for name in needed:
      if name.startswith('native-composed/vision.')or name.startswith('native-composed/processor.'):require(sha(dest/name.split('/',1)[1])==sha(out/'baseline-old'/name.split('/',1)[1]),'unchanged composed vision/processor')
   row=np.fromfile(dest/'prefill.logits.f32',dtype='<f4');require(row.size==248320 and np.isfinite(row).all()and sha(dest/'prefill.logits.f32')==sha(dest/'teacher.0.logits.f32')==sha(dest/'greedy.0.logits.f32'),'same actual first row');d=row.astype('f8')-source_row;merged=np.fromfile(dest/'vision.merged.f32',dtype='<f4');reference=np.fromfile(feature,dtype='<f4');require(merged.shape==reference.shape and np.isfinite(merged).all()and np.isfinite(reference).all(),'finite merged geometry');delta=merged.astype('f8')-reference
   r['arms'][arm]=dict(complete=summary['eos'],ids=generated,decoded=text,summary=summary,score=ex.score(gold,text,summary['eos']),raw_rows=ex.audit_rows(dest,generated),first_score_vs_source=dict(rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),source_max_abs=float(abs(source_row).max())),merged_vs_source=dict(bit_exact=merged.tobytes()==reference.tobytes(),rms=float(np.sqrt(np.mean(delta*delta))),max_abs=float(abs(delta).max())),resources=r['commands'][arm]);save()
  r['primary_meets_provisional_source_floor']=ex.floor(r['arms']['corrected-composed']['score'],r['reference_scores']['upstream-off'],case['id']=='inventory-table');r['all_primary_complete']=all(v['complete']for v in r['arms'].values());require(sha(candidate)==cfg['model_sha256'],'weights immutable');r['status']='measured_provisional_q6_heldout';r['qualified']=False
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  deadline.close()
  for sig,handler in prior.items():signal.signal(sig,handler)
  r['elapsed_seconds']=time.monotonic()-start;owned={label+'.log'for label in r['commands']}|{'CMakeCache.txt','physical-pack.json','pre-forward-inputs.json'};r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)};save()
if __name__=='__main__':main()
