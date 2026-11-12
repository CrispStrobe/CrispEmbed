#!/usr/bin/env python3
"""Observed-only physical RoPE A/B; tested candidate bundle, historical fusedON OFF closure."""
import hashlib,importlib.util,json,os,platform,shutil,signal,stat,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
shared=load('q6_ab',ROOT/'tools/q6-tiled-full-ab/worker.py');base=shared.base;f=shared.f;require=shared.require;sha=shared.sha
Deadline=shared.Deadline;command=shared.command
GATE='CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE'
PROFILE={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
CFG_SHA='69b8fdb438404dc9beef6582ae8ccf2af8da009a1d8a04e5c6a4ce363116d10a'
HISTORICAL='67ff19db0230ddc329300cc3f6a0256eb85b3cdbbcf06a578b23eb93c2688ba1'
def admit(path):require(sha(path)==CFG_SHA,'immutable observed protocol');return json.loads(Path(path).read_text())
def gates(env):
 off=dict(env);off[GATE]='0';on=dict(off);on[GATE]='1';require({k for k in off if off[k]!=on[k]}=={GATE},'only physical RoPE changes');require(off.get('CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU')==on.get('CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU')=='1','fused SwiGLU remains enabled');return off,on
def stage(path):require(path.is_file(),'owned binary');path.chmod(path.stat().st_mode|stat.S_IXUSR)
def admit_controls(controls,candidate_source):
 require(controls['status']=='measured_source_fed_rope_controls'and controls['passed']and controls['candidate_source']==candidate_source and controls['checkpoint_loaded']is False and controls['all_source_stages_exact']and controls['actual_cache_toggle_recovery_pass'],'completed source-bound component controls')
 require(set(controls['unit_controls'])=={'test-lighton4b-physical-rope','test-lighton4b-source-prefill-chunks','test-lighton4b-physical-swiglu','test-lighton4b-physical-gdn-source-ops','test-lighton4b-physical-native-ops','test-lighton4b-physical-f16-attention','physical-rope-unsupported','physical-rope-wasm-guard'}and all('pass'in v.lower()and 'skip'not in v.lower()for v in controls['unit_controls'].values()),'actual supported/guard unit results')

def main():
 import numpy as np
 from PIL import Image
 cfg=admit(Path(__file__).parent/'protocol.json');require(np.__version__=='2.3.5'and Image.__version__=='11.3.0'and Path('/kaggle/input').is_dir(),'CPU serializer profile');require(cfg['cases']==['simple_table','simple_form','arabic_printed_line']and not cfg['heldout_data_allowed']and cfg['cap']==512 and cfg['threads']==4,'fixed observed scope')
 out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'physical-rope-observed';work.mkdir();started=time.monotonic();deadline=Deadline(7200);deadline.__enter__();prior={s:signal.signal(s,lambda sig,frame:(_ for _ in()).throw(SystemExit(128+sig)))for s in[signal.SIGTERM,signal.SIGINT]}
 r=dict(schema=cfg['schema'],status='failed',commands={},cases={},quality_qualified=False,heldout_data_accessed=False,runtime_rebuilt=False,reference_forward_executed=False)
 def save():(out/'physical-rope-observed-ab.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def run(label,args,env=None,limit=900,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,7200-(time.monotonic()-started)),r['commands'][label],env=env,cwd=ROOT,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  found=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(found)==1,'one frozen '+name);return found[0],json.loads(found[0].read_text())
 def verify(p,row):require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'immutable '+p.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE']==cfg['controller_source'],'exact controller source');candidate_source=os.environ['LIGHTON4B_CANDIDATE_SOURCE'];control_sha=os.environ['LIGHTON4B_CANDIDATE_CONTROLS_SHA256'];require(candidate_source==cfg['candidate_source']and len(control_sha)==64 and all(c in'0123456789abcdef'for c in control_sha),'immutable candidate/control bindings')
  flags=next(v.split(':',1)[1].split()for v in Path('/proc/cpuinfo').read_text().splitlines()if v.startswith('flags'));require(platform.system()=='Linux'and platform.machine().lower()in['x86_64','amd64']and {'avx2','fma','f16c'}<=set(flags),'actual admitted CPU')
  r['capacity']=dict(memory=base.memory(),scratch=shutil.disk_usage(work).free,output=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=10<<30 and r['capacity']['scratch']>=25<<30 and r['capacity']['output']>=6<<30,'pre-model resources')
  arm_paths={}
  cp,controls=locate('physical-rope-controls.json',control_sha);admit_controls(controls,candidate_source)
  cache=(cp.parent/'CMakeCache.txt').read_text().splitlines();verify(cp.parent/'CMakeCache.txt',controls['cmake_cache'])
  for k,v in PROFILE.items():require(f'{k}:BOOL={v}'in cache,'actual candidate profile '+k)
  shutil.copyfile(cp.parent/'CMakeCache.txt',out/'CMakeCache.txt');bundle=work/'runtime';require(2<=len(controls['runtime_bundle'])<=20,'bounded tested bundle')
  for name,row in controls['runtime_bundle'].items():
   n=Path(name);require(not n.is_absolute()and '..'not in n.parts and n.parts[0]in['bin','lib'],'closed tested bundle path');src=cp.parent/'runtime'/n;verify(src,row);dest=bundle/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);verify(dest,row)
  r['candidate_controls_sha256']=sha(cp);r['runtime_bundle']=controls['runtime_bundle'];r['runtime_producer_source']=candidate_source;native=bundle/'bin/lighton4b-native-document';stage(native)
  bp,baseline=locate('swiglu-observed-ab.json',HISTORICAL);ph,physical=locate('physical-decode.json',shared.PHYSICAL);tp,table=locate('physical-f16-table512.json',shared.TABLE)
  require(baseline['status']=='measured_swiglu_observed_ab'and baseline['runtime_producer_source']=='519c3d9765c234cd93e9069d5efdc239e3cad3f2'and baseline['all_inputs_authenticated_before_weights']and physical['status']=='measured'and table['status']=='measured_extended_table'and table['source_observer_full_neutral'],'complete frozen fusedON baseline');baseline['artifacts']={**baseline['reused_owned_artifacts'],**baseline['artifacts']};r['historical_fused_receipt_sha256']=sha(bp);r['historical_raw_scope']=cfg['baseline_scope']
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(baseline['cases']['simple_form']['arms']['on']['gates']);env['LD_LIBRARY_PATH']=str(bundle/'lib');off,on=gates(env)
  for k in ['CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS','CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK','CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION','CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION','CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE']:require(off[k]=='1','unchanged corrected pipeline')
  require(off['CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL']=='1'and off['CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU']==on['CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU']=='1','fixed plannerON and fused SwiGLUON')
  for k in ['CRISPEMBED_LIGHTON4B_HEAD_BF16','CRISPEMBED_LIGHTON4B_BF16_LINEAR','CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS']:require(off[k]=='0','unchanged text/head precision')
  try:run('cli',[native,work/'missing',work/'missing.rgb',1,1,work/'missing.ids',out,512,4,'greedy-full','--observed-composed-cap512'],off,30)
  except subprocess.CalledProcessError as e:require(e.returncode==4,'actual cap512 CLI')
  else:raise RuntimeError('missing model admitted')
  prepared=[]
  for case,count in [('simple_table',218),('simple_form',167),('arabic_printed_line',6)]:
   receiptp,receipt,prefix,gold=bp,baseline,case+'/on/',baseline['cases'][case]['arms']['on']
   require(len(gold['generated_ids'])==count and gold['summary']['eos'],'complete OFF golden')
   art=physical['cases'][case]['artifacts'];folder=ph.parent/case
   for name in ['input.rgb','original.json','retained-upstream-response.json','observer-on/source.logits.0.f32']:verify(folder/name,art[name])
   original=json.loads((folder/'original.json').read_text());upfile=tp.parent/'upstream-off/response.json'if case=='simple_table'else folder/'retained-upstream-response.json';uprow=table['artifacts']['upstream-off/response.json']if case=='simple_table'else art['retained-upstream-response.json'];verify(upfile,uprow);up=json.loads(upfile.read_text());require(up['stop']is True and up['stop_type']=='eos'and not up['truncated']and up['tokens'][-1]in[248044,248046]and up['tokens_predicted']==len(up['tokens'])==len(up['completion_probabilities']),'complete actual upstream reference')
   scorefile=tp.parent/'upstream-on/source.logits.0.f32'if case=='simple_table'else folder/'observer-on/source.logits.0.f32';scorerow=table['artifacts']['upstream-on/source.logits.0.f32']if case=='simple_table'else art['observer-on/source.logits.0.f32'];verify(scorefile,scorerow);reference=np.fromfile(scorefile,dtype='<f4');require(reference.size==248320 and np.isfinite(reference).all(),'finite actual upstream vocabulary')
   visual=[name[len(prefix):]for name in receipt['artifacts']if name.startswith(prefix+'vision.')or name.startswith(prefix+'processor.')];anchors=sorted({name[len(prefix):]for name in receipt['artifacts']if name.startswith(prefix)and(name.endswith('.f32')or name[len(prefix):]in['prompt.ids','modalities.ids'])}|set(visual))
   require(all(prefix+name in receipt['artifacts']and receipt['artifacts'][prefix+name]['bytes']>=0 and len(receipt['artifacts'][prefix+name]['sha256'])==64 for name in anchors),'complete frozen prior stage/row identities');require(all('greedy.%d.logits.f32'%i in anchors for i in range(count)),'all prior vocabulary rows declared')
   protocol=json.loads((ROOT/'docs/lighton4b-upstream-physical-decode-v1.json').read_text());c=next(v for v in protocol['cases']if v['id']==case);image=Image.open(ROOT/c['image']).convert('RGB');require(sha(ROOT/c['image'])==c['image_sha256']and image.tobytes()==(folder/'input.rgb').read_bytes(),'same frozen RGB');prepared.append((case,count,receiptp,receipt,prefix,gold,folder,original,up,reference,visual,anchors,image))
  fp,form_trace=locate('form-text-trace.json','2723be06d2b1a4836d51b75d50a23f61a07972127c907e01d0505c778dcf7260');require(form_trace['all170_scores_observer_neutral']and not form_trace['f32_model_instantiated']and not form_trace['native_forward_executed'],'neutral original form source probe');source_form={}
  for label,name,width in[('input','model.input_embed',2560),('gate','ffn_gate-0',9216),('up','ffn_up-0',9216),('activation','ffn_swiglu-0',9216),('down','ffn_out-0',2560)]:
   file=fp.parent/f'on/graph-1.{name}.bin';verify(file,form_trace['artifacts'][f'on/graph-1.{name}.bin']);require(file.stat().st_size==4*width*4,'exact source first4 geometry');source_form[label]=np.fromfile(file,dtype='<f4');require(np.isfinite(source_form[label]).all(),'finite source probe')
  position_file=fp.parent/'on/graph-1.inp_pos.bin';verify(position_file,form_trace['artifacts']['on/graph-1.inp_pos.bin']);require(position_file.stat().st_size==16*4,'exact source four-axis position geometry');source_form['positions']=np.fromfile(position_file,dtype='<i4').reshape(4,4)[:3].astype('<f4')
  r['source_form_probe_receipt_sha256']=sha(fp);r['all_inputs_authenticated_before_weights']=True;save();sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
  def download(url,path,size,digest):
   r['phase']='download-'+Path(path).name;records=r.setdefault('downloads',{}).setdefault(Path(path).name,[]);save()
   try:shared.transfer(base.download,url,path,size,digest,records)
   finally:save()
  model=json.loads((ROOT/'docs/lighton4b-composed-form-vision-v1.json').read_text());v=model['models']['Q4_K_M'];upmodel=work/v['file'];download('https://huggingface.co/'+model['model_repo']+'/resolve/'+model['model_revision']+'/'+v['file'],upmodel,v['bytes'],v['sha256']);v=model['base_model'];protected=work/v['file'];download('https://huggingface.co/'+v['repository']+'/resolve/'+v['revision']+'/'+v['file'],protected,v['bytes'],v['sha256']);candidate=work/'candidate.gguf';audit=pack.convert(protected,upmodel,candidate,out/'physical-pack.json');require(audit['candidate_sha256']==cfg['model_sha256'],'unchanged physical weights');protected.unlink()
  for case,count,receiptp,receipt,prefix,gold,folder,original,up,reference,visual,anchors,image in prepared:
   teacher=work/(case+'.ids');teacher.write_text(str(original['generated_ids'][0])+'\n');r['cases'][case]=dict(arms={},actual_upstream_vs_original=f.metric(base,up['tokens'],up['content'],original['generated_ids'],original['decoded']))
   for arm,child in [('off',off),('on',on)]:
    require(base.memory()['MemAvailable']>=10<<30 and shutil.disk_usage(out).free>=2<<30,'per-arm resources');dest=out/case/arm;arm_paths[(case,arm)]=dest;dest.mkdir(parents=True)
    checked=[0]
    def tick():
     if arm!='off':return
     n=shared.progress_count(dest)
     if n is None:return
     while checked[0]<min(count,n-1):
      j=checked[0];require(sha(dest/f'greedy.{j}.logits.f32')==receipt['artifacts'][prefix+f'greedy.{j}.logits.f32']['sha256'],'early OFF closed row');checked[0]+=1
    label=case+'-'+arm
    run(label,[native,candidate,folder/'input.rgb',image.height,image.width,teacher,dest,512,4,'greedy-full','--observed-composed-cap512'],child,900,tick)
    summary=json.loads((dest/'native-summary.json').read_text());ids=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode();require(summary['generated_tokens']==len(ids)and ids and len(ids)<=512 and summary['eos']==(ids[-1]in[248044,248046])and(summary['eos']or len(ids)==512)and not summary['source_fed_vision']and summary['teacher_positions']==1,'independent mode/count/EOS');shared.rows(dest,ids,np)
    require(summary['physical_source_prefill_chunks']and summary['physical_q6_tiled']and summary['physical_q4_repack']and summary['physical_repack_entries']==216 and 0<summary['physical_repack_bytes']<=2<<30 and summary['converted_f32_weight_entries']==summary['converted_f32_weight_bytes']==0,'actual mode/bounded quantized cache')
    sizes=list(map(int,(dest/'prefill.chunk_sizes.ids').read_text().split()));require(sizes and all(1<=n<=16 for n in sizes)and sum(sizes)==summary['prompt_tokens'],'actual bounded complete chunk plan')
    if arm=='off':
     require(summary['physical_rope']is False and {k:summary[k]for k in gold['summary']}==gold['summary']and set(summary)==set(gold['summary'])|{'physical_rope'},'OFF original summary fields exact; only new false gate telemetry added')
     require(ids==gold['generated_ids']and text==gold['decoded']and summary['eos'],'complete OFF text/IDs/EOS')
     for name in anchors:require(sha(dest/name)==receipt['artifacts'][prefix+name]['sha256'],'all OFF frozen rows/stages')
     r['cases'][case]['off_all_rows_stages_exact']=True
    else:
     require(summary['physical_rope']is True,'actual ON context gate')
     for layer in range(3):require(sha(dest/f'prefill.layer.{layer}.f32')==sha(arm_paths[(case,'off')]/f'prefill.layer.{layer}.f32'),'unchanged layers0to2 before first full attention')
     for name in visual+['prompt.ids','modalities.ids','text.embedded.f32','prefill.input.f32','prefill.positions.f32']:require(sha(dest/name)==sha(arm_paths[(case,'off')]/name),'unchanged vision and decoder input')
    if case=='simple_form':
     probe={};input_values=np.fromfile(dest/'prefill.input.f32',dtype='<f4')[:4*2560];require(input_values.tobytes()==source_form['input'].tobytes(),'same real source first4 form input')
     actual_positions=np.fromfile(dest/'prefill.positions.f32',dtype='<f4').reshape(3,-1)[:,:4];require(actual_positions.tobytes()==source_form['positions'].tobytes(),'same source first4 positions')
     for probe_name in['gate','up','activation','down']:
      actual=np.fromfile(dest/f'prefill.first_chunk.layer0.{probe_name}.f32',dtype='<f4');reference_values=source_form[probe_name];require(actual.shape==reference_values.shape and np.isfinite(actual).all(),'source probe actual geometry');d=actual.astype('f8')-reference_values;probe[probe_name]=dict(bit_exact=actual.tobytes()==reference_values.tobytes(),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),candidate_rms_magnitude=float(np.sqrt(np.mean(actual.astype('f8')**2))),source_rms_magnitude=float(np.sqrt(np.mean(reference_values.astype('f8')**2))))
     require(probe['gate']['bit_exact']and probe['up']['bit_exact'],'actual source same gate/up inputs');require(probe['activation']['bit_exact']and probe['down']['bit_exact'],'both arms retain actual fused source first4 activation/down');r['cases'][case].setdefault('source_form_first4',{})[arm]=probe
    x=np.fromfile(dest/'prefill.logits.f32',dtype='<f4').astype('f8');d=x-reference;vs=f.metric(base,ids,text,original['generated_ids'],original['decoded']);floor=r['cases'][case]['actual_upstream_vs_original'];r['cases'][case]['arms'][arm]=dict(summary=summary,gates={k:v for k,v in child.items()if k.startswith('CRISPEMBED_')},chunk_sizes=sizes,generated_ids=ids,decoded=text,vs_original=vs,vs_upstream=f.metric(base,ids,text,up['tokens'],up['content']),meets_observed_upstream_floor=bool(summary['eos']and vs['character_edits']<=floor['character_edits']and vs['word_edits']<=floor['word_edits']),firstscore_vs_upstream=dict(rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),candidate_rms_magnitude=float(np.sqrt(np.mean(x*x))),reference_rms_magnitude=float(np.sqrt(np.mean(reference.astype('f8')**2)))),resources=r['commands'][label]);save()
   paired=[]
   for layer in range(32):
    x=np.fromfile(arm_paths[(case,'on')]/f'prefill.layer.{layer}.f32',dtype='<f4');y=np.fromfile(arm_paths[(case,'off')]/f'prefill.layer.{layer}.f32',dtype='<f4');require(x.size and x.shape==y.shape and np.isfinite(x).all()and np.isfinite(y).all(),'paired finite hidden');d=x.astype('f8')-y;paired.append(dict(layer=layer,bit_exact=x.tobytes()==y.tobytes(),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),off_rms_magnitude=float(np.sqrt(np.mean(y.astype('f8')**2))),on_rms_magnitude=float(np.sqrt(np.mean(x.astype('f8')**2)))))
   r['cases'][case]['paired_native_layers']=dict(scope='Native OFF/ON, not actual source hidden',first_different_layer=next((v['layer']for v in paired if not v['bit_exact']),None),layers=paired);save()
  require(sha(candidate)==cfg['model_sha256'],'weights unchanged');r['status']='measured_physical_rope_observed_ab';r['limits']=['Observed calibration only; no heldout data accessed or qualification.','One teacher row plus full independent greedy; Native tiles remain at most16; full upstream graph equivalence is not assumed.','Source-exact RoPE/component evidence is not decoded quality or speed proof; observed source floors/EOS remain separate gates.']
 except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  deadline.close()
  for sig,handler in prior.items():signal.signal(sig,handler)
  r['elapsed_seconds']=time.monotonic()-started;r['artifacts']=shared.artifacts(out,r['commands']);save()
if __name__=='__main__':main()
