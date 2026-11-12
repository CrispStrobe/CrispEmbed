#!/usr/bin/env python3
"""Receipt-first aligned fixed diagnostic; immutable RoPEON baselines, private book outputs."""
import hashlib,importlib.util,json,os,shutil,signal,stat,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
shared=load('fixed_shared',ROOT/'tools/q6-tiled-full-ab/worker.py');base=shared.base;require=shared.require;sha=shared.sha
sys.path.insert(0,str(ROOT/'tools/heldout'));import execution as ex
GATE='CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL_ALIGNED'
CFG_SHA='396e2c2c740f015c3603315eaadf17f443ec87508895583e2e3c425b5fab2803'
def admit(path):
 require(sha(path)==CFG_SHA,'immutable fixed regression protocol');cfg=json.loads(Path(path).read_text());require(isinstance(cfg['observed_receipt_sha256'],str)and len(cfg['observed_receipt_sha256'])==64,'freeze observed receipt before fixed execution');require([case['id']for case in cfg['cases']]==cfg['case_order']==['historical-receipt','two-column-page','inventory-table']and cfg['arm_order']==[name+'/'+arm for name in cfg['case_order']for arm in ['off','on']]and cfg['continue_after_decoded_negative'],'predeclared receipt-first six-arm order');selection=ROOT/'docs/lighton4b-heldout-v1.json';require(sha(selection)==cfg['selection_protocol_sha256'],'unchanged selection');selected=json.loads(selection.read_text());ex.prepare.admit(selected)
 require(sha(ROOT/'docs/lighton4b-heldout-executor-v1.json')==cfg['frozen_executor_sha256'],'unchanged original model/reference protocol')
 for case in selected['cases']:
  for key in ['source','gold']:require(sha(ROOT/case[key])==case[key+'_sha256'],'unchanged source/gold')
 for name,digest in selected['reference']['scorer'].items():require(sha(ROOT/name)==digest,'unchanged scorer')
 require(sha(ROOT/'tests/regression/images/cc0/ground_truth.json')==selected['reference']['gold_registry_sha256'],'unchanged registry');return cfg,selected

def gates(before):
 off={k:v for k,v in before.items()if k.startswith('CRISPEMBED_')};off[GATE]='0';off['CRISPEMBED_LIGHTON4B_PROVISIONAL_HELDOUT']='1';off['CRISPEMBED_LIGHTON4B_FROZEN_CASES']='1';on=dict(off);on[GATE]='1'
 require({k for k in off if off[k]!=on[k]}=={GATE},'only physical source alignment changes')
 for key in ['PHYSICAL_SOURCE_PREFILL','PHYSICAL_GDN_SOURCE_OPS','PHYSICAL_Q4_REPACK','PHYSICAL_Q6_TILED','PHYSICAL_PILLOW_RESIZE','PHYSICAL_F16_VISION','PHYSICAL_F16_ATTENTION']:require(off['CRISPEMBED_LIGHTON4B_'+key]=='1','fixed corrected/planner tuple')
 for key in ['HEAD_BF16','BF16_LINEAR','BF16_QUANT_WEIGHTS']:require(off['CRISPEMBED_LIGHTON4B_'+key]=='0','fixed text/head precision')
 require(off.get('CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU')==off.get('CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE')=='1','fused SwiGLU and source RoPE remainON');return off,on

def stop_state(summary,ids,expected_tokens=None,expected_eos=None,cap=1536,prompt_tokens=None,image_tokens=None):
 require(not any(token in [248044,248046]for token in ids[:-1]),'no ignored earlier EOS')
 require(summary['generated_tokens']==len(ids)and 1<=len(ids)<=cap and summary['eos']==(ids[-1]in[248044,248046])and(summary['eos']or len(ids)==cap),'actual EOS/cap')
 require(summary['cap']==cap and summary['upstream_physical']and 1<=summary['prompt_tokens']<=512 and 0<summary['image_tokens']<=256 and summary['prompt_tokens']+cap<=2048,'actual physical admission/cap')
 if prompt_tokens is not None:require(summary['prompt_tokens']==prompt_tokens and summary['image_tokens']==image_tokens,'actual sealed token/image counts')
 if expected_tokens is not None:require(len(ids)==expected_tokens and summary['eos']is expected_eos,'frozen baseline stop/count')
 require(summary['physical_source_prefill_chunks']and summary['physical_q6_tiled']and summary['physical_q4_repack']and summary['physical_repack_entries']==216 and 0<summary['physical_repack_bytes']<=2<<30,'actual bounded quantized modes')
 require(summary['converted_f32_weight_entries']==summary['converted_f32_weight_bytes']==0 and summary['teacher_positions']==1 and not summary['source_fed_vision'],'independent pixel arm/no F32 text cache')

def admit_observed(observed,cfg,control_sha,bundle):
 require(observed['status']=='measured_source_aligned_observed_ab'and observed['runtime_producer_source']==cfg['candidate_source']and observed['candidate_controls_sha256']==control_sha and observed['runtime_bundle']==bundle and set(observed['cases'])=={'simple_table','simple_form','arabic_printed_line'},'same candidate completed observed screen')
 require(all(case['off_all_rows_stages_exact']and set(case['arms'])=={'off','on'}for case in observed['cases'].values()),'all6 observed arms and exact old RoPEON OFF closure')
 # Decoded floors remain binding results; they do not skip declared cases.
 for case in observed['cases'].values():
  for arm in case['arms'].values():
   summary=arm['summary'];require(1<=summary['generated_tokens']<=512 and(summary['eos']or summary['generated_tokens']==512),'observed EOS or explicit cap-negative')

def archive(root,arm):
 row=arm['raw_rows'];a=row['archive'];require(a['file']=='greedy-raw-rows.zip','closed historical archive name');p=Path(root)/a['file'];require(p.stat().st_size==a['bytes']and sha(p)==a['sha256'],'whole historical raw archive identity')
 with zipfile.ZipFile(p)as z:
  names=[f'greedy.{i}.logits.f32'for i in range(len(row['rows']))];require(len(z.infolist())==len(names)and set(z.namelist())==set(names),'exact historical ZIP census')
  for name,v in zip(names,row['rows']):require(z.getinfo(name).file_size==v['bytes']==248320*4,'full historical vocabulary geometry')
 return a

def main():
 import numpy as np
 from PIL import Image
 cfg,selection=admit(Path(__file__).with_name('fixed-protocol.json'));require(np.__version__=='2.3.5'and Image.__version__=='11.3.0'and Path('/kaggle/input').is_dir(),'hosted serializer profile');out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'source-aligned-fixed';work.mkdir();start=time.monotonic();deadline=shared.Deadline(cfg['worker_seconds']);deadline.__enter__();prior={s:signal.signal(s,lambda sig,frame:(_ for _ in()).throw(SystemExit(128+sig)))for s in[signal.SIGTERM,signal.SIGINT]};r=dict(schema=cfg['schema'],status='failed',commands={},cases={},qualified=False,references_expert_adjudicated=False,reference_forward_executed=False,source_feature_forward_executed=False,heldout_forward_executed=False,model_acquisition=False)
 def save():(out/'source-aligned-fixed-panel.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
 def run(label,args,env=None,limit=60,on_tick=None):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:shared.command(args,out/(label+'.log'),min(limit,cfg['worker_seconds']-(time.monotonic()-start)),r['commands'][label],env=env,cwd=ROOT,on_tick=on_tick)
  finally:save()
 def locate(name,digest):
  matches=[p for root in [Path('/kaggle/input'),out]for p in root.rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(matches)==1,'one pinned '+name);return matches[0],json.loads(matches[0].read_text())
 def verify(p,row):require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'immutable owned artifact '+p.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['controller_source']=(out/'source.log').read_text().strip();require(r['controller_source']==cfg['controller_source']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'fixed controller');r['capacity']=dict(memory=base.memory(),scratch=shutil.disk_usage(work).free,output=shutil.disk_usage(out).free);require(r['capacity']['memory']['MemAvailable']>=10<<30 and r['capacity']['scratch']>=25<<30 and r['capacity']['output']>=12<<30,'resources before model acquisition')
  cp,control=locate('receipt-aligned-controls.json',cfg['candidate_controls_sha256']);require(control['status']=='measured_exact_receipt_aligned_controls'and control['passed']and control['candidate_source']==cfg['candidate_source']and control['all22_legacy_stage_cache_hashes_exact']and control['all56_source_stage_comparisons_exact']and control['source_image_end_state']['bit_exact']and control['source_image_end_fifo']['bit_exact'],'exact aligned source controls');require(len(control['unit_controls'])==5 and all('pass'in value.lower()and 'skip'not in value.lower()for value in control['unit_controls'].values()),'actual five supported units');r['candidate_source']=control['candidate_source'];r['controls_sha256']=sha(cp);r['runtime_bundle']=control['runtime_bundle'];bundle=work/'runtime'
  for name,row in control['runtime_bundle'].items():
   n=Path(name);require(not n.is_absolute()and '..'not in n.parts and n.parts[0]in['bin','lib'],'closed tested runtime member');src=cp.parent/'runtime'/n;verify(src,row);dest=bundle/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);verify(dest,row)
  native=bundle/'bin/lighton4b-native-document';preflight=Path(os.environ['LIGHTON4B_PREFLIGHT_BINARY']);require(sha(preflight)==os.environ['LIGHTON4B_PREFLIGHT_SHA256'],'exact metadata preflight');r['preflight_binary_sha256']=sha(preflight);require(sha(native)==cfg['native_binary_sha256'],'exact tested native-document bytes');r['preflight_candidate_source']=os.environ['LIGHTON4B_PREFLIGHT_SOURCE'];require(r['preflight_candidate_source']==cfg['candidate_source'],'source-identical metadata probe')
  for p in [native,preflight]:p.chmod(p.stat().st_mode|stat.S_IXUSR)
  observedp,observed=locate('source-aligned-observed-ab.json',cfg['observed_receipt_sha256']);admit_observed(observed,cfg,sha(cp),control['runtime_bundle'])
  for name,row in observed['artifacts'].items():verify(observedp.parent/name,row)
  r['observed_receipt_sha256']=sha(observedp);r['observed_controls_reused_without_forward']=True
  prepared=[]
  for case_cfg in cfg['cases']:
   name=case_cfg['id'];bp,baseline=locate(case_cfg['baseline_receipt'],case_cfg['baseline_sha256']);op,original=locate('heldout-result.json',case_cfg['reference_sha256']);ap=ROOT/'tests/results/qwen35-ocr'/case_cfg['audit_receipt'];require(sha(ap)==case_cfg['audit_sha256'],'unchanged independent raw audit');audit=json.loads(ap.read_text());require(audit['status']==case_cfg['audit_status']and audit['receipt_sha256']==sha(bp)and audit['all_raw_finite_argmax_sha']and audit['all_owned_artifact_hashes']and name in audit['cases'],'completed independent RoPE raw audit');require(original['status']=='measured_provisional_heldout'and original['case']==name and original['source_observer_full_neutral'],'frozen complete source references');require(baseline['candidate_source']==cfg['prior_rope_candidate_source'],'exact old RoPE runtime');oldcase=baseline['cases'][name];oldarm=oldcase['arms']['on'];require(len(oldarm['ids'])==case_cfg['baseline_tokens']and oldarm['summary']['eos']is case_cfg['baseline_eos']and oldcase['pre_forward_input_seal_sha256']==case_cfg['seal_sha256']==original['pre_forward_input_seal_sha256'],'audited RoPE baseline stop/seal');archive(bp.parent/case_cfg['baseline_prefix'],oldarm);require(sha(bp.parent/name/'pre-forward-inputs.json')==case_cfg['seal_sha256'],'actual retained RoPE input seal')
   prefix=case_cfg['baseline_prefix']+'/';stages=[n for n in baseline['artifacts']if n.startswith((prefix+'prefill.',prefix+'vision.',prefix+'processor.'))or n in[prefix+'prompt.ids',prefix+'modalities.ids',prefix+'text.embedded.f32']]
   for n in stages:verify(bp.parent/n,baseline['artifacts'][n])
   needed=['inputs/input.png','inputs/input.rgb','upstream-off/response.json','upstream-on/response.json','upstream-on/first-score.f32','upstream-on/source.vision.0.f32','source-preflight.prompt.ids','source-preflight.modalities.ids','source-preflight.positions.f32']
   for n in needed:verify(op.parent/n,original['artifacts'][n])
   off_response=json.loads((op.parent/'upstream-off/response.json').read_text());on_response=json.loads((op.parent/'upstream-on/response.json').read_text());ex.neutral(off_response,on_response);require(ex.source_response(off_response,off_response['tokens_evaluated']),'complete unchanged source reference');selected=next(v for v in selection['cases']if v['id']==name);png=op.parent/'inputs/input.png';rgb=op.parent/'inputs/input.rgb';image=Image.open(png).convert('RGB');require(sha(png)==selected['prepared']['png_sha256']and sha(rgb)==selected['prepared']['rgb_sha256']and image.tobytes()==rgb.read_bytes(),'frozen prepared pixels');prepared.append((case_cfg,bp,baseline,oldarm,op,original,stages,selected,image,off_response))
  r['all_three_historical_inputs_baselines_audits_authenticated_before_models']=True;save();off_gates,on_gates=gates(baseline['gates']['on'])
  def environment(values):
   env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(values);env['LD_LIBRARY_PATH']=str(bundle/'lib');return env
  envs={'off':environment(off_gates),'on':environment(on_gates)};r['gates']={'off':off_gates,'on':on_gates};teacher=work/'one.ids';teacher.write_text('248046\n');original_cfg=json.loads((ROOT/'docs/lighton4b-heldout-executor-v1.json').read_text());sys.path.insert(0,str(ROOT/'tools'));import lighton4b_upstream_physical_pack as pack
  def download(url,path,size,digest):
   r['phase']='download-'+Path(path).name;records=r.setdefault('downloads',{}).setdefault(Path(path).name,[]);save()
   try:shared.transfer(base.download,url,path,size,digest,records)
   finally:save()
  retained=observed['retained_quantized_model'];require(retained['file']=='physical-model.gguf'and retained['sha256']==cfg['model_sha256'],'exact retained quantized model');candidate=observedp.parent/retained['file'];verify(candidate,retained);require(observed['artifacts'][retained['file']]=={key:retained[key]for key in ['bytes','sha256']},'same retained model manifest');r['retained_model_reused']=True;r['model_acquisition']=False
  pack_source=observedp.parent/'physical-pack.json';verify(pack_source,observed['artifacts']['physical-pack.json']);shutil.copy2(pack_source,out/'physical-pack.json');packed=json.loads(pack_source.read_text());require(packed['candidate_sha256']==cfg['model_sha256']and not packed['requantization_executed']and not packed['full_f32_model_created'],'unchanged physical weights')
  for case_cfg,bp,baseline,oldarm,op,original,stages,selected,image,response in prepared:
   name=case_cfg['id'];folder=out/name;folder.mkdir();r['cases'][name]=dict(arms={},historical_baseline_sha256=sha(bp),historical_raw_audit_sha256=case_cfg['audit_sha256'],historical_reference_sha256=sha(op),historical_baseline_completed=case_cfg['baseline_eos']);case=r['cases'][name];rgb=op.parent/'inputs/input.rgb';pre=folder/'native-preflight';pre.mkdir();run(name+'-preflight',[preflight,candidate,rgb,image.height,image.width,pre,'lighton4b-upstream-q4km-physical-v1'],envs['off'],120);ids=list(map(int,(pre/'prompt.ids').read_text().split()));mods=list(map(int,(pre/'modalities.ids').read_text().split()));grid=list(map(int,(pre/'grid.ids').read_text().split()));source_ids=list(map(int,(op.parent/'source-preflight.prompt.ids').read_text().split()));source_mods=list(map(int,(op.parent/'source-preflight.modalities.ids').read_text().split()));seal=folder/'pre-forward-inputs.json';seal_sha=ex.seal_inputs(seal,selected,ids,mods,grid,(pre/'prefill.positions.f32').read_bytes(),source_ids,source_mods,(op.parent/'source-preflight.positions.f32').read_bytes(),original_cfg['expected_media_prompt'],original_cfg);require(seal_sha==case_cfg['seal_sha256'],'actual same-source pre-forward seal');case['pre_forward_input_seal_sha256']=seal_sha;gold=json.loads((ROOT/selected['gold']).read_text());source_score=np.fromfile(op.parent/'upstream-on/first-score.f32',dtype='<f4').astype('f8');require(source_score.size==248320 and np.isfinite(source_score).all(),'finite full source vocabulary');case['reference_score']=ex.score(gold,response['content'],True);save()
   for arm in ['off','on']:
    admission=dict(memory=base.memory(),output_free=shutil.disk_usage(out).free,actual_seal_sha256=sha(seal),expected_seal_sha256=seal_sha);admission['accepted']=bool(admission['memory']['MemAvailable']>=10<<30 and admission['output_free']>=2<<30 and admission['actual_seal_sha256']==seal_sha);r.setdefault('admissions',{})[name+'-'+arm]=admission;save();require(admission['accepted'],'per-arm resources/seal');dest=folder/arm;dest.mkdir();checked=[0]
    def tick():
     if arm!='off':return
     n=shared.progress_count(dest)
     if n is None:return
     while checked[0]<min(len(oldarm['ids']),n-1):
      i=checked[0];require(sha(dest/f'greedy.{i}.logits.f32')==oldarm['raw_rows']['rows'][i]['sha256'],'early exact historical OFF row');checked[0]+=1
    r['heldout_forward_attempted']=True;run(name+'-'+arm,ex.native_args(native,candidate,rgb,image.height,image.width,teacher,dest,'native-composed'),envs[arm],case_cfg['arm_seconds'],tick);summary=json.loads((dest/'native-summary.json').read_text());generated=list(map(int,(dest/'generated.ids').read_text().split()));text=(dest/'generated.txt').read_bytes().decode();r['heldout_forward_executed']=True;stop_state(summary,generated,case_cfg['baseline_tokens']if arm=='off'else None,case_cfg['baseline_eos']if arm=='off'else None,1536,len(ids),mods.count(1));require(summary['provisional_heldout_cap']==1536,'unchanged actual diagnostic cap')
    if arm=='off':
     require(generated==oldarm['ids']and text==oldarm['decoded'],'all historical OFF IDs/text including capped book')
     for i,row in enumerate(oldarm['raw_rows']['rows']):require(sha(dest/f'greedy.{i}.logits.f32')==row['sha256'],'every historical OFF vocabulary row')
     prefix=case_cfg['baseline_prefix']+'/'
     for n in stages:require(sha(dest/n[len(prefix):])==baseline['artifacts'][n]['sha256'],'all historical OFF text/vision stages')
     require(summary['physical_source_prefill_aligned']is False and {key:summary[key]for key in oldarm['summary']}==oldarm['summary']and set(summary)==set(oldarm['summary'])|{'physical_source_prefill_aligned'},'OFF old summary exact plus only aligned false telemetry');case['off_all_historical_rows_stages_stop_exact']=True
    else:
     require(summary['physical_source_prefill_aligned']is True and summary['physical_rope']is True,'actual ON aligned/source RoPE context')
     names=[p.name for p in(folder/'off').iterdir()if p.is_file()and(p.name.startswith(('vision.','processor.'))or p.name in['prompt.ids','modalities.ids','text.embedded.f32','prefill.input.f32','prefill.positions.f32'])]
     for n in names:require(sha(dest/n)==sha(folder/'off'/n),'unchanged pixels/input across RoPE')
    x=np.fromfile(dest/'prefill.logits.f32',dtype='<f4').astype('f8');require(x.shape==source_score.shape and np.isfinite(x).all()and sha(dest/'prefill.logits.f32')==sha(dest/'teacher.0.logits.f32')==sha(dest/'greedy.0.logits.f32'),'actual first-row identity');d=x-source_score;score=ex.score(gold,text,summary['eos']);case['arms'][arm]=dict(summary=summary,chunk_sizes=list(map(int,(dest/'prefill.chunk_sizes.ids').read_text().split())),ids=generated,decoded=text,score=score,raw_rows=ex.audit_rows(dest,generated),meets_provisional_source_floor=ex.floor(score,case['reference_score'],name=='inventory-table'),firstscore=dict(rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),candidate_rms_magnitude=float(np.sqrt(np.mean(x*x))),source_rms_magnitude=float(np.sqrt(np.mean(source_score*source_score)))),resources=r['commands'][name+'-'+arm]);save()
   pairs=[]
   for layer in range(32):
    a=np.fromfile(folder/'off'/f'prefill.layer.{layer}.f32',dtype='<f4').astype('f8');b=np.fromfile(folder/'on'/f'prefill.layer.{layer}.f32',dtype='<f4').astype('f8');require(a.shape==b.shape and a.size and np.isfinite(a).all()and np.isfinite(b).all(),'paired finite native geometry');d=b-a;pairs.append(dict(layer=layer,bit_exact=np.array_equal(a,b),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),off_rms_magnitude=float(np.sqrt(np.mean(a*a))),on_rms_magnitude=float(np.sqrt(np.mean(b*b)))))
   case['paired_native_layers']=dict(scope='same-source native OFF/ON, not unavailable source hidden',first_different_layer=next((v['layer']for v in pairs if not v['bit_exact']),None),layers=pairs);save()
  require(sha(candidate)==cfg['model_sha256'],'immutable final model');r['status']='measured_source_aligned_fixed_panel';r['all_on_complete']=all(c['arms']['on']['summary']['eos']for c in r['cases'].values());r['on_source_floors_passed']=sum(c['arms']['on']['meets_provisional_source_floor']for c in r['cases'].values())
 except BaseException as error:r.update(error_type=type(error).__name__,error=str(error));raise
 finally:
  deadline.close()
  for sig,handler in prior.items():signal.signal(sig,handler)
  r['elapsed_seconds']=time.monotonic()-start;r['artifacts']=shared.artifacts(out,r['commands']);save()
if __name__=='__main__':main()
