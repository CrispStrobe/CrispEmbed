#!/usr/bin/env python3
"""Independent artifact-only inventory, raw-row and decoded-score audit; no inference."""
import argparse,hashlib,importlib.util,json,signal,sys,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];VOCAB=248320
PROTOCOL_SHA='0d3c75097b78fa477e460d6d137dac6bafd9e283c7da93f35fa2ae9cd20cb419'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def owned(root,name):
 n=Path(name);require(not n.is_absolute()and '..'not in n.parts,'closed relative artifact path');p=Path(root)/n;require(p.is_file()and not p.is_symlink()and p.resolve().is_relative_to(Path(root).resolve()),'owned regular artifact');return p
def raw_archive(root,record,ids,known):
 a=record['archive'];p=owned(root,a['file']);require(p.stat().st_size==a['bytes']and known[p.resolve()]==a['sha256'],'actual whole archived bytes');rows=record['rows'];require(len(rows)==len(ids)and 1<=len(ids)<=1536,'raw row count');chain=hashlib.sha256()
 with zipfile.ZipFile(p)as z:
  names=[f'greedy.{i}.logits.f32'for i in range(len(ids))];require(len(z.infolist())==len(names)and set(z.namelist())==set(names),'closed raw ZIP members')
  for i,(name,row,token)in enumerate(zip(names,rows,ids)):
   require(row['position']==i and row['bytes']==VOCAB*4 and row['argmax']==token and type(token)is int and 0<=token<VOCAB,'frozen actual row identity');info=z.getinfo(name);require(info.file_size==VOCAB*4 and info.compress_type==zipfile.ZIP_STORED,'bounded lossless actual vocabulary member');b=z.read(name);require(len(b)==VOCAB*4 and hashlib.sha256(b).hexdigest()==row['sha256'],'all archived row SHA');x=np.frombuffer(b,dtype='<f4');require(np.isfinite(x).all()and int(x.argmax())==token,'all finite actual greedy argmaxes');chain.update(bytes.fromhex(row['sha256']))
 return dict(rows=len(ids),archive_sha256=a['sha256'],ordered_row_sha256=chain.hexdigest())
def edit_count(a,b):
 if not a:return len(b)
 mask=(1<<len(a))-1;high=1<<(len(a)-1);peq={}
 for i,item in enumerate(a):peq[item]=peq.get(item,0)|(1<<i)
 pv=mask;mv=0;score=len(a)
 for item in b:
  eq=peq.get(item,0);xv=eq|mv;xh=(((eq&pv)+pv)^pv)|eq;ph=mv|~(xh|pv);mh=pv&xh;score+=bool(ph&high)-bool(mh&high);ph=((ph<<1)|1)&mask;mh=(mh<<1)&mask;pv=(mh|~(xv|ph))&mask;mv=ph&xv
 return score
def independent_edits(gold,text,score):
 result={}
 for view,reference,actual in [('raw',gold['text'],text),('content',' '.join(gold['text'].split()),score.get('parsed_text'))]:
  if actual is None:result[view]=None;continue
  ce=edit_count(reference,actual);we=edit_count(reference.split(),actual.split());require(ce/max(1,len(reference))==score[view]['cer']and we/max(1,len(reference.split()))==score[view]['wer'],'independent bit-parallel character/word distances');result[view]=dict(character_edits=ce,reference_codepoints=len(reference),word_edits=we,reference_words=len(reference.split()))
 return result
def safe_score(score):
 return {k:v for k,v in score.items()if k in ['policy','complete','raw','characters','lines','parse_ok','content','table_gold_available','content_exact','table','exact_quality','expected_table_shape','table_shape_valid']}
def actual_summary(s,ids,cap,expected_n=None,expected_eos=None):
 require(s['generated_tokens']==len(ids)and 1<=len(ids)<=cap and s['cap']==cap and s['eos']==(ids[-1]in[248044,248046])and(s['eos']or len(ids)==cap),'actual count/EOS/cap')
 if expected_n is not None:require(len(ids)==expected_n and s['eos']is expected_eos,'historical exact count/stop')
 require(s['upstream_physical']and not s['source_fed_vision']and s['teacher_positions']==1 and s['physical_source_prefill_chunks']and s['physical_q6_tiled']and s['physical_q4_repack'],'actual pixel/planner/quantized mode');require(s['physical_repack_entries']==216 and 0<s['physical_repack_bytes']<=2<<30 and s['converted_f32_weight_entries']==s['converted_f32_weight_bytes']==0,'bounded quantized-only cache');require(1<=s['prompt_tokens']<=512 and 0<s['image_tokens']<=256 and s['prompt_tokens']+cap<=2048,'actual context/image bounds')

def audit(receipt,digest):
 require(np.__version__=='2.3.5','pinned serializer');require(len(digest)==64 and all(c in'0123456789abcdef'for c in digest)and sha(receipt)==digest,'exact completed panel receipt');r=json.loads(receipt.read_text());root=receipt.parent;protocol=Path(__file__).with_name('protocol.json');require(sha(protocol)==PROTOCOL_SHA,'exact frozen panel protocol');cfg=json.loads(protocol.read_text());require(r['status']=='measured_swiglu_fixed_regression_panel'and r['qualified']is False and r['candidate_source']==cfg['candidate_source']and r['controller_source']==cfg['controller_source']and not r['reference_forward_executed']and not r['source_feature_forward_executed'],'completed exact native-only diagnostic');require(set(r['cases'])=={c['id']for c in cfg['cases']},'all fixed cases');sys.path.insert(0,str(ROOT/'tools/heldout'));import execution as ex
 require(sha(ROOT/'docs/lighton4b-heldout-v1.json')==cfg['selection_protocol_sha256'],'frozen source selection');selection=json.loads((ROOT/'docs/lighton4b-heldout-v1.json').read_text());ex.prepare.admit(selection)
 for p,h in selection['reference']['scorer'].items():require(sha(ROOT/p)==h,'frozen scorer source')
 require(sha(ROOT/'tests/regression/images/cc0/ground_truth.json')==selection['reference']['gold_registry_sha256'],'frozen registry')
 def locate(name,h):
  found=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==h];require(len(found)==1,'one exact historical receipt');return found[0],json.loads(found[0].read_text())
 known={};total=0
 for name,row in r['artifacts'].items():
  p=owned(root,name);require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'all actual owned artifact sizes/SHA');known[p.resolve()]=row['sha256'];total+=p.stat().st_size
 cp=owned(root,'swiglu-fixed-controls.json');require(sha(cp)==r['controls_sha256'],'actual candidate control receipt');control=json.loads(cp.read_text());require(control['passed']and not control['checkpoint_loaded']and control['source_sha']==cfg['candidate_source']and len(control['unit_control_outputs'])==8,'actual same-session supported controls');require(all(control['source_files'][n]['sha256']==h for n,h in cfg['source_files'].items()),'frozen production source/CLI/preflight')
 for n,row in control['runtime_bundle'].items():require(known[owned(root,'runtime/'+n).resolve()]==row['sha256'],'actual tested executable/backend bundle')
 retained_logs=0;overwritten=[]
 for command in control['commands']:
  p=owned(root,command['label']+'.log');actual=sha(p)
  if actual==command['sha256']:retained_logs+=1;continue
  require(command['label']=='source'and p.read_text().strip()==cfg['controller_source']and command['sha256']==hashlib.sha256((cfg['candidate_source']+'\n').encode()).hexdigest(),'only declared candidate/controller git-rev log collision')
  overwritten.append(dict(file='source.log',original_candidate_log_sha256=command['sha256'],retained_controller_log_sha256=actual))
 require(len(overwritten)==1,'exact known source-log provenance limitation')
 off,on=r['gates']['off'],r['gates']['on'];require(set(off)==set(on)and {k for k in off if off[k]!=on[k]}==set(cfg['changed_gates'])and off[cfg['changed_gates'][0]]=='0'and on[cfg['changed_gates'][0]]=='1','single frozen gate intervention')
 report=dict(schema='lighton4b-swiglu-fixed-independent-raw-audit/v1',status='complete',receipt_sha256=digest,owned_artifacts=len(known),owned_artifact_bytes=total,all_owned_artifact_hashes=True,all_raw_finite_argmax_sha=True,candidate_controls_sha256=sha(cp),source_feature_forward_executed=False,model_forward_executed=False,qualified=False,rights='Book recognized text and token IDs remain private; audit publishes counts, metrics and hashes only.',observed={},cases={});report['model_free_command_log_retention']=dict(retained_log_hashes_verified=retained_logs,overwritten_logs=overwritten,original_candidate_git_rev_log_bytes_available=False,original_candidate_git_rev_log_independently_rehashed=False,pinned_source_file_hashes_and_tested_runtime_bundle_verified=True,scope='The model worker overwrote root source.log with the controller revision. Original candidate git-rev log bytes were not retained or independently rehashed. Its recorded hash remains in the immutable control receipt; pinned source files, tested bundle and other retained command logs are checked. No claim that all original control logs were rehashed.')
 row_total=0
 for arm in ['off','on']:
  folder=root/('observed-'+arm);s=json.loads(owned(folder,'native-summary.json').read_text());ids=list(map(int,owned(folder,'generated.ids').read_text().split()));actual_summary(s,ids,512,cfg['observed_controls'][arm],True);require(s==r['observed-'+arm]['summary'],'actual observed summary matches report');raw=raw_archive(folder,r['observed-'+arm]['raw_rows'],ids,known);report['observed'][arm]=dict(tokens=len(ids),eos=s['eos'],raw=raw);row_total+=len(ids)
  h=cfg['observed_off_receipt_sha256']if arm=='off'else cfg['observed_on_receipt_sha256'];p,old=locate('source-prefill-observed-ab.json'if arm=='off'else'swiglu-observed-ab.json',h);gold=old['cases']['simple_form']['arms']['on'];require(ids==gold['generated_ids']and owned(folder,'generated.txt').read_bytes().decode()==gold['decoded'],'actual observed text/IDs');prefix='simple_form/on/'
  for n,row in old['artifacts'].items():
   if n.startswith((prefix+'prefill.',prefix+'vision.',prefix+'processor.'))or n in[prefix+'prompt.ids',prefix+'modalities.ids',prefix+'text.embedded.f32']:require(known[owned(folder,n[len(prefix):]).resolve()]==row['sha256'],'all actual observed historical stages')
  require([v['sha256']for v in r['observed-'+arm]['raw_rows']['rows']]==[old['artifacts'][prefix+f'greedy.{i}.logits.f32']['sha256']for i in range(len(ids))],'all actual observed historical rows')
 for c in cfg['cases']:
  name=c['id'];v=r['cases'][name];folder=root/name;bp,baseline=locate(c['baseline_receipt'],c['baseline_sha256']);ap,old_audit=locate('raw-audit.json',c['audit_sha256']);op,original=locate('heldout-result.json',c['reference_sha256']);require(old_audit['status']=='complete'and old_audit['receipt_sha256']==sha(bp)and old_audit['all_raw_finite_argmax_sha']and old_audit['all_owned_artifact_hashes'],'independently audited original baseline');old=baseline['arm']if name=='two-column-page'else baseline['arms']['corrected-composed'];a=old['raw_rows']['archive'];oldzip=owned(bp.parent/c['baseline_prefix'],a['file']);require(sha(oldzip)==a['sha256']and oldzip.stat().st_size==a['bytes'],'original historical archive retained');require(sha(owned(folder,'pre-forward-inputs.json'))==c['seal_sha256']==v['pre_forward_input_seal_sha256']==original['pre_forward_input_seal_sha256'],'actual original input seal');selected=next(x for x in selection['cases']if x['id']==name)
  for rel,key in [('inputs/input.png','png_sha256'),('inputs/input.rgb','rgb_sha256')]:require(sha(owned(op.parent,rel))==selected['prepared'][key]==original['artifacts'][rel]['sha256'],'actual frozen prepared image bytes')
  require(sha(ROOT/selected['gold'])==selected['gold_sha256'],'fixed provisional label');gold=json.loads((ROOT/selected['gold']).read_text())
  responses=[]
  for role in ['upstream-off','upstream-on']:
   n=role+'/response.json';p=owned(op.parent,n);require(sha(p)==original['artifacts'][n]['sha256'],'actual immutable source response');responses.append(json.loads(p.read_text()))
  ex.neutral(*responses);require(ex.source_response(responses[0],responses[0]['tokens_evaluated']),'complete original source response');source_score=ex.score(gold,responses[0]['content'],True);require(source_score==v['reference_score'],'unchanged provisional source floor');sourcep=owned(op.parent,'upstream-on/first-score.f32');require(sha(sourcep)==original['artifacts']['upstream-on/first-score.f32']['sha256'],'actual source score anchor');source=np.fromfile(sourcep,dtype='<f4').astype('f8');require(source.size==VOCAB and np.isfinite(source).all(),'finite full source vocabulary');seal=json.loads(owned(folder,'pre-forward-inputs.json').read_text());pre=folder/'native-preflight';require(list(map(int,owned(pre,'prompt.ids').read_text().split()))==seal['prompt_ids']and list(map(int,owned(pre,'modalities.ids').read_text().split()))==seal['modalities']and list(map(int,owned(pre,'grid.ids').read_text().split()))==seal['grid']and sha(owned(pre,'prefill.positions.f32'))==seal['positions_sha256'],'actual preflight metadata seals');feature=owned(op.parent,'upstream-on/source.vision.0.f32');require(sha(feature)==original['artifacts']['upstream-on/source.vision.0.f32']['sha256'],'actual immutable upstream merged feature');out_case=dict(reference_score=safe_score(source_score),reference_independent_edits=independent_edits(gold,responses[0]['content'],source_score),arms={})
  for arm in ['off','on']:
   av=v['arms'][arm];dest=folder/arm;s=json.loads(owned(dest,'native-summary.json').read_text());ids=list(map(int,owned(dest,'generated.ids').read_text().split()));text=owned(dest,'generated.txt').read_bytes().decode();actual_summary(s,ids,1536,c['baseline_tokens']if arm=='off'else None,c['baseline_eos']if arm=='off'else None);require(s==av['summary']and ids==av['ids']and text==av['decoded'],'actual full summary/IDs/text');seal=json.loads(owned(folder,'pre-forward-inputs.json').read_text());require(s['prompt_tokens']==seal['prompt_tokens']and s['image_tokens']==seal['image_tokens'],'actual sealed prompt/image counts');require(s['provisional_heldout_cap']==1536,'actual diagnostic cap');require(list(map(int,owned(dest,'prompt.ids').read_text().split()))==seal['prompt_ids']and list(map(int,owned(dest,'modalities.ids').read_text().split()))==seal['modalities']and sha(owned(dest,'prefill.positions.f32'))==seal['positions_sha256'],'actual arm input seal');require(known[owned(dest,'vision.merged.f32').resolve()]==sha(feature),'actual native merged feature equals original source');raw=raw_archive(dest,av['raw_rows'],ids,known);row_total+=len(ids);first=owned(dest,'prefill.logits.f32');require(known[first.resolve()]==known[owned(dest,'teacher.0.logits.f32').resolve()]==av['raw_rows']['rows'][0]['sha256'],'actual first row triple identity');scores=ex.score(gold,text,s['eos']);require(scores==av['score'],'all actual decoded quality fields');floor=ex.floor(scores,source_score,name=='inventory-table');require(floor==av['meets_provisional_source_floor'],'unchanged EOS/parse/content/table floor');x=np.fromfile(first,dtype='<f4').astype('f8');d=x-source;numeric=dict(rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),candidate_rms_magnitude=float(np.sqrt(np.mean(x*x))),source_rms_magnitude=float(np.sqrt(np.mean(source*source))));require(numeric==av['firstscore'],'independent actual first-score errors/magnitudes');out_case['arms'][arm]=dict(tokens=len(ids),eos=s['eos'],score=safe_score(scores),independent_edits=independent_edits(gold,text,scores),source_floor=floor,raw=raw,firstscore=numeric)
   if arm=='off':
    require(ids==old['ids']and text==old['decoded'],'historical OFF actual complete or capped text/IDs');require([x['sha256']for x in av['raw_rows']['rows']]==[x['sha256']for x in old['raw_rows']['rows']],'every historical OFF raw row');prefix=c['baseline_prefix']+'/'
    for n,row in baseline['artifacts'].items():
     if n.startswith((prefix+'prefill.',prefix+'vision.',prefix+'processor.'))or n in[prefix+'prompt.ids',prefix+'modalities.ids',prefix+'text.embedded.f32']:require(known[owned(dest,n[len(prefix):]).resolve()]==row['sha256'],'all actual historical OFF stages')
   else:
    for p in(folder/'off').iterdir():
     if p.is_file()and(p.name.startswith(('vision.','processor.'))or p.name in['prompt.ids','modalities.ids','text.embedded.f32','prefill.input.f32','prefill.positions.f32']):require(known[owned(dest,p.name).resolve()]==known[p.resolve()],'actual unchanged vision/inputs')
  layers=[]
  for i in range(32):
   a=np.fromfile(owned(folder/'off',f'prefill.layer.{i}.f32'),dtype='<f4').astype('f8');b=np.fromfile(owned(folder/'on',f'prefill.layer.{i}.f32'),dtype='<f4').astype('f8');require(a.size and a.shape==b.shape and np.isfinite(a).all()and np.isfinite(b).all(),'paired finite full native layer');d=b-a;layers.append(dict(layer=i,bit_exact=bool(np.array_equal(a,b)),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),off_rms_magnitude=float(np.sqrt(np.mean(a*a))),on_rms_magnitude=float(np.sqrt(np.mean(b*b)))))
  require(layers==v['paired_native_layers']['layers'],'all paired-layer actual errors/magnitudes');out_case['paired_native_layers']=dict(scope='Same-source native OFF/ON, not actual source hidden states.',first_different_layer=next((x['layer']for x in layers if not x['bit_exact']),None),layers=layers);report['cases'][name]=out_case
 report['raw_rows']=row_total;report['all_on_complete']=all(c['arms']['on']['eos']for c in report['cases'].values());report['on_source_floors_passed']=sum(c['arms']['on']['source_floor']for c in report['cases'].values());require(report['all_on_complete']==r['all_on_complete']and report['on_source_floors_passed']==r['on_source_floors_passed'],'actual overall diagnostic inventory');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();start=time.monotonic();r=dict(status='failed',model_forward_executed=False,qualified=False)
 def interrupted(sig,frame):raise TimeoutError('fixed artifact-only audit bound')
 for sig in[signal.SIGALRM,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupted)
 signal.alarm(1800)
 try:r=audit(a.receipt,a.receipt_sha)
 except BaseException as error:r['error_type']=type(error).__name__;raise
 finally:signal.alarm(0);r['elapsed_seconds']=time.monotonic()-start;a.out.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
