"""Independent timeout-artifact integrity. No models, builds or text/ID publication."""
import argparse,hashlib,json,signal,time,zipfile
from pathlib import Path
import numpy as np
VOCAB=248320
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while b:=f.read(8<<20):h.update(b)
 return h.hexdigest()
def owned(root,name):
 p=Path(name);require(not p.is_absolute()and '..'not in p.parts,'owned relative path');f=root/p;require(f.is_file()and not f.is_symlink()and f.resolve().is_relative_to(root.resolve()),'owned regular artifact');return f
def audit_zip(root,record,ids,known):
 a=record['archive'];p=owned(root,a['file']);require(p.stat().st_size==a['bytes']and known[p.resolve()]==a['sha256'],'whole archived bytes');rows=record['rows'];require(len(rows)==len(ids)and 1<=len(ids)<=1536,'archive row count');chain=hashlib.sha256()
 with zipfile.ZipFile(p)as z:
  names=[f'greedy.{i}.logits.f32'for i in range(len(ids))];require(len(z.infolist())==len(names)and set(z.namelist())==set(names),'archive members')
  for i,(n,row,token)in enumerate(zip(names,rows,ids)):
   require(row['position']==i and row['bytes']==VOCAB*4 and row['argmax']==token and type(token)is int and 0<=token<VOCAB,'record argmax');b=z.read(n);require(len(b)==VOCAB*4 and hashlib.sha256(b).hexdigest()==row['sha256'],'archive row SHA');x=np.frombuffer(b,dtype='<f4');require(np.isfinite(x).all()and int(x.argmax())==token,'actual archive finite/argmax');chain.update(bytes.fromhex(row['sha256']))
 return dict(rows=len(ids),archive_sha256=a['sha256'],ordered_row_sha256=chain.hexdigest())
def audit(new,previous,observed,reference):
 require(np.__version__=='2.3.5','pinned NumPy');bindings=[(new,'f231ab48c9bff411a1f9c96cfb79eda084f2f0b7cee46f9ea747ccc734d5f8e6'),(previous,'e3fb9b5b8dddcbed51678c27d203fbc2bfec645966e94f846fd31f956c84f9a3'),(observed,'393bfbd65f0f39c9d15e68110eb8174fb912797200cc68b1c8339e7a0797edcb'),(reference,'9b88195d5a8058dea4fab87472009d3826bc1fa3b5d0575aa2e33698316d5722')]
 for p,h in bindings:require(sha(p)==h,'immutable receipt binding')
 root=new.parent;r=json.loads(new.read_text());old=json.loads(previous.read_text());obs=json.loads(observed.read_text());ref=json.loads(reference.read_text());require(r['status']=='failed'and r['phase']=='corrected-composed'and r['error_type']=='TimeoutExpired'and r['source_sha']=='a6164205620c71427a39c97ab386591406c4ca73'and r['case']=='two-column-page'and r['qualified']is False and not r['reference_forward_executed'],'fixed timeout source');require(set(r['arms'])=={'baseline-old'}and r['old_baseline_all_rows_stages_exact'],'no completed planner arm');cmd=r['commands']['corrected-composed'];require(cmd['error_type']=='TimeoutExpired'and 1200<=cmd['elapsed_seconds']<1230 and cmd['returncode']in[-15,-9],'fixed arm timeout')
 require(old['source_sha']=='52e1cfcec5b515b4f4483fbeb86817250868f1f4'and old['status']=='measured_provisional_q6_heldout'and obs['status']=='measured_source_prefill_observed_ab'and ref['status']=='measured_provisional_heldout'and ref['source_observer_full_neutral'],'completed historical inputs');require(r['pre_forward_input_seal_sha256']==old['pre_forward_input_seal_sha256']==ref['pre_forward_input_seal_sha256']and r['previous_corrected_receipt_sha256']==sha(previous)and r['historical_reference_sha256']==sha(reference),'original seal/reference chain')
 require(len(r['artifacts'])+1==1457 and new.stat().st_size+sum(x['bytes']for x in r['artifacts'].values())==2735343103,'frozen artifact census');total=new.stat().st_size;known={}
 for n,row in r['artifacts'].items():
  p=owned(root,n);require(p.suffix not in['.gguf','.safetensors']and p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'all actual owned SHA');known[p.resolve()]=row['sha256'];total+=row['bytes']
 require(sha(owned(root,'pre-forward-inputs.json'))==r['pre_forward_input_seal_sha256'],'actual current seal');prior=old['arms']['corrected-composed'];baseline=r['arms']['baseline-old'];require(baseline['ids']==prior['ids']and baseline['decoded']==prior['decoded']and baseline['complete']and len(baseline['ids'])==660,'complete prior baseline');require([x['sha256']for x in baseline['raw_rows']['rows']]==[x['sha256']for x in prior['raw_rows']['rows']],'all660 baseline row identities');stages=0
 for n,row in old['artifacts'].items():
  if n.startswith(('corrected-composed/prefill.','corrected-composed/vision.','corrected-composed/processor.'))or n in['corrected-composed/text.embedded.f32','corrected-composed/prompt.ids','corrected-composed/modalities.ids']:
   require(sha(owned(root/'baseline-old',n.split('/',1)[1]))==row['sha256'],'prior baseline actual stages');stages+=1
 archives={'baseline-old':audit_zip(root/'baseline-old',baseline['raw_rows'],baseline['ids'],known)}
 for tag,count in [('observed-old',167),('observed-corrected',168)]:
  record=r[tag]['raw_rows'];ids=[x['argmax']for x in record['rows']];require(len(ids)==count,'observed complete count');archives[tag]=audit_zip(root/tag,record,ids,known);expected=[x['sha256']for x in old['observed-corrected']['raw_rows']['rows']]if tag=='observed-old'else[obs['artifacts'][f'simple_form/on/greedy.{i}.logits.f32']['sha256']for i in range(count)];require([x['sha256']for x in record['rows']]==expected,'observed frozen row identity')
 changed={'CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL'};off=r['native_gates']['off'];on=r['native_gates']['on'];require({k for k in off if off[k]!=on[k]}==changed and off[next(iter(changed))]=='0'and on[next(iter(changed))]=='1','only planner changes');require(all(off[k]==on[k]=='1'for k in['CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK','CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED','CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS']),'old corrections unchanged')
 feature=owned(reference.parent,'upstream-on/source.vision.0.f32');require(sha(feature)==ref['artifacts']['upstream-on/source.vision.0.f32']['sha256'],'original actual source feature');feature_sha=sha(feature)
 for arm in['baseline-old','corrected-composed']:
  folder=root/arm;require(sha(owned(folder,'vision.merged.f32'))==feature_sha,'actual upstream merged feature');
  for name in['prompt.ids','modalities.ids','text.embedded.f32','prefill.input.f32','prefill.positions.f32']:require(known[owned(folder,name).resolve()]==known[owned(root/'baseline-old',name).resolve()],'unchanged actual inputs')
 for n,row in old['artifacts'].items():
  if n.startswith(('corrected-composed/vision.','corrected-composed/processor.')):require(known[owned(root/'corrected-composed',n.split('/',1)[1]).resolve()]==row['sha256'],'actual full vision unchanged')
 part=root/'corrected-composed';progress=json.loads(owned(part,'generated-progress.json').read_text());require(progress['generated_tokens']==1054 and progress['eog']is False,'retained progress1054/noEOS');names=[f'greedy.{i}.logits.f32'for i in range(1054)];actual={n.split('/',1)[1]for n in r['artifacts']if n.startswith('corrected-composed/greedy.')and n.endswith('.logits.f32')};require(actual==set(names),'exact consecutive partial rows');chain=hashlib.sha256();eogs=0;last=None
 for n in names:
  p=owned(part,n);require(p.stat().st_size==VOCAB*4,'full partial vocabulary size');x=np.fromfile(p,dtype='<f4');require(np.isfinite(x).all(),'finite partial scores');last=int(x.argmax());eogs+=last in[248044,248046];chain.update(bytes.fromhex(known[p.resolve()]))
 require(last==progress['last_id']and eogs==0,'partial lastargmax/progress/noEOS');require(sha(owned(part,'prefill.logits.f32'))==sha(owned(part,'teacher.0.logits.f32'))==known[owned(part,names[0]).resolve()],'partial first row closure')
 paired=[]
 for layer in range(32):
  name=f'prefill.layer.{layer}.f32';b=np.fromfile(owned(root/'baseline-old',name),dtype='<f4').astype('f8');a=np.fromfile(owned(part,name),dtype='<f4').astype('f8');require(b.size==a.size==baseline['summary']['prompt_tokens']*2560 and np.isfinite(a).all()and np.isfinite(b).all(),'paired finite geometry');d=a-b;paired.append(dict(layer=layer,rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),before_rms_magnitude=float(np.sqrt(np.mean(b*b))),after_rms_magnitude=float(np.sqrt(np.mean(a*a))),bit_exact=known[owned(root/'baseline-old',name).resolve()]==known[owned(part,name).resolve()]))
 return dict(schema='lighton4b-planner-book-timeout-independent-audit/v1',status='complete-timeout-integrity',receipt_sha256=sha(new),owned_files=len(r['artifacts'])+1,owned_bytes=total,complete_archived_rows=995,partial_rows=1054,total_vocabulary_rows=2049,raw_archives=archives,partial_ordered_row_sha256=chain.hexdigest(),all_owned_sha=True,all_rows_finite_argmax_audited=True,last_argmax_matches_progress=True,eog_count=0,completed_planner_output=False,source_feature_pair_executed=False,prior_stages=stages,source_merged_actual=True,paired_native_layers=dict(scope='Native prior-corrected versus partial planner prefill, not unavailable source hidden states.',first_different_layer=next((x['layer']for x in paired if not x['bit_exact']),None),layers=paired),candidate_final_cache_summary_available=False,qualified=False,reference_forward_executed=False,model_forward_executed=False,scope='Private timeout bytes only. No recognized text or token-ID sequence is published; no complete CER/WER or floor decision for the planner arm.')
def main():
 p=argparse.ArgumentParser();
 for n in['new','previous','observed','reference','out']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();start=time.monotonic();r=dict(status='failed',qualified=False,model_forward_executed=False)
 def stop(sig,frame):raise TimeoutError('bounded timeout audit interrupted')
 for sig in[signal.SIGALRM,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
 signal.alarm(600)
 try:r=audit(a.new,a.previous,a.observed,a.reference)
 finally:signal.alarm(0);r['elapsed_seconds']=time.monotonic()-start;a.out.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
