"""Independent completed-artifact audit; no model, build, network or text publication."""
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

def audit_zip(root,record,ids,known=None):
 a=record['archive'];p=owned(root,a['file']);require(p.stat().st_size==a['bytes']and(known.get(p.resolve())==a['sha256']if known is not None else sha(p)==a['sha256']),'whole ZIP identity');rows=record['rows'];require(len(rows)==len(ids)and 1<=len(ids)<=1536,'raw row census');chain=hashlib.sha256()
 with zipfile.ZipFile(p)as z:
  expected=[f'greedy.{i}.logits.f32'for i in range(len(ids))];require(len(z.infolist())==len(expected)and set(z.namelist())==set(expected),'closed ZIP names')
  for i,(name,row,token)in enumerate(zip(expected,rows,ids)):
   require(row['position']==i and row['bytes']==VOCAB*4 and row['argmax']==token and type(token)is int and 0<=token<VOCAB,'record position/vocabulary/argmax')
   info=z.getinfo(name);require(info.file_size==VOCAB*4 and not info.is_dir(),'full vocabulary member');b=z.read(name);require(hashlib.sha256(b).hexdigest()==row['sha256'],'each exact raw row SHA');x=np.frombuffer(b,dtype='<f4');require(x.size==VOCAB and np.isfinite(x).all()and int(x.argmax())==token,'actual finite row/argmax');chain.update(bytes.fromhex(row['sha256']))
 return dict(rows=len(ids),archive_bytes=a['bytes'],archive_sha256=a['sha256'],ordered_row_sha256=chain.hexdigest())

def audit(new_path,old_path,observed_path,new_sha,old_sha,case):
 require(np.__version__=='2.3.5','pinned NumPy');require(sha(new_path)==new_sha and sha(old_path)==old_sha,'immutable result receipts');require(sha(observed_path)=='f5d0077052b907e2d674caef9e7c4ae09b08456ff48a79f327cbdfd030b2dcc9','immutable observed producer')
 root=new_path.parent;oldroot=old_path.parent;r=json.loads(new_path.read_text());old=json.loads(old_path.read_text());observed=json.loads(observed_path.read_text());require(r['source_sha']=='52e1cfcec5b515b4f4483fbeb86817250868f1f4'and r['status']=='measured_provisional_q6_heldout'and r['case']==old['case']==case and r['qualified']is False and not r['reference_forward_executed'],'fixed completed diagnostic');require(old['status']=='measured_provisional_heldout'and old['source_sha']=='d378c6cc055e3a31b0d4cc9dad54a5b41f10d792'and old['qualified']is False and r['old_baseline_all_rows_stages_exact']and r['pre_forward_input_seal_sha256']==old['pre_forward_input_seal_sha256'],'retained baseline/input seal');require(len(r['artifacts'])<=2000 and sum(x['bytes']for x in r['artifacts'].values())<=12<<30,'bounded artifact inventory')
 total=new_path.stat().st_size;known={}
 for name,row in r['artifacts'].items():
  f=owned(root,name);require(f.suffix not in['.gguf','.safetensors'],'no model payload in auditor');require(f.stat().st_size==row['bytes']and sha(f)==row['sha256'],'all owned artifact bytes');known[f.resolve()]=row['sha256'];total+=row['bytes']
 oldseal=owned(oldroot,'pre-forward-inputs.json');require(sha(oldseal)==old['pre_forward_input_seal_sha256']==sha(owned(root,'pre-forward-inputs.json')),'actual original input seal identity')
 baseline=r['arms']['baseline-old'];prior=old['arms']['native-composed'];require(baseline['ids']==prior['ids']and baseline['decoded']==prior['decoded']and baseline['complete'],'old complete native output identity');require([x['sha256']for x in baseline['raw_rows']['rows']]==[x['sha256']for x in prior['raw_rows']['rows']],'old full rows reference identity')
 compared=0
 for name,row in old['artifacts'].items():
  if name.startswith('native-composed/prefill.')or name.startswith('native-composed/vision.')or name.startswith('native-composed/processor.')or name in['native-composed/text.embedded.f32','native-composed/prompt.ids','native-composed/modalities.ids']:
   actual=owned(root,'baseline-old/'+name.split('/',1)[1]);require(sha(actual)==row['sha256'],'old actual stage/vision byte identity');compared+=1
 changed={'CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS','CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK','CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED'};off=r['native_gates']['off'];on=r['native_gates']['on'];require({k for k in off if off[k]!=on[k]}==changed and all(off[k]=='0'and on[k]=='1'for k in changed),'exact three-gate diagnostic')
 source_feature=owned(oldroot,'upstream-on/source.vision.0.f32');require(sha(source_feature)==old['artifacts']['upstream-on/source.vision.0.f32']['sha256'],'actual original upstream merged feature');source_feature_sha=sha(source_feature)
 out={}
 for arm in ['baseline-old','corrected-composed','corrected-source-features']:
  data=r['arms'][arm];folder=root/arm;actual_ids=list(map(int,owned(folder,'generated.ids').read_text().split()));require(actual_ids==data['ids']and owned(folder,'generated.txt').read_bytes().decode()==data['decoded'],'actual text/IDs identity');summary=json.loads(owned(folder,'native-summary.json').read_text());require(summary==data['summary']and summary['generated_tokens']==len(actual_ids)and summary['eos']==(actual_ids[-1]in[248044,248046])and(summary['eos']or len(actual_ids)==1536),'actual count/EOS/cap');require(summary['provisional_heldout_cap']==1536 and summary['source_fed_vision']==(arm=='corrected-source-features'),'actual arm role');require(summary['converted_f32_weight_entries']==summary['converted_f32_weight_bytes']==0,'no F32 text cache');mode=arm!='baseline-old';require(summary['physical_q4_repack']==summary['physical_q6_tiled']==mode and summary['physical_repack_entries']==(216 if mode else 0)and 0<=summary['physical_repack_bytes']<=2<<30 and bool(summary['physical_repack_bytes'])==mode,'actual corrected cache modes');require(sha(owned(folder,'vision.merged.f32'))==source_feature_sha,'all actual merged features upstream exact')
  for name in ['prefill.logits.f32','teacher.0.logits.f32']:require(sha(owned(folder,name))==data['raw_rows']['rows'][0]['sha256'],'actual first vocabulary row closure')
  for name in ['prompt.ids','modalities.ids','text.embedded.f32','prefill.input.f32','prefill.positions.f32']:
   require(known[owned(folder,name).resolve()]==known[owned(root/'baseline-old',name).resolve()],'all actual decoder inputs unchanged')
  if arm=='corrected-composed':
   for name,row in old['artifacts'].items():
    if name.startswith('native-composed/vision.')or name.startswith('native-composed/processor.'):require(known[owned(folder,name.split('/',1)[1]).resolve()]==row['sha256'],'corrected actual full vision/processor unchanged')
  out[arm]=audit_zip(folder,data['raw_rows'],actual_ids,known)
 for tag in ['observed-old','observed-corrected']:
  record=r[tag]['raw_rows'];ids=[x['argmax']for x in record['rows']];require(len(ids)==167,'observed full167 census');out[tag]=audit_zip(root/tag,record,ids,known)
  expected=[x['sha256']for x in old['observed_control_rows']['rows']]if tag=='observed-old'else[observed['artifacts'][f'simple_form/on/greedy.{i}.logits.f32']['sha256']for i in range(167)];require([x['sha256']for x in record['rows']]==expected,'observed rows actual frozen reference')
 same_corrected_rows=[x['sha256']for x in r['arms']['corrected-composed']['raw_rows']['rows']]==[x['sha256']for x in r['arms']['corrected-source-features']['raw_rows']['rows']]
 return dict(corrected_arms_all_rows_identical=same_corrected_rows,corrected_arms_text_identical=r['arms']['corrected-composed']['decoded']==r['arms']['corrected-source-features']['decoded'],schema='lighton4b-q6-heldout-independent-raw-audit/v1',status='complete',case=case,receipt_sha256=new_sha,old_receipt_sha256=old_sha,owned_files=len(r['artifacts'])+1,owned_bytes=total,raw_rows=sum(x['rows']for x in out.values()),raw_archives=out,old_reference_stages_compared=compared,all_raw_finite_argmax_sha=True,all_owned_artifact_hashes=True,source_merged_all_arms_exact=True,reference_forward_executed=False,model_forward_executed=False,qualified=False,scope='Completed diagnostic bytes only. Historical reference rights/provisional-label limits remain binding. No text or IDs published by this auditor.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--new',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--observed',type=Path,required=True);p.add_argument('--new-sha',required=True);p.add_argument('--old-sha',required=True);p.add_argument('--case',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();start=time.monotonic();r=dict(status='failed',qualified=False,model_forward_executed=False)
 def stop(sig,frame):raise TimeoutError('bounded artifact audit interrupted')
 for sig in[signal.SIGTERM,signal.SIGINT,signal.SIGALRM]:signal.signal(sig,stop)
 signal.alarm(600)
 try:r=audit(a.new,a.old,a.observed,a.new_sha,a.old_sha,a.case)
 except Exception as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:signal.alarm(0);r['elapsed_seconds']=time.monotonic()-start;a.out.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
