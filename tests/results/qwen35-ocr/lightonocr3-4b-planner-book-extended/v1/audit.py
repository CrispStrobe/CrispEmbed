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

def audit(path,receipt_sha,failed_path,original_path,bundle_path):
 require(np.__version__=='2.3.5','pinned NumPy');require(len(receipt_sha)==64 and sha(path)==receipt_sha,'exact new receipt')
 r=json.loads(path.read_text());failed=json.loads(failed_path.read_text());original=json.loads(original_path.read_text());bundle=json.loads(bundle_path.read_text());root=path.parent
 require(sha(failed_path)=='f231ab48c9bff411a1f9c96cfb79eda084f2f0b7cee46f9ea747ccc734d5f8e6' and sha(original_path)=='9b88195d5a8058dea4fab87472009d3826bc1fa3b5d0575aa2e33698316d5722' and sha(bundle_path)=='540aa9983faf570e46e9e6708f745dbd3cbd5f8354276f3fddb301e7fe9e65d7','frozen original/timeout/bundle')
 require(r['source_sha']=='4a32434e822ac6c8fd69ad77252a59a2cd4d5950'and r['status']=='measured_extended_planner_book'and r['qualified']is False,'closed controller')
 require(r['runtime_rebuilt']is False and r['reference_forward_executed']is False and r['old_baseline_decode_executed']is False and r['source_feature_decode_executed']is False,'single native primary replay only')
 require(r['strict_1054_prefix_and_stages_reproduced']and r['prior_baseline_authenticated_without_decode']and r['all_actual_model_free_control_members_authenticated'],'closed replay guards')
 require(r['runtime_manifest_sha256']==sha(bundle_path)and r['native_binary_sha256']==bundle['native_binary_sha256']and r['native_gates']==failed['native_gates']['on'],'same admitted native bundle/gates')
 require(r['timeout_receipt_sha256']==sha(failed_path)and r['timeout_audit_sha256']=='4906aa6e171cb5f92816a7a9e98e177a7ccbfd5f5406f2f2b1e353acb852ef06','actual audited timeout prefix')
 require(len(r['artifacts'])<=2000 and sum(x['bytes']for x in r['artifacts'].values())<=8<<30,'bounded child artifacts');known={};total=path.stat().st_size
 for name,row in r['artifacts'].items():
  f=owned(root,name);require(f.suffix not in['.gguf','.safetensors']and f.stat().st_size==row['bytes']and sha(f)==row['sha256'],'all owned artifact bytes');known[f.resolve()]=row['sha256'];total+=row['bytes']
 arm=r['arm'];folder=root/'planner-on';ids=arm['ids'];actual=list(map(int,owned(folder,'generated.ids').read_text().split()));require(actual==ids and owned(folder,'generated.txt').read_bytes().decode()==arm['decoded'],'actual final text/IDs')
 summary=json.loads(owned(folder,'native-summary.json').read_text());require(summary==arm['summary']and summary['generated_tokens']==len(ids)and summary['eos']==(ids[-1]in[248044,248046])and(summary['eos']or len(ids)==1536),'actual EOS/cap/count')
 require(summary['provisional_heldout_cap']==1536 and not summary['source_fed_vision']and summary['physical_source_prefill_chunks']and summary['physical_q6_tiled']and summary['physical_q4_repack']and summary['physical_repack_entries']==216 and 0<summary['physical_repack_bytes']<=2<<30 and summary['converted_f32_weight_entries']==summary['converted_f32_weight_bytes']==0,'actual modes/no F32 text cache')
 rows=arm['raw_rows']['rows'];require(len(rows)>=1054 and [x['sha256']for x in rows[:1054]]==[failed['artifacts'][f'corrected-composed/greedy.{i}.logits.f32']['sha256']for i in range(1054)],'reproduced actual1054 row prefix')
 stages=0
 for name,row in failed['artifacts'].items():
  if name.startswith(('corrected-composed/prefill.','corrected-composed/vision.','corrected-composed/processor.'))or name in['corrected-composed/prompt.ids','corrected-composed/modalities.ids','corrected-composed/text.embedded.f32']:
   require(sha(owned(folder,name.split('/',1)[1]))==row['sha256'],'all actual earlier prefill/vision stages');stages+=1
 for name in['prefill.logits.f32','teacher.0.logits.f32']:require(sha(owned(folder,name))==rows[0]['sha256'],'actual first-row triple')
 require(sha(owned(root,'pre-forward-inputs.json'))==failed['pre_forward_input_seal_sha256']==original['pre_forward_input_seal_sha256'],'original seal')
 require(sha(owned(folder,'vision.merged.f32'))==original['artifacts']['upstream-on/source.vision.0.f32']['sha256'],'same actual source vision')
 zipped=audit_zip(folder,arm['raw_rows'],ids,known)
 return dict(schema='lighton4b-planner-book-extended-independent-audit/v1',status='complete',receipt_sha256=receipt_sha,owned_files=len(r['artifacts'])+1,owned_bytes=total,raw_rows=len(rows),raw_archive=zipped,reproduced_prefix_rows=1054,prior_stages=stages,eos=summary['eos'],all_owned_artifact_hashes=True,all_raw_finite_argmax_sha=True,model_forward_executed=False,qualified=False,scope='Integrity only. No protected text or IDs published; provisional label limits and binding receipt failure remain unchanged.')
def main():
 p=argparse.ArgumentParser();p.add_argument('--new',type=Path,required=True);p.add_argument('--new-sha',required=True);p.add_argument('--failed',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();start=time.monotonic();r=dict(status='failed',qualified=False,model_forward_executed=False)
 def stop(sig,frame):raise TimeoutError('bounded artifact audit canceled')
 for s in[signal.SIGALRM,signal.SIGTERM,signal.SIGINT]:signal.signal(s,stop)
 signal.alarm(600)
 try:r=audit(a.new,a.new_sha,a.failed,a.original,a.bundle)
 except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:signal.alarm(0);r['elapsed_seconds']=time.monotonic()-start;a.out.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
