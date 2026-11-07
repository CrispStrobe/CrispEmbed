#!/usr/bin/env python3
"""Bounded retained177-row analysis; never runs a model or compares mismatched prefixes."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
ORIGINAL='170bc06c8b5e929e45d334a5eadc284024f4265b2549f2feddb46607cb49251a'
NATIVE='0a0c2084baa4e46aa80f09f7ae94fd4b1c56ccae2bbf2f9ef6210e81e8cf1d01'
GOLD='1694d990aa2b23c17685025134b0c565aff844369ef823a46bfc6a8933a219a1'
NATIVE_SOURCE='4a63d73bee7289deab41fb20f67765d3c6800182'
CANDIDATE='843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43'
def require(ok,why):
 if not ok:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def locate(root,pattern,digest):
 paths=list(root.rglob(pattern));require(len(paths)<=4096,'bounded artifact census');matches=[p for p in paths if p.is_file()and p.stat().st_size<=2**21 and sha(p)==digest];require(1<=len(matches)<=4,'bounded hash-identical artifact copies: '+pattern);return sorted(matches)[0]
def verify(p,row):
 require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'actual artifact length/hash: '+p.name);return p

def source_revision():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()

def metrics(reference,candidate,selected):
 import numpy as np
 require(reference.shape==candidate.shape and reference.ndim==1 and np.isfinite(reference).all()and np.isfinite(candidate).all(),'finite equal full vocabulary rows');require(0<=selected<reference.size,'selected token domain')
 x=reference.astype('f8');y=candidate.astype('f8');e=y-x;lx=x-(np.log(np.exp(x-x.max()).sum())+x.max());ly=y-(np.log(np.exp(y-y.max()).sum())+y.max());px=np.exp(lx);py=np.exp(ly);rx=int(x.argmax());ry=int(y.argmax());den=float(np.linalg.norm(x)*np.linalg.norm(y));order=np.partition(y,-2)[-2:]
 return {'elements':int(x.size),'reference_argmax':rx,'candidate_argmax':ry,'argmax_equal':rx==ry,'teacher_id':selected,'candidate_matches_teacher':ry==selected,'reference_matches_teacher':rx==selected,'reference_rms_magnitude':float(np.sqrt(np.mean(x*x))),'candidate_rms_magnitude':float(np.sqrt(np.mean(y*y))),'reference_max_abs':float(abs(x).max()),'candidate_max_abs':float(abs(y).max()),'error_rms':float(np.sqrt(np.mean(e*e))),'error_max_abs':float(abs(e).max()),'cosine':float(np.dot(x,y)/den)if den else None,'kld_reference_to_candidate':max(0.,float(np.sum(px*(lx-ly)))),'reference_probability_teacher':float(px[selected]),'candidate_probability_teacher':float(py[selected]),'candidate_probability_reference_argmax':float(py[rx]),'candidate_teacher_rank_1based':int(np.count_nonzero(y>y[selected])+np.count_nonzero(y[:selected]==y[selected])+1),'candidate_top2_margin':float(order.max()-order.min()),'different_bit_patterns':int(np.count_nonzero(reference.view('<u4')!=candidate.view('<u4'))),'different_numerical_values':int(np.count_nonzero(reference!=candidate))}

def main(mount,output):
 import numpy as np
 require(np.__version__=='2.3.5','frozen array reader');root=Path(mount);output=Path(output);require(not output.exists(),'fresh output');output.mkdir();started=time.monotonic();result={'schema':'lighton4b-f16-teacher-readback/v1','status':'failed','model_inference_executed':False,'large_artifact_copy_executed':False,'rows':[]}
 def save():(output/'teacher-readback.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 try:
  protocol=json.loads((Path(__file__).resolve().parents[2]/'docs/lighton4b-f16-teacher-readback-v1.json').read_text());require(protocol['native_report_sha256']==NATIVE and protocol['original_bf16_report_sha256']==ORIGINAL and protocol['positions']==177 and protocol['vocabulary']==248320 and protocol['model_inference_allowed']is False,'sealed protocol');result['protocol_sha256']=sha(Path(__file__).resolve().parents[2]/'docs/lighton4b-f16-teacher-readback-v1.json')
  op=locate(root,'*.json',ORIGINAL);npth=locate(root,'physical-f16-full-form.json',NATIVE);gp=locate(root,'physical-native-ops-smoke.json',GOLD);original=json.loads(op.read_text());native=json.loads(npth.read_text());gold=json.loads(gp.read_text());base=npth.parent;gbase=gp.parent/'simple_form'
  require(original['source_sha']=='07543dcc80ebcecad644815d9397be2570a51dd4'and original['original_forward_executed']and original['parameter_dtype_counts'].keys()=={'torch.bfloat16'}and len(original['scores'])==177,'genuine original BF16 capture');require(native['status']=='measured_full_form'and native['source_sha']==NATIVE_SOURCE and native['candidate_sha256']==CANDIDATE and native['golden_report_sha256']==GOLD and native['arms']['f16-full']['teacher_positions']==177,'exact completed native scope');require(gold['status']=='measured'and gold['candidate_sha256']==CANDIDATE,'authenticated upstream source-feature capture')
  art=native['artifacts'];gart=gold['cases']['simple_form']['artifacts'];verify(gbase/'original.json',gart['original.json']);require(sha(gbase/'original.json')==ORIGINAL,'actual original teacher sequence used by native producer');ids=original['generated_ids'];require(len(ids)==177 and ids[-1]==248046 and [x['selected_id']for x in original['scores']]==ids,'independent original sequence and rows/EOS')
  # The sealed producer constructs the teacher file from this exact original.json.
  prompt=verify(base/'f16-full/prompt.ids',art['f16-full/prompt.ids']);inputrow=original['prepared_inputs']['input_ids'];ip=locate(root,inputrow['file'],inputrow['sha256']);inputs=np.load(ip,allow_pickle=False);require(inputs.shape==(1,154)and inputs.dtype==np.int64 and [int(x)for x in prompt.read_text().split()]==inputs.reshape(-1).tolist(),'same original prompt token IDs')
  for name,gname in [('prefill.input.f32','off/prefill.input.f32'),('prefill.positions.f32','off/prefill.positions.f32'),('vision.merged.f32','observer-on/source.vision.0.f32')]:
   verify(base/'f16-full'/name,art['f16-full/'+name]);verify(gbase/gname,gart[gname]);require(art['f16-full/'+name]['sha256']==gart[gname]['sha256'],'same actual pinned upstream input/features/positions')
  result.update(source_sha=source_revision(),original_receipt_sha256=ORIGINAL,native_receipt_sha256=NATIVE,golden_receipt_sha256=GOLD,native_source_sha=NATIVE_SOURCE,candidate_sha256=CANDIDATE,original_source_sha=original['source_sha'],teacher_ids=ids,teacher_contexts_authenticated=True,original_parameter_dtype_counts=original['parameter_dtype_counts'],decoded_comparisons=native['arms']['f16-full']['vs_original_bf16'],upstream_decoded_comparison=native['arms']['f16-full']['vs_retained_upstream'])
  scoreparent=locate(root,original['scores'][0]['file'],original['scores'][0]['sha256']).parent
  def row(p,record):
   verify(p,record);require(record['bytes']==993280,'whole248320-row only');v=np.fromfile(p,dtype='<f4');require(v.shape==(248320,)and np.isfinite(v).all(),'finite complete vocabulary');return v
  for i,s in enumerate(original['scores']):
   require(s['position']==i and s['shape']==[1,248320],'source position/shape');a=row(scoreparent/s['file'],s);require(int(a.argmax())==ids[i],'original score/selection');name=f'f16-full/teacher.{i}.logits.f32';b=row(base/name,art[name]);result['rows'].append({'position':i,'reference_sha256':s['sha256'],'native_sha256':art[name]['sha256'],**metrics(a,b,ids[i])});save()
  source_name='observer-on/source.logits.0.f32';u=row(gbase/source_name,gart[source_name]);b=row(base/'f16-full/teacher.0.logits.f32',art['f16-full/teacher.0.logits.f32']);result['pinned_upstream_position0']={'scope':'same source-fed vision and causal prompt; upstream has no original177-teacher capture, so no later upstream score comparison','reference_sha256':gart[source_name]['sha256'],**metrics(u,b,int(u.argmax()))};first=native['arms']['f16-full']['source_first_score'];require(abs(first['rms']-result['pinned_upstream_position0']['error_rms'])<1e-12 and abs(first['max_abs']-result['pinned_upstream_position0']['error_max_abs'])<1e-12,'actual source first-score closure')
  rows=result['rows'];result['aggregate']={'positions':177,'elements':177*248320,'error_rms':float(np.sqrt(np.mean([x['error_rms']**2 for x in rows]))),'reference_rms_magnitude':float(np.sqrt(np.mean([x['reference_rms_magnitude']**2 for x in rows]))),'candidate_rms_magnitude':float(np.sqrt(np.mean([x['candidate_rms_magnitude']**2 for x in rows]))),'reference_max_abs':max(x['reference_max_abs']for x in rows),'candidate_max_abs':max(x['candidate_max_abs']for x in rows),'error_max_abs':max(x['error_max_abs']for x in rows),'mean_kld_bf16_to_native':float(np.mean([x['kld_reference_to_candidate']for x in rows])),'teacher_top1_matches':sum(x['candidate_matches_teacher']for x in rows),'teacher_top1_disagreements':[x['position']for x in rows if not x['candidate_matches_teacher']],'mean_native_probability_original_token':float(np.mean([x['candidate_probability_teacher']for x in rows]))};result['status']='readback-verified';result['limitations']=['Original unquantized BF16 versus quantized physical native: includes weight quantization, source vision features and runtime effects; not same-weight parity.','Teacher rows use original prefixes and do not prove independent greedy quality.','Pinned upstream has no177 original-teacher rows: numerical comparison restricted to admitted position0.','No new inference or source intermediate capture; large arrays remain remote.'];return result
 finally:result['seconds']=time.monotonic()-started;save()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mount',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.mount,a.output)
