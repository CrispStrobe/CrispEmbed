#!/usr/bin/env python3
"""Read-only remote artifact audit: complete payloads, scores and causal stage deltas."""
import argparse,hashlib,json,re,subprocess,time,urllib.request
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
from gguf_file_type_canonicalize import parse,span,sha
import lighton4b_hf_tiled_out_pack as profile
import lighton4b_hf_tiled_out_reproduce as proof
MODEL_SOURCE='b2a3778e424559b7216e1b11f21c7b38b71b02d8'
OLD_MANIFEST_SHA='36c6d9deeeea000e560a5532a24e27a9019f0376f80e8e14e13a6a1e0316f64e'
OLD_CAL_SHA='1f296b54b74e2660f81f7fa7d9aae9850bb3d22fb19c5024c7a28908248090e7'
OLD_FORM_SHA='a483fe4d56cfdde2c3835d2cf7ca1f322d09c4594664405e0faa8da073230643'
def require(ok,why):
 if not ok:raise ValueError(why)
def locate(root,name,digest):
 require(re.fullmatch('[0-9a-f]{64}',digest)is not None,'sealed receipt SHA required');paths=[p for p in root.rglob(name)if p.stat().st_size<2**20 and sha(p)==digest];require(len(paths)==1,'single immutable receipt required: '+name);return paths[0]
def distance(a,b):
 previous=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  current=[i]
  for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
  previous=current
 return previous[-1]
def first_difference(a,b):
 return next((i for i,(x,y)in enumerate(zip(a,b))if x!=y),None if len(a)==len(b)else min(len(a),len(b)))
def causal_rows(a,b):
 n=min(len(a),len(b));different=first_difference(a,b);return min(n,n if different is None else different+1)
def main(mount,receipt_sha,output):
 import numpy as np
 require(np.__version__=='2.3.5','frozen F32 serializer required');mount=Path(mount);output=Path(output);require(not output.exists(),'fresh readback directory required');output.mkdir();started=time.monotonic();result={'schema':'lighton4b-hf-tiled-out-readback/v1','status':'failed','model_inference_executed':False,'weight_quantization_executed':False,'full_f32_model_instantiated':False,'cases':{}}
 def save():(output/'readback.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 try:
  current_path=locate(mount,'hf-tiled-out-complete-decode.json',receipt_sha);current=json.loads(current_path.read_text());require(current['status']=='measured'and current['source_sha']==MODEL_SOURCE and set(current['cases'])=={'simple_form','simple_table','arabic_printed_line'},'complete immutable B2 threecase report required');base=current_path.parent;result.update(protocol_sha256=sha(ROOT/'docs/lighton4b-hf-tiled-out-readback-protocol-v1.json'),model_source_sha=MODEL_SOURCE,readback_source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),producer_receipt_sha256=receipt_sha)
  pack_path=base/'candidate-pack.json';require(pack_path.stat().st_size<2**20,'bounded pack receipt required');pack=json.loads(pack_path.read_text());require(pack['status']=='packed'and len(pack['tensors'])==723 and pack['candidate_sha256']==current['candidate']['candidate_sha256']and pack['base_sha256']==profile.BASE_SHA,'sealed full pack audit required');result['pack_receipt_sha256']=sha(pack_path);candidate=base/(profile.RECIPE+'.gguf');require(candidate.stat().st_size==pack['candidate_bytes']<=4050000000,'complete retained model size differs');model_sha=sha(candidate);require(model_sha==pack['candidate_sha256'],'complete retained model hash differs');result['candidate_sha256']=model_sha
  url='https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/3f4b02d2c7b7531a815816512f61bf1fa4f93732/tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/lighton-q4km-candidate-results/candidate-payload-audit.json'
  with urllib.request.urlopen(url,timeout=60)as f:raw=f.read(2**20+1)
  require(len(raw)<=2**20 and hashlib.sha256(raw).hexdigest()==OLD_MANIFEST_SHA,'independent old723-payload manifest differs');(output/'old-payload-audit.json').write_bytes(raw);old_manifest=json.loads(raw);require(old_manifest['sha256']==profile.BASE_SHA and len(old_manifest['tensor_payload_sha256'])==723,'immutable old5d manifest required')
  g=parse(candidate,profile.TYPES);selected={r['name']:r for r in proof.protocol()['selected']};records={r['name']:r for r in pack['tensors']};require(set(g['tensors'])==set(records)==set(old_manifest['tensor_payload_sha256']),'complete723 name domains differ');require(all(g['metadata'][k]['kind']==8 and g['metadata'][k]['value']==v for k,v in profile.strings().items())and all(g['metadata'][k]['kind']==4 and g['metadata'][k]['value']==v for k,v in profile.numbers().items()),'actual typed HF-layout metadata differs');types={};payloads=[]
  with candidate.open('rb')as f:
   for name,t in g['tensors'].items():
    actual=span(f,t['data_offset'],t['bytes']);old=old_manifest['tensor_payload_sha256'][name];expected=selected[name]['expected_publisher_tiled_q4_sha256']if name in selected else old;require(actual==expected==records[name]['sha256']and old==records[name]['source_sha256']and t['type']==records[name]['type']and t['bytes']==records[name]['bytes'],'actual retained tensor bytes differ: '+name);payloads.append({'name':name,'type':t['type'],'bytes':t['bytes'],'sha256':actual,'old_sha256':old,'protected':name not in selected});types[t['type']]=types.get(t['type'],0)+1
  require(types=={0:474,12:216,14:33}and sum(x['protected']for x in payloads)==699,'actual complete dtype/protected census differs');(output/'actual723-payloads.json').write_text(json.dumps(payloads,indent=2,sort_keys=True)+'\n');result.update(all723_payloads_read_and_hashed=True,all699_match_independent_old_manifest=True,all24_match_original_f16_proven_publisher_payloads=True,types=types);save();print('PHASE full723-payload-readback-passed',flush=True)
  old_cal_path=locate(mount,'q4km-complete-decode.json',OLD_CAL_SHA);old_form_path=locate(mount,'q4km-complete-decode.json',OLD_FORM_SHA);old_cases={**{k:(old_cal_path,json.loads(old_cal_path.read_text())['cases'][k])for k in('simple_table','arabic_printed_line')},'simple_form':(old_form_path,json.loads(old_form_path.read_text())['cases']['simple_form'])}
  def vector(folder,record,name):
   path=folder/name;spec=record['artifacts'][name];require(path.stat().st_size==spec['bytes']and path.stat().st_size<=16*2**20 and path.stat().st_size%4==0,'bounded authenticated F32 artifact required: '+name);v=np.fromfile(path,dtype='<f4');require(hashlib.sha256(v.tobytes()).hexdigest()==spec['sha256'],'actual F32 artifact hash differs: '+name);require(np.isfinite(v).all(),'finite artifact required');return v
  def delta(a,b):
   require(a.shape==b.shape and a.size>0,'same nonempty shape required');x=a.astype(np.float64);y=b.astype(np.float64);e=y-x;return {'elements':a.size,'reference_max_abs':float(np.abs(x).max()),'candidate_max_abs':float(np.abs(y).max()),'reference_rms_magnitude':float(np.sqrt(np.mean(x*x))),'candidate_rms_magnitude':float(np.sqrt(np.mean(y*y))),'error_rms':float(np.sqrt(np.mean(e*e))),'error_max_abs':float(np.abs(e).max()),'different_bit_patterns':int(np.count_nonzero(a.view('<u4')!=b.view('<u4'))),'different_numerical_values':int(np.count_nonzero(a!=b))}
  for case,row in current['cases'].items():
   print('PHASE causal-readback-'+case,flush=True);folder=base/case/'native';old_path,old=old_cases[case];old_folder=old_path.parent/case/'native';ids=[int(x)for x in(folder/'generated.ids').read_text().split()];text=(folder/'generated.txt').read_bytes().decode();require(1<=len(ids)<=256 and ids==row['ids']and text==row['decoded']and len(ids)==len(row['greedy_rows']),'raw decoded output/row count differs');old_ids=[int(x)for x in(old_folder/'generated.ids').read_text().split()];old_text=(old_folder/'generated.txt').read_bytes().decode();require(old_ids==old['generated_ids']and old_text==old['decoded'],'old raw decoded output differs')
   for source_folder,record in((folder,row),(old_folder,old)):
    for name in('generated.ids','generated.txt'):require(sha(source_folder/name)==record['artifacts'][name]['sha256'],'raw output hash differs')
   for name in('prefill.input.f32','prefill.positions.f32'):
    left=vector(old_folder,old,name);right=vector(folder,row,name);require(np.array_equal(left.view('<u4'),right.view('<u4')),'causal prefill inputs differ; no layer comparison admitted')
   stages={}
   for name in ['prefill.layer.'+str(i)+'.f32'for i in range(32)]+['prefill.norm.f32','prefill.logits.f32']:stages[name]=delta(vector(old_folder,old,name),vector(folder,row,name))
   valid=causal_rows(ids,old_ids);greedy=[]
   for i,token in enumerate(ids):
    name='greedy.'+str(i)+'.logits.f32';v=vector(folder,row,name);require(v.size==248320 and int(v.argmax())==token and row['greedy_rows'][i]['position']==i and row['greedy_rows'][i]['id']==token and row['artifacts'][name]['sha256']==row['greedy_rows'][i]['sha256'],'actual independent greedy row differs')
    if i<valid:greedy.append({'position':i,'old_id':old_ids[i],'candidate_id':token,**delta(vector(old_folder,old,name),v)})
   eos=bool(ids and ids[-1]==248046);require(eos==row['actual_eos']and(row['cap_hit']==(not eos))and(eos or len(ids)==256),'independent EOS/cap differs');case_out=output/case;case_out.mkdir();(case_out/'generated.txt').write_bytes((folder/'generated.txt').read_bytes());(case_out/'generated.ids').write_bytes((folder/'generated.ids').read_bytes());result['cases'][case]={'tokens':len(ids),'eos':eos,'reference_native_source_sha':json.loads(old_path.read_text())['source_sha'],'candidate_native_source_sha':MODEL_SOURCE,'cap_hit':not eos,'raw_text_sha256':sha(folder/'generated.txt'),'old_raw_text_sha256':sha(old_folder/'generated.txt'),'ids_equal_old':ids==old_ids,'text_equal_old':text==old_text,'first_id_difference_old':first_difference(ids,old_ids),'causal_greedy_positions_compared':valid,'post_divergence_rows_not_compared':len(ids)-valid,'same_prefill_input_and_positions_bit_exact':True,'earliest_observed_prefill_layer_difference':next((i for i in range(32)if stages['prefill.layer.'+str(i)+'.f32']['different_bit_patterns']),None),'stage_deltas':stages,'causal_greedy_deltas':greedy,'raw_cer_to_old_candidate':{'edits':distance(text,old_text),'reference_codepoints':len(old_text)},'raw_wer_to_old_candidate':{'edits':distance(text.split(),old_text.split()),'reference_words':len(old_text.split())}};save()
  require(sha(candidate)==pack['candidate_sha256'],'retained model changed during read-only audit');result['status']='readback-verified';result['limitations']=['This compares two different quantization-coordinate recipes on matched native prefill/prefix inputs, not same-weight original BF16 runtime parity.','Only retained layer boundaries/final norm/logits are available; no new source operator stages were inferred.','Exact payload agreement does not authenticate publisher original float checkpoint identity.','No decoded-quality promotion or heldout claim follows.'];return result
 finally:result['seconds']=time.monotonic()-started;save()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mount',type=Path,required=True);p.add_argument('--receipt-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.mount,a.receipt_sha256,a.output)
