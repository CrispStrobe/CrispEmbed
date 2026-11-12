#!/usr/bin/env python3
"""Hosted source-only prefill capture, then artifact-only fixed receipt attribution."""
import hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('receipt_arrays',ROOT/'tools/attention252-probe/worker.py');arrays=importlib.util.module_from_spec(spec);spec.loader.exec_module(arrays)
require,sha,owned,source_array,delta=arrays.require,arrays.sha,arrays.owned,arrays.source_array,arrays.delta
CFG=json.loads((HERE/'protocol.json').read_text())
def comparison(candidate,reference):
 import numpy as np
 row=delta(candidate,reference);d=np.abs(candidate.astype('f8')-reference.astype('f8'));row['first_different_flat_index']=next((int(i)for i in np.flatnonzero(candidate.view('u4').reshape(-1)!=reference.view('u4').reshape(-1))[:1]),None);row['max_error_flat_index']=int(d.argmax());return row

def main():
 import numpy as np
 require(Path('/kaggle/input').is_dir()and np.__version__=='2.3.5','hosted pinned arrays');require(not any(CFG[k]for k in ['full_decode_allowed','native_model_forward_allowed','full_f32_model_allowed']),'closed inference scope')
 out=Path(os.environ['HEAVY_OUT']);start=time.monotonic();r=dict(schema=CFG['schema'],status='failed',candidate_source=CFG['candidate_source'],native_model_forward_executed=False,full_decode_executed=False,full_f32_model_instantiated=False,qualified=False)
 def save():(out/'receipt-prefill-trace.json').write_text(json.dumps(r,indent=2)+'\n')
 def locate(name,digest):
  found=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(found)==1,'one exact '+name);return found[0],json.loads(found[0].read_text())
 try:
  fp,fixed=locate('physical-rope-fixed-panel.json',CFG['fixed_receipt_sha256']);op,original=locate('heldout-result.json',CFG['reference_sha256']);require(fixed['status']=='measured_physical_rope_fixed_panel'and fixed['candidate_source']==CFG['candidate_source']and original['case']=='historical-receipt'and original['source_observer_full_neutral'],'frozen measured source/native receipts');r['fixed_receipt_sha256']=sha(fp);r['reference_receipt_sha256']=sha(op)
  sealpath=owned(op.parent,'pre-forward-inputs.json',original['artifacts']);seal=json.loads(sealpath.read_text());require(sha(sealpath)==original['pre_forward_input_seal_sha256']==fixed['cases']['historical-receipt']['pre_forward_input_seal_sha256'],'same retained original/native input seal')
  prefix='historical-receipt/on/';names=['prefill.input.f32','prefill.positions.f32','prefill.norm.f32','prefill.logits.f32','prefill.chunk_sizes.ids','prompt.ids','modalities.ids']+['prefill.layer.%d.f32'%i for i in range(32)];names += [n[len(prefix):]for n in fixed['artifacts']if n.startswith(prefix+'prefill.first_chunk.layer0.')]
  for n in names:owned(fp.parent,prefix+n,fixed['artifacts'])
  require(list(map(int,(fp.parent/(prefix+'prompt.ids')).read_text().split()))==seal['prompt_ids']and list(map(int,(fp.parent/(prefix+'modalities.ids')).read_text().split()))==seal['modalities'],'same actual native prompt IDs/modalities')
  packp=owned(fp.parent,'physical-pack.json',fixed['artifacts']);pack=json.loads(packp.read_text());require(pack['candidate_sha256']=='843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43'and not pack['full_f32_model_created']and not pack['requantization_executed']and all(t['quant_blocks_unchanged']for t in pack['tensors']if not t['protected_vision']),'unchanged physical pack/whole-row permutation of frozen source weights');r['physical_pack_sha256']=sha(packp)
  actual=out/'source';actual.mkdir();env=dict(os.environ);env['HEAVY_OUT']=str(actual);r['phase']='source-neutral-prefill';save();print('PHASE source-neutral-prefill',flush=True);subprocess.run([sys.executable,HERE/'source_worker.py'],env=env,check=True,timeout=3650)
  sr=json.loads((actual/'receipt-prefill-source.json').read_text());require(sr['status']=='measured_neutral_receipt-prefill_source'and sr['firstscore_observer_neutral'],'actual source firstscore observer neutrality');r['source_firstscore_observer_neutral']=True
  for n in sr['artifacts']:owned(actual,n,sr['artifacts'])
  indexes=sr['graph_indexes'];counts=CFG['source_prompt_rows'];require(len(indexes)==5 and sum(counts)==256,'auxiliary plus four frozen prompt graphs')
  def read(g,name):return source_array(actual/'on',indexes[g],name)
  embeddings=np.concatenate([read(g,'model.input_embed').reshape(n,2560)for g,n in zip(range(1,5),counts)])
  positions=np.concatenate([read(g,'inp_pos').reshape(4,n)[:3]for g,n in zip(range(1,5),counts)],axis=1).astype('<f4')
  native=fp.parent/'historical-receipt/on';require(embeddings.tobytes()==(native/'prefill.input.f32').read_bytes()and positions.tobytes()==(native/'prefill.positions.f32').read_bytes(),'actual source/native all256 input embedding and position bits');r['all256_source_native_inputs_positions_bit_exact']=True;r['actual_source_prompt_rows']=counts;r['native_chunk_sizes']=list(map(int,(native/'prefill.chunk_sizes.ids').read_text().split()))
  comparisons=[];offset=0
  for g,n in zip(range(1,5),counts):
   for layer in range(31):
    candidate=np.fromfile(native/('prefill.layer.%d.f32'%layer),dtype='<f4').reshape(256,2560)[offset:offset+n];ref=read(g,'l_out-'+str(layer)).reshape(n,2560);comparisons.append(dict(graph=g,rows=n,layer=layer,**comparison(candidate,ref)))
   offset+=n
  candidate=np.fromfile(native/'prefill.layer.31.f32',dtype='<f4').reshape(256,2560)[-1:];comparisons.append(dict(graph=4,rows=1,layer=31,scope='source last-layer output is cropped to final prompt token',**comparison(candidate,read(4,'l_out-31').reshape(1,2560))))
  r['hidden_comparisons']=comparisons;r['first_execution_order_hidden_difference']=next((v for v in comparisons if not v['bit_exact']),None)
  aliases={'input_norm':'attn_norm','qkv':'linear_attn_qkv_mixed','z':'z','a':'alpha','b':'beta','attention':'linear_attn_out','attention_residual':'attn_residual','post_norm':'attn_post_norm','gate':'ffn_gate','up':'ffn_up','activation':'ffn_swiglu','down':'ffn_out'};r['first_chunk_layer0_stages']=[]
  for role,stage in aliases.items():
   candidate=np.fromfile(native/('prefill.first_chunk.layer0.'+role+'.f32'),dtype='<f4');ref=read(1,stage+'-0').reshape(-1);r['first_chunk_layer0_stages'].append(dict(stage=stage,**comparison(candidate,ref)))
  final_norm=np.fromfile(native/'prefill.norm.f32',dtype='<f4').reshape(256,2560)[-1:];r['final_norm']=comparison(final_norm,read(4,'result_norm').reshape(1,2560));source_score=np.fromfile(actual/'on/source.logits.0.f32',dtype='<f4');require(read(4,'result_output').reshape(-1).tobytes()==source_score.tobytes(),'actual graph head output equals sampled source score0')
  r['firstscore']=comparison(np.fromfile(native/'prefill.logits.f32',dtype='<f4'),source_score);r['source_graph_indexes']=indexes;r['status']='measured_neutral_receipt_prefill_trace';r['limits']=['No native model execution or source full decode; retained a8ON native vectors only.','Detailed source stages are captured for layers0to3; native full hidden boundaries and first text chunk layer0 stages are available for comparison.','Earliest unequal hidden boundary identifies where further same-input component instrumentation is needed; no unique operator or batching causality is assumed.','The original failed receipt floor and all fixed/observed negative results remain unchanged.']
 except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:r['elapsed_seconds']=time.monotonic()-start;r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and p.name!='receipt-prefill-trace.json'};save()
if __name__=='__main__':main()
