"""Read retained live tensors; only compare authenticated, aligned causal rows."""
import csv,hashlib,math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from source_patch import STAGES,INPUTS

def require(ok,why):
 if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def delta(candidate,reference):
 import numpy as np
 x=np.asarray(candidate,dtype='<f4');y=np.asarray(reference,dtype='<f4');require(x.shape==y.shape and x.size and np.isfinite(x).all()and np.isfinite(y).all(),'same finite causal arrays');d=x.astype('f8')-y.astype('f8')
 return {'elements':int(x.size),'bit_exact':x.tobytes()==y.tobytes(),'different_bits':int(np.count_nonzero(x.view('<u4')!=y.view('<u4'))),'different_numeric':int(np.count_nonzero(x!=y)),'error_rms':float(np.sqrt(np.mean(d*d))),'error_max_abs':float(abs(d).max()),'candidate_rms_magnitude':float(np.sqrt(np.mean(x.astype('f8')**2))),'reference_rms_magnitude':float(np.sqrt(np.mean(y.astype('f8')**2))),'candidate_max_abs':float(abs(x).max()),'reference_max_abs':float(abs(y).max())}
def row_map(role,width):
 def p(h):return 16*(h%2)+h//2
 if role=='qkv':return list(range(4096))+[4096+128*p(i//128)+i%128 for i in range(4096)]
 if role=='z':return [128*p(i//128)+i%128 for i in range(width)]
 if role in ['a','b']:return [p(i)for i in range(width)]
 return list(range(width))
def load_graphs(folder):
 import numpy as np
 folder=Path(folder);paths=sorted(folder.glob('graph-*.tsv'),key=lambda p:int(p.stem.split('-')[1]));require(1<=len(paths)<=16,'bounded actual prefill graphs');require([int(p.stem.split('-')[1])for p in paths]==list(range(len(paths))),'contiguous graph indices');graphs=[];total=0
 for i,path in enumerate(paths):
  rows=list(csv.DictReader(path.open(),delimiter='\t'));require(len(rows)<=len(STAGES)+len(INPUTS)and len({r['name']for r in rows})==len(rows),'bounded unique graph tensor names');records={};arrays={}
  for row in rows:
   name=row['name'];require(name in STAGES+INPUTS,'closed selected tensor');ne=[int(row['ne'+str(j)])for j in range(4)];size=int(row['bytes']);require(size>=0 and all(v>=0 for v in ne),'nonnegative source geometry');require(int(row['sample_count'])in[0,126],'closed captured sampling phase');records[name]={**row,'ne':ne}
   if row['status']=='captured':
    file=folder/row['file'];require(row['file']==f'graph-{i}.{name}.bin'and 0<size<=8<<20 and file.stat().st_size==size and int(row['type'])in[0,26]and size==math.prod(ne)*4,'captured full F32/I32 tensor');total+=size;require(total<=128<<20,'bounded live source bytes');a=np.fromfile(file,dtype='<f4'if int(row['type'])==0 else'<i4');require(int(row['type'])!=0 or np.isfinite(a).all(),'finite live source values');arrays[name]=a;records[name]['sha256']=sha(file)
  require('model.input_embed'in arrays,'actual model input required');emb=records['model.input_embed'];require(int(emb['type'])==0 and emb['ne'][0]==2560 and emb['ne'][2:]==[1,1],'actual decoder input matrix');n=emb['ne'][1];require(1<=n<=128 and 'inp_pos'in arrays,'bounded actual token block/positions');phases={int(r['sample_count'])for r in rows};require(len(phases)==1,'one graph sampling phase');graphs.append({'index':i,'tokens':n,'sample_count':phases.pop(),'records':records,'arrays':arrays})
 require([int(x)for x in(folder/'sampling-steps.tsv').read_text().split()]==list(range(282)),'actual282 sampling calls, captured only declared phases');return graphs,total

def select_prefill(graphs):
 """Exclude only pinned common_context_can_seq_rm's independently identified probe."""
 import numpy as np
 prefill=[g for g in graphs if g['sample_count']==0];excluded=[]
 if sum(g['tokens']for g in prefill)==88:
  g=prefill[0];a=g['arrays'];require(g['index']==0 and g['tokens']==2 and a.get('inp_tokens',np.array([])).tolist()==[0,0]and a.get('inp_pos',np.array([])).tolist()==[0,1]*3+[0,0],'exact pinned sequence-removal compatibility probe')
  excluded=[{'graph':0,'tokens':[0,0],'positions_4axes':[[0,1]]*3+[[0,0]],'reason':'Pinned common_context_can_seq_rm probes two zero IDs then clears memory and synchronizes before actual prompt.','common_cpp_sha256':'1b41b3613a10d579e2808bf5e1a73531980ca836d0f4d23dba22d7130d78985e','server_context_cpp_sha256':'e0e43f22fd2357a1e76ad6b8996b099d62615dd973de33feaa02fa0ee1a2a973'}];prefill=prefill[1:]
 require(sum(g['tokens']for g in prefill)==86,'actual table prompt token census');return prefill,excluded

def analyze(folder,native_folder,first_score,step_score,source_ids):
 import numpy as np
 all_graphs,total=load_graphs(folder);graphs,excluded=select_prefill(all_graphs);steps=[g for g in all_graphs if g['sample_count']==126];require(len(steps)==1 and steps[0]['tokens']==1,'one actual same-prefix126 graph');native=Path(native_folder);count=sum(g['tokens']for g in graphs);require(count==86,'actual table prompt token census');input_native=np.fromfile(native/'prefill.input.f32',dtype='<f4').reshape(86,2560);pos_native=np.fromfile(native/'prefill.positions.f32',dtype='<f4').reshape(3,86);emb=np.concatenate([g['arrays']['model.input_embed'].reshape(g['tokens'],2560)for g in graphs]);pos=[];offset=0
 for g in graphs:
  n=g['tokens'];a=g['arrays']['inp_pos'];require(a.size in[3*n,4*n],'source MRoPE axes');p=a.reshape(a.size//n,n);pos.append(p[:3]);g['offset']=offset;offset+=n
 positions=np.concatenate(pos,axis=1);same_input=emb.tobytes()==input_native.tobytes();same_pos=np.array_equal(positions.astype('f4'),pos_native)
 result={'excluded_nonprompt_graphs':excluded,'actual_graph_token_counts':[g['tokens']for g in graphs],'captured_bytes':total,'total_prompt_tokens':count,'same_model_input_bits':same_input,'same_positions_first3_axes':bool(same_pos),'input_delta':delta(input_native,emb),'tensor_metadata':[{k:v for k,v in g.items()if k!='arrays'}for g in all_graphs],'comparisons':{},'unsupported':[],'scope':'Actual source batching is retained; no tensor values or missing stages are inferred.'}
 source_first=np.fromfile(first_score,dtype='<f4');require(source_first.size==248320 and np.isfinite(source_first).all(),'authenticated first source vocabulary row');matches=[]
 for g in graphs:
  a=g['arrays'].get('result_output')
  if a is not None:
   require(g['records']['result_output']['ne'][0]==248320,'source vocabulary dimension')
   for j,v in enumerate(a.reshape(-1,248320)):
    if v.tobytes()==source_first.tobytes():matches.append([g['index'],j])
 require(len(matches)==1,'live output closes actual sampler first row');result['live_output_matches_first_score']=matches
 step=steps[0];a=step['arrays'].get('result_output');reference=np.fromfile(step_score,dtype='<f4');require(a is not None and a.tobytes()==reference.tobytes(),'live step126 output closes actual sampler');token=step['arrays'].get('inp_tokens');require(token is not None and token.tolist()==[source_ids[125]],'actual126 input token');candidate=np.fromfile(native/'greedy.126.logits.f32',dtype='<f4');result['same_prefix126_logits']=delta(candidate,reference);result['same_prefix126_logits'].update(native_argmax=int(candidate.argmax()),source_argmax=int(reference.argmax()),source_token=source_ids[126],source_top2_margin=float(np.partition(reference,-2)[-1]-np.partition(reference,-2)[-2]),native_top2_margin=float(np.partition(candidate,-2)[-1]-np.partition(candidate,-2)[-2]));result['firstscore_logits']=delta(np.fromfile(native/'prefill.logits.f32',dtype='<f4'),source_first)
 if not(same_input and same_pos):result['comparison_status']='blocked-input-alignment';return result
 for stage,filename in [(f'l_out-{i}',f'prefill.layer.{i}.f32')for i in range(32)]+[('result_norm','prefill.norm.f32')]:
  reference=np.fromfile(native/filename,dtype='<f4').reshape(86,2560);pairs=[]
  for g in graphs:
   name=stage;a=g['arrays'].get(name)
   if a is None:continue
   require(g['records'][name]['ne'][0]==2560,'source hidden width');x=a.reshape(-1,2560);n=g['tokens'];indices=list(range(n))if len(x)==n else g['arrays'].get('out_ids',np.array([],dtype='<i4')).tolist()
   if len(indices)!=len(x)or any(not 0<=j<n for j in indices):result['unsupported'].append({'graph':g['index'],'stage':name,'reason':'output crop mapping unavailable'});continue
   selected=reference[np.asarray(indices)+g['offset']];pairs.append({'graph':g['index'],'native_global_token_rows':[int(j)+g['offset']for j in indices],'source_batch_tokens':n,**delta(selected,x)})
  result['comparisons'][stage]=pairs
 stages=[('attn_norm-0','input_norm',2560,None),('linear_attn_qkv_mixed-0','qkv',8192,'qkv'),('z-0','z',4096,'z'),('alpha-0','a',32,'a'),('beta-0','b',32,'b'),('linear_attn_out-0','attention',2560,None),('attn_residual-0','attention_residual',2560,None),('attn_post_norm-0','post_norm',2560,None),('ffn_gate-0','gate',9216,None),('ffn_up-0','up',9216,None),('ffn_swiglu-0','activation',9216,None),('ffn_out-0','down',2560,None)]
 result['comparisons']['layer0_first_block']=[];result['layer0_source_native_projection_inputs_bits_equal']=[]
 for g in graphs:
  offset=g['offset'];take=min(16-offset,g['tokens'])
  if take<=0:continue
  raw=g['arrays'];norm=raw.get('attn_norm-0');same_projected=False
  if norm is not None:
   a=norm.reshape(-1,2560)[:take];b=np.fromfile(native/'prefill.first_chunk.layer0.input_norm.f32',dtype='<f4').reshape(-1,2560)[offset:offset+take];same_projected=a.tobytes()==b.tobytes()
  result['layer0_source_native_projection_inputs_bits_equal'].append({'graph':g['index'],'native_global_token_rows':list(range(offset,offset+take)),'bit_exact':same_projected})
  for source,name,width,role in stages:
   if source not in raw:result['unsupported'].append({'graph':g['index'],'stage':source,'reason':'not captured/materialized'});continue
   a=raw[source].reshape(-1,width)[:take];a=a[:,row_map(role,width)]if role else a;b=np.fromfile(native/f'prefill.first_chunk.layer0.{name}.f32',dtype='<f4').reshape(-1,width)[offset:offset+take];result['comparisons']['layer0_first_block'].append({'graph':g['index'],'native_global_token_rows':list(range(offset,offset+take)),'source_stage':source,'native_stage':name,'tokens':take,'source_graph_tokens':g['tokens'],'native_chunk_tokens':16,**delta(b,a)})
 result['comparison_status']='aligned-actual-stages';return result
