#!/usr/bin/env python3
"""Hosted-only same-input layer0 and quantized projection attribution; no model loader."""
import hashlib,importlib.util,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
spec=importlib.util.spec_from_file_location('receipt_arrays',ROOT/'tools/attention252-probe/worker.py');arrays=importlib.util.module_from_spec(spec);spec.loader.exec_module(arrays)
require,sha,owned,delta,source_array=arrays.require,arrays.sha,arrays.owned,arrays.delta,arrays.source_array
CFG=json.loads((HERE/'protocol.json').read_text())
FLAGS={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_AVX512':'OFF','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_BLAS':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF','GGML_BACKEND_DL':'OFF','GGML_METAL':'OFF','CRISPEMBED_BUILD_SHARED':'OFF'}
PAIRS=[('attn_norm','input_norm',2560),('linear_attn_qkv_mixed','qkv',8192),('z','z',4096),('alpha','a',32),('beta','b',32),('final_output','postgdn',4096),('linear_attn_out','attention',2560),('attn_residual','attention_residual',2560),('attn_post_norm','post_norm',2560),('ffn_gate','gate',9216),('ffn_up','up',9216),('ffn_swiglu','activation',9216),('ffn_out','down',2560),('l_out','hidden',2560)]
def permutation(role,width):
 head=lambda h:16*(h%2)+h//2
 if role=='qkv':return list(range(4096))+[4096+128*head(i//128)+i%128 for i in range(4096)]
 if role in ['z','postgdn']:return [128*head(i//128)+i%128 for i in range(width)]
 if role in ['a','b']:return [head(i)for i in range(width)]
 return list(range(width))
def compare(x,y):
 import numpy as np
 r=delta(x,y);indices=np.flatnonzero(x.view('u4').reshape(-1)!=y.view('u4').reshape(-1));r['first_different_flat_index']=int(indices[0])if len(indices)else None
 r['different_rows']=np.flatnonzero(np.any(x.view('u4')!=y.view('u4'),axis=1)).tolist();return r

def main():
 import numpy as np
 require(Path('/kaggle/input').is_dir()and np.__version__=='2.3.5'and not list(Path('/dev').glob('nvidia[0-9]*')),'hosted CPU/pinned arrays')
 require(not any(CFG[k]for k in ['model_forward_allowed','source_capture_allowed','full_f32_model_allowed','runtime_changes_allowed']),'closed operator-only scope')
 out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'receipt-layer0';work.mkdir();started=time.monotonic();deadline=Deadline(CFG['worker_seconds']);deadline.__enter__();r=dict(schema=CFG['schema'],status='failed',commands={},model_forward_executed=False,source_capture_executed=False,full_f32_model_instantiated=False,qualified=False)
 def save():(out/'receipt-layer0.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,args,env=None,limit=900,expected=0):
  r['phase']=label;r['commands'][label]={};save();print('PHASE '+label,flush=True)
  try:
   try:command(args,out/(label+'.log'),min(limit,CFG['worker_seconds']-(time.monotonic()-started)),r['commands'][label],cwd=ROOT,env=env)
   except subprocess.CalledProcessError as e:require(e.returncode==expected,'expected bounded control exit')
   require(r['commands'][label]['returncode']==expected,'actual command return code')
  finally:save()
 def locate(name,digest):
  paths=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(paths)==1,'one exact '+name);return paths[0],json.loads(paths[0].read_text())
 try:
  require(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==CFG['candidate_source'],'exact production candidate checkout');r['candidate_source']=CFG['candidate_source'];r['tools_source']=os.environ['LIGHTON4B_TOOLS_SOURCE']
  mem=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024;cap=Path('/sys/fs/cgroup/memory.max').read_text().strip()
  if cap!='max':mem=min(mem,int(cap)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
  r['capacity']=dict(available_bytes=mem,scratch_free=shutil.disk_usage(work).free,output_free=shutil.disk_usage(out).free,load=os.getloadavg());require(mem>=6<<30 and r['capacity']['scratch_free']>=6<<30 and r['capacity']['output_free']>=2<<30,'bounded component resource admission')
  sp,s=locate('receipt-prefill-source.json',CFG['source_report_sha256']);tp,t=locate('receipt-prefill-trace.json',CFG['trace_sha256']);fp,f=locate('physical-rope-fixed-panel.json',CFG['fixed_sha256']);require(s['firstscore_observer_neutral']and t['all256_source_native_inputs_positions_bit_exact']and f['candidate_source']==CFG['candidate_source'],'closed exact source/native captures')
  r['source_report_sha256']=sha(sp);r['trace_report_sha256']=sha(tp);r['fixed_report_sha256']=sha(fp)
  def source(g,stage):
   owned(sp.parent,'on/graph-%d.%s.bin'%(g,stage),s['artifacts']);return source_array(sp.parent/'on',s['graph_indexes'][g],stage)
  actual=work/'actual';actual.mkdir();native=fp.parent/'historical-receipt/on'
  for src,dst in [('prefill.input.f32','input.f32'),('prefill.layer.0.f32','hidden.f32')]:shutil.copy2(owned(fp.parent,'historical-receipt/on/'+src,f['artifacts']),actual/dst)
  embeddings=np.concatenate([source(g,'model.input_embed').reshape(n,2560)for g,n in zip(range(1,5),CFG['source_prompt_rows'])]);require(embeddings.tobytes()==(actual/'input.f32').read_bytes(),'all256 exact component input rows')
  packp=owned(fp.parent,'physical-pack.json',f['artifacts']);pack=json.loads(packp.read_text());require(pack['candidate_sha256']=='843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43'and not pack['full_f32_model_created']and not pack['requantization_executed'],'unchanged physical transport')
  rows=[row for row in pack['tensors']if row.get('physical_name')and row['physical_name'].startswith('blk.0.')];require(len(rows)==14 and all(row['quant_blocks_unchanged']for row in rows),'one exact linear layer14 tensors');require(sum(row['bytes']for row in rows)<=CFG['maximum_parameter_bytes'],'bounded component parameters');r['parameter_manifest']=rows
  run('submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==CFG['ggml_source'],'exact native GGML')
  build=work/'build';run('configure',['cmake','-S',HERE,'-B',build,'-DCRISPEMBED_SOURCE='+str(ROOT),'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in FLAGS.items()]])
  cache=(build/'CMakeCache.txt').read_text().splitlines();require(all(f'{k}:BOOL={v}'in cache for k,v in FLAGS.items()),'same exact candidate CPU build profile');shutil.copy2(build/'CMakeCache.txt',out/'CMakeCache.txt')
  run('baseline',[sys.executable,ROOT/'scripts/check-cpu-baseline.py',build]);run('build',['cmake','--build',build,'--target','receipt-layer0-component','receipt-projection-component','--parallel','2'],limit=1500)
  program=build/'receipt-layer0-component';proj=build/'receipt-projection-component';run('arity-layer0',[program],expected=1);run('arity-projection',[proj],expected=1)
  z=work/'zero.q4';z.write_bytes(bytes(2560//256*144*32));zx=work/'zero-input.f32';np.ones((10,2560),dtype='<f4').tofile(zx);run('supported-zero-repack',[proj,z,zx,12,2560,32,10,'repack',work/'zero']);require(not np.any(np.fromfile(work/'zero.f32',dtype='<f4')),'actual preferred Q4 zero-matrix control')
  weights=work/'weights';weights.mkdir();raw_paths={}
  for row in rows:
   begin,size=row['source_offset'],row['bytes'];url=CFG['source_model_url']+'?download=true&receipt_range=%d-%d'%(begin,begin+size-1);request=urllib.request.Request(url,headers={'Range':'bytes=%d-%d'%(begin,begin+size-1),'Accept-Encoding':'identity'})
   with urllib.request.urlopen(request,timeout=120)as response:
    require(response.status==206 and response.headers.get('Content-Range')=='bytes %d-%d/%d'%(begin,begin+size-1,CFG['source_model_bytes']),'bounded public tensor range');raw=response.read(size+1)
   require(len(raw)==size and hashlib.sha256(raw).hexdigest()==row['source_packed_sha256'],'whole unchanged source tensor SHA');rawp=weights/(row['physical_name']+'.source');rawp.write_bytes(raw);raw_paths[row['physical_name']]=rawp
   name=row['name'];order=None
   if name.endswith('.linear_attn.conv1d.weight'):order=permutation('qkv',8192)
   elif name.endswith(('.linear_attn.A_log','.linear_attn.dt_bias')):order=permutation('a',32)
   elif row['whole_rows_permuted']:
    role=next(role for key,role in [('in_proj_qkv','qkv'),('in_proj_z','z'),('in_proj_a','a'),('in_proj_b','b')]if key in name);order=permutation(role,row['source_shape'][1])
   transformed=np.frombuffer(raw,dtype='u1').reshape(len(order),-1)[order].copy().tobytes()if order else raw
   require(hashlib.sha256(transformed).hexdigest()==row['sha256'],'exact native whole-row transport SHA');(weights/(row['physical_name']+'.bin')).write_bytes(transformed)
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update(f['gates']['on']);require(all(env[key]=='1'for key in ['CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU','CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE','CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL','CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS','CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK','CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED']),'exact historical experimental gate tuple');r['component_gates']={k:v for k,v in env.items()if k.startswith('CRISPEMBED_')};run('actual-layer0',[program,weights,actual,out/'native'],env,limit=600);r['all256_actual_native_hidden0_byte_exact']=True;r['duplicate_production_gdn_cache_exact']=True
  comparisons=[];offset=0
  for g,n in zip(range(1,5),CFG['source_prompt_rows']):
   for stage,role,width in PAIRS:
    ref=source(g,stage+'-0').reshape(n,width)[:,permutation(role,width)];candidate=np.fromfile(out/'native'/(role+'.f32'),dtype='<f4').reshape(256,width)[offset:offset+n];comparisons.append(dict(graph=g,stage=stage,role=role,**compare(candidate,ref)))
   offset+=n
  r['stage_comparisons']=comparisons;r['first_image_stage_difference']=next((v for v in comparisons if v['graph']==2 and not v['bit_exact']),None);save()
  # Fixed input intervention: preferred Q4 projections with native10 versus source242 batches.
  projection_results={}
  for physical,stage,width in [('blk.0.attn_gate.weight','z',4096),('blk.0.ssm_alpha.weight','alpha',32),('blk.0.ssm_beta.weight','beta',32),('blk.0.ffn_gate.weight','ffn_gate',9216),('blk.0.ffn_up.weight','ffn_up',9216)]:
   x=source(2,'attn_norm-0'if stage in ['z','alpha','beta']else'attn_post_norm-0').reshape(242,2560);reference=source(2,stage+'-0').reshape(242,width);projection_results[stage]={}
   for count,begin in [(242,0),(16,224),(10,224),(8,234)]:
    inp=work/(stage+str(count)+'.f32');x[begin:begin+count].astype('<f4').tofile(inp);prefix=out/(stage+'-'+str(count));run(stage+'-'+str(count),[proj,raw_paths[physical],inp,12,2560,width,count,'repack',prefix],env,120);value=np.fromfile(str(prefix)+'.f32',dtype='<f4').reshape(count,width);projection_results[stage][str(count)]=dict(source_input_rows=[begin,begin+count],**compare(value,reference[begin:begin+count]))
  r['same_input_preferred_projection_interventions']=projection_results;r['status']='measured_receipt_layer0_same_input';r['limits']=['One production layer0 replay on existing exact receipt embeddings; no model loader, decoder, source forward, recipe or runtime change.','Source and native projection head order is canonicalized before every comparison.','Source-input projection interventions establish local batch-route effects only; full-model decoded causality and qualification remain unresolved.','All fixed receipt/observed BF16 failures and default-off gates remain binding.']
 except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  deadline.close();r['seconds']=time.monotonic()-started;r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and p.name!='receipt-layer0.json'};save()
if __name__=='__main__':main()
