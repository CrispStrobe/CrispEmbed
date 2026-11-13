#!/usr/bin/env python3
"""Hosted-only frozen layer3 replay and actual quantized source prefill capture."""
import hashlib,importlib.util,json,os,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
CFG=json.loads((HERE/'protocol.json').read_text())

def require(ok,why):
 if not ok:raise ValueError(why)

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def owned(root,name,manifest):
 root=Path(root).resolve();p=(root/name).resolve();require(p.is_relative_to(root)and name in manifest,'closed owned path');r=manifest[name]
 require(p.is_file()and p.stat().st_size==r['bytes']and sha(p)==r['sha256'],'owned artifact SHA/size');return p

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def delta(x,y):
 import numpy as np
 require(x.shape==y.shape and x.size>0 and np.isfinite(x).all()and np.isfinite(y).all(),'finite equal arrays');a=x.astype('f8');b=y.astype('f8');d=a-b
 return dict(elements=int(x.size),bit_exact=x.tobytes()==y.tobytes(),error_rms=float(np.sqrt(np.mean(d*d))),error_max_abs=float(np.abs(d).max()),native_rms=float(np.sqrt(np.mean(a*a))),source_rms=float(np.sqrt(np.mean(b*b))),native_max_abs=float(np.abs(a).max()),source_max_abs=float(np.abs(b).max()))

def source_array(folder,index,name):
 import numpy as np
 rows=[x for x in index['records']if x['name']==name];require(len(rows)==1,'unique actual source stage');r=rows[0];require(r['status']=='captured'and r['type']in['0','1','26'],'actual supported stage')
 ne=[int(r['ne%d'%i])for i in range(4)];nb=[int(r['nb%d'%i])for i in range(4)];dtype={'0':'<f4','1':'<f2','26':'<i4'}[r['type']];p=folder/r['file'];raw=p.read_bytes();require(len(raw)==int(r['bytes']),'complete source span')
 return np.ndarray(shape=tuple(reversed(ne)),dtype=dtype,buffer=raw,strides=tuple(reversed(nb))).copy()

def main():
 import numpy as np
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir()and np.__version__=='2.3.5','hosted pinned arrays only')
 require(not any(CFG[k]for k in['full_f32_model_allowed','native_full_model_forward_allowed','full_decode_allowed']),'closed forward scope')
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(exist_ok=True);scratch=Path(os.environ['HEAVY_SCRATCH'])/'attention252';scratch.mkdir();started=time.monotonic();deadline=Deadline(CFG['total_timeout_seconds']);deadline.__enter__()
 r=dict(schema=CFG['schema'],status='failed',commands={},qualified=False,full_f32_model_instantiated=False,native_full_model_forward_executed=False,full_decode_executed=False)
 def save():(out/'attention252.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,args,env=None,limit=1200):
  r['phase']=label;r['commands'][label]={};save();left=CFG['total_timeout_seconds']-(time.monotonic()-started);require(left>0,'whole deadline')
  try:command(args,out/(label+'.log'),min(left,limit),r['commands'][label],cwd=ROOT,env=env)
  finally:save()
 def locate(name,digest):
  candidates=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(candidates)==1,'one frozen report');return candidates[0]
 try:
  r['diagnostic_source_commit']=os.environ['LIGHTON4B_TOOLS_SOURCE'];require(len(r['diagnostic_source_commit'])==40 and all(c in '0123456789abcdef'for c in r['diagnostic_source_commit']),'exact diagnostic source identity');r['diagnostic_tool_files']={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in HERE.iterdir()if p.is_file()}
  r['source_sha']=subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact worker source');require(subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD:src'],text=True).strip()==CFG['candidate_src_tree'],'production sources unchanged')
  mem=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024;limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
  if limit!='max':mem=min(mem,int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
  r['capacity']=dict(available_bytes=mem,scratch_free=shutil.disk_usage(scratch).free,output_free=shutil.disk_usage(out).free,load=os.getloadavg());require(mem>=10<<30 and r['capacity']['scratch_free']>=12<<30 and r['capacity']['output_free']>=4<<30,'bounded hosted resources')
  sp=locate('book-text-trace.json',CFG['source_receipt_sha256']);sr=json.loads(sp.read_text());npth=locate('swiglu-fixed-panel.json',CFG['native_receipt_sha256']);nr=json.loads(npth.read_text());require(sr['all654_scores_observer_neutral']and nr['status']=='failed'and nr['phase']=='inventory-table-preflight','immutable negative sources')
  require(nr['candidate_source']==CFG['candidate_source']and nr['gates']['on']['CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU']=='1','fixed runtime/gates')
  # Authenticate all needed frozen native arrays before rebuilding or acquiring weights.
  actual=scratch/'actual';actual.mkdir()
  for name in ['prefill.layer.2.f32','prefill.layer.3.f32','prefill.positions.f32','prefill.input.f32']:
   p=owned(npth.parent,'two-column-page/on/'+name,nr['artifacts']);shutil.copy2(p,actual/name)
  source_hidden=[]
  for g,count in zip(range(1,5),[4,252,6,4]):
   for stage in ['l_out-2','l_out-3','model.input_embed','inp_pos']:
    owned(sp.parent,'on/graph-%d.%s.bin'%(g,stage),sr['artifacts'])
   source_hidden.append(np.fromfile(sp.parent/('on/graph-%d.l_out-2.bin'%g),dtype='<f4').reshape(count,2560))
  require(np.concatenate(source_hidden).tobytes()==(actual/'prefill.layer.2.f32').read_bytes(),'source/native exact all266 layer2 input')
  inputs=np.concatenate([source_array(sp.parent/'on',sr['graph_indexes'][g],'model.input_embed').reshape(count,2560)for g,count in zip(range(1,5),[4,252,6,4])])
  positions=np.concatenate([source_array(sp.parent/'on',sr['graph_indexes'][g],'inp_pos').reshape(4,count)[:3]for g,count in zip(range(1,5),[4,252,6,4])],axis=1).astype('<f4')
  require(inputs.tobytes()==(actual/'prefill.input.f32').read_bytes()and positions.tobytes()==(actual/'prefill.positions.f32').read_bytes(),'actual all266 embeddings and positions equality')
  r['actual_same_input_all266_bits']=True;r['actual_prompt_rows']=[4,252,6,4]
  # Exact source tensors only: bounded range transfers never acquire an F32 model.
  packp=owned(npth.parent,'physical-pack.json',nr['artifacts']);pack=json.loads(packp.read_text());weights=scratch/'weights';weights.mkdir()
  for row in CFG['tensors']:
   mapped=[v for v in pack['tensors']if v.get('physical_name')==row['physical_name']];require(len(mapped)==1 and mapped[0]['source_packed_sha256']==row['source_packed_sha256']and mapped[0]['sha256']==row['source_packed_sha256'],'unchanged source/native layer3 tensor')
   start,size=row['source_offset'],row['bytes'];p=weights/(row['physical_name']+'.bin');request=urllib.request.Request(CFG['source_url'],headers={'Range':'bytes=%d-%d'%(start,start+size-1)})
   with urllib.request.urlopen(request,timeout=120)as response:
    require(response.status==206 and response.headers.get('Content-Range')=='bytes %d-%d/%d'%(start,start+size-1,CFG['source_bytes']),'exact frozen tensor range');b=response.read(size+1)
   require(len(b)==size and hashlib.sha256(b).hexdigest()==row['source_packed_sha256'],'whole quantized tensor seal');p.write_bytes(b)
  r['same_quantized_weight_tensors']=len(CFG['tensors']);r['quantized_weight_bytes']=sum(x['bytes']for x in CFG['tensors'])
  run('native-submodule',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==CFG['native_ggml_commit'],'exact native backend')
  flags={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_AVX512':'OFF','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_CPU_REPACK':'ON','GGML_BLAS':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF','GGML_BACKEND_DL':'OFF'};build=scratch/'native-build'
  run('native-configure',['cmake','-S',HERE,'-B',build,'-DCRISPEMBED_SOURCE='+str(ROOT),'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]])
  cache=(build/'CMakeCache.txt').read_text().splitlines()
  for k,v in flags.items():require(f'{k}:BOOL={v}'in cache,'actual native build profile')
  shutil.copy2(build/'CMakeCache.txt',out/'native-CMakeCache.txt');run('native-build',['cmake','--build',build,'--target','attention252-native','--parallel','2'])
  env=dict(os.environ);env.update(nr['gates']['on']);native=out/'native';run('native-layer3-closure',[build/'attention252-native',weights,actual,native],env,900);require('ACTUAL_NATIVE_LAYER3_ALL266_BYTE_CLOSURE PASS'in(out/'native-layer3-closure.log').read_text(),'actual native closure marker');r['actual_native_all266_hidden3_bits']=True
  # Fresh quantized source prefill + first sample only; original 654-token output is never regenerated.
  source_out=out/'source';source_out.mkdir();source_env=dict(os.environ);source_env['HEAVY_OUT']=str(source_out)
  run('source-prefill-observer',[sys.executable,HERE/'source_worker.py'],source_env,3600)
  new=json.loads((source_out/'attention252-source.json').read_text());require(new['status']=='measured_neutral_attention252_source'and new['firstscore_observer_neutral'],'actual source observer neutrality');r['source_firstscore_observer_neutral']=True
  # Authenticate each fresh array and demand every old available hidden boundary unchanged.
  for name in new['artifacts']:owned(source_out,name,new['artifacts'])
  for g in range(1,5):
   for stage in ['model.input_embed','inp_pos']+['l_out-'+str(i)for i in range(32)]:
    name='on/graph-%d.%s.bin'%(g,stage)
    if name in sr['artifacts']:
     require(name in new['artifacts']and new['artifacts'][name]==sr['artifacts'][name],'frozen source hidden observer closure')
  r['source_retained_hidden_stages_exact']=True
  widths={'input_norm':2560,'q_full':8192,'k_raw':1024,'v_raw':1024,'q_norm':4096,'k_norm':1024,'q_rope':4096,'k_rope':1024,'attn_gate':4096,'pooled':4096,'attention':2560,'attention_residual':2560,'post_norm':2560,'ffn_gate':9216,'ffn_up':9216,'activation':9216,'down':2560,'hidden':2560}
  pairs=[('attn_norm','input_norm'),('Qcur_full','q_full'),('Kcur_raw','k_raw'),('Vcur_raw','v_raw'),('Qcur_normed','q_norm'),('Kcur_normed','k_norm'),('Qcur','q_rope'),('Kcur','k_rope'),('gate_reshaped','attn_gate'),('attn_gated','pooled'),('attn_output','attention'),('attn_residual','attention_residual'),('attn_post_norm','post_norm'),('ffn_gate','ffn_gate'),('ffn_up','ffn_up'),('ffn_swiglu','activation'),('ffn_out','down'),('l_out','hidden')];comparisons=[];offset=0
  for g,count in zip(range(1,5),[4,252,6,4]):
   index=new['graph_indexes'][g]
   for stage,role in pairs:
    ref=source_array(source_out/'on',index,stage+'-3').reshape(count,widths[role]);candidate=np.fromfile(native/(role+'.f32'),dtype='<f4').reshape(266,widths[role])[offset:offset+count]
    comparisons.append(dict(graph=g,stage=stage,**delta(candidate,ref)))
   offset+=count
  cache_checks=[]
  for g,count,tag in [(1,4,'cache4'),(2,256,'cache256')]:
   index=new['graph_indexes'][g]
   key=source_array(source_out/'on',index,'operand-kq-3-0').astype('<f4').reshape(4,256,256)[:,:count]
   value=source_array(source_out/'on',index,'operand-kqv-3-0').astype('<f4').reshape(4,256,256)[:,:,:count].transpose(0,2,1).copy()
   for role,reference in [('keys',key),('values',value)]:
    candidate=np.fromfile(native/(tag+'.'+role+'.f32'),dtype='<f4').reshape(4,count,256)
    row=dict(scope=tag,stage=role,**delta(candidate,reference));cache_checks.append(row)
    if g==1:require(row['bit_exact'],'actual equal four-token attention context/cache')
  q_operand=[]
  for g,count,offset in [(1,4,0),(2,252,4)]:
   reference=source_array(source_out/'on',new['graph_indexes'][g],'operand-kq-3-1').reshape(16,count,256).transpose(1,0,2).reshape(count,4096)
   candidate=np.fromfile(native/'q_rope.f32',dtype='<f4').reshape(266,4096)[offset:offset+count]
   q_operand.append(dict(graph=g,stage='actual_QK_Q_operand_vs_reconstructed_native_RoPE',**delta(candidate,reference)))
  r['actual_q_operand_comparisons']=q_operand
  r['cache_comparisons']=cache_checks;r['actual_four_token_cache_bits']=True
  r['comparisons']=comparisons;r['first_image_stage_difference']=next((x for x in comparisons if x['graph']==2 and not x['bit_exact']),None)
  r['source_graph_indexes']=new['graph_indexes'];r['status']='measured_actual_layer3_stages';r['limits']=['Native RoPE vectors are reconstructed from actual normalized Q/K using the same production scalar arithmetic; actual pooled/full block byte closure is separately required.','Earliest differing stage isolates a boundary only; direct same-input interventions precede any unique operator/dispatch claim.','Source runs only prefill plus one sample; source654 EOS/native1536 cap remain the unchanged frozen decoded evidence.','No production source change, default promotion, F32 model or full decoded rerun.']
 except BaseException as error:r['error_type'],r['error']=type(error).__name__,str(error);raise
 finally:
  deadline.close();r['elapsed_seconds']=time.monotonic()-started;r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and p.name!='attention252.json'};save()

if __name__=='__main__':main()
