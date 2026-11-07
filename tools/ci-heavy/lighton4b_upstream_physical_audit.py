#!/usr/bin/env python3
"""No-inference physical Q4/Q6 import feasibility audit, bounded chunks only."""
import hashlib,json,math,os,shutil,signal,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def require(x,s):
 if not x:raise RuntimeError(s)
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest_span(path,offset,size):
 h=hashlib.sha256()
 with path.open('rb')as f:
  f.seek(offset)
  while size:
   b=f.read(min(size,8*1024**2));require(bool(b),'truncated span');h.update(b);size-=len(b)
 return h.hexdigest()
def resources():
 mem={k:int(v.split()[0])*1024 for k,v in(line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if k in ('MemAvailable','SwapTotal','SwapFree')};observed=0;bounds=[]
 for line in Path('/proc/self/cgroup').read_text().splitlines():
  h,c,r=line.split(':',2)
  if h=='0'and c=='':base=Path('/sys/fs/cgroup');ln='memory.max';un='memory.current'
  elif 'memory'in c.split(','):base=Path('/sys/fs/cgroup/memory');ln='memory.limit_in_bytes';un='memory.usage_in_bytes'
  else:continue
  for node in (base/r.lstrip('/'),*(base/r.lstrip('/')).parents):
   if node!=base and base not in node.parents:break
   try:
    raw=(node/ln).read_text().strip();used=int((node/un).read_text());observed+=1
    if raw!='max'and int(raw)<2**60:bounds.append(max(0,int(raw)-used))
   except(OSError,ValueError):pass
 require(observed>0,'cgroup inspection');mem['finite_cgroup_bounds']=len(bounds)
 if bounds:mem['MemAvailable']=min(mem['MemAvailable'],*bounds)
 return mem
def physical_name(native):
 if native=='l.embed_tokens.weight':return'token_embd.weight'
 if native=='l.norm.weight':return'output_norm.weight'
 parts=native.split('.');layer=parts[2];suffix='.'.join(parts[3:]);names={'input_layernorm.weight':'attn_norm.weight','post_attention_layernorm.weight':'post_attention_norm.weight','mlp.gate_proj.weight':'ffn_gate.weight','mlp.up_proj.weight':'ffn_up.weight','mlp.down_proj.weight':'ffn_down.weight','self_attn.q_proj.weight':'attn_q.weight','self_attn.k_proj.weight':'attn_k.weight','self_attn.v_proj.weight':'attn_v.weight','self_attn.o_proj.weight':'attn_output.weight','self_attn.q_norm.weight':'attn_q_norm.weight','self_attn.k_norm.weight':'attn_k_norm.weight','linear_attn.in_proj_qkv.weight':'attn_qkv.weight','linear_attn.in_proj_z.weight':'attn_gate.weight','linear_attn.in_proj_a.weight':'ssm_alpha.weight','linear_attn.in_proj_b.weight':'ssm_beta.weight','linear_attn.out_proj.weight':'ssm_out.weight','linear_attn.conv1d.weight':'ssm_conv1d.weight','linear_attn.A_log':'ssm_a','linear_attn.dt_bias':'ssm_dt.bias','linear_attn.norm.weight':'ssm_norm.weight'}
 return'blk.'+layer+'.'+names[suffix]
def head_perm(width=128):return[(16*(h%2)+h//2)*width+c for h in range(32)for c in range(width)]
def rows_for(name,rows):
 if name.endswith('.linear_attn.in_proj_qkv.weight'):return list(range(4096))+[4096+x for x in head_perm()]
 if name.endswith('.linear_attn.in_proj_z.weight'):return head_perm()
 if name.endswith(('.linear_attn.in_proj_a.weight','.linear_attn.in_proj_b.weight')):return head_perm(1)
 return list(range(rows))
def main():
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir(),'authorized Kaggle CPU only')
 import numpy as np
 require(np.__version__=='2.3.5'and sys.byteorder=='little','independent serializer and endian pin')
 sys.path.insert(0,str(ROOT/'tools'));from gguf_file_type_canonicalize import parse
 cfgpath=ROOT/'docs/lighton4b-upstream-physical-audit-v1.json';cfg=json.loads(cfgpath.read_text());out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'physical-audit';scratch.mkdir();active=None;report={'schema':'lighton4b-upstream-physical-audit-result/v1','status':'failed','source_sha':subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip(),'protocol_sha256':sha(cfgpath),'inference_executed':False,'full_f32_model_created':False,'quantization_or_requantization_executed':False,'quality_qualified':False,'preflight':{'memory':resources(),'load':os.getloadavg(),'scratch_free':shutil.disk_usage(scratch).free,'output_free':shutil.disk_usage(out).free}}
 def save(phase):report['phase']=phase;(out/'physical-audit.json').write_text(json.dumps(report,indent=2)+'\n');print('phase',phase,flush=True)
 def stop():
  nonlocal active
  if active is not None and active.poll()is None:
   os.killpg(active.pid,signal.SIGTERM)
   try:active.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
  active=None
 def interrupted(signum,frame):stop();save('interrupted');raise SystemExit(128+signum)
 signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(cfg['total_timeout_seconds'])
 def command(label,args,timeout=1800):
  nonlocal active
  save(label)
  with(out/(label+'.log')).open('w')as log:
   active=subprocess.Popen(list(map(str,args)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=active.wait(timeout=timeout);require(code==0,label+' failed')
   finally:stop()
 def download(repository,revision,row):
  save('download-'+row['file']);path=scratch/row['file'];require(not path.exists(),'task-owned download already exists');count=0;h=hashlib.sha256()
  with urllib.request.urlopen('https://huggingface.co/'+repository+'/resolve/'+revision+'/'+row['file'],timeout=120)as src,path.open('xb')as f:
   while b:=src.read(8*1024**2):count+=len(b);require(count<=row['bytes'],'immutable download bound');h.update(b);f.write(b)
  require(count==row['bytes']and h.hexdigest()==row['sha256'],'whole immutable file identity');return path
 try:
  require(report['preflight']['memory']['MemAvailable']>=cfg['minimum_available_ram_bytes'],'available RAM before transfers');require(report['preflight']['scratch_free']>=cfg['minimum_scratch_free_bytes']and report['preflight']['output_free']>=cfg['minimum_output_free_bytes'],'disk bounds before transfers')
  p=head_perm();require(sorted(p)==list(range(4096)),'full V permutation bijection');inverse=[p.index(i)for i in range(4096)];require(all(inverse[p[i]]==i for i in range(4096)),'full V inverse');report['geometry']={'bijection_lanes':4096,'all_inverse_lanes_exact':True,'half_block_columns':128,'physical_quant_block':256,'permutation_preserves256_block_groups':all({p[i]//256 for i in range(j,j+256)}=={p[j]//256}for j in range(0,4096,256))};require(not report['geometry']['permutation_preserves256_block_groups'],'audit must retain physical out quant blocks')
  upstream=scratch/'upstream';command('upstream-init',['git','init',upstream]);command('upstream-remote',['git','-C',upstream,'remote','add','origin','https://github.com/ggml-org/llama.cpp']);command('upstream-fetch',['git','-C',upstream,'fetch','--depth','1','origin',cfg['upstream_source']]);command('upstream-checkout',['git','-C',upstream,'checkout','--detach','FETCH_HEAD']);require(subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()==cfg['upstream_source'],'independent dequant source')
  command('native-submodules',['git','-C',ROOT,'submodule','update','--init','--depth','1','--recursive']);build=scratch/'build';command('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DGGML_AVX2=ON','-DGGML_AVX512=OFF','-DGGML_CUDA=OFF','-DGGML_BLAS=OFF','-DCMAKE_C_FLAGS=-ffp-contract=off','-DCMAKE_CXX_FLAGS=-ffp-contract=off']);command('build',['cmake','--build',build,'--target','lighton4b-upstream-dequant-audit','-j','4']);binary=build/'lighton4b-upstream-dequant-audit';report['native_helper_sha256']=sha(binary);shutil.copy(build/'CMakeCache.txt',out/'CMakeCache.txt');command('cpu',['lscpu'])
  model=download(cfg['model_repository'],cfg['model_revision'],cfg['upstream_file']);native=download(cfg['native_repository'],cfg['native_revision'],cfg['native_file']);types={0:(1,4),8:(32,34),12:(256,144),14:(256,210)};up=parse(model,types);base=parse(native,types);require(len(up['tensors'])==426 and len(base['tensors'])==723,'complete file inventories');require(up['data_offset']==cfg['upstream_file']['header_bytes']and digest_span(model,0,up['data_offset'])==cfg['upstream_file']['header_sha256'],'previous physical header continuity');require(up['metadata']['general.name']['value'].lower()==cfg['original_revision_declared_by_upstream_general_name'],'declared checkpoint revision');report['declared_original_revision']=up['metadata']['general.name']['value'];report['original_checkpoint_tensor_content_identity_independently_established']=False
  text={k:v for k,v in base['tensors'].items()if k.startswith('l.')};mapped={physical_name(k):k for k in text};require(len(mapped)==426 and set(mapped)==set(up['tensors']),'all426 name mapping');mapping_identity={name:{'native_name':mapped[name],'upstream_shape':row['stored_shape'],'type':row['type']}for name,row in up['tensors'].items()};require(hashlib.sha256(json.dumps(mapping_identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()==cfg['predeclared_mapping_sha256'],'predeclared complete name/shape/type mapping');census={str(t):sum(x['type']==t for x in up['tensors'].values())for t in {x['type']for x in up['tensors'].values()}};require(census==cfg['expected_upstream_census'],'actual physical types');report['census']=census;report['tied_embedding_head']={'physical_embedding':'token_embd.weight','type':up['tensors']['token_embd.weight']['type'],'separate_output_tensor_present':'output.weight'in up['tensors'],'loader_rule':'missing output aliases token_embd'};require(report['tied_embedding_head']['type']==14 and not report['tied_embedding_head']['separate_output_tensor_present'],'actual tied Q6 head')
  quant={k:v for k,v in up['tensors'].items()if v['type']in(12,14)};plan=out/'native-dequant-plan.tsv';plan.write_text(''.join(f'{name}\t{row["type"]}\t{row["data_offset"]}\t{math.prod(row["stored_shape"])}\t{row["stored_shape"][0]}\n'for name,row in quant.items()));command('native-dequant',[binary,model,plan,out/'native-dequant.tsv'],2400)
  native_hashes={}
  for line in(out/'native-dequant.tsv').read_text().splitlines():
   name,typ,n,raw,f32,maxabs,squares=line.split('\t');native_hashes[name]={'type':int(typ),'elements':int(n),'packed_sha256':raw,'f32_sha256':f32,'max_abs':float(maxabs),'sum_squares':float(squares)}
  require(set(native_hashes)==set(quant),'full native dequant coverage');sys.path.insert(0,str(upstream/'gguf-py'));from gguf import GGMLQuantizationType;from gguf.quants import dequantize
  results={};small={};dotprobes={};save('independent-weight-operators')
  def values(path,row):
   require(row['type']==0 and row['bytes']<=131072,'small protected vector bound')
   with path.open('rb')as f:f.seek(row['data_offset']);raw=f.read(row['bytes'])
   require(len(raw)==row['bytes'],'small vector complete');return np.frombuffer(raw,dtype='<f4').copy()
  def bf16(x):
   bits=np.asarray(x,dtype='<f4').view('<u4');return(((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view('<f4'))
  for name,row in up['tensors'].items():
   nname=mapped[name];original=text[nname];require(math.prod(row['stored_shape'])==math.prod(original['stored_shape']),'shape element mapping')
   if row['type']in(12,14):
    count=math.prod(row['stored_shape']);block,width=types[row['type']];h=hashlib.sha256();rawhash=hashlib.sha256();maximum=0
    with model.open('rb')as f:
     f.seek(row['data_offset'])
     for begin in range(0,count,65536):
      n=min(65536,count-begin);data=f.read(n//block*width);require(len(data)==n//block*width,'independent chunk complete');rawhash.update(data);val=dequantize(np.frombuffer(data,dtype=np.uint8),GGMLQuantizationType(row['type'])).astype('<f4',copy=False);require(val.size==n and np.isfinite(val).all(),'bounded independent decode');h.update(val.tobytes());maximum=max(maximum,float(np.abs(val).max()))
    actual=native_hashes[name];require(rawhash.hexdigest()==actual['packed_sha256'],'physical packed-byte hash');entry={'native_name':nname,**actual,'independent_f32_sha256':h.hexdigest(),'native_independent_f32_bit_exact':h.hexdigest()==actual['f32_sha256'],'independent_max_abs':maximum,'layout':'retained physical columns; native activation bridge required'if nname.endswith('.linear_attn.out_proj.weight')else'inverse row mapping'if rows_for(nname,row['stored_shape'][1])!=list(range(row['stored_shape'][1]))else'identity'}
    rows=row['stored_shape'][1];order=rows_for(nname,rows);require(len(order)==rows and sorted(order)==list(range(rows)),'whole-row bijection');rowbytes=row['bytes']//rows
    if order!=list(range(rows)):
     canonical=hashlib.sha256();physical_rows=[]
     with model.open('rb')as f:
      for i in range(rows):f.seek(row['data_offset']+i*rowbytes);physical_rows.append(hashlib.sha256(f.read(rowbytes)).hexdigest())
      for i in order:f.seek(row['data_offset']+i*rowbytes);canonical.update(f.read(rowbytes))
     inverse_rows=[0]*rows
     for i,j in enumerate(order):inverse_rows[j]=i
     require(all(physical_rows[order[inverse_rows[i]]]==physical_rows[i]for i in range(rows)),'physical row roundtrip');entry.update(canonical_inverse_rows_packed_sha256=canonical.hexdigest(),whole_rows_permuted=rows,row_bytes=rowbytes,quant_blocks_unchanged=True)
    results[name]=entry
    if nname.endswith('.linear_attn.out_proj.weight'):
     x=np.array([((37*i)%257-128)/256 for i in range(4096)],dtype=np.float64);xu=np.empty_like(x);xu[p]=x;probes=[]
     with model.open('rb')as f:
      for index in cfg['fixed_out_projection_rows']:
       f.seek(row['data_offset']+index*rowbytes);data=f.read(rowbytes);w=dequantize(np.frombuffer(data,dtype=np.uint8),GGMLQuantizationType(row['type'])).reshape(-1).astype(np.float64);hf=w[p];a=math.fsum((w*xu).tolist());b=math.fsum((hf*x).tolist());wrong=math.fsum((w*x).tolist());require(a==b,'exact real-valued permutation control');probes.append({'row':index,'physical_row_sha256':hashlib.sha256(data).hexdigest(),'source_tiled_dot_fsum':a,'equivalent_grouped_dot_fsum':b,'unpermuted_negative_dot':wrong,'negative_abs_error':abs(wrong-a)})
     dotprobes[name]=probes
   else:
    a=values(model,row);b=values(native,original);require(np.isfinite(a).all()and np.isfinite(b).all(),'finite small original parameters');kind='identity';expected=b
    if nname.endswith('.linear_attn.conv1d.weight'):a=a.reshape(8192,4)[list(range(4096))+[4096+x for x in p]].reshape(-1);kind='V channel inverse rows'
    elif nname.endswith(('.linear_attn.A_log','.linear_attn.dt_bias')):a=a[head_perm(1)];kind='head inverse permutation'
    if nname.endswith('.linear_attn.A_log'):require(np.all(a<0),'negative pretransformed A');expected=-np.exp(b,dtype=np.float32);kind+=' and direct pretransformed A';inverse_value=np.log(-a,dtype=np.float32)
    elif nname.endswith('norm.weight')and not nname.endswith('.linear_attn.norm.weight'):expected=b+np.float32(1);kind='direct +1 RMS scale';inverse_value=a-np.float32(1)
    else:inverse_value=b
    small[name]={'native_name':nname,'transform':kind,'elements':a.size,'max_abs_error_vs_f32_transform':float(np.max(np.abs(a-expected))),'different_vs_f32_transform':int(np.count_nonzero(a.view('<u4')!=expected.view('<u4'))),'max_abs_error_vs_bf16_rounded_transform':float(np.max(np.abs(a-bf16(expected)))),'inverse_transform_different_from_original':int(np.count_nonzero(inverse_value.view('<u4')!=b.view('<u4'))),'inverse_transform_max_abs_error':float(np.max(np.abs(inverse_value-b))),'physical_sha256':digest_span(model,row['data_offset'],row['bytes'])}
  require(len(results)==249 and len(small)==177 and len(dotprobes)==24,'all physical operator classes covered');report.update(quant_matrices=results,small_parameters=small,out_projection_probes=dotprobes,all249_native_independent_dequant_exact=all(v['native_independent_f32_bit_exact']for v in results.values()),quant_inverse_row_tensors=sum('whole_rows_permuted'in v for v in results.values()),out_projection_rows_tested=sum(len(v)for v in dotprobes.values()),all_physical_out_blocks_retained=True,maximum_dequant_chunk_elements=65536,status='completed');(out/'physical-tensor-map.json').write_text(json.dumps({name:{'native_name':mapped[name],**row}for name,row in up['tensors'].items()},indent=2)+'\n')
 finally:stop();save(report.get('phase','failed'))
if __name__=='__main__':main()
