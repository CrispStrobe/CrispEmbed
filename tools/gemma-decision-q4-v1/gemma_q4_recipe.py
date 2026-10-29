"""Strict complete-Gemma inventory and actual-C-quantizer mixed Q4 plans."""
import hashlib,math,re,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from gguf_file_type_canonicalize import parse,sha,span,require
F32_SHA='cfd0ab416c2ffafd0f42bf7b4562b2d610b89d15e43a1648097a919572cc9abe'
F32_BYTES=1102182464
RECIPES={'broad-mixed':{0:218,8:1,12:217},'protected-input':{0:268,12:168}}

def array(path,meta,name):
 m=meta[name];require(m['kind']==9 and m['value']['count']==24 and m['value']['element']in [4,5,11],'Exact24 source-supported integer layer metadata required')
 fmt={4:'I',5:'i',11:'q'}[m['value']['element']];n=24*struct.calcsize(fmt)
 with Path(path).open('rb')as f:f.seek(m['offset']+12);data=f.read(n)
 require(len(data)==n,'Truncated layer array');return list(struct.unpack('<24'+fmt,data))

def canonical_shape(shape):
 shape=list(shape)
 while len(shape)>1 and shape[-1]==1:shape.pop()
 return shape

def inventory(path,authenticated=True):
 if authenticated:require(Path(path).stat().st_size==F32_BYTES and sha(path)==F32_SHA,'Exact published full F32 required')
 p=parse(path);m=p['metadata'];rows=p['tensors']
 require(m['general.architecture']['value']=='gemma_decision','Complete typed Gemma required')
 for name,value in [('hidden_size',512),('embedding_dim',768),('n_layer',24),('n_head',4),('intermediate_size',2048),('vocab_size',262144),('sliding_window',512),('head_width',256)]:
  require(m['gemma_decision.'+name]['kind']==4 and m['gemma_decision.'+name]['value']==value,'Unsupported exact Gemma geometry: '+name)
 hd=array(path,m,'gemma_decision.layer_head_dim');kv=array(path,m,'gemma_decision.layer_kv_heads')
 expected={'enc.embed_tokens.weight':[512,262144],'enc.ple.per_layer_model_projection.weight':[512,12288],
 'enc.ple.per_layer_projection_norm.weight':[512],'enc.norm.weight':[512],'enc.embedding_projection.weight':[512,768]}
 for i in range(24):
  require(hd[i]in[256,512]and kv[i]>0 and 4%kv[i]==0,'Unsupported head metadata')
  root='enc.layers.'+str(i)+'.'
  for name in ['input_layernorm.weight','post_attention_layernorm.weight','pre_feedforward_layernorm.weight','post_feedforward_layernorm.weight','ple_block.post_per_layer_input_norm.weight']:expected[root+name]=[512]
  expected[root+'layer_scalar']=[1]
  for qk in ['q','k']:expected[root+'self_attn.'+qk+'_norm.weight']=[hd[i]]
  for key,shape in {'q_proj':[512,hd[i]*4],'k_proj':[512,hd[i]*kv[i]],'v_proj':[512,hd[i]*kv[i]],'o_proj':[hd[i]*4,512]}.items():expected[root+'self_attn.'+key+'.weight']=shape
  for key,shape in {'gate_proj':[512,2048],'up_proj':[512,2048],'down_proj':[2048,512]}.items():expected[root+'mlp.'+key+'.weight']=shape
  for key in ['per_layer_input_gate','per_layer_projection']:expected[root+'ple_block.'+key+'.weight']=[512,512]
 for key in ['state_proj','question_proj','option_proj']:expected['head.'+key+'.weight']=[768,256];expected['head.'+key+'.bias']=[256]
 for key in ['field.0','field.3','residual.0','residual.3']:
  cols={'field.0':768,'field.3':256,'residual.0':1024,'residual.3':256}[key];n=1 if key=='residual.3'else 256
  expected['head.'+key+'.weight']=[cols,n];expected['head.'+key+'.bias']=[n]
 for key in ['field_norm','option_norm']:
  for suffix in ['weight','bias']:expected['head.'+key+'.'+suffix]=[256]
 expected['head.type_emb.weight']=[256,3];expected['temperatures']=[3]
 for key in ['gate','log_joint_scale','log_prior_scale']:expected['head.'+key]=[1]
 require(len(expected)==436 and set(rows)==set(expected),'Exact436 tensor names required; no unreviewed extra/missing weights')
 for name,r in rows.items():require(r['type']==0 and canonical_shape(r['stored_shape'])==canonical_shape(expected[name]),'Exact F32 tensor/semantic shape required: '+name)
 return p

def plan(rows,recipe):
 require(recipe in RECIPES,'Predeclared recipe required');result={}
 for name,r in rows.items():
  eligible=name.startswith('enc.')and name.endswith('.weight')and r['storage_rank']==2 and name!='enc.embedding_projection.weight'and 'norm'not in name
  if recipe=='protected-input':eligible=bool(re.fullmatch(r'enc\.layers\.(?:[0-9]|1[0-9]|2[0-3])\.(?:self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.weight',name))
  kind=8 if eligible and name=='enc.embed_tokens.weight'else 12 if eligible else 0
  require(not kind or r['stored_shape'][0]%(32 if kind==8 else 256)==0,'Fallback forbidden: '+name)
  result[name]=kind
 require({k:sum(v==k for v in result.values())for k in RECIPES[recipe]}==RECIPES[recipe]and len(rows)==436,'Actual source role/type census differs')
 return result

def finite_and_hash(path,row):
 import numpy as np
 h=hashlib.sha256();block,width={0:(1,4),8:(32,34),12:(256,144)}[row['type']]
 with Path(path).open('rb')as f:
  f.seek(row['data_offset']);left=row['bytes']
  while left:
   n=min(left,(4*2**20//width)*width);data=f.read(n);require(len(data)==n,'Tensor span truncated');h.update(data);left-=n
   if row['type']==0:require(np.isfinite(np.frombuffer(data,dtype='<f4')).all(),'Nonfinite F32 payload')
   else:
    b=np.frombuffer(data,dtype='u1').reshape(-1,width);scales=b[:,:4 if row['type']==12 else 2].copy().view('<f2')
    require(np.isfinite(scales).all(),'Nonfinite actual quantization scales')
 return h.hexdigest()

def verify(source,output,recipe):
 original=inventory(source);result=parse(output);rules=plan(original['tensors'],recipe);m=result['metadata']
 require(m['general.file_type']['kind']==4 and m['general.file_type']['value']==14
         and m['general.quantization_version']['kind']==4 and m['general.quantization_version']['value']==2,'Standard14/QNTv2 required')
 allowed={'general.file_type','general.quantization_version'}
 require(set(m)==set(original['metadata'])|allowed,'Only standard precision metadata additions allowed')
 for name,value in original['metadata'].items():
  if name not in allowed:require(m[name]['raw_sha256']==value['raw_sha256'],'Unchanged model/tokenizer metadata required: '+name)
 require(set(result['tensors'])==set(rules),'Exact complete output inventory')
 records=[]
 for name,before in original['tensors'].items():
  after=result['tensors'][name];require(after['type']==rules[name]and canonical_shape(after['stored_shape'])==canonical_shape(before['stored_shape']),'Actual type/semantic shape/fallback differs: '+name)
  if before['stored_shape']!=after['stored_shape']:require(name=='head.residual.3.weight'and before['stored_shape']==[256,1]and after['stored_shape']==[256],'Undeclared physical rank change')
  source_sha=finite_and_hash(source,before);out_sha=finite_and_hash(output,after)
  if rules[name]==0:require(source_sha==out_sha and before['bytes']==after['bytes'],'Critical F32 payload changed: '+name)
  records.append(dict(name=name,type=after['type'],source_type=0,source_shape=before['stored_shape'],output_shape=after['stored_shape'],semantic_shape=canonical_shape(before['stored_shape']),bytes=after['bytes'],source_sha256=source_sha,sha256=out_sha,critical_f32_byte_exact=source_sha==out_sha if rules[name]==0 else None))
 return dict(schema='crispembed-gemma-decision-q4-conversion-v1',recipe=recipe,model_sha256=sha(output),model_bytes=Path(output).stat().st_size,
             source_sha256=F32_SHA,source_bytes=F32_BYTES,types=RECIPES[recipe],tensors=records,selected=False,publication_qualified=False)
