"""Tiny actual-C type/guard fixture. It is not a model or original reference."""
import hashlib,math,struct
from pathlib import Path
import gemma_q4_recipe as R

def write(path):
 names=['enc.embed_tokens.weight','enc.ple.per_layer_model_projection.weight','enc.layers.0.ple_block.per_layer_input_gate.weight',
        'enc.layers.0.self_attn.q_proj.weight','enc.layers.0.mlp.up_proj.weight','enc.embedding_projection.weight','head.field.0.weight','enc.norm.weight']
 rows=[];offset=0
 def text(s):b=s.encode();return struct.pack('<Q',len(b))+b
 for name in names:
  shape=[256]if name=='enc.norm.weight'else[256,8];n=math.prod(shape)
  data=b''.join(struct.pack('<f',((i%37)-18)/32)for i in range(n));rows.append((name,shape,data,offset));offset=(offset+len(data)+31)//32*32
 header=b'GGUF'+struct.pack('<IQQ',3,len(rows),1)+text('general.architecture')+struct.pack('<I',8)+text('gemma_decision')
 for name,shape,data,start in rows:header+=text(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)+struct.pack('<IQ',0,start)
 at=(len(header)+31)//32*32
 with Path(path).open('xb')as f:
  f.write(header);f.write(b'\0'*(at-len(header)))
  for name,shape,data,start in rows:f.write(b'\0'*(at+start-f.tell()));f.write(data)
 return names

def run(out,command):
 root=out/'raw/quantizer-fixture';root.mkdir();source=root/'input.gguf';names=write(source);before=R.parse(source);records=[]
 for recipe in ['broad-mixed','protected-input']:
  target=root/(recipe+'.gguf');args=[out/'build/crispembed-quantize',source,target,'q4_k']
  if recipe=='protected-input':
   for name in names:
    if name=='enc.embed_tokens.weight'or '.ple'in name:args+=['--tensor-type',name+'=f32']
  command(out,'synthetic-'+recipe,args,timeout=120)
  after=R.parse(target);expected={name:0 if name in ['enc.embedding_projection.weight','head.field.0.weight','enc.norm.weight']else 8 if name=='enc.embed_tokens.weight'else 12 for name in names}
  if recipe=='protected-input':expected.update({name:0 for name in names if name=='enc.embed_tokens.weight'or '.ple'in name})
  R.require(after['metadata']['general.file_type']['value']==14 and after['metadata']['general.quantization_version']['value']==2,'Synthetic standard Q4 metadata')
  for name,row in after['tensors'].items():
   R.require(row['type']==expected[name]and row['stored_shape']==before['tensors'][name]['stored_shape'],'Actual C policy/type/shape fixture failed')
   actual=R.finite_and_hash(target,row)
   if row['type']==0:R.require(actual==R.finite_and_hash(source,before['tensors'][name]),'Critical fixture F32 bytes changed')
  records.append(dict(recipe=recipe,types=expected,model_sha256=R.sha(target),passed=True))
 return dict(passed=True,actual_c_quantizer=True,model_free=True,controls=records)
