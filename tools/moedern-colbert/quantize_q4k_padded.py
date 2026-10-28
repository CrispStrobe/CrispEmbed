#!/usr/bin/env python3
"""Hosted-only real Q4_K with explicit zero-padded row storage, all3 MoEdern sizes.

No fallback/relabeling: delegates ONLY the numeric block encoding to the existing
C++ quantizer; independently verifies actual GGUF types and all F32 exclusions.
"""
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize_gguf import quantizable

ROOT = Path(__file__).resolve().parents[2]
BLOCK, TYPE_BYTES = 256, 144
PROFILES = {
 '1.5M': {'publication':'publication-1p5m.json','revision':'9d360b8b88ae11bea5784bf9012ff509fdd5c2da','tensors':32,'parameters':1484448,'q4k':20,'format':'crispembed-moedern-text-f32-v1'},
 '27M': {'publication':'publication-27m.json','revision':'d2b7b1feaaac1bf87510ca9525208c3daeabcc5a','tensors':32,'parameters':27124000,'q4k':20,'format':'crispembed-moedern-family-f32-v1'},
 '115M': {'publication':'publication-115m.json','revision':'66380d2ade3b5027bba83bdcdba052d0bb86dcdb','tensors':94,'parameters':114696576,'q4k':52,'format':'crispembed-moedern-family-f32-v1'},
}
def require(ok, why):
 if not ok: raise ValueError(why)
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  while block:=f.read(1024*1024): h.update(block)
 return h.hexdigest()
def plan(name, logical_shape):
 require(isinstance(name,str)and name and isinstance(logical_shape,(list,tuple))and
         1<=len(logical_shape)<=4 and all(isinstance(n,numbers.Integral)and not isinstance(n,bool)and 0<n<=30002 for n in logical_shape),'bounded semantic shape required')
 shape=list(map(int,logical_shape))
 logical=shape[0];stored=(logical+255)//256*256
 selected=quantizable(name) and len(shape)>=2 and stored//BLOCK*TYPE_BYTES<logical*4
 storage=[stored,*shape[1:]]if selected else shape[:]
 rows=math.prod(shape[1:]);before=math.prod(shape)*4;after=rows*(stored//BLOCK*TYPE_BYTES)if selected else before
 return {'name':name,'logical_shape':shape,'storage_shape':storage,'type':'Q4_K'if selected else'F32','original_bytes':before,'bytes':after,'bytes_saved':before-after}
def publication(variant):
 cfg=PROFILES[variant];row=json.loads((ROOT/'tests/results/moedern-colbert'/cfg['publication']).read_text())
 require(row['status']=='published and re-verified','reviewed F32 publication required');return row

def quantize(source, output, variant, quantizer):
 # Never install/import array frameworks or read models on the shared source host.
 kaggle=Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')
 require(os.environ.get('GITHUB_ACTIONS')=='true'or kaggle,'hosted GHA/Kaggle execution required')
 require(variant in PROFILES,'explicit supported variant required')
 source,output,quantizer=Path(source),Path(output),Path(quantizer);cfg=PROFILES[variant];pin=publication(variant)
 require(source.is_file()and not source.is_symlink()and source.stat().st_size==pin['bytes']and sha_file(source)==pin['sha256'],'immutable published F32 input differs')
 require(quantizer.is_file()and not output.exists(),'existing quantizer/fresh output required')
 import numpy as np
 import gguf
 reader=gguf.GGUFReader(source)
 def field(name):return reader.fields[name].contents()
 require(field('general.architecture')=='modernbert_hash_moe_v10'and field('modernbert_hash_moe_v10.source_revision')==cfg['revision']and field('moedern.packed.format')==cfg['format'],'source profile differs')
 require(len(reader.tensors)==cfg['tensors']and all(t.tensor_type==gguf.GGMLQuantizationType.F32 for t in reader.tensors),'exact F32 tensor inventory required')
 rows=[plan(t.name,list(t.shape))for t in reader.tensors];require(len({r['name']for r in rows})==len(rows),'duplicate input tensor')
 require(sum(math.prod(r['logical_shape'])for r in rows)==cfg['parameters']and sum(r['type']=='Q4_K'for r in rows)==cfg['q4k'],'semantic inventory/Q4_K coverage differs')
 output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_name(output.name+'.padded-f32.gguf');require(not temporary.exists(),'fresh padded scratch required')
 writer=gguf.GGUFWriter(str(temporary),arch='modernbert_hash_moe_v10',use_temp_file=True)
 for f in reader.fields.values():
  if f.name.startswith('GGUF.')or f.name in('general.architecture','moedern.packed.format','modernbert_hash_moe_v10.quant_recipe'):continue
  kind=f.types[0];sub=f.types[-1]if kind==gguf.GGUFValueType.ARRAY else None
  writer.add_key_value(f.name,f.contents(),kind,sub_type=sub)
 writer.add_string('moedern.packed.format','crispembed-moedern-q4k-padded-v1')
 writer.add_string('modernbert_hash_moe_v10.quant_recipe','q4_k_padded_f32_critical');writer.add_string('moedern.q4k.variant',variant);writer.add_string('moedern.q4k.format','q4_k-padded-columns-v1');writer.add_string('moedern.q4k.source_sha256',pin['sha256']);writer.add_string('moedern.q4k.source_format',cfg['format'])
 selected=[r for r in rows if r['type']=='Q4_K']
 writer.add_array('moedern.q4k.names',[r['name']for r in selected])
 for key,index in [('logical_columns','logical_shape'),('storage_columns','storage_shape')]:writer.add_key_value('moedern.q4k.'+key,[r[index][0]for r in selected],gguf.GGUFValueType.ARRAY,sub_type=gguf.GGUFValueType.INT32)
 for t,row in zip(reader.tensors,rows):
  a=np.asarray(t.data,dtype='<f4').reshape(tuple(reversed(row['logical_shape'])))
  row['original_f32_sha256']=hashlib.sha256(a.tobytes()).hexdigest()
  padding=row['storage_shape'][0]-row['logical_shape'][0]
  if padding:a=np.pad(a,[(0,0)]*(a.ndim-1)+[(0,padding)],mode='constant',constant_values=0)
  require(list(reversed(a.shape))==row['storage_shape']and np.isfinite(a).all(),'padded tensor shape/finiteness differs')
  row['padded_f32_sha256']=hashlib.sha256(a.tobytes()).hexdigest()
  writer.add_tensor(t.name,a,raw_dtype=gguf.GGMLQuantizationType.F32)
 writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
 # Existing generic fallback is never trusted: every selected row now256-aligned,
 # every nonselected exact name forced F32; readback below rejects all other types.
 args=[str(quantizer),str(temporary),str(output),'q4_k']
 for row in rows:
  if row['type']=='F32':args+=['--tensor-type',row['name']+'=f32']
 log=output.with_suffix(output.suffix+'.quantize.log')
 with log.open('w')as stream:subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=900,env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')})
 require('(fallback 'not in log.read_text(),'generic fallback occurred; cannot claim Q4_K')
 result=gguf.GGUFReader(output);require(len(result.tensors)==len(rows),'output tensor count differs')
 padded=gguf.GGUFReader(temporary)
 changed_metadata={'general.quantization_version','general.file_type'}
 expected_metadata={k for k in padded.fields if not k.startswith('GGUF.')}
 actual_metadata={k for k in result.fields if not k.startswith('GGUF.')}
 require(actual_metadata==expected_metadata|changed_metadata,'quantizer metadata names differ')
 for name in expected_metadata-changed_metadata:
  a,b=padded.fields[name],result.fields[name]
  require(a.types==b.types and np.array_equal(a.contents(),b.contents()),'tokenizer/profile metadata changed: '+name)
 # Pinned ggml.h GGML_FTYPE_MOSTLY_Q4_K=12; this is not llama.cpp's
 # separate mixed-Q4 file-type enum. Hosted C++ controls assert the same ABI.
 require(result.fields['general.file_type'].contents()==12,'actual file type is not Q4_K')
 actual={t.name:t for t in result.tensors};require(set(actual)=={r['name']for r in rows},'output names differ')
 for row in rows:
  t=actual[row['name']];expected=gguf.GGMLQuantizationType.Q4_K if row['type']=='Q4_K'else gguf.GGMLQuantizationType.F32
  require(t.tensor_type==expected and list(map(int,t.shape))==row['storage_shape']and t.data.nbytes==row['bytes'],'actual output type/shape/bytes differ: '+row['name'])
  row['stored_payload_sha256']=hashlib.sha256(t.data.tobytes()).hexdigest()
  if row['type']=='F32':require(row['stored_payload_sha256']==row['original_f32_sha256'],'critical F32 payload changed')
  else:
   decoded=gguf.quants.dequantize(t.data,t.tensor_type).reshape(tuple(reversed(row['storage_shape'])))
   require(np.isfinite(decoded).all(),'nonfinite Q4_K dequantized weight')
   original=next(x for x in reader.tensors if x.name==row['name']);a=np.asarray(original.data,dtype='<f4').reshape(tuple(reversed(row['logical_shape'])))
   row['max_absolute_weight_error']=float(np.max(np.abs(decoded[...,:row['logical_shape'][0]]-a)))
   row['padding_columns']=row['storage_shape'][0]-row['logical_shape'][0]
 require(output.stat().st_size<source.stat().st_size,'Q4_K package is not smaller than F32')
 return {'schema':'crispembed-moedern-q4k-padded-v1','variant':variant,'completed':True,'source':pin,'source_sha256':pin['sha256'],'output_sha256':sha_file(output),'output_bytes':output.stat().st_size,'q4k_tensors':len(selected),'f32_tensors':len(rows)-len(selected),'tensors':rows,'quantizer_sha256':sha_file(quantizer),'quantizer_log_sha256':sha_file(log),'temporary_padded_f32_sha256':sha_file(temporary),'semantic_widths_unchanged':True,'quantized_publication_qualified':False}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--variant',choices=list(PROFILES),required=True);p.add_argument('--quantizer',required=True);p.add_argument('--report',required=True);a=p.parse_args()
 report=quantize(a.input,a.output,a.variant,a.quantizer);Path(a.report).write_text(json.dumps(report,indent=2)+'\n')
