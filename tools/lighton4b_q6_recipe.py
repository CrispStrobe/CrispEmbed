#!/usr/bin/env python3
"""Model-free exact Q4_K_M-style recipe; build only from original BF16/F16 tensors."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RECIPE='lighton4b-q4km-q6-q8head-v1'
UPSTREAM='18ab62a8ec5144ae5d2f65e78374ba69652d75dd'
HEADER_SHA='a1e36638b195f2d5943e000fcb069b275cb22ccc28b1b6184ddee7bdea7ef081'
def use_more_bits(i):return 0<=i<32 and(i<4 or i>=28 or(i-4)%3==2)
def plan():
 raw=(ROOT/'tests/fixtures/lightonocr3-4b-source/model.safetensors.header.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=HEADER_SHA:raise ValueError('original723 tensor header differs')
 original=json.loads(raw);rows=[];counts={0:0,8:0,12:0,14:0};extra=0
 for source,r in sorted(original.items()):
  if source=='__metadata__':continue
  name=('l.'+source.removeprefix('model.language_model.')if source.startswith('model.language_model.')else'v.'+source.removeprefix('model.visual.'))
  shape=r['shape'];typ=0
  if name=='l.embed_tokens.weight':typ=8
  elif name.startswith('l.')and len(shape)==2:
   typ=12;parts=name.split('.');layer=int(parts[2]);value=name.endswith('.linear_attn.in_proj_qkv.weight')or name.endswith('.self_attn.v_proj.weight')
   if use_more_bits(layer)and(value or name.endswith('.mlp.down_proj.weight')):typ=14
  if typ==14:extra+=shape[0]*shape[1]//256*(210-144)
  counts[typ]+=1;rows.append({'name':name,'source':source,'source_shape':shape,'type':typ})
 if counts!={0:474,8:1,12:216,14:32}:raise ValueError('exact protected/Q8/Q4/Q6 inventory differs')
 names=sorted(r['name']for r in rows if r['type']==14)
 return {'schema':'lighton4b-selective-q6-recipe/v1','recipe':RECIPE,'upstream':{'repository':'ggml-org/llama.cpp','revision':UPSTREAM,'policy':'src/llama-quant.cpp category_is_attn_v/use_more_bits; src/llama-model-loader.h natural layer order','exception':'tied output/embedding explicitlyQ8_0 instead of defaultQ6_K'},'source_header_sha256':HEADER_SHA,'source_policy':'original BF16 checkpoint or authenticated unquantized F16 baseline; quantized Q8/Q4 inputs prohibited','general_file_type':15,'metadata':{'qwen35_ocr.mixed_quant_recipe':RECIPE},'q6_names':names,'q6_names_sha256':hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest(),'types':{str(k):v for k,v in counts.items()},'extra_payload_bytes_vs_plain_q4':extra,'all_tensors':rows,'quality_qualified':False,'held_out_selection':False}
def validate_source_inventory(tensors,p):
 expected={r['name']:r for r in p['all_tensors']}
 if set(tensors)!=set(expected):raise ValueError('unquantized source full inventory differs')
 for name,r in tensors.items():
  wanted=expected[name];shape=wanted['source_shape'];shape=([shape[0],shape[1]*shape[2]*shape[3]]if name=='v.patch_embed.proj.weight'else shape)
  if r['stored_shape']!=list(reversed(shape))or r['type']not in(0,1)or(wanted['type']==0 and r['type']!=0):raise ValueError('source must retain original F32 protection and unquantized matrices: '+name)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.write_text(json.dumps(plan(),indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
