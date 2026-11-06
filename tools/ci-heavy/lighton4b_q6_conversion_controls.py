#!/usr/bin/env python3
"""Small synthetic BF16-row conversion vs independent F32 input quantizer; no model."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_q6_convert as m

def main():
 m.require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable model-free scope required')
 out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);build=scratch/'q6-conversion-controls'
 report={'schema':'lighton4b-q6-conversion-controls/v1','status':'failed','model_payload_bytes':0,'model_inference_executed':False,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 def run(label,args):
  with(out/(label+'.log')).open('w')as f:subprocess.run([str(x)for x in args],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
 try:
  run('python-header-controls',[sys.executable,ROOT/'tests/test_lighton4b_q6_convert.py'])
  run('python-recipe-controls',[sys.executable,ROOT/'tests/test_lighton4b_q6_recipe.py'])
  run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
  run('build',['cmake','--build',build,'--target','lighton4b-q6-rows','crispembed-quantize','--parallel','2'])
  values=[];bf16=bytearray()
  for i in range(256*257):
   value=(i%513-256)/256;bits=struct.unpack('<I',struct.pack('<f',value))[0]&0xffff0000;bf16.extend(struct.pack('<H',bits>>16));values.append(struct.unpack('<f',struct.pack('<I',bits))[0])
  source=scratch/'synthetic.bf16';source.write_bytes(b'P'*16+bf16+b'Q'*16);target=scratch/'synthetic.q6';n=len(values)//256*210;target.write_bytes(b'X'*64+b'\0'*n+b'Y'*32);receipt=out/'synthetic-rows.json';binary=build/'lighton4b-q6-rows'
  args=[binary,source,16,len(values),256,target,64,hashlib.sha256(bf16).hexdigest(),receipt,65536];run('bounded-row-controls',args)
  raw=target.read_bytes();m.require(raw[:64]==b'X'*64 and raw[-32:]==b'Y'*32,'bounded converter changed neighboring bytes');q=json.loads(receipt.read_text());m.require(q['max_chunk_elements']==65536 and q['bytes']==n and not q['requantized'],'row-boundary source receipt differs')
  # Independent source fixture: generic existing quantizer reads true F32 values,
  # not the framed BF16 stream used by the new bounded row converter.
  meta=[m.text('general.architecture')+struct.pack('<I',8)+m.text('synthetic'),m.text('general.file_type')+struct.pack('<II',4,0),m.text('general.alignment')+struct.pack('<II',4,32)]
  name='l.layers.0.mlp.down_proj.weight';head=b'GGUF'+struct.pack('<IQQ',3,1,len(meta))+b''.join(meta)+m.text(name)+struct.pack('<IQQIQ',2,256,257,0,0);head+=b'\0'*((-len(head))%32)
  original=scratch/'synthetic-f32.gguf';original.write_bytes(head+struct.pack('<'+'f'*len(values),*values));reference=scratch/'synthetic-q6-reference.gguf';run('independent-generic-quantizer',[build/'crispembed-quantize',original,reference,'q6_k','--standard-file-type'])
  parsed=m.parse(reference,m.TYPES);r=parsed['tensors'][name]
  with reference.open('rb')as f:expected=m.span(f,r['data_offset'],r['bytes'])
  m.require(expected==q['q6_sha256']==hashlib.sha256(raw[64:-32]).hexdigest(),'BF16 chunk conversion differs from independent F32 source quantization')
  m.require(m.independent_bf16(target,64,n)==q['dequant_bf16_sha256'],'independent NumPy/gguf Q6 BF16 decoder differs')
  bad=list(args);bad[7]='0'*64;bad[8]=out/'bad-sha-must-not-exist.json';m.require(subprocess.run([str(x)for x in bad],capture_output=True).returncode==2 and not bad[8].exists(),'wrong original BF16 hash admitted')
  report.update(status='passed',synthetic_elements=len(values),max_chunk_elements=q['max_chunk_elements'],two_chunks_required=True,independent_quantizer_exact=True,independent_q6_bf16_exact=True,wrong_source_sha_rejected=True,neighbor_bytes_unchanged=True,converter_binary_sha256=m.digest(binary))
 finally:(out/'q6-conversion-controls.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
