"""Independent synthetic GGUF metadata/copy checks; no model or framework."""
import copy
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_q6_convert as m

def fixture(path):
 rows=[('l.layers.0.mlp.down_proj.weight',12,[256,2]),('l.layers.1.mlp.up_proj.weight',12,[256,2]),('l.embed_tokens.weight',8,[256,2]),('v.protected.weight',0,[2,2])]
 meta=[m.text('general.file_type')+struct.pack('<II',4,12),m.text('general.alignment')+struct.pack('<II',4,32),m.text('tokenizer.ggml.tokens')+struct.pack('<IIQ',9,8,3)+b''.join(m.text(x)for x in ('<','é','○')),m.text('test.float')+struct.pack('<If',6,-0.75)]
 desc=[];data=bytearray();plan=[]
 for name,kind,shape in rows:
  while len(data)%32:data.append(0)
  off=len(data);block,width=m.TYPES[kind];size=shape[0]*shape[1]//block*width;data.extend(bytes([kind])*size)
  desc.append(m.text(name)+struct.pack('<IQQIQ',2,*shape,kind,off));plan.append({'name':name,'source':name,'source_shape':list(reversed(shape)),'type':14 if name==rows[0][0]else kind})
 header=b'GGUF'+struct.pack('<IQQ',3,len(rows),len(meta))+b''.join(meta)+b''.join(desc);header+=b'\0'*((-len(header))%32);path.write_bytes(header+data)
 return {'all_tensors':plan,'q6_names_sha256':'a'*64}
class Controls(unittest.TestCase):
 def test_metadata_clone_unicode_arrays_and_payload_layout(self):
  with tempfile.TemporaryDirectory()as d:
   path=Path(d)/'base.gguf';p=fixture(path);base=m.parse(path);head,rows,size=m.header(base,path,p);out=Path(d)/'new.gguf';out.write_bytes(head+b'\0'*(size-len(head)))
   with path.open('rb')as src,out.open('r+b')as dst:
    for row in rows:
     if row['copied']:src.seek(row['old_offset']);dst.seek(row['data_offset']);dst.write(src.read(row['bytes']))
   new=m.parse(out,m.TYPES);self.assertEqual(new['metadata']['general.file_type']['value'],15)
   for k,v in base['metadata'].items():
    if k!='general.file_type':self.assertEqual(v['raw_sha256'],new['metadata'][k]['raw_sha256'])
   self.assertEqual(sum(r['copied']for r in rows),3);self.assertEqual(new['tensors'][rows[0]['name']]['type'],14)
   for row in rows:
    self.assertEqual(new['tensors'][row['name']]['data_offset'],row['data_offset'])
    if row['copied']:
     with path.open('rb')as src,out.open('rb')as dst:self.assertEqual(m.span(src,row['old_offset'],row['old_bytes']),m.span(dst,row['data_offset'],row['bytes']))
 def test_wrong_or_missing_base_tensors_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   path=Path(d)/'base.gguf';p=fixture(path);base=m.parse(path)
   for mutate in ('q8_selected','missing','shape'):
    bad=copy.deepcopy(base)
    if mutate=='missing':bad['tensors'].pop('l.embed_tokens.weight')
    elif mutate=='shape':bad['tensors']['l.embed_tokens.weight']['stored_shape']=[32,16]
    else:bad['tensors']['l.layers.0.mlp.down_proj.weight']['type']=8
    with self.assertRaises(ValueError):m.header(bad,path,p)
 def test_checkpoint_identity_closed(self):
  self.assertEqual(m.SOURCE_SHA,'6b0dab74a2dab02d5b10ca56322255024ed8dca35c02746b42e9f296aaeb4e67');self.assertNotEqual(m.SOURCE_SHA,m.BASE_SHA)
if __name__=='__main__':unittest.main()
