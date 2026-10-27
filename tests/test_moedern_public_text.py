#!/usr/bin/env python3
"""Bounded stdlib/fake binding and authored tiny GGUF controls, no CPP/model."""
from pathlib import Path
import ctypes as C
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
pack=module('moedern_draft_pack','tools/pack_moedern_text_f32.py')
guard=module('moedern_draft_guard','tools/moedern-public/public_contract.py')
def binding(lib):
 package=types.ModuleType('moedern_test_package');package.__path__=[]
 base=types.ModuleType('moedern_test_package._binding');base._load_library=Mock(return_value=lib);base.CrispEmbed=types.SimpleNamespace(resolve_model=Mock(return_value='owned.gguf'));base.set_offline=Mock()
 with patch.dict(sys.modules,{package.__name__:package,base.__name__:base}):value=module('moedern_test_package._moedern_text','python/crispembed/_moedern_text.py')
 return value,base

def tiny(path,tensors,metadata):
 records=b''.join(metadata);table=b'';payload=b''
 for name,values in tensors:
  payload+=b'\0'*(-len(payload)%32);offset=len(payload);payload+=struct.pack('<'+'f'*len(values),*values)
  table+=pack.string(name)+struct.pack('<IQIQ',1,len(values),0,offset)
 header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(metadata))+records+table;header+=b'\0'*(-len(header)%32);path.write_bytes(header+payload)

class Controls(unittest.TestCase):
 def test_legacy_capability_before_resolution(self):
  m,b=binding(types.SimpleNamespace());b._load_library.assert_not_called()
  with self.assertRaisesRegex(RuntimeError,'does not provide'):m.CrispMoEdern('x.gguf')
  b.CrispEmbed.resolve_model.assert_not_called()
 def test_binding_roles_copy_lifetime_score_and_close(self):
  lib=types.SimpleNamespace(**{'crispembed_moedern_'+n:Mock(return_value=v)for n,v in [('init',7),('free',None),('encode',None),('score',0),('error',b'actual rejected')]});m,b=binding(lib)
  ids=(C.c_int32*2)(2,30001);mask=(C.c_uint8*2)(1,1);vectors=(C.c_float*256)(*([.5]*256));r=m._Result(1,C.sizeof(m._Result),1,2,2,128,0,ids,mask,mask,mask,None,vectors,vectors);lib.crispembed_moedern_encode.return_value=C.pointer(r)
  with m.CrispMoEdern('owned.gguf',n_threads=0,offline=True,auto_download=False,lib_path='new.so')as ctx:
   result=ctx.encode('Ä','document');self.assertEqual(result['ids'],[2,30001]);self.assertEqual(result['expansion_positions'],[]);self.assertEqual(len(result['all_vectors']),256)
   lib.crispembed_moedern_encode.assert_called_with(7,'Ä'.encode(),1);ids[0]=999;vectors[0]=42;self.assertEqual(result['ids'][0],2);self.assertEqual(result['all_vectors'][0],.5)
   with self.assertRaisesRegex(ValueError,'Explicit'):ctx.encode('x','dense')
   lib.crispembed_moedern_encode.return_value=None
   with self.assertRaisesRegex(ValueError,'actual rejected'):ctx.encode('x','query')
   with self.assertRaisesRegex(ValueError,'actual rejected'):ctx.score('q','d')
   self.assertIsNone(lib.crispembed_moedern_free.restype)
   b.set_offline.assert_called_once_with(True,lib_path='new.so');b.CrispEmbed.resolve_model.assert_called_once_with('owned.gguf',auto_download=False,lib_path='new.so')
  ctx.close();lib.crispembed_moedern_free.assert_called_once_with(7)
  with self.assertRaisesRegex(RuntimeError,'closed'):ctx.encode('x','query')
 def test_text_bounds_unicode_and_constructor_controls(self):
  m,_=binding(types.SimpleNamespace())
  self.assertEqual(len(m.CrispMoEdern._text('Ä'*32768)),65536)
  for v in ['Ä'*32769,'x\0y',None,2,'\ud800']:
   with self.assertRaises((ValueError,UnicodeEncodeError)):m.CrispMoEdern._text(v)
  for n in [-1,3,True,1.5]:
   with self.assertRaises(ValueError):m.CrispMoEdern('x',n_threads=n)
 def test_tiny_byte_preserving_pack_and_f64_metadata(self):
  with tempfile.TemporaryDirectory()as folder:
   p=Path(folder);model=p/'m.gguf';tok=p/'t.gguf';out=p/'o.gguf'
   tiny(model,[('weight.a',[1.,-2.]),('weight.b',[3.])],[pack.scalar('general.architecture','modernbert_hash_moe_v10'),pack.scalar('general.license','apache-2.0')])
   raw=pack.string('scores')+struct.pack('<IIQd',9,12,1,.19999999999999998)
   tiny(tok,[],[pack.scalar('general.architecture','moedern_unigram_tokenizer'),pack.scalar('general.license','apache-2.0'),raw])
   result=pack._pack(pack.parse(model),pack.parse(tok),out,{'moedern.packed.format':'owned-fixture'})
   after=pack.parse(out);self.assertEqual(after['metadata']['scores']['raw'],raw);self.assertFalse(result['tensor_math_performed']);self.assertEqual(len(result['tensors']),2)
   self.assertEqual(result['tensors']['weight.a']['sha256'],pack.hashlib.sha256(struct.pack('<ff',1.,-2.)).hexdigest())
   with self.assertRaisesRegex(ValueError,'fresh'):pack._pack(pack.parse(model),pack.parse(tok),out,{})
 def test_production_host_barrier_and_malformed_fixture(self):
  with patch.dict('os.environ',{'GITHUB_ACTIONS':'false'}):
   with self.assertRaisesRegex(ValueError,'hosted'):pack.pack('missing','missing','missing','missing')
  with tempfile.TemporaryDirectory()as folder:
   p=Path(folder)/'bad.gguf';p.write_bytes(b'GGUF'+struct.pack('<IQQ',3,33,0))
   with self.assertRaisesRegex(ValueError,'count'):pack.parse(p)
 def test_old_engine_and_inference_math_body_unchanged(self):
  self.assertEqual((ROOT/'src/moedern_text_engine.cpp').read_bytes(),subprocess.check_output(['git','show',guard.BASE+':src/moedern_text_engine.cpp'],cwd=ROOT))
  before=subprocess.check_output(['git','show',guard.BASE+':src/moedern_text_frontend.cpp'],cwd=ROOT).decode();after=(ROOT/'src/moedern_text_frontend.cpp').read_text();marker='std::string text_frontend::normalize('
  self.assertEqual(before[before.index(marker):],after[after.index(marker):])
  for path in ['CMakeLists.txt','src/crispembed.h','examples/cli/main.cpp','examples/server/server.cpp','python/crispembed/__init__.py','crispembed/src/lib.rs','flutter/crispembed/lib/crispembed.dart']:
   self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show',guard.BASE+':'+path],cwd=ROOT))
 def test_source_declaration_and_honest_unfinished_status(self):
  cfg=json.loads(guard.PATH.read_text());self.assertFalse(cfg['compiled']);self.assertFalse(cfg['public_model_proof']);self.assertEqual(set(cfg['owned_sha256']),guard.NEW|guard.CHANGED)
  for name,value in cfg['owned_sha256'].items():self.assertEqual(guard.sha((ROOT/name).read_bytes()),value,name)
  self.assertEqual(set(cfg['protected_objects']),(guard.old_src()-guard.CHANGED)|guard.ROOTS)
  source=(ROOT/'src/moedern_text_api.cpp').read_text();self.assertIn('core_env::on("CRISPEMBED_MOEDERN_PUBLIC_TEXT")',source);self.assertIn('ctx->clear();',source)
  self.assertNotIn('moedern_text_testing',(ROOT/'src/moedern_text_api.h').read_text())
if __name__=='__main__':unittest.main()
