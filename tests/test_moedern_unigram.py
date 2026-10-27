#!/usr/bin/env python3
"""Small stdlib controls only; no Rust/tokenizer/CPP/model execution locally."""
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/moedern-unigram'))
import contract as c
import original as oracle

def driver():
 spec=importlib.util.spec_from_file_location('moedern_unigram_driver',ROOT/'tools/ci-heavy/moedern_unigram.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class Controls(unittest.TestCase):
 def test_metadata_roundtrip_f64_not_f32(self):
  number=.19999999999999998
  values={'s':'é','empty_string_array':[''],'ids':[0,30000],'scores':('f64',[0.1,number])}
  binary=c.gguf(values);self.assertEqual(struct.unpack_from('<IQQ',binary,4),(3,0,4));self.assertEqual(c.read_gguf(binary),values)
  self.assertNotEqual(number,struct.unpack('<f',struct.pack('<f',number))[0])
  self.assertEqual(struct.pack('<d',c.read_gguf(binary)['scores'][1][1]),struct.pack('<d',number))
  for bad in [b'BAD!'+binary[4:],binary[:-1]+b'x',binary[:12]]:
   with self.assertRaises((ValueError,struct.error)):c.read_gguf(bad)
 def test_no_weight_payload_allowlist(self):
  self.assertEqual(set(c.PAYLOADS),{'tokenizer.json','tokenizer_config.json','config_sentence_transformers.json','sentence_bert_config.json','config.json','modeling_modernbert_hash_moe_v10.py','LICENSE'})
  self.assertEqual(c.PAYLOADS['tokenizer.json'],{'bytes':1956542,'sha256':c.TOKENIZER})
  self.assertFalse(any('safetensors'in p or p.endswith('.gguf') for p in c.PAYLOADS))
 def test_original_frozen_ten_case_scope(self):
  cfg=json.loads(c.PATH.read_text());rows=cfg['primary_cases'];self.assertEqual(len(rows),10)
  self.assertEqual([r['id']for r in rows],['q-de','d-de','q-empty','d-punctuation','q-markers','d-unicode','q-truncate','d-truncate','d-one','d-space'])
  self.assertEqual([len(r['token_ids'])for r in rows],[32,28,32,23,32,21,32,512,4,3])
  for row in rows:
   self.assertEqual(row['effective_attention_mask'],[1]*len(row['token_ids'])if row['role']=='query'else row['supplied_attention_mask'])
 def test_registration_ast_is_source_only(self):
  tree=ast.parse((ROOT/'tools/moedern-unigram/original.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='register')
  self.assertTrue(any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='parse' for n in ast.walk(fn)))
  source=(ROOT/'tools/moedern-unigram/original.py').read_text();self.assertIn("n.name=='_register_prefix_tokens'",source);self.assertNotIn('method.decorator_list=[]',source)
  self.assertNotIn('tokenizers',sys.modules);self.assertNotIn('torch',sys.modules)
 def test_replacement_order_and_space_composition_control(self):
  mapping={ord('＇'):"'",ord('ﬁ'):'fi',ord('Ａ'):'a',0x301:''}
  self.assertEqual(oracle.mapped("''",mapping),'"')
  self.assertEqual(oracle.mapped('＇＇',mapping),"''")
  self.assertEqual(oracle.mapped('Ａ  ﬁe\u0301',mapping),'a fie')
  self.assertEqual(oracle.mapped('\t  Ａ\n',mapping),'\t a\n')
 def test_owned_context_combining_and_overlaps_declared(self):
  fixture=json.loads((ROOT/'tests/fixtures/moedern-unigram-v1.json').read_text());self.assertEqual(len(fixture['cases']),24)
  self.assertEqual(len(fixture['normalization_sequences']),16);self.assertEqual(fixture['composition_random_cases'],1024)
  self.assertIn('a\u0315\u0300',fixture['normalization_sequences']);self.assertIn('a\u0300\u0315',fixture['normalization_sequences'])
  self.assertTrue(any('[MASK]'in r['text']and '[Q]'in r['text']for r in fixture['cases']))
  self.assertTrue(any('▁'in r['text']for r in fixture['cases']));self.assertEqual(len(fixture['negative_inputs']),4)
 def test_strict_f64_viterbi_and_no_generic_normalizer(self):
  source=(ROOT/'src/moedern_text_frontend.cpp').read_text()
  self.assertIn('GGUF_TYPE_FLOAT64',source);self.assertIn('double score',source);self.assertIn('score > best[end].score',source);self.assertIn('minimum - 10.0',source)
  self.assertNotIn('SentencePieceTokenizer',source);self.assertNotIn('core_spm::normalize',source);self.assertNotIn('core_bert::',source)
  self.assertIn('reverse.back() != 1',source);self.assertIn('static_cast<unsigned char>(text[end++])',source)
  self.assertIn('core_env::on("CRISPEMBED_MOEDERN_TEXT_FRONTEND")',source);self.assertEqual(source.count('core_env::on('),1)
  self.assertIn('gguf_get_n_tensors(meta.get()) == 0',source)
 def test_original_first_boundaries_and_no_packages_except_rust(self):
  workflow=(ROOT/'.github/workflows/moedern-unigram.yml').read_text()
  self.assertLess(workflow.index('--phase prepare'),workflow.index('pip --python'));self.assertLess(workflow.index('--phase original'),workflow.index('cmake -S'))
  self.assertIn('--no-deps --only-binary=:all: --require-hashes',workflow);self.assertIn('timeout-minutes: 25',workflow);self.assertIn('-j2',workflow)
  self.assertEqual(c.WHEEL_SHA,'369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67')
  self.assertIn('tokenizers-0.22.2',c.WHEEL)
  cmake=(ROOT/'tests/moedern-unigram/CMakeLists.txt').read_text();self.assertIn('set(GGML_CPU OFF',cmake);self.assertIn('PUBLIC ggml-base',cmake);self.assertNotIn('moedern_colbert.cpp',cmake)
 def test_dependency_budget_rejection_before_download(self):
  d=driver();pin={'run':1,'source':'x','artifact_id':2,'bytes':2000000,'sha256':'0'*64,'member':'x','member_sha256':'0'*64}
  run={'head_sha':'x','status':'completed','conclusion':'success'};artifact={'workflow_run':{'id':1,'head_sha':'x'},'expired':False,'size_in_bytes':2000000,'digest':'sha256:'+'0'*64}
  with mock.patch.object(d,'api',side_effect=[run,artifact]),mock.patch.object(d.subprocess,'run')as download:
   with self.assertRaisesRegex(ValueError,'budget'):d.dependency(pin,Path('unused'))
   download.assert_not_called()
 def test_only_new_metadata_registration(self):
  old=subprocess.check_output(['git','show',c.BASE+':.github/workflows/ovisocr2-metadata.yml'],cwd=ROOT,text=True)
  expected=old.replace('options: [','options: [moedern-unigram, ',1)+"\n  moedern-unigram:\n    if: github.event_name == 'workflow_dispatch' && inputs.phase == 'moedern-unigram'\n    uses: ./.github/workflows/moedern-unigram.yml\n"
  self.assertEqual((ROOT/'.github/workflows/ovisocr2-metadata.yml').read_text(),expected)
 def test_old_native_frozen_bytes_and_guards_untouched(self):
  for name in c.OLD|c.old_src():self.assertEqual(c.git('rev-parse','HEAD:'+name),c.git('rev-parse',c.BASE+':'+name),name)
  for name in ['src/tokenizer_spm.cpp','src/tokenizer_bpe.cpp','src/moedern_colbert.cpp','tools/moedern-native-v2/contract_v2.py']:
   self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show',c.BASE+':'+name],cwd=ROOT))
 def test_closed_sources_and_omission_negative(self):
  cfg=json.loads(c.PATH.read_text());self.assertEqual(set(cfg['source_sha256']),c.LANE);self.assertEqual(set(cfg['protected_objects']),c.OLD|c.old_src())
  for name,value in cfg['source_sha256'].items():self.assertEqual(c.sha((ROOT/name).read_bytes()),value,name)
  bad=deepcopy(cfg);del bad['protected_objects']['src/tokenizer_spm.cpp']
  with mock.patch.object(c.json,'loads',return_value=bad):
   with self.assertRaisesRegex(ValueError,'closure omitted'):c.source()
 def test_clean_exit_and_test_only_diagnostics(self):
  h=(ROOT/'src/moedern_text_frontend.h').read_text();s=(ROOT/'src/moedern_text_frontend.cpp').read_text();probe=(ROOT/'tests/moedern-unigram/probe.cpp').read_text()
  self.assertIn('#ifdef CRISPEMBED_MOEDERN_TEXT_TESTING',h);self.assertIn('#ifdef CRISPEMBED_MOEDERN_TEXT_TESTING',s);self.assertIn('core_util::clean_exit(rc)',probe)
  self.assertIn('segment_utf8_boundaries',probe);self.assertIn('test_unigram',probe);self.assertRegex(s,r'out = \{\};\s*error.clear\(\);')

if __name__=='__main__':unittest.main()
