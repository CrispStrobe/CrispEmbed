#!/usr/bin/env python3
"""Source-only preservation controls; runtime candidate remains unverified."""
from pathlib import Path
import os
import subprocess
import unittest
ROOT=Path(__file__).resolve().parents[1]
GGML=Path(os.environ.get('GGML_SOURCE',ROOT/'ggml'))
BASE='890278a8342c620197c90e702e1188bcab94f510'
PRODUCTION_BASE='973b200edd5d1d7429edecff1bd3ff42f0d80ca7'
# Commit that declared the default-off candidate (wasm: prepare default-off per-backend Q8_0 rounding candidate).
DECLARATION='2283626fae5ab1c44dd0d11ea13eadb6f5c9ee8c'
BLOCK_START='#if defined(__EMSCRIPTEN__)\n    // Conditional encoder candidate:'
import hashlib,sys
sys.path.insert(0,str(ROOT/'tools'))
from frozen_guard_amendments import expected

def amended(path,frozen):return expected('wasm-q8-rounding-source',path,frozen)

def listing(rows):return hashlib.sha256('\n'.join(rows).encode()).hexdigest()

def original(path):
 return subprocess.check_output(['git','show',BASE+':'+path],cwd=GGML,text=True)

def function(source,name):
 start=source.index('void '+name+'(');brace=source.index('{',start);depth=1;end=brace+1
 while depth and end<len(source):
  if source[end]=='{':depth+=1
  elif source[end]=='}':depth-=1
  end+=1
 if depth:raise ValueError('Unbalanced function')
 return source[start:end]

class Preservation(unittest.TestCase):
 def test_legacy_and_q8_1_converter_bytes_preserved(self):
  path='src/ggml-cpu/arch/wasm/quants.c';before=original(path);after=(GGML/path).read_text()
  for name in ('quantize_row_q8_0','quantize_row_q8_1'):
   self.assertEqual(function(before,name),function(after,name))
 def test_only_new_converter_changes_one_rounding_expression(self):
  path='src/ggml-cpu/arch/wasm/quants.c';before=function(original(path),'quantize_row_q8_0')
  after=function((GGML/path).read_text(),'quantize_row_q8_0_wasm_nearest')
  expected=before.replace('void quantize_row_q8_0(', 'void quantize_row_q8_0_wasm_nearest(').replace(
   'wasm_i32x4_trunc_sat_f32x4(v)', 'wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(v))')
  self.assertEqual(after,expected)
 def test_global_type_traits_are_byte_identical(self):
  path='src/ggml-cpu/ggml-cpu.c'
  def table(text):
   start=text.index('static const struct ggml_type_traits_cpu type_traits_cpu')
   return text[start:text.index('\n};',start)+3]
  self.assertEqual(table(original(path)),table((GGML/path).read_text()))
 def test_no_production_probe_or_wrapper_change(self):
  # Anchored to the commit that declared the candidate (DECLARATION): BASE..DECLARATION
  # changed only src/crispembed.cpp, and only by the gated candidate block. Diffing
  # against the working tree failed on every later unrelated src/ commit (the
  # runtime A/B guard keeps HEAD's encoder closure pinned through its replay).
  changed=subprocess.check_output(['git','diff',PRODUCTION_BASE,DECLARATION,'--name-only','--','src','wasm'],cwd=ROOT,text=True).splitlines()
  self.assertEqual(changed,['src/crispembed.cpp'])
  before=subprocess.check_output(['git','show',PRODUCTION_BASE+':src/crispembed.cpp'],cwd=ROOT,text=True)
  declared=subprocess.check_output(['git','show',DECLARATION+':src/crispembed.cpp'],cwd=ROOT,text=True)
  start=declared.index(BLOCK_START)
  end=declared.index('#endif\n\n',start)+len('#endif\n\n')
  addition=declared[start:end]
  self.assertIn('core_env::on("CRISPEMBED_WASM_Q8_0_ROUND")',addition)
  self.assertEqual(declared[:start]+declared[end:],before)
  # HEAD still carries the declared candidate block byte for byte.
  after=(ROOT/'src/crispembed.cpp').read_text()
  head=after.index(BLOCK_START)
  self.assertEqual(after[head:after.index('#endif\n\n',head)+len('#endif\n\n')],addition)
if __name__=='__main__':unittest.main()
