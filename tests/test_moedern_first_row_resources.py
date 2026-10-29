"""Resource and packed-initializer successor controls: no frameworks/compiler/model access."""
import ast
import copy
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/moedern-first-row'))
import resource_profile as budget
import common
OLD='024f3e7a58cc4ed8e219689d1bea387342dd9556'
class Controls(unittest.TestCase):
 def test_exact_observed_initial_host_now_admitted(self):
  r=budget.values(32672014336,20940525568,4)
  self.assertEqual(r['policy']['minimum_free_disk_bytes'],10*2**30)
  self.assertEqual(r['policy']['minimum_available_ram_bytes'],14*2**30)
  self.assertEqual(sum(r['policy']['disk_budget_gib'].values()),10)
 def test_ram_cpu_and_disk_gates_not_bypassed(self):
  for values in [(14*2**30-1,20*2**30,4),(32*2**30,10*2**30-1,4),(32*2**30,20*2**30,1)]:
   with self.assertRaises(ValueError):budget.values(*values)
 def test_exact_caps_and_reserve_reject_without_deleting_evidence(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);p=root/'evidence';p.write_bytes(b'keep')
   self.assertEqual(budget.live(root,free_disk=2*2**30)['output_bytes'],4)
   for free,size in [(2*2**30-1,4),(3*2**30,2*2**30+1)]:
    with self.assertRaises(ValueError):budget.live(root,free_disk=free,extent=dict(bytes=size,files=1))
    self.assertEqual(p.read_bytes(),b'keep')
 def test_inherited_file_limit_is_declared_not_actual_host_mutation(self):
  calls=[]
  fake=types.SimpleNamespace(RLIMIT_FSIZE=1,RLIM_INFINITY=-1,getrlimit=lambda _: (-1,-1),setrlimit=lambda *a:calls.append(a))
  with patch.object(budget,'resource',fake):self.assertEqual(budget.single_file_limit()['maximum_single_file_bytes'],2*2**30)
  self.assertEqual(calls,[(1,(2*2**30,2*2**30))])
 def test_original_compare_cmake_byte_exact(self):
  for name in ['tools/moedern-first-row/original.py','tools/moedern-first-row/compare.py',
       'tests/moedern-first-row/CMakeLists.txt']:
   self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show',OLD+':'+name],cwd=ROOT),name)
 def test_masks_mapping_and_anchoring_functions_unchanged(self):
  def functions(text):return {n.name:ast.get_source_segment(text,n)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
  before=functions(subprocess.check_output(['git','show',OLD+':tools/moedern-first-row/common.py'],cwd=ROOT,text=True))
  after=functions((ROOT/'tools/moedern-first-row/common.py').read_text())
  for name in ['authenticated_input','actual_mapping','hosted','sha','atomic']:self.assertEqual(before[name],after[name],name)
 def test_profile_changes_only_explicit_initializer_and_source_fields(self):
  before=json.loads(subprocess.check_output(['git','show',OLD+':docs/moedern-first-row-stages-v1.json'],cwd=ROOT,text=True))
  after=common.declaration()
  changed={k for k in set(before)|set(after)if before.get(k)!=after.get(k)}
  self.assertEqual(changed,{'base','owned_sha256','packed_initializer','attempt_history'})
  self.assertEqual(after['limits'],before['limits']);self.assertEqual(after['gates'],before['gates'])
 def test_all_three_actual_admission_points_share_same_policy(self):
  for name in ['tools/moedern-first-row/prepare.py','tools/ci-heavy/moedern_first_row_v1.py','tools/moedern-first-row/kernel.py']:
   text=(ROOT/name).read_text();self.assertIn('budget.admit(',text);self.assertNotIn('>=24*2**30',text)
  self.assertEqual(common.declaration()['resource_profile'],budget.POLICY)
 def test_only_probe_constructor_dispatch_changes(self):
  name='tests/moedern-first-row/probe.cpp'
  before=subprocess.check_output(['git','show',OLD+':'+name],cwd=ROOT,text=True)
  after=(ROOT/name).read_text()
  self.assertIn('std::is_same_v<decltype(&mt::init), mt::context * (*)(const char *, const char *, int, std::string &)>',after)
  self.assertIn('std::is_same_v<decltype(&mt::init_packed), mt::context * (*)(const char *, int, std::string &)>',after)
  predicate='model && tokenizer && *model && *tokenizer && std::strcmp(model, tokenizer) == 0'
  self.assertIn('const bool packed = '+predicate+';',after)
  self.assertIn('return packed ? mt::init_packed(model, threads, error) : mt::init(model, tokenizer, threads, error);',after)
  after=after.replace('static_assert(std::is_same_v<decltype(&mt::init), mt::context * (*)(const char *, const char *, int, std::string &)>);\n','')
  after=after.replace('static_assert(std::is_same_v<decltype(&mt::init_packed), mt::context * (*)(const char *, int, std::string &)>);\n','')
  begin=after.index('size_t packed_inits =');end=after.index('void save_counts()',begin)
  after=after[:begin]+after[end:]
  after=after.replace('init_context("must-not-open-packed", "must-not-open-packed", 2, error)',
                      'mt::init("must-not-open-model", "must-not-open-tokenizer", 2, error)')
  after=after.replace('init_context(argv[1], argv[2],','mt::init(argv[1], argv[2],')
  # Only formatter line wrapping changes outside those exact constructor substitutions.
  self.assertEqual(re.sub(r'\s+','',before),re.sub(r'\s+','',after))
 def test_actual_packed_constructor_gate_precedes_file_access(self):
  native=(ROOT/'src/moedern_text.cpp').read_text();frontend=(ROOT/'src/moedern_text_frontend.cpp').read_text()
  packed=native.split('context * init_packed(',1)[1].split('void free(',1)[0]
  self.assertLess(packed.index('gates();'),packed.index('pending->tokenizer.load_packed('))
  self.assertLess(packed.index('"MoEdern packed gate is OFF"'),packed.index('pending->tokenizer.load_packed('))
  self.assertIn('pending->tokenizer.load(tokenizer, error)',native.split('context * init_packed(',1)[0])
  self.assertIn('gguf_get_n_tensors(meta.get()) == 0, "metadata-only tokenizer component required"',frontend)
  self.assertIn('(packed ? "modernbert_hash_moe_v10" : "moedern_unigram_tokenizer")',frontend)
  self.assertIn('return load_file(path, false, error)',frontend)
  self.assertIn('return load_file(path, true, error)',frontend)
 def test_actual_caller_argv_and_route_receipts(self):
  text=(ROOT/'tools/ci-heavy/moedern_first_row_v1.py').read_text();tree=ast.parse(text)
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('probe_argv','packed_route')]
  scope=dict(require=common.require,json=json)
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'actual-owned-caller','exec'),scope)
  for name in ('first-row-plain-probe','first-row-observed-probe'):
   actual=scope['probe_argv'](name,'authenticated-packed.gguf','input.tsv','output')
   self.assertEqual(actual,[name,'authenticated-packed.gguf','authenticated-packed.gguf','input.tsv','output'])
  with self.assertRaises(ValueError):scope['probe_argv']('probe','','input','output')
  for name,mode in [('first-row-plain-probe','plain'),('first-row-observed-probe','observed')]:
   self.assertIn("probe_argv(tests/'"+name+"',model,text_packet,raw/'"+mode+"')",text)
   self.assertIn("packed_route(raw/'"+mode+"')",text)
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp)
   for counts,okay in [(dict(packed_calls=5,component_calls=0),True),(dict(packed_calls=0,component_calls=5),False),(dict(packed_calls=4,component_calls=0),False)]:
    (root/'init-route.json').write_text(json.dumps(counts))
    if okay:self.assertEqual(scope['packed_route'](root),counts)
    else:
     with self.assertRaises(ValueError):scope['packed_route'](root)
 def test_previous_one_call_failure_and_retry_budgets(self):
  c=common.declaration();r=json.loads((ROOT/c['packed_initializer']['failure_receipt']).read_text())
  self.assertEqual(r['source'],OLD);self.assertEqual(r['original']['targeted_calls'],1)
  self.assertEqual(r['native']['counts']['full_computations'],0);self.assertFalse(r['native']['observers_installed'])
  self.assertEqual(r['native']['counts']['init_rejections'],3)
  self.assertEqual(r['small_member_count'],48);self.assertEqual(sum(p['bytes']for p in r['small_members'].values()),347938)
  self.assertTrue(all(p['max_absolute']==0 and p['cpp_ref']['passed']for p in r['original']['anchor'].values()))
  self.assertEqual(c['attempt_history']['aggregate_targeted_original_max'],1+c['limits']['original_targeted_forward_max'])
if __name__=='__main__':unittest.main()
