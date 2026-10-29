"""Resource-only successor controls: no frameworks/compiler/model access."""
import ast
import copy
import json
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
OLD='e7ff72f1a2fe1290f5d7b879d41e3fd5b6da26b8'
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
 def test_original_compare_probe_cmake_byte_exact(self):
  for name in ['tools/moedern-first-row/original.py','tools/moedern-first-row/compare.py',
       'tests/moedern-first-row/probe.cpp','tests/moedern-first-row/CMakeLists.txt']:
   self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show',OLD+':'+name],cwd=ROOT),name)
 def test_masks_mapping_and_anchoring_functions_unchanged(self):
  def functions(text):return {n.name:ast.get_source_segment(text,n)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
  before=functions(subprocess.check_output(['git','show',OLD+':tools/moedern-first-row/common.py'],cwd=ROOT,text=True))
  after=functions((ROOT/'tools/moedern-first-row/common.py').read_text())
  for name in ['authenticated_input','actual_mapping','hosted','sha','atomic']:self.assertEqual(before[name],after[name],name)
 def test_profile_changes_only_resource_and_source_fields(self):
  before=json.loads(subprocess.check_output(['git','show',OLD+':docs/moedern-first-row-stages-v1.json'],cwd=ROOT,text=True))
  after=common.declaration()
  changed={k for k in set(before)|set(after)if before.get(k)!=after.get(k)}
  self.assertEqual(changed,{'base','protected_sha256','owned_sha256','resource_profile','resource_predecessor'})
  self.assertEqual(after['limits'],before['limits']);self.assertEqual(after['gates'],before['gates'])
 def test_all_three_actual_admission_points_share_same_policy(self):
  for name in ['tools/moedern-first-row/prepare.py','tools/ci-heavy/moedern_first_row_v1.py','tools/moedern-first-row/kernel.py']:
   text=(ROOT/name).read_text();self.assertIn('budget.admit(',text);self.assertNotIn('>=24*2**30',text)
  self.assertEqual(common.declaration()['resource_profile'],budget.POLICY)
if __name__=='__main__':unittest.main()
