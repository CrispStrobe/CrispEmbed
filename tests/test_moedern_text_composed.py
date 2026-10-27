#!/usr/bin/env python3
"""Source/stdlib only. Pending dependency must stop before API/model access."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/moedern-text'))
import contract as c

def load_driver():
 spec=importlib.util.spec_from_file_location('moedern_text_driver',ROOT/'tools/ci-heavy/moedern_text_composed.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def valid_pin():
 return {'reviewed':True,'completed_parity':True,'run':7,'source':'1'*40,
 'review':{'id':8,'bytes':100,'sha256':'2'*64,'name':'moedern-unigram-review-7'},
 'metadata':{'id':9,'bytes':200,'sha256':'3'*64,'name':'moedern-unigram-metadata-7'},
 'report_sha256':'4'*64,'original_report_sha256':'5'*64,'component_sha256':'6'*64,'component_bytes':300,
 'component_member':'moedern-unigram/tokenizer.gguf','gold_member':'moedern-unigram/original-gold.json'}

class Controls(unittest.TestCase):
 def test_pending_rejection_precedes_git_and_api(self):
  cfg=json.loads(c.PATH.read_text());cfg['unigram']['reviewed']=False
  with tempfile.TemporaryDirectory()as folder:
   path=Path(folder)/'pending.json';path.write_text(json.dumps(cfg))
   with mock.patch.object(c,'PATH',path),mock.patch.object(c,'git',side_effect=AssertionError('Git before pending barrier')),mock.patch.object(subprocess,'check_output',side_effect=AssertionError('network/process before pending barrier')):
    with self.assertRaisesRegex(ValueError,'pins pending'):c.source()
 def test_actual_cli_pending_negative_creates_evidence_without_api(self):
  d=load_driver()
  with tempfile.TemporaryDirectory()as folder:
   cfg=json.loads(c.PATH.read_text());cfg['unigram']['reviewed']=False;path=Path(folder)/'contract.json';path.write_text(json.dumps(cfg));out=Path(folder)/'out'
   args=['driver','--phase','preflight','--output',str(out),'--scratch',str(Path(folder)/'scratch')]
   with mock.patch.object(c,'PATH',path),mock.patch.object(sys,'argv',args),mock.patch.dict('os.environ',{'GITHUB_ACTIONS':'true','GITHUB_SHA':'x'}),mock.patch.object(d,'api')as api,mock.patch.object(c,'git',side_effect=AssertionError('unexpected Git access')):
    with self.assertRaisesRegex(ValueError,'pins pending'):d.main()
    api.assert_not_called()
   rejection=json.loads((out/'review/rejection-preflight.json').read_text());self.assertFalse(rejection['qualified']);self.assertFalse(rejection['primary_replaced']);self.assertFalse((Path(folder)/'scratch').exists())
 def test_no_half_filled_pin_or_member_guess(self):
  cfg={'unigram':valid_pin()};c.pending_guard(cfg)
  for key,value in [('reviewed',False),('completed_parity',False),('run',None),('source','latest'),('report_sha256',None),('component_bytes',0),('component_member','../tokenizer.gguf'),('gold_member','x//original-gold.json'),('gold_member',None)]:
   bad=deepcopy(cfg);bad['unigram'][key]=value
   with self.assertRaises(ValueError):c.pending_guard(bad)
  for key in ('id','bytes','sha256','name'):
   bad=deepcopy(cfg);bad['unigram']['review'][key]=None
   with self.assertRaises((ValueError,TypeError)):c.pending_guard(bad)
 def test_kernel_arithmetic_adaptation_exact(self):
  old=subprocess.check_output(['git','show',c.BASE+':src/moedern_colbert.cpp'],cwd=ROOT,text=True)
  actual=(ROOT/'src/moedern_text_engine.cpp').read_text();expected=c.adapt_engine(old)
  self.assertEqual(c.tokens(actual),c.tokens(expected))
  self.assertNotEqual(c.tokens(actual.replace('1e-5f','2e-5f',1)),c.tokens(expected))
  self.assertIn('#define emit(callback, label, values, ...)',actual)
  self.assertIn('#define emit_ids(...) ((void)0)',actual)
 def test_internal_public_header_accepts_raw_text_no_probes(self):
  header=(ROOT/'src/moedern_text.h').read_text()
  for forbidden in ('observer','std::function','encode_tokens','tensor_view','moedern_colbert.h','frontend.h'):self.assertNotIn(forbidden,header)
  self.assertIn('const std::string & text',header);self.assertIn('const std::string & query',header)
  source=(ROOT/'src/moedern_text.cpp').read_text();self.assertRegex(source,r'ctx->tokenizer.encode\(\s*text');self.assertIn('detail::encode_tokens(',source)
  self.assertIn('detail::select_rows(pending.all_vectors, pending.scoring_mask',source)
  self.assertNotIn('drop_cls',source);self.assertNotIn('mean_pool',source)
 def test_value_gates_output_clear_and_reviewed_matcher_fixes_preserved(self):
  source=(ROOT/'src/moedern_text.cpp').read_text();self.assertIn('core_env::on("CRISPEMBED_MOEDERN_TEXT_COMPOSED")',source);self.assertIn('core_env::on("CRISPEMBED_MOEDERN_COLBERT")',source)
  self.assertRegex(source,r'output = \{\};');self.assertIn('value = 0;',source)
  self.assertIn('added_token_state(x)',(ROOT/'tools/moedern-unigram/original.py').read_text())
  self.assertNotIn('json.loads(x.__getstate__())',(ROOT/'tools/moedern-unigram/original.py').read_text())
  self.assertIn('normalized_history=normalized_matchers(raw,effective)',(ROOT/'tools/moedern-unigram/original.py').read_text())
  self.assertIn('ids.push_back(normalized_ids_[selected])',(ROOT/'src/moedern_text_frontend.cpp').read_text())
  self.assertIn("['revision','flags','matcher-id','matcher-content','score-type','score-nonfinite']",(ROOT/'tools/ci-heavy/moedern_text_composed.py').read_text())
 def test_all_old_paths_and_sources_unchanged(self):
  for name in c.OLD|c.old_src():self.assertEqual(c.git('rev-parse','HEAD:'+name),c.git('rev-parse',c.BASE+':'+name),name)
  for name in ('src/moedern_colbert.cpp','src/moedern_colbert.h','src/moedern_text_frontend.cpp','src/moedern_text_frontend.h','tools/moedern-unigram/original.py'):
   self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show',c.BASE+':'+name],cwd=ROOT))
 def test_full_history_and_dependency_before_optional_work(self):
  workflow=(ROOT/'.github/workflows/moedern-text-composed.yml').read_text();self.assertIn('fetch-depth: 0',workflow)
  self.assertLess(workflow.index('--phase preflight'),workflow.index('--phase prepare'));self.assertLess(workflow.index('--phase prepare'),workflow.index('pip install'))
  self.assertIn('timeout-minutes: 60',workflow);self.assertNotIn('pip install torch',workflow)
  driver=(ROOT/'tools/ci-heavy/moedern_text_composed.py').read_text();self.assertLess(driver.index("verify_tokenizer_reports(cfg,native,original)"),driver.index("primary=cfg['primary']"))
  self.assertIn("pin['component_member']",driver);self.assertIn("pin['gold_member']",driver)
 def test_ref_source_and_fixed_primary_budgets(self):
  self.assertEqual((ROOT/'tests/moedern-text/diff.cpp').read_bytes(),(ROOT/'tests/moedern-native-v2/diff.cpp').read_bytes())
  self.assertEqual(c.GATES,{'max_absolute':.002,'min_row_cosine':.99999,'maxsim_max_absolute':.002})
  source=(ROOT/'tests/moedern-text/diff.cpp').read_text();self.assertIn('original.compare(name, b.first, b.second, 0)',source)
  self.assertIn('double(r.max_abs) <= 0.002',source);self.assertIn('double(r.cos_min) >= 0.99999',source)
 def test_probe_only_raw_text_inputs_and_full_panel(self):
  source=(ROOT/'tests/moedern-text/probe.cpp').read_text();self.assertIn('unhex(',source);self.assertIn('row.text',source);self.assertNotIn('MOEDERN_TOKENS_V1',source)
  self.assertIn('cases == 10',source);self.assertIn('pairs == 3',source);self.assertIn('any_failed ? 1 : 0',source)
  self.assertIn('core_util::clean_exit(rc)',source);self.assertIn('"rejection.json"',source)
  self.assertIn('core_util::clean_exit(controls_main(', (ROOT/'tests/moedern-text/controls.cpp').read_text())
 def test_target_specific_probe_isolation(self):
  cmake=(ROOT/'tests/moedern-text/CMakeLists.txt').read_text();self.assertNotIn('/src/moedern_colbert.cpp',cmake)
  self.assertIn('moedern-text-observed PRIVATE CRISPEMBED_MOEDERN_COMPOSED_TESTING',cmake)
  self.assertNotIn('moedern-text-plain PRIVATE CRISPEMBED_MOEDERN_COMPOSED_TESTING',cmake)
  self.assertIn('#error',(ROOT/'src/moedern_text_testing.h').read_text())
 def test_metadata_only_new_phase_and_job(self):
  old=subprocess.check_output(['git','show',c.BASE+':.github/workflows/ovisocr2-metadata.yml'],cwd=ROOT,text=True)
  expected=old.replace('options: [moedern-unigram, ','options: [moedern-text-composed, moedern-unigram, ',1)+"\n  moedern-text-composed:\n    if: github.event_name == 'workflow_dispatch' && inputs.phase == 'moedern-text-composed'\n    uses: ./.github/workflows/moedern-text-composed.yml\n"
  self.assertEqual((ROOT/'.github/workflows/ovisocr2-metadata.yml').read_text(),expected)
 def test_closed_config_source_hashes_and_old_refs(self):
  cfg=json.loads(c.PATH.read_text());self.assertEqual(cfg['base'],c.BASE);self.assertEqual(cfg['native_v2'],c.NATIVE_V2);self.assertEqual(cfg['gates'],c.GATES)
  self.assertEqual(set(cfg['source_sha256']),c.LANE);self.assertEqual(set(cfg['protected_objects']),c.OLD|c.old_src())
  for name,expected in cfg['source_sha256'].items():self.assertEqual(c.sha((ROOT/name).read_bytes()),expected,name)
  self.assertEqual(cfg['primary'],json.loads((ROOT/'docs/moedern-colbert-native-contract-v2.json').read_text())['primary'])
  self.assertNotIn('torch',sys.modules);self.assertNotIn('tokenizers',sys.modules)

if __name__=='__main__':unittest.main()
