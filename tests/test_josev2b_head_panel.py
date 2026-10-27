#!/usr/bin/env python3
"""Source/stdlib controls only; no compile, array imports or model payloads."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('head_panel',ROOT/'tools/josev2b_head_panel.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
class Controls(unittest.TestCase):
    def test_frozen_native_source_and_public_reference(self):
        for name in p.FROZEN_FILES:self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show',p.FROZEN_HEAD+':'+name],cwd=ROOT))
        report=json.loads(p.artifact.PUBLIC.read_text());self.assertEqual(hashlib.sha256(p.artifact.PUBLIC.read_bytes()).hexdigest(),p.RECEIPT)
        self.assertEqual(report['git_commit'],p.SOURCE);self.assertEqual(len(report['arrays']),251)
        self.assertEqual(sum(r['stage'].startswith(('head/','logits/','probabilities/')) for r in report['arrays']),123)
    def test_pinned_function_rejects_module_drift_before_exec(self):
        with patch.object(p,'_compile_answer',side_effect=AssertionError('must not execute changed source')):
            with self.assertRaisesRegex(ValueError,'exact pinned'):p.official_answer(b'def systemone_answer(q,p): return {}')
    def test_ast_extract_executes_function_without_top_level_import(self):
        tree=ast.parse('import nonexistent_framework\ndef systemone_answer(question, probabilities):\n return {"id": question["id"], "p": probabilities["x"]}\n')
        answer=p._compile_answer(tree.body[1]);self.assertEqual(answer({'id':'owned'},{'x':.25}),{'id':'owned','p':.25})
    def test_local_and_cpu_only_gpu_task_fail_before_payload(self):
        with patch.dict(p.os.environ,{},clear=True),patch.object(p.Path,'is_dir',return_value=False),patch.object(p.subprocess,'check_output',side_effect=AssertionError('must not query GPU')):
            with self.assertRaises(ValueError):p.hosted(['cpu'])
            with self.assertRaises(ValueError):p.hosted(['CUDA0'])
        with patch.dict(p.os.environ,{'GITHUB_ACTIONS':'true'},clear=True):self.assertEqual(p.hosted(['cpu'])['service'],'GitHub Actions')
    def test_evidence_member_bytes_hash_and_symlink_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'small';path.write_bytes(b'owned');declaration=dict(bytes=5,sha256=hashlib.sha256(b'owned').hexdigest());p.member(path,declaration)
            for altered in (dict(declaration,bytes=4),dict(declaration,sha256='0'*64)):
                with self.assertRaises(ValueError):p.member(path,altered)
            other=Path(tmp)/'link';other.symlink_to(path)
            with self.assertRaises(ValueError):p.member(other,declaration)
    def test_actual_ref_control_recipe_collects_positive_numeric_and_structural(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen=[]
            def execute(args,log,env,timeout):
                case=Path(args[-1]).parent.name;seen.append(case);log.write_text('synthetic stdlib process control\n')
                if case=='wrong-shape':result={'status':'structural_rejection'};rc=2
                else:
                    bad=case=='bad-row';result=dict(status='numeric_rejection' if bad else 'passed',stages=[dict(found=True,elements=4,ggml_shape=[2,2],max_absolute=.0002 if bad else 0,min_row_cosine=-1 if bad else 1,passed=not bad)]);rc=int(bad)
                    self.assertEqual(env['CRISPEMBED_DIFF_COS_THRESHOLD'],'0.0001')
                Path(args[-1]).write_text(json.dumps(result));return rc
            with patch.object(p,'execute',side_effect=execute):proof=p.ref_controls(Path('never-executed'),Path(tmp),{})
            self.assertEqual(seen,['positive','bad-row','wrong-shape']);self.assertEqual([r['exit_code'] for r in proof],[0,1,2])
    def test_standalone_cmake_keeps_cuda_optional_and_observer_separate(self):
        text=(ROOT/'tests/josev-head/CMakeLists.txt').read_text()
        self.assertIn('option(GGML_CUDA',text);self.assertNotIn('set(GGML_CUDA OFF CACHE',text)
        self.assertIn('target_compile_definitions(josev-head-observed PRIVATE CRISPEMBED_JOSEV_HEAD_TESTING)',text)
        self.assertNotIn('target_compile_definitions(josev-head-plain',text);self.assertIn('add_executable(josev-head-diff diff.cpp)',text)
    def test_fixed_ref_budget_and_numeric_loop_source(self):
        cpp=(ROOT/'tests/josev-head/diff.cpp').read_text();self.assertIn('original.compare(name, b.first, b.second, 0)',cpp)
        self.assertIn('double(r.max_abs) <= 0.002 && double(r.cos_min) >= 0.99999',cpp);self.assertNotIn('r.is_pass(',cpp)
        source=(ROOT/'tools/josev2b_head_panel.py').read_text()
        self.assertIn('for backend in backends:',source);self.assertIn("for case,record in zip(report['cases'],freeze['records']):",source)
        self.assertIn('adapter.collect(rc,',source);self.assertNotIn('check=True,timeout=120',source)
        self.assertIn("if not row['numeric_passed']:result['numeric_failures'].append",source)
if __name__=='__main__':unittest.main()
