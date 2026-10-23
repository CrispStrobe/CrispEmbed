"""Pure controls; never import torch or load processor/model payloads."""
import ast
import importlib.util
from pathlib import Path
import unittest

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ovis_metadata',root/'tools/ci-heavy/ovisocr2_metadata.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class MetadataControls(unittest.TestCase):
    def test_tied_head_policy_and_unknown_shape_missing_mtp_rejected(self):
        expected={'model.weight':[2,3],'lm_head.weight':[4,3]};actual={'model.weight':{'shape':[2,3]}}
        self.assertTrue(p.check_schema(expected,actual,True)['tied_head_absent'])
        for e,a,tied in [(expected,actual,False),(expected,{'model.weight':{'shape':[2,4]}},True),
                         (expected,{**actual,'mtp.weight':{'shape':[2,3]}},True),
                         ({**expected,'missing.weight':[1]},actual,True)]:
            with self.assertRaises(ValueError):p.check_schema(e,a,tied)
    def test_original_literal_prompt_and_executable_or_ambiguous_rejected(self):
        self.assertEqual(p.original_prompt("```python\nprompt = 'a' + 'b'\n```"),'ab')
        for source in ["prompt = open('x').read()", "prompt = 'a'\nprompt = 'b'", 'pass']:
            with self.assertRaises(ValueError):p.original_prompt('```python\n'+source+'\n```')
    def test_pixel_override_must_change_small_default_grid(self):
        result=p.pixel_override_evidence([[1,14,20]],[[1,24,36]])
        self.assertTrue(result['min_override_effect_proved'])
        for before,after in [([[1,14,20]],[[1,14,20]]),([[1,28,28]],[[1,28,28]]),
                             ([[1,14,20]],[[1,12,16]]),([[1,14,20]],[[1,200,200]]),
                             ([[1,14,20]],[[2,24,36]]),([[1,14,20]],[[1,23,36]])]:
            with self.assertRaises(ValueError):p.pixel_override_evidence(before,after)
    def test_failure_summary_redacts_urls_credentials_and_host_paths(self):
        secret='hf_'+'x'*40
        detail=p.failure_detail(ValueError('source https://example.org/x?token='+secret+' file /private/operator/model '+secret))
        self.assertNotIn(secret,detail)
        self.assertNotIn('/private/operator',detail)
        self.assertNotIn('example.org',detail)
        self.assertIn('source',detail)
        self.assertLessEqual(len(p.failure_detail(ValueError('x'*1000))),800)

    def test_no_model_imports_when_loading_controls(self):
        import sys
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('transformers',sys.modules)

if __name__=='__main__':unittest.main()
