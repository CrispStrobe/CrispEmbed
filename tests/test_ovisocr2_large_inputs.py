"""Small fail-closed controls; native bounds execute on hosted build."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
import tempfile
import platform
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ci-heavy'))
import ovisocr2_large_inputs as driver

class Controls(unittest.TestCase):
    def test_frozen_panel_precedes_outputs(self):
        p=driver.checked_panel()
        self.assertEqual(p['max_new_tokens'],32)
        self.assertEqual(p['rejection_case']['patches'],2176)
    def test_default_limits_or_cap_drift_rejected(self):
        p=driver.checked_panel()
        for key,value in [('default_visual_patch_limit',2048),('optin_visual_patch_limit',8192),('max_new_tokens',64)]:
            changed=deepcopy(p);changed[key]=value
            with mock.patch.object(driver,'PANEL',mock.Mock(read_text=mock.Mock(return_value=json.dumps(changed)))):
                with self.assertRaises(ValueError):driver.checked_panel()
    def test_unrelated_arithmetic_must_remain_frozen(self):
        closure=driver.source_contract()
        self.assertIn('src/qwen35_image.cpp',closure['intentional_changed_files'])
        original=driver.audit.sha
        def modified(path):return '0'*64 if str(path).endswith('src/qwen35_norm.cpp') else original(path)
        with mock.patch.object(driver.audit,'sha',side_effect=modified):
            with self.assertRaisesRegex(ValueError,'arithmetic'):driver.source_contract()
    def test_unreviewed_public_wrapper_cannot_pass(self):
        original=driver.audit.sha
        def modified(path):return '0'*64 if str(path).endswith('src/crispembed.cpp') else original(path)
        with mock.patch.object(driver.audit,'sha',side_effect=modified):
            with self.assertRaisesRegex(ValueError,'public wrapper'):driver.source_contract()
    @unittest.skipUnless(platform.system()=='Linux' and Path('/usr/bin/time').is_file(),'GNU time on hosted Linux')
    def test_actual_resource_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            report={'commands':{}}
            self.assertTrue(driver.timed_command('probe',[sys.executable,'-c','print(42)'],Path(directory),report))
            self.assertGreater(report['commands']['probe']['resources']['max_rss_kib'],0)
    def test_actual_environment_override_with_existing_baseline_keys(self):
        with mock.patch.dict(driver.os.environ,{'CRISPEMBED_DIFF_COS_THRESHOLD':'0.5'}):
            frozen=driver.native_environment('frozen-control')
            larger=driver.native_environment('large-boundary')
        for env in (frozen,larger):
            self.assertEqual(env['CRISPEMBED_QWEN35_VISION_PRECISE_NORM'],'1')
            self.assertEqual(env['CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'],'1')
            self.assertEqual(env['CRISPEMBED_QWEN35_VISION_CASCADE_NORM'],'0')
            self.assertNotIn('CRISPEMBED_DIFF_COS_THRESHOLD',env)
        self.assertEqual(frozen['CRISPEMBED_OVISOCR2_LARGE_IMAGES'],'0')
        self.assertEqual(larger['CRISPEMBED_OVISOCR2_LARGE_IMAGES'],'1')
    def test_missing_full_layer_raw_cannot_pass(self):
        with self.assertRaises(ValueError):driver.raw_manifest(Path('.'),{'stages':{},'generated_ids':[[1,248046]]},'PASS prefill.decoder.layer.0: cos_min=1','f16-decoder-native-frontend')

if __name__=='__main__':unittest.main()
