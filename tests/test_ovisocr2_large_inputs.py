"""Small fail-closed controls; native bounds execute on hosted build."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
import tempfile
import platform
from contextlib import contextmanager
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ci-heavy'))
import ovisocr2_large_inputs as driver

class Controls(unittest.TestCase):
    @contextmanager
    def historical_source(self):
        # The old larger-input audit must keep rejecting the new cache runtime.
        # Exercise its original contract against a synthetic historical tree,
        # rather than widening its arithmetic allowlist for a new experiment.
        old=json.loads((driver.ROOT/'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']['native_source_sha256']
        def historical_sha(path):
            name=str(Path(path).relative_to(driver.ROOT))
            if name=='src/qwen35_image.cpp':return '1'*64
            return old.get(name,'2'*64)
        sources=tuple(name for name in driver.SOURCES if name!='src/qwen35_weight_cache.h')
        with mock.patch.object(driver.audit,'sha',side_effect=historical_sha),mock.patch.object(driver,'SOURCES',sources):
            yield
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
        with self.historical_source():
            closure=driver.source_contract()
            self.assertIn('src/qwen35_image.cpp',closure['intentional_changed_files'])
            original=driver.audit.sha.side_effect
            def modified(path):return '0'*64 if str(path).endswith('src/qwen35_norm.cpp') else original(path)
            with mock.patch.object(driver.audit,'sha',side_effect=modified):
                with self.assertRaisesRegex(ValueError,'arithmetic'):driver.source_contract()
    def test_historical_driver_rejects_new_cache_runtime(self):
        # Reach the decoder-arithmetic guard using the historical wrapper;
        # unrelated current wrapper changes must not mask this specific check.
        with self.historical_source():
            original=driver.audit.sha.side_effect
            def cached_decoder(path):
                return '0'*64 if str(path).endswith('src/qwen35_decoder.cpp') else original(path)
            with mock.patch.object(driver.audit,'sha',side_effect=cached_decoder):
                with self.assertRaisesRegex(ValueError,'arithmetic'):driver.source_contract()
    def test_historical_driver_rejects_new_cache_header(self):
        with self.historical_source():
            sources=driver.SOURCES+('src/qwen35_weight_cache.h',)
            with mock.patch.object(driver,'SOURCES',sources):
                with self.assertRaisesRegex(ValueError,'new native source'):driver.source_contract()
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
            self.assertFalse(report['commands']['probe']['resources_recovered_from_gnu_failure_prefix'])
    def test_observed_gnu_failure_prefix_retains_resources_and_failure(self):
        measured={'max_rss_kib':9762208,'wall_seconds':307.06}
        parsed,recovered=driver.parse_gnu_time_resources(
            'Command exited with non-zero status 1\n'+json.dumps(measured)+'\n',1)
        self.assertEqual(parsed,measured)
        self.assertTrue(recovered)
        self.assertEqual(driver.parse_gnu_time_resources(json.dumps(measured),0),(measured,False))
    @unittest.skipUnless(platform.system()=='Linux' and Path('/usr/bin/time').is_file(),'GNU time on hosted Linux')
    def test_actual_failed_command_retains_resources_without_passing(self):
        with tempfile.TemporaryDirectory() as directory:
            report={'commands':{}}
            self.assertFalse(driver.timed_command('probe',[sys.executable,'-c','raise SystemExit(1)'],Path(directory),report))
            command=report['commands']['probe']
            self.assertEqual(command['exit_code'],1)
            self.assertFalse(command['passed'])
            self.assertTrue(command['resources_recovered_from_gnu_failure_prefix'])
            self.assertGreater(command['resources']['max_rss_kib'],0)
    def test_gnu_resource_parser_rejects_junk_status_and_invalid_values(self):
        good='{"max_rss_kib":100,"wall_seconds":0.01}'
        bad=[
            ('garbage\n'+good,1),
            ('Command exited with non-zero status 2\n'+good,1),
            ('Command exited with non-zero status 1\n'+good,0),
            ('Command exited with non-zero status 1\n'+good+'junk',1),
            ('{"max_rss_kib":true,"wall_seconds":0.01}',0),
            ('{"max_rss_kib":0,"wall_seconds":0.01}',0),
            ('{"max_rss_kib":-1,"wall_seconds":0.01}',0),
            ('{"max_rss_kib":100,"wall_seconds":-1}',0),
            ('{"max_rss_kib":100,"wall_seconds":false}',0),
            ('{"max_rss_kib":100,"wall_seconds":NaN}',0),
            ('{"max_rss_kib":100,"wall_seconds":Infinity}',0),
            ('{"max_rss_kib":100,"wall_seconds":"1"}',0),
            ('{"max_rss_kib":100,"wall_seconds":0.01,"extra":1}',0),
        ]
        for contents,exit_code in bad:
            with self.subTest(contents=contents):
                with self.assertRaises(ValueError):
                    driver.parse_gnu_time_resources(contents,exit_code)
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
    def test_native_vision_decoder_uses_actual_score_count(self):
        stages={f'{phase}.decoder.layer.{layer}':[1]
                for phase in ('prefill','cached','full_prefix') for layer in range(24)}
        stages.update({f'generated.scores.{i}':[1] for i in range(3)})
        stages.update({f'{phase}.cache.0.conv_states':[1] for phase in ('prefill','cached')})
        reference={'stages':stages,'generated_ids':[[7,8,248046]]}
        log='\n'.join(f'PASS {name}: cos_min=1' for name in stages)
        with tempfile.TemporaryDirectory() as directory:
            for name in stages:(Path(directory)/(name+'.f32')).write_bytes(bytes(4))
            result=driver.raw_manifest(Path(directory),reference,log,'f16-decoder-native-vision')
            self.assertTrue(result['coverage_complete'])
            self.assertEqual(result['file_count'],len(stages))
    def test_missing_full_layer_raw_cannot_pass(self):
        with self.assertRaises(ValueError):driver.raw_manifest(Path('.'),{'stages':{},'generated_ids':[[1,248046]]},'PASS prefill.decoder.layer.0: cos_min=1','f16-decoder-native-frontend')

if __name__=='__main__':unittest.main()
