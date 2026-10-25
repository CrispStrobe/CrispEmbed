"""Small fail-closed controls; actual C++ and model execution stay hosted."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ci-heavy'))
import ovisocr2_weight_cache as driver


class CacheControls(unittest.TestCase):
    def fixture(self,two=False):
        reference=json.loads((driver.ROOT/'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']['native_cases']['f16']['dev-hello']
        return [dict(label=label,contexts=2 if two else 1,init_seconds=1,recognize_seconds=38,
                     token_ids=reference['token_ids'],confidences=reference['confidences'],
                     raw_text_utf8_hex=reference['raw_text'].encode().hex(),stop_kind=1,
                     mean_confidence=reference['mean_confidence'])
                for label in (['cold','warm','independent','survivor'] if two else ['cold','warm'])]
    def logs(self,rows):
        return '\n'.join('CACHE_BENCH_RESULT '+json.dumps(row) for row in rows)
    def test_complete_cold_warm_survivor_answers(self):
        for two in (False,True):
            rows=self.fixture(two)
            self.assertEqual(driver.benchmark_rows(self.logs(rows),two),rows)
            driver.require_exact(rows)
    def test_bad_order_missing_stop_and_nonfinite_rejected(self):
        for key,value in [('label','warm'),('stop_kind',2),('token_ids',[1]),('recognize_seconds',float('nan')),('confidences',[1]*8)]:
            rows=self.fixture();rows[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):driver.benchmark_rows(self.logs(rows))
        with self.assertRaises(ValueError):driver.benchmark_rows(self.logs(self.fixture()[:-1]))
    def test_probability_drift_rejected(self):
        rows=self.fixture();rows[1]['confidences']=list(rows[1]['confidences']);rows[1]['confidences'][0]-=1e-6
        with self.assertRaises(ValueError):driver.require_exact(rows)
    def test_actual_decoder_schema_without_file_and_delta_variants(self):
        # Regression for the actual e6 postprocessing KeyError. Both schemas
        # retain their payload metadata; path reconstruction cannot drop a row.
        decoder={'coverage_complete':True,'files':[{'stage':'prefill.decoder.layer.0','shape':[1,2],
                 'bytes':8,'sha256':'a'*64}]}
        normalized=driver.normalize_manifest(decoder,'cache-1-f16-decoder-prepared')
        self.assertEqual(normalized['files'][0]['file'],'native-diff/cache-1-f16-decoder-prepared/prefill.decoder.layer.0.f32')
        self.assertEqual({k:v for k,v in normalized['files'][0].items() if k!='file'},decoder['files'][0])
        self.assertNotIn('file',decoder['files'][0])
        delta={'coverage_complete':True,'files':[{'stage':'prefill.delta.output','variant':'reset',
               'file':'native-diff/f16-delta/reset/prefill.delta.output.f32','bytes':8,'sha256':'b'*64}]}
        self.assertEqual(driver.normalize_manifest(delta,'cache-0-f16-delta')['files'][0]['file'],
                         'native-diff/cache-0-f16-delta/reset/prefill.delta.output.f32')
        bad=deepcopy(delta);bad['files'][0]['variant']='../escape'
        with self.assertRaises(ValueError):driver.normalize_manifest(bad,'cache-0-f16-delta')
    def test_ambient_experiments_cannot_reach_either_arm(self):
        with patch.dict(os.environ,{'CRISPEMBED_QWEN35_PRECISE_RMS':'1','CRISPEMBED_DIFF_COS_THRESHOLD':'0.1',
                                    'CRISPEMBED_QWEN35_WEIGHT_CACHE_BYTES':'9999999999',
                                    'CRISPEMBED_NEW_UNREVIEWED_GATE':'1','CRISPEMBED_QWEN35_PARALLEL_F16':'1'}):
            for on in (False,True):
                env=driver.environment(on)
                self.assertEqual(env['CRISPEMBED_QWEN35_WEIGHT_CACHE'],str(int(on)))
                self.assertEqual(env['CRISPEMBED_QWEN35_WEIGHT_CACHE_BYTES'],str(driver.BUDGET))
                self.assertEqual(env['CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'],'1')
                self.assertEqual(env['CRISPEMBED_QWEN35_PRECISE_RMS'],'0')
                self.assertEqual(env['CRISPEMBED_QWEN35_PARALLEL_F16'],'0')
                self.assertNotIn('CRISPEMBED_DIFF_COS_THRESHOLD',env)
                self.assertNotIn('CRISPEMBED_NEW_UNREVIEWED_GATE',env)
    def test_source_pins_reject_arbitrary_native_change(self):
        actual=driver.audit.sha
        required={name for name in driver.NATIVE_PINS if name.startswith('src/') or name in
                  ('CMakeLists.txt','tests/test_qwen35_decoder.cpp')}
        def fake_git(*args):
            return '\n'.join(sorted(required)) if args[:3]==('diff','--name-only',driver.BASE) else ''
        with patch.object(driver,'git',side_effect=fake_git),patch.object(driver.audit,'sha',side_effect=lambda p:'0'*64 if str(p).endswith('qwen35_linear.cpp') else actual(p)):
            with self.assertRaisesRegex(ValueError,'prototype hashes'):driver.source_contract()
    def test_full_first_f16_quality_and_confidences(self):
        source=json.loads((driver.ROOT/'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']
        fake=type('Report',(),{'read_text':lambda self:json.dumps(source)})()
        with patch.object(driver,'Path',return_value=fake):
            result=driver.verify_quality('unused')
        self.assertEqual(result['token_count'],1841)
        source=deepcopy(source);source['native_cases']['f16']['dev-hello']['confidences'][0]-=1e-6
        with patch.object(driver,'Path',return_value=type('Report',(),{'read_text':lambda self:json.dumps(source)})()):
            with self.assertRaises(ValueError):driver.verify_quality('unused')

if __name__=='__main__':unittest.main()
