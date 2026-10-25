"""No native execution: preserved failed-producer analysis must fail closed."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import unittest
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/ci-heavy'))
import ovisocr2_weight_cache_reanalysis as recovery


class PreservedControls(unittest.TestCase):
    def producer(self):
        return json.loads((recovery.ROOT/'tests/results/qwen35-ocr/initial-weight-cache-manifest-failure.json').read_text())['producer']
    def test_actual_failed_producer_has_only_known_schema_errors(self):
        self.assertEqual(len(recovery.require_source_producer(self.producer())),37)
    def test_real_program_failure_cannot_be_waived(self):
        producer=self.producer();producer['commands']['cache-1-f16-decoder-prepared']['exit_code']=1
        with self.assertRaisesRegex(ValueError,'program/benchmark'):recovery.require_source_producer(producer)
    def test_unknown_decoder_failure_is_not_reclassified(self):
        producer=self.producer();producer['arms']['1']['f16-decoder-prepared']['error']='Raw stage set differs'
        with self.assertRaisesRegex(ValueError,'manifest field error'):recovery.require_source_producer(producer)
    def test_incomplete_original_coverage_rejects(self):
        producer=self.producer();del producer['arms']['0']['f32-vision']
        with self.assertRaisesRegex(ValueError,'fourteen-arm'):recovery.require_source_producer(producer)
    def test_unexpected_initial_failure_rejects(self):
        producer=self.producer();producer['frozen_failures'].append('unexpected')
        with self.assertRaisesRegex(ValueError,'failure set'):recovery.require_source_producer(producer)
    def test_wrong_source_run_rejects_before_transfer(self):
        with self.assertRaisesRegex(ValueError,'exact preserved'):recovery.preflight(1)
    def test_unsafe_or_oversized_archive_rejects(self):
        for name,limit in [('../escape',1024),('/absolute',1024),('valid',1)]:
            b=io.BytesIO()
            with zipfile.ZipFile(b,'w') as z:z.writestr(name,b'1234')
            b.seek(0)
            with zipfile.ZipFile(b) as z:
                with self.assertRaises(ValueError):recovery.checked_members(z,maximum=limit)

if __name__=='__main__':unittest.main()
