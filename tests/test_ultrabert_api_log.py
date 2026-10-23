"""Model-free controls: pooled qualification cannot trust forged native verdicts."""
import importlib.util
from pathlib import Path
import unittest
import tempfile

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ultrabert_api',root/'tools/ci-heavy/ultrabert_api.py')
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
GOOD=('IDS\t0\t0\t42\n'
      'POOLED\tcls_raw\t1\t0.0001\t0.00001\t0.00002\t80\t80\t1\n'
      'POOLED\tcls_normalized\t1\t0.000001\t0.0000001\t0.0000002\t1\t1\t1\n'
      'TOTAL\t1\t2\t0\n')

class EvidenceControls(unittest.TestCase):
    def test_cmake_cli_output_name_and_missing_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            build=Path(folder);(build/'bin').mkdir()
            actual=build/'bin/crispembed';actual.touch()
            self.assertEqual(api.find_binary(build,'crispembed-cli'),actual)
            with self.assertRaisesRegex(ValueError,'Missing built executable: crispembed-server'):
                api.find_binary(build,'crispembed-server')

    def test_positive_and_honest_numerical_failure(self):
        self.assertEqual(api.parse_pooled(GOOD)['failures'],0)
        bad=GOOD.replace('1\t0.0001','1\t0.01').replace('80\t80\t1','80\t80\t0').replace('TOTAL\t1\t2\t0','TOTAL\t1\t2\t1')
        self.assertEqual(api.parse_pooled(bad)['failures'],1)
    def test_forged_pass_and_nonfinite_rejected(self):
        for bad in (GOOD.replace('1\t0.0001','1\t0.01'),GOOD.replace('1\t0.0001','nan\t0.0001')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):api.parse_pooled(bad)
    def test_missing_duplicate_and_wrong_total_rejected(self):
        for bad in (GOOD.replace('POOLED\tcls_raw','IGNORED\tcls_raw'),GOOD+GOOD,GOOD.replace('TOTAL\t1','TOTAL\t2')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):api.parse_pooled(bad)
    def test_missing_invalid_or_unattended_bos_ids_rejected(self):
        for bad in (GOOD.replace('IDS\t0\t0\t42\n',''),GOOD.replace('IDS\t0\t0','IDS\t1\t0'),
                    GOOD.replace('IDS\t0\t0','IDS\t0\t15'),GOOD.replace('\t42\n','\t49152\n')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):api.parse_pooled(bad)

if __name__=='__main__':unittest.main()
