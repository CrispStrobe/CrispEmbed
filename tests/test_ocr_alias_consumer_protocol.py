"""Model-free controls for the hosted OCR alias smoke's failure handling."""
import importlib.util
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
with patch.dict(os.environ, {'HEAVY_OUT': '/tmp/ocr-alias-control-out', 'HEAVY_SCRATCH': '/tmp/ocr-alias-control-scratch'}):
    spec = importlib.util.spec_from_file_location('ocr_alias_consumer', ROOT / 'tools/ci-heavy/ocr_experimental_alias_consumer.py')
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)


def result(status, stdout='', stderr=''):
    return subprocess.CompletedProcess([], status, stdout, stderr)


class ConsumerProtocolTest(unittest.TestCase):
    def test_zero_exit_is_not_enough_to_prove_recognition(self):
        with self.assertRaises(ValueError):
            smoke.decoded(result(0, '', 'error: OCR recognition failed'))
        with self.assertRaises(ValueError):
            smoke.decoded(result(0, '{"text":"  "}'))
        self.assertEqual(smoke.decoded(result(0, '{"text":"fox"}')), {'text': 'fox'})

    def test_gate_rejection_requires_engine_diagnostic_and_no_output(self):
        diagnostic = 'unlimited_ocr: Unlimited-OCR-2 model requires CRISPEMBED_UNLIMITED_OCR2=1'
        smoke.check_gate_rejected(result(0, '', diagnostic), 'unlimited')
        with self.assertRaises(AssertionError):
            smoke.check_gate_rejected(result(1, '', 'unrelated file error'), 'unlimited')
        with self.assertRaises(AssertionError):
            smoke.check_gate_rejected(result(0, '{"text":"fox"}', diagnostic), 'unlimited')


if __name__ == '__main__':
    unittest.main()
