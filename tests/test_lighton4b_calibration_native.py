#!/usr/bin/env python3
"""Bounded staging negatives without model data, network or execution."""
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('calibration',ROOT/'tools/ci-heavy/lighton4b_calibration_native.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class TransportControls(unittest.TestCase):
 def test_bounded_verified_transfer(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(module.urllib.request,'urlopen',return_value=io.BytesIO(b'actual')):
   path=Path(tmp)/'row';self.assertEqual(module.bounded_download('unused',path,hashlib.sha256(b'actual').hexdigest(),6,6),6)
 def test_overrun_and_wrong_hash_rejected(self):
  for payload,digest,maximum in [(b'too large','0'*64,3),(b'bad','0'*64,3)]:
   with tempfile.TemporaryDirectory() as tmp,patch.object(module.urllib.request,'urlopen',return_value=io.BytesIO(payload)):
    with self.assertRaises(module.CalibrationError):module.bounded_download('unused',Path(tmp)/'row',digest,maximum)
 def test_only_reviewed_calibration_cases(self):
  self.assertEqual([x[0] for x in module.CASES],['simple_table','arabic_printed_line'])
  self.assertEqual([x[2] for x in module.CASES],[222,6])
if __name__=='__main__':unittest.main()
