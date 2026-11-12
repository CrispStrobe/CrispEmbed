#!/usr/bin/env python3
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('trace',Path(__file__).with_name('audit.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class Controls(unittest.TestCase):
    def test_metrics_and_nonfinite(self):
        row=audit.delta(np.array([3.,4.],dtype='<f4'),np.array([0.,0.],dtype='<f4'))
        self.assertEqual(row['error_max_abs'],4.)
        self.assertAlmostEqual(row['error_rms'],(25/2)**.5)
        self.assertFalse(row['bit_exact'])
        with self.assertRaises(ValueError):audit.delta(np.array([np.inf]),np.array([0.]))
        with self.assertRaises(ValueError):audit.delta(np.zeros(2),np.zeros(1))

    def test_seals_and_closed_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'owned';p.write_bytes(b'abcd')
            seal={'owned':{'bytes':4,'sha256':hashlib.sha256(b'abcd').hexdigest()}}
            self.assertEqual(audit.owned(root,'owned',seal),p)
            p.write_bytes(b'efgh')
            with self.assertRaises(ValueError):audit.owned(root,'owned',seal)
            with self.assertRaises(ValueError):audit.owned(root,'../escape',seal)

    def test_signed_zero_is_not_bit_exact(self):
        row=audit.delta(np.array([0.],dtype='<f4'),np.array([-0.],dtype='<f4'))
        self.assertFalse(row['bit_exact']);self.assertEqual(row['error_max_abs'],0.)


if __name__=='__main__':unittest.main()
