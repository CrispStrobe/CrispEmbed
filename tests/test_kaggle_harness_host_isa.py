#!/usr/bin/env python3
"""kaggle_harness keys ccache on the resolved -march=native ISA (docs/kaggle-ccache.md).

A seed compiled on an AVX-512 Kaggle host handed AVX-512 objects to an AVX2 host
(SIGILL): ccache keys on command-line text, and `-march=native` is the same text
everywhere. Stdlib only; the end-to-end proof with a real AVX2 consumer is
tools/kaggle/ccache_isa_proof.py (receipt tests/results/kaggle-ccache-host-isa.json).
"""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('ce_kaggle_harness_isa', ROOT / 'kaggle_harness.py')
kh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kh)
HAS_GCC = shutil.which('gcc') is not None and os.uname().machine in ('x86_64', 'AMD64')


class HostIsaKey(unittest.TestCase):
    @unittest.skipUnless(HAS_GCC, 'requires an x86-64 gcc driver')
    def test_avx2_and_avx512_hosts_get_different_keys(self):
        with mock.patch.dict(os.environ, {'CC': 'gcc'}):
            v3, v4 = kh.host_isa_fingerprint('x86-64-v3'), kh.host_isa_fingerprint('x86-64-v4')
        self.assertIn('-mavx2 [enabled]', v3)
        self.assertNotIn('-mavx512f [enabled]', v3)
        self.assertIn('-mavx512f [enabled]', v4)
        self.assertNotEqual(v3, v4)
        self.assertEqual(v3, kh.host_isa_fingerprint('x86-64-v3'))  # stable text, stable key

    def test_extrafiles_points_at_fingerprint_and_keeps_existing_entries(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {'CCACHE_EXTRAFILES': '/existing.txt'}), \
                mock.patch.object(kh, 'host_isa_fingerprint', return_value='-march= haswell\n'):
            digest = kh._key_cache_on_host_isa(Path(d))
            files = os.environ['CCACHE_EXTRAFILES'].split(os.pathsep)
            self.assertEqual(files[0], '/existing.txt')
            self.assertEqual(Path(files[1]).read_text(), '-march= haswell\n')
            self.assertIn(digest, files[1])
            self.assertEqual(os.environ['CE_CCACHE_HOST_ISA'], digest)
            kh._key_cache_on_host_isa(Path(d))  # idempotent
            self.assertEqual(os.environ['CCACHE_EXTRAFILES'].split(os.pathsep), files)

    def test_fallback_without_compiler_uses_cpu_flags(self):
        with mock.patch.dict(os.environ, {'CC': '/nonexistent/cc'}):
            text = kh.host_isa_fingerprint()
        self.assertTrue(text.strip())

    def test_install_keys_cache_before_warming_and_records_seed_isa(self):
        src = (ROOT / 'kaggle_harness.py').read_text()
        body = src[src.index('def install_build_toolchain'):src.index('def crispasr_cmake_flags')]
        self.assertLess(body.index('_key_cache_on_host_isa('), body.index('_warm_ccache_from_dataset('))
        self.assertIn('"host_isa": os.environ.get("CE_CCACHE_HOST_ISA")', src)


if __name__ == '__main__':
    unittest.main()
