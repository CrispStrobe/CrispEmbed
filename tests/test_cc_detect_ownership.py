#!/usr/bin/env python3
"""Actual CC C-ABI allocator/ownership controls without ggml or model builds.

Compile the two production wrapper functions against the real public headers,
substituting only the lower detector and allocator to force bounded failures.
"""

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_preprocess_bindings import BINDING, NativePreprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTP = ctypes.POINTER(BINDING._CrispOcrResult)


def extract_function(source, prefix):
    start = source.index(prefix)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def native_fixture(directory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise unittest.SkipTest("C++ compiler required for tiny native interface fixture")
    source = (ROOT / "src/crispembed.cpp").read_text()
    functions = "\n".join(extract_function(source, prefix) for prefix in (
        'extern "C" crispembed_ocr_result * crispembed_cc_detect(',
        'extern "C" void crispembed_cc_detect_free('))
    harness = r'''
#define CRISPEMBED_BUILD
#include "crispembed.h"
#include "cc_detect.h"
#include <cstdlib>
static int mode = 0, intermediate_frees = 0, result_frees = 0, calls = 0;
static void * outstanding = nullptr;
static cc_text_region regions[3] = {{1, 2, 3, 4}, {101, 102, 103, 104}, {201, 202, 203, 204}};
extern "C" void test_reset(int value) {
    mode = value; intermediate_frees = result_frees = calls = 0;
}
extern "C" int test_intermediate_frees() { return intermediate_frees; }
extern "C" int test_result_frees() { return result_frees; }
extern "C" int test_calls() { return calls; }
extern "C" int test_live() { return outstanding != nullptr; }
extern "C" cc_text_region * cc_detect_lines(const uint8_t * gray, int w, int h, int * n) {
    ++calls;
    *n = 0;
    if (!gray || w <= 0 || h <= 0 || mode == 1) return nullptr;
    if (mode == 2) return regions; // non-null empty lower-level allocation
    *n = 3;
    if (mode == 4) return nullptr; // lower-level failure with stale count
    return regions;
}
extern "C" void cc_detect_free(cc_text_region * ptr) {
    if (ptr) ++intermediate_frees;
}
static void * test_calloc(size_t count, size_t size) {
    if (mode == 3) return nullptr;
    if (outstanding) std::abort();
    outstanding = std::calloc(count, size);
    return outstanding;
}
static void test_free(void * ptr) {
    if (!ptr) return;
    if (ptr != outstanding) std::abort();
    ++result_frees;
    std::free(ptr);
    outstanding = nullptr;
}
#define calloc test_calloc
#define free test_free
'''
    cpp = Path(directory) / "fixture.cpp"
    harness = harness.replace('extern "C" ', 'extern "C" CRISPEMBED_API ')
    cpp.write_text(harness + functions)
    library = Path(directory) / ("cc_fixture.dylib" if sys.platform == "darwin" else "cc_fixture.so")
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden",
                    "-I", str(ROOT / "src"), str(cpp), "-o", str(library)],
                   check=True, text=True, timeout=30)
    # The real API's CRISPEMBED_API declarations must export the production pair
    # even though the compiler hides all symbols by default.
    return library, harness, functions


class CcDetectOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="cc-interface-")
        cls.addClassCleanup(cls.temp.cleanup)
        library, cls.harness, cls.functions = native_fixture(cls.temp.name)
        cls.library = library
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.test_reset.argtypes = [ctypes.c_int]
        for name in ("test_intermediate_frees", "test_result_frees", "test_calls", "test_live"):
            getattr(cls.lib, name).restype = ctypes.c_int
        cls.lib.crispembed_cc_detect.argtypes = [ctypes.POINTER(ctypes.c_uint8),
                                               ctypes.c_int, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_int)]
        cls.lib.crispembed_cc_detect.restype = RESULTP
        cls.lib.crispembed_cc_detect_free.argtypes = [RESULTP]
        cls.lib.crispembed_cc_detect_free.restype = None

    def setUp(self):
        self.lib.test_reset(0)
        native = NativePreprocess()
        native.lib.crispembed_cc_detect = self.lib.crispembed_cc_detect
        native.lib.crispembed_cc_detect_free = self.lib.crispembed_cc_detect_free
        self.pp = BINDING.CrispPreprocess.__new__(BINDING.CrispPreprocess)
        self.pp._lib = native.lib
        self.pixels = np.zeros((2, 3), dtype=np.uint8)

    def tearDown(self):
        self.assertEqual(self.lib.test_live(), 0, "native CC result allocation leaked")

    def test_three_regions_copy_real_header_layout_and_use_library_free(self):
        with patch.object(ctypes.cdll, "LoadLibrary", side_effect=AssertionError("foreign allocator")):
            result = self.pp.cc_detect(self.pixels, 3, 2)
        self.assertEqual(result, [dict(x=1, y=2, w=3, h=4),
                                  dict(x=101, y=102, w=103, h=104),
                                  dict(x=201, y=202, w=203, h=204)])
        self.assertEqual(self.lib.test_intermediate_frees(), 1)
        self.assertEqual(self.lib.test_result_frees(), 1)

    def test_empty_and_allocation_failures_clear_count_and_free_intermediates(self):
        for mode, expected_free in ((1, 0), (2, 1), (3, 1), (4, 0)):
            with self.subTest(mode=mode):
                self.lib.test_reset(mode)
                n = ctypes.c_int(99)
                ptr = self.lib.crispembed_cc_detect(
                    self.pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 3, 2, ctypes.byref(n))
                self.assertFalse(ptr)
                self.assertEqual(n.value, 0)
                self.assertEqual(self.lib.test_intermediate_frees(), expected_free)
                self.assertEqual(self.pp.cc_detect(self.pixels, 3, 2), [])
                self.assertEqual(self.lib.test_result_frees(), 0)

    def test_null_free_and_optional_output_count(self):
        self.lib.crispembed_cc_detect_free(None)
        ptr = self.lib.crispembed_cc_detect(
            self.pixels.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 3, 2, None)
        self.assertTrue(ptr)
        self.lib.crispembed_cc_detect_free(ptr)
        self.assertEqual(self.lib.test_result_frees(), 1)

    def test_python_copy_error_still_frees_native_results(self):
        def fail(row):
            raise ValueError("copy failed")
        with patch.object(BINDING._CrispOcrResult, "x", property(fail)):
            with self.assertRaisesRegex(ValueError, "copy failed"):
                self.pp.cc_detect(self.pixels, 3, 2)
        self.assertEqual(self.lib.test_result_frees(), 1)

    def test_nonnull_empty_or_invalid_count_still_releases_library_allocation(self):
        original = self.pp._lib.crispembed_cc_detect
        for count in (0, -1):
            with self.subTest(count=count):
                self.lib.test_reset(0)
                def detect(gray, w, h, out_n):
                    ptr = original(gray, w, h, out_n)
                    out_n[0] = count
                    return ctypes.cast(ptr, ctypes.c_void_p).value
                callback = ctypes.CFUNCTYPE(ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int))(detect)
                self.pp._lib.crispembed_cc_detect = callback
                if count < 0:
                    with self.assertRaisesRegex(RuntimeError, "invalid result count"):
                        self.pp.cc_detect(self.pixels, 3, 2)
                else:
                    self.assertEqual(self.pp.cc_detect(self.pixels, 3, 2), [])
                self.assertEqual(self.lib.test_result_frees(), 1)

    def test_null_pointer_with_positive_count_rejects_before_indexing(self):
        def detect(gray, w, h, out_n):
            out_n[0] = 3
            return None
        callback = ctypes.CFUNCTYPE(ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int))(detect)
        self.pp._lib.crispembed_cc_detect = callback
        with self.assertRaisesRegex(RuntimeError, "invalid result count"):
            self.pp.cc_detect(self.pixels, 3, 2)
        self.assertEqual(self.lib.test_result_frees(), 0)

    def test_actual_rust_cc_method_uses_same_library_owner(self):
        compiler = shutil.which("rustc")
        if not compiler:
            self.skipTest("rustc required for focused Rust interface test")
        raw = (ROOT / "crispembed-sys/src/lib.rs").read_text()
        safe = (ROOT / "crispembed/src/lib.rs").read_text()
        result = extract_function(raw, "pub struct CrispembedOcrResult {")
        region = extract_function(safe, "pub struct OcrResult {")
        region += "\n" + next(line for line in safe.splitlines()
                                if line.startswith("pub type OcrRegion ="))
        method = extract_function(safe, "pub fn cc_detect(")
        decl_start = raw.index("    pub fn crispembed_cc_detect(")
        decl_end = raw.index(";", raw.index("pub fn crispembed_cc_detect_free", decl_start)) + 1
        declarations = raw[decl_start:decl_end]
        harness = """
mod crispembed_sys {
    use std::ffi::{c_char, c_int, c_float};
    #[repr(C)]
    RESULT
    extern "C" { DECLARATIONS }
}
REGION
METHOD
extern "C" {
    fn test_reset(mode: i32);
    fn test_result_frees() -> i32;
    fn test_intermediate_frees() -> i32;
    fn test_live() -> i32;
}
fn main() {
    unsafe { test_reset(0); }
    let results = cc_detect(&[0; 6], 3, 2);
    assert_eq!(results.len(), 3);
    for (i, row) in results.iter().enumerate() {
        let offset = (i * 100) as f32;
        assert_eq!((row.x, row.y, row.w, row.h),
                   (offset+1.0, offset+2.0, offset+3.0, offset+4.0));
        assert_eq!(row.confidence, 1.0);
    }
    unsafe {
        assert_eq!(test_live(), 0);
        assert_eq!(test_result_frees(), 1);
        assert_eq!(test_intermediate_frees(), 1);
        for mode in [1, 2, 3, 4] {
            test_reset(mode);
            assert!(cc_detect(&[0; 6], 3, 2).is_empty());
            assert_eq!(test_live(), 0);
            assert_eq!(test_result_frees(), 0);
        }
    }
}
"""
        for key, value in (("RESULT", result), ("DECLARATIONS", declarations),
                           ("REGION", region), ("METHOD", method)):
            harness = harness.replace(key, value)
        directory = Path(self.temp.name)
        source = directory / "cc_method.rs"
        binary = directory / "cc_method"
        source.write_text(harness)
        subprocess.run([compiler, "--edition=2021", str(source), "-o", str(binary),
                        "-C", "link-arg=" + str(self.library),
                        "-C", "link-arg=-Wl,-rpath," + str(directory)],
                       check=True, text=True, timeout=30)
        subprocess.run([str(binary)], check=True, text=True, timeout=10)

    def test_older_runtime_only_blocks_cc_before_native_allocation(self):
        native = NativePreprocess()
        del native.lib.crispembed_cc_detect_free
        with patch.object(BINDING, "_load_library", return_value=native.lib):
            pp = BINDING.CrispPreprocess()
        np.testing.assert_array_equal(pp.background_norm(self.pixels, 3, 2), self.pixels)
        native.calls.clear()
        with self.assertRaisesRegex(RuntimeError, "update the native library"):
            pp.cc_detect(self.pixels, 3, 2)
        self.assertEqual(native.calls, [])


if __name__ == "__main__":
    unittest.main()
