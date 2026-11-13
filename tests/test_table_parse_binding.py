#!/usr/bin/env python3
"""TableParse buffer/HTML ownership via actual public wrappers and internal free."""

import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_preprocess_bindings import BINDING
from test_cc_detect_ownership import extract_function

ROOT = Path(__file__).resolve().parents[1]


class TableParseBindingTests(unittest.TestCase):
    @staticmethod
    def build_fixture(directory):
        compiler = shutil.which("c++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler required for tiny interface fixture")
        source = (ROOT / "src/crispembed.cpp").read_text()
        wrappers = "\n".join(extract_function(source, prefix) for prefix in (
            'extern "C" void * crispembed_table_parse_init(',
            'extern "C" void crispembed_table_parse_free(',
            'extern "C" char * crispembed_table_parse_to_html(',
            'extern "C" void crispembed_table_parse_free_string(',
            'extern "C" int crispembed_table_parse_detect_grid('))
        real_free = extract_function((ROOT / "src/table_parse.cpp").read_text(),
                                     "void table_parse_free_string(")
        grid = extract_function((ROOT / "src/table_parse.cpp").read_text(), "struct Grid {") + ";"
        real_grid = extract_function((ROOT / "src/table_parse.cpp").read_text(),
                                     "int table_parse_detect_grid(")
        harness = r'''
#include "crispembed.h"
#include "table_parse.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <vector>
static int calls=0, length=0, mode=0, frees=0, context_frees=0, null_model=0;
static void * outstanding=nullptr;
static std::vector<uint8_t> seen;
extern "C" void test_reset(int size,int value) {
    if (outstanding) std::abort();
    calls=frees=context_frees=0; length=size; mode=value; seen.clear();
}
extern "C" int test_calls() { return calls; }
extern "C" int test_frees() { return frees; }
extern "C" int test_context_frees() { return context_frees; }
extern "C" int test_live() { return outstanding!=nullptr; }
extern "C" int test_null_model() { return null_model; }
extern "C" int test_byte(int index) { return seen.at(index); }
int ce_resolve_threads(int threads) { return threads; }
extern "C" table_parse_context * table_parse_init(const char * path,int) {
    null_model=path==nullptr; return reinterpret_cast<table_parse_context *>(1);
}
extern "C" void table_parse_free(table_parse_context *) { ++context_frees; }
extern "C" char * table_parse_to_html(table_parse_context *,const uint8_t * pixels,int w,int h) {
    ++calls;
    if (w<=0 || h<=0 || w>INT_MAX-31 || w>INT_MAX/h || length<w*h || mode==1) return nullptr;
    seen.assign(pixels,pixels+w*h);
    const char * text=mode==2 ? "" : mode==3 ? "\xff" : "<table><tr><td>日本語 €</td></tr></table>";
    outstanding=std::malloc(std::strlen(text)+1);
    std::memcpy(outstanding,text,std::strlen(text)+1);
    return static_cast<char *>(outstanding);
}
static void audited_free(void * pointer) {
    if (!pointer) return;
    if (pointer!=outstanding) std::abort();
    ++frees; std::free(pointer); outstanding=nullptr;
}
#define free audited_free
'''
        grid_sink = r'''
static Grid detect_grid(const uint8_t * pixels,int w,int h) {
    ++calls; Grid grid;
    if (w<=0 || h<=0 || w>INT_MAX-31 || w>INT_MAX/h || length<w*h || mode==1) return grid;
    seen.assign(pixels,pixels+w*h);
    grid.row_bounds={0,1,2}; grid.col_bounds={0,1,2,3}; return grid;
}
'''
        directory = Path(directory)
        cpp, library = directory / "native.cpp", directory / "native.so"
        cpp.write_text(harness + real_free + '\n#undef free\n' + grid + grid_sink + real_grid + wrappers)
        subprocess.run([compiler, "-std=c++17", "-fPIC", "-shared", "-I", str(ROOT / "src"),
                        str(cpp), "-o", str(library)], check=True, text=True, timeout=30)
        return library

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="table-parse-interface-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.lib = ctypes.CDLL(str(cls.build_fixture(cls.temp.name)))
        cls.lib.test_reset.argtypes = [ctypes.c_int, ctypes.c_int]
        cls.lib.test_byte.argtypes = [ctypes.c_int]

    def setUp(self):
        self.lib.test_reset(0, 0)
        with patch.object(BINDING, "_load_library", return_value=self.lib):
            self.parser = BINDING.CrispTableParse()
        self.addCleanup(self.parser.__del__)

    def tearDown(self):
        self.assertEqual(self.lib.test_live(), 0, "native HTML allocation leaked")

    def test_flat_extra_dtype_and_strided_pixels_keep_native_prefix(self):
        base = np.arange(12, dtype=np.int16).reshape(3, 4)
        for image in (base, base[::-1, ::-1], base.T, np.arange(15, dtype=np.uint8)):
            with self.subTest(shape=image.shape, strides=image.strides):
                self.lib.test_reset(image.size, 0)
                before = image.copy()
                text = self.parser.to_html(image, 4, 3)
                self.assertEqual(text, "<table><tr><td>日本語 €</td></tr></table>")
                expected = image.flatten().astype(np.uint8)[:12].tolist()
                self.assertEqual([self.lib.test_byte(i) for i in range(12)], expected)
                np.testing.assert_array_equal(image, before)
                self.assertEqual(self.lib.test_frees(), 1)

    def test_invalid_geometry_or_short_input_never_reaches_native(self):
        for w, h, size in ((0, 1, 0), (1, 0, 0), (-1, 1, 0), (-1, -1, 1),
                           (2**31, 1, 0), (2**31-1, 1, 0), (65536, 65536, 0),
                           (1.5, 1, 3), (1, 1.5, 3), (4, 3, 11)):
            with self.subTest(width=w, height=h, size=size):
                with self.assertRaises(ValueError):
                    self.parser.to_html(np.zeros(size, dtype=np.uint8), w, h)
                self.assertEqual(self.lib.test_calls(), 0)
        # Width padding (w+31) must reject before converting/copying an enormous image.
        with patch.object(BINDING.np, "asarray", side_effect=AssertionError("copy before width guard")):
            with self.assertRaises(ValueError):
                self.parser.to_html(object(), 2**31-1-30, 1)

    def test_empty_null_and_decode_failure_match_free_contract(self):
        for mode in (1, 2, 3):
            with self.subTest(mode=mode):
                self.lib.test_reset(12, mode)
                if mode == 1:
                    with self.assertRaisesRegex(RuntimeError, "Table parsing failed"):
                        self.parser.to_html(np.zeros(12, dtype=np.uint8), 4, 3)
                elif mode == 2:
                    self.assertEqual(self.parser.to_html(np.zeros(12, dtype=np.uint8), 4, 3), "")
                else:
                    with self.assertRaises(UnicodeDecodeError):
                        self.parser.to_html(np.zeros(12, dtype=np.uint8), 4, 3)
                self.assertEqual(self.lib.test_frees(), 0 if mode == 1 else 1)

    def test_copy_exception_still_uses_matching_free(self):
        self.lib.test_reset(12, 0)
        real_cast = ctypes.cast
        def fail_output_copy(pointer, target):
            if target is ctypes.c_char_p:
                raise MemoryError("output copy failed")
            return real_cast(pointer, target)
        with patch.object(BINDING.ctypes, "cast", side_effect=fail_output_copy):
            with self.assertRaises(MemoryError):
                self.parser.to_html(np.zeros(12, dtype=np.uint8), 4, 3)
        self.assertEqual(self.lib.test_frees(), 1)

    def test_context_cleanup_is_idempotent(self):
        self.parser.__del__()
        self.parser.__del__()
        self.assertEqual(self.lib.test_context_frees(), 1)

    def test_model_path_nul_before_load_and_none_empty_defaults(self):
        for path in ("bad\0path", b"bad\0path"):
            with patch.object(BINDING, "_load_library", side_effect=AssertionError("native load")):
                with self.assertRaises(ValueError):
                    BINDING.CrispTableParse(path)
        for path in (None, "", Path("ocr.gguf")):
            with self.subTest(path=path):
                with patch.object(BINDING, "_load_library", return_value=self.lib):
                    parser = BINDING.CrispTableParse(path)
                self.addCleanup(parser.__del__)
                self.assertEqual(self.lib.test_null_model(), int(path in (None, "")))


if __name__ == "__main__":
    unittest.main()
