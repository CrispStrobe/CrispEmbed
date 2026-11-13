#!/usr/bin/env python3
"""One-shot renderer ownership tests using actual C wrappers and public headers.

The tiny native fixture replaces the renderer backend and allocator to force
failures; it compiles no models or ggml. Public Python/Rust methods run unchanged.
"""

import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_cc_detect_ownership import BINDING, extract_function

ROOT = Path(__file__).resolve().parents[1]


def native_fixture(directory):
    compiler = shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise unittest.SkipTest("C++ compiler required for tiny render interface fixture")
    source = (ROOT / "src/crispembed.cpp").read_text()
    functions = "\n".join(extract_function(source, prefix) for prefix in (
        'extern "C" char * crispembed_ocr_render(',
        'extern "C" void crispembed_ocr_render_free('))
    harness = r'''
#define CRISPEMBED_BUILD
#include "crispembed.h"
#include "ocr_render.h"
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
struct ocr_renderer { std::string output; };
static int mode = 0, lifecycle_frees = 0, output_frees = 0, calls = 0, selected_format = -1;
static void * outstanding = nullptr;
extern "C" CRISPEMBED_API void test_reset(int value) {
    mode=value; lifecycle_frees=output_frees=calls=0; selected_format=-1;
}
extern "C" CRISPEMBED_API int test_lifecycle_frees() { return lifecycle_frees; }
extern "C" CRISPEMBED_API int test_output_frees() { return output_frees; }
extern "C" CRISPEMBED_API int test_calls() { return calls; }
extern "C" CRISPEMBED_API int test_format() { return selected_format; }
extern "C" CRISPEMBED_API int test_live() { return outstanding != nullptr; }
extern "C" ocr_renderer * ocr_render_create(ocr_render_format format) {
    ++calls; selected_format = (int)format; return new ocr_renderer;
}
extern "C" void ocr_render_begin(ocr_renderer *) {}
extern "C" void ocr_render_add_page(ocr_renderer * r, const ocr_render_page * page) {
    for (int i=0; i<page->n_lines; ++i) {
        if (i) r->output += "|";
        r->output += page->lines[i].words[0].text;
    }
    if (mode == 3) r->output = "\xff";
    if (mode == 4) r->output.clear();
}
extern "C" void ocr_render_end(ocr_renderer *) {}
extern "C" const char * ocr_render_error(const ocr_renderer *) {
    return mode == 1 ? "forced renderer error" : nullptr;
}
extern "C" int ocr_render_output_size(const ocr_renderer * r) { return (int)r->output.size(); }
extern "C" const char * ocr_render_output(const ocr_renderer * r) { return r->output.c_str(); }
extern "C" void ocr_render_free(ocr_renderer * r) { ++lifecycle_frees; delete r; }
static void * test_malloc(size_t size) {
    if (mode == 2) return nullptr;
    if (outstanding) std::abort();
    return outstanding = std::malloc(size);
}
static void test_free(void * ptr) {
    if (!ptr) return;
    if (ptr != outstanding) std::abort();
    ++output_frees; std::free(ptr); outstanding = nullptr;
}
#define malloc test_malloc
#define free test_free
'''
    cpp = Path(directory) / "fixture.cpp"
    cpp.write_text(harness + functions)
    library = Path(directory) / ("render_fixture.dylib" if sys.platform == "darwin" else "render_fixture.so")
    subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden",
                    "-I", str(ROOT / "src"), str(cpp), "-o", str(library)],
                   check=True, text=True, timeout=30)
    return library


class RenderOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="render-interface-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.library = native_fixture(cls.temp.name)
        cls.lib = ctypes.CDLL(str(cls.library))
        cls.lib.test_reset.argtypes = [ctypes.c_int]
        for name in ("test_lifecycle_frees", "test_output_frees", "test_calls", "test_live", "test_format"):
            getattr(cls.lib, name).restype = ctypes.c_int
        cls.lib.crispembed_ocr_render_free.argtypes = [ctypes.c_void_p]
        cls.lib.crispembed_ocr_render_free.restype = None

    def setUp(self):
        self.lib.test_reset(0)
        self.rows = [dict(text=text, x=i*100+1, y=i*100+2, w=30, h=40)
                     for i, text in enumerate(("Hello", "日本語 €", "Grüße"))]

    def tearDown(self):
        self.assertEqual(self.lib.test_live(), 0, "render output leaked")

    def render(self, **kwargs):
        with patch.object(BINDING, "_load_library", return_value=self.lib):
            return BINDING.ocr_render(self.rows, 640, 480, **kwargs)

    def test_unicode_output_is_owned_and_library_freed_without_pipeline_symbols(self):
        with patch.object(ctypes.cdll, "LoadLibrary", side_effect=AssertionError("foreign allocator")):
            result = self.render()
        self.assertEqual(result, "Hello|日本語 €|Grüße")
        self.assertEqual(self.lib.test_output_frees(), 1)
        self.assertEqual(self.lib.test_lifecycle_frees(), 1)
        self.assertEqual(self.render(), result)
        self.assertEqual(result, "Hello|日本語 €|Grüße")

    def test_format_dispatch_preserved(self):
        for format, code in (("text", 0), ("hocr", 1), ("alto", 2), ("pdf", 3)):
            with self.subTest(format=format):
                self.lib.test_reset(0)
                self.assertEqual(self.render(format=format), "Hello|日本語 €|Grüße")
                self.assertEqual(self.lib.test_format(), code)
                self.assertEqual(self.lib.test_output_frees(), 1)

    def test_backend_and_output_allocation_failures_cleanup(self):
        for mode in (1, 2):
            with self.subTest(mode=mode):
                self.lib.test_reset(mode)
                self.assertEqual(self.render(), "")
                self.assertEqual(self.lib.test_lifecycle_frees(), 1)
                self.assertEqual(self.lib.test_output_frees(), 0)

    def test_empty_output_is_freed_and_invalid_utf8_keeps_replacement_behavior(self):
        for mode, expected in ((3, "\ufffd"), (4, "")):
            with self.subTest(mode=mode):
                self.lib.test_reset(mode)
                self.assertEqual(self.render(), expected)
                self.assertEqual(self.lib.test_output_frees(), 1)

    def test_copy_exception_still_frees_output(self):
        with patch.object(BINDING.ctypes, "cast", side_effect=ValueError("copy failed")):
            with self.assertRaisesRegex(ValueError, "copy failed"):
                self.render()
        self.assertEqual(self.lib.test_output_frees(), 1)

    def test_empty_input_needs_no_native_library(self):
        with patch.object(BINDING, "_load_library", side_effect=AssertionError("unneeded native load")):
            self.assertEqual(BINDING.ocr_render([], 640, 480), "")

    def test_older_runtime_fails_before_rendering(self):
        render = Mock()
        old = SimpleNamespace(crispembed_ocr_render=render)
        with patch.object(BINDING, "_load_library", return_value=old):
            with self.assertRaisesRegex(RuntimeError, "update the native library"):
                BINDING.ocr_render(self.rows, 640, 480)
        render.assert_not_called()

    def test_actual_rust_renderer_keeps_owned_unicode_and_matching_free(self):
        compiler = shutil.which("rustc")
        if not compiler:
            self.skipTest("rustc required for focused Rust interface test")
        raw = (ROOT / "crispembed-sys/src/lib.rs").read_text()
        safe = (ROOT / "crispembed/src/lib.rs").read_text()
        result = extract_function(raw, "pub struct CrispembedOcrResult {")
        region = extract_function(safe, "pub struct OcrResult {")
        region += "\n" + next(line for line in safe.splitlines()
                                if line.startswith("pub type OcrRegion ="))
        method = extract_function(safe, "pub fn ocr_render(results:")
        start = raw.index("    pub fn crispembed_ocr_render(")
        end = raw.index(";", raw.index("pub fn crispembed_ocr_render_free", start)) + 1
        harness = r"""
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
    fn test_output_frees() -> i32;
    fn test_lifecycle_frees() -> i32;
    fn test_calls() -> i32;
    fn test_live() -> i32;
}
fn main() {
    let rows: Vec<OcrRegion> = ["Hello", "日本語 €", "Grüße"].iter()
        .map(|text| OcrRegion { text: text.to_string(), x: 1.0, y: 2.0,
            w: 30.0, h: 40.0, confidence: 0.75, orientation_corrected: false }).collect();
    unsafe { test_reset(0); }
    let owned = ocr_render(&rows, 640, 480, "text").unwrap();
    assert_eq!(owned, "Hello|日本語 €|Grüße");
    unsafe {
        assert_eq!(test_live(), 0);
        assert_eq!(test_output_frees(), 1);
        assert_eq!(test_lifecycle_frees(), 1);
        for mode in [1, 2] {
            test_reset(mode);
            assert!(ocr_render(&rows, 640, 480, "text").is_none());
            assert_eq!(test_live(), 0);
            assert_eq!(test_output_frees(), 0);
            assert_eq!(test_lifecycle_frees(), 1);
        }
        for (mode, expected) in [(3, "\u{FFFD}"), (4, "")] {
            test_reset(mode);
            assert_eq!(ocr_render(&rows, 640, 480, "text").unwrap(), expected);
            assert_eq!(test_live(), 0);
            assert_eq!(test_output_frees(), 1);
        }
        test_reset(0);
        assert!(ocr_render(&[], 640, 480, "text").is_none());
        assert_eq!(test_calls(), 0);
    }
    assert_eq!(owned, "Hello|日本語 €|Grüße");
}
"""
        for key, value in (("RESULT", result), ("DECLARATIONS", raw[start:end]),
                           ("REGION", region), ("METHOD", method)):
            harness = harness.replace(key, value)
        directory = Path(self.temp.name)
        source = directory / "render_method.rs"
        binary = directory / "render_method"
        source.write_text(harness)
        subprocess.run([compiler, "--edition=2021", str(source), "-o", str(binary),
                        "-C", "link-arg=" + str(self.library),
                        "-C", "link-arg=-Wl,-rpath," + str(directory)],
                       check=True, text=True, timeout=30)
        subprocess.run([str(binary)], check=True, text=True, timeout=10)

    def test_native_null_output_free_is_safe(self):
        self.lib.crispembed_ocr_render_free(None)
        self.assertEqual(self.lib.test_output_frees(), 0)


if __name__ == "__main__":
    unittest.main()
