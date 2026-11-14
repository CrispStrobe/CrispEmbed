#!/usr/bin/env python3
"""Same-size restoration RGB input and matching-free controls using public C wrappers."""

import ctypes as C
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

from test_cc_detect_ownership import extract_function
from test_preprocess_bindings import BINDING

ROOT = Path(__file__).resolve().parents[1]
ENGINES = (("restormer", BINDING.CrispRestormer), ("adair", BINDING.CrispAdaIR),
           ("instructir", BINDING.CrispInstructIR))


class SameSizeRestorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.temp = tempfile.TemporaryDirectory(prefix="same-size-restoration-")
        cls.addClassCleanup(cls.temp.cleanup)
        native = (ROOT / "src/crispembed.cpp").read_text()
        wrappers = []
        for family in ("restormer", "adair", "instructir"):
            for result, suffix in (("void *", "init"), ("void", "free"),
                                   ("int", "process"), ("void", "free_image")):
                body = extract_function(native, f'extern "C" {result} crispembed_{family}_{suffix}(')
                if suffix == "free_image":
                    body = body.replace("{", "{ ++image_free_calls;", 1)
                wrappers.append(body)
        wrappers.append(extract_function(native, 'extern "C" int crispembed_instructir_n_tasks('))
        harness = r'''
#include "crispembed.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>
struct restormer_context {};
struct adair_context {};
struct instructir_context {};
static restormer_context restormer;
static adair_context adair;
static instructir_context instructir;
static std::set<void *> live;
static std::vector<uint8_t> seen;
static int mode=0, calls=0, allocations=0, frees=0, image_free_calls=0, context_frees=0;
static int tile_seen=0, overlap_seen=0, input_length=0, task_seen=0;
static int ce_resolve_threads(int n) { return n; }
static restormer_context * restormer_init(const char *,int) { return &restormer; }
static adair_context * adair_init(const char *,int) { return &adair; }
static instructir_context * instructir_init(const char *,int) { return &instructir; }
static int instructir_get_n_tasks(const instructir_context *) { return 7; }
static void instructir_free(instructir_context *) { ++context_frees; }
static void restormer_free(restormer_context *) { ++context_frees; }
static void adair_free(adair_context *) { ++context_frees; }
static void * test_alloc(size_t count) {
    if (mode==4) return nullptr;
    void * p=std::malloc(count); if (!p) std::abort();
    live.insert(p); ++allocations; return p;
}
static void test_release(void * p) {
    if (!p) return;
    if (live.erase(p)!=1) std::abort();
    ++frees; std::memset(p,255,3); std::free(p);
}
static int capture(const uint8_t * p,int w,int h) {
    ++calls;
    // Keep fault probes safe even if a future Python regression bypasses preflight.
    if (!p || w<=0 || h<=0 || w>INT_MAX/h || w*h>INT_MAX/3 || input_length<w*h*3) return -1;
    seen.assign(p,p+w*h*3); return 0;
}
static int restormer_process(restormer_context *,const uint8_t * p,int w,int h,int tile,int overlap,uint8_t ** out) {
    tile_seen=tile; overlap_seen=overlap;
    if (capture(p,w,h) || mode==1) return -1;
    if (mode==2) return 0;
    auto * result=static_cast<uint8_t *>(test_alloc(w*h*3));
    if (!result) return -1;
    for (int i=0;i<w*h*3;++i) result[i]=uint8_t(i);
    *out=result; return mode==3?-1:0;
}
static void restormer_free_image(uint8_t * p) { test_release(p); }
static int adair_process(adair_context *,const uint8_t * p,int w,int h,uint8_t * out) {
    if (capture(p,w,h) || mode==1 || mode==3) return -1;
    for (int i=0;i<w*h*3;++i) out[i]=uint8_t(i);
    return 0;
}
static int instructir_process(instructir_context *,int task,const uint8_t * p,int w,int h,uint8_t * out) {
    task_seen=task;
    if (task<0 || task>=7) { ++calls; return -1; }
    return adair_process(&adair,p,w,h,out);
}
extern "C" int test_task() { return task_seen; }
extern "C" void test_reset(int length,int value) {
    if (!live.empty()) std::abort();
    calls=allocations=frees=image_free_calls=context_frees=0; seen.clear();
    input_length=length; mode=value;
}
extern "C" int test_calls() { return calls; }
extern "C" int test_allocations() { return allocations; }
extern "C" int test_frees() { return frees; }
extern "C" int test_image_frees() { return image_free_calls; }
extern "C" int test_context_frees() { return context_frees; }
extern "C" int test_live() { return int(live.size()); }
extern "C" int test_byte(int at) { return seen.at(at); }
extern "C" int test_tile() { return tile_seen; }
extern "C" int test_overlap() { return overlap_seen; }
#define malloc test_alloc
#define free test_release
'''
        cpp, library = Path(cls.temp.name)/"native.cpp", Path(cls.temp.name)/"native.so"
        cpp.write_text(harness + "\n".join(wrappers))
        subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(ROOT / "src"),
                        str(cpp), "-o", str(library)], check=True, timeout=30)
        cls.lib = C.CDLL(str(library))
        cls.lib.test_reset.argtypes = [C.c_int, C.c_int]
        cls.lib.test_byte.argtypes = [C.c_int]

    def create(self, constructor, length=18, mode=0):
        self.lib.test_reset(length, mode)
        with patch.object(BINDING, "_load_library", return_value=self.lib):
            model = constructor("fixture.gguf")
        self.addCleanup(model.__del__)
        return model

    def tearDown(self):
        self.assertEqual(self.lib.test_live(), 0)
        self.assertEqual(self.lib.test_allocations(), self.lib.test_frees())

    def test_rgb_prefix_dtype_strides_and_owned_output(self):
        base = np.arange(36, dtype=np.int16).reshape(2, 6, 3)-4
        arrays = (np.arange(18, dtype=np.uint8), np.arange(23, dtype=np.uint8),
                  np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
                  base[:, ::2], base[::-1, ::-2])
        for _, constructor in ENGINES:
            for pixels in arrays:
                with self.subTest(engine=constructor.__name__, shape=pixels.shape, strides=pixels.strides):
                    before = pixels.copy()
                    model = self.create(constructor, length=pixels.size)
                    out = model.process(pixels, 3, 2)
                    np.testing.assert_array_equal(out, np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
                    self.assertEqual(out.dtype, np.uint8)
                    self.assertEqual(self.lib.test_image_frees(), 1)
                    self.assertEqual([self.lib.test_byte(i) for i in range(18)],
                                     pixels.astype(np.uint8).flatten()[:18].tolist())
                    np.testing.assert_array_equal(pixels, before)

    def test_invalid_input_never_reaches_native_or_array_conversion(self):
        invalid_geometry = ((0, 1), (1, 0), (-1, 1), (1, -1), (-1, -1),
                            (2**31, 1), (1, 2**31), (2**30, 1), (65536, 65536), (1.5, 1), (1, 1.5))
        for _, constructor in ENGINES:
            model = self.create(constructor)
            for width, height in invalid_geometry:
                with patch.object(BINDING.np, "asarray", side_effect=AssertionError("conversion before guard")):
                    with self.assertRaises(ValueError):
                        model.process([], width, height)
            for count in (0, 1, 17):
                with self.assertRaises(ValueError):
                    model.process(np.zeros(count, dtype=np.uint8), 3, 2)
            self.assertEqual(self.lib.test_calls(), 0)

    def test_native_error_allocation_ownership_and_null_results(self):
        for family, constructor in ENGINES:
            for mode in (1, 3, 4):
                with self.subTest(family=family, mode=mode):
                    model = self.create(constructor, mode=mode)
                    with self.assertRaises(RuntimeError):
                        model.process(np.zeros(18, dtype=np.uint8), 3, 2)
                    self.assertEqual(self.lib.test_live(), 0)
                    self.assertEqual(self.lib.test_image_frees(), int(family=="restormer" and mode==3))
            model = self.create(constructor)
            model.__del__(); model.__del__()
            self.assertEqual(self.lib.test_context_frees(), 1)
        model = self.create(BINDING.CrispRestormer, mode=2)
        with self.assertRaises(RuntimeError):
            model.process(np.zeros(18, dtype=np.uint8), 3, 2)
        self.assertEqual(self.lib.test_image_frees(), 0)

    def test_defensive_nonnull_error_result_uses_matching_free(self):
        # AdaIR/InstructIR currently free lower-error output internally. Inject a changed
        # status after public success to also check defensive Python cleanup.
        for family, constructor in ENGINES:
            model = self.create(constructor)
            name = "crispembed_"+family+"_process"
            native = getattr(self.lib, name)
            def allocated_error(*args):
                self.assertEqual(native(*args), 0)
                return -1
            with patch.object(self.lib, name, allocated_error):
                with self.assertRaises(RuntimeError):
                    model.process(np.zeros(18, dtype=np.uint8), 3, 2)
            self.assertEqual(self.lib.test_image_frees(), 1)
            self.assertEqual(self.lib.test_live(), 0)

    def test_matching_free_on_array_view_copy_and_reshape_exceptions(self):
        for _, constructor in ENGINES:
            for stage in ("view", "copy", "reshape"):
                with self.subTest(engine=constructor.__name__, stage=stage):
                    model = self.create(constructor)
                    if stage == "view":
                        args = {"side_effect": MemoryError("view")}
                    elif stage == "copy":
                        args = {"return_value": SimpleNamespace(copy=Mock(side_effect=MemoryError("copy")))}
                    else:
                        args = {"return_value": SimpleNamespace(copy=lambda: SimpleNamespace(
                            reshape=Mock(side_effect=MemoryError("reshape"))))}
                    with patch.object(np.ctypeslib, "as_array", **args):
                        with self.assertRaises(MemoryError):
                            model.process(np.zeros(18, dtype=np.uint8), 3, 2)
                    self.assertEqual(self.lib.test_image_frees(), 1)
                    self.assertEqual(self.lib.test_live(), 0)

    def test_instructir_tasks_and_native_invalid_task_behavior_are_preserved(self):
        for task in range(7):
            model = self.create(BINDING.CrispInstructIR)
            self.assertEqual(model.n_tasks, 7)
            out = model.process(np.zeros(18, dtype=np.uint8), 3, 2, task=task)
            self.assertEqual(self.lib.test_task(), task)
            np.testing.assert_array_equal(out.flatten(), np.arange(18, dtype=np.uint8))
            self.assertEqual(self.lib.test_image_frees(), 1)
        for task in (-1, 7):
            model = self.create(BINDING.CrispInstructIR)
            with self.assertRaisesRegex(RuntimeError, "InstructIR processing failed"):
                model.process(np.zeros(18, dtype=np.uint8), 3, 2, task=task)
            self.assertEqual(self.lib.test_task(), task)
            self.assertEqual(self.lib.test_calls(), 1)
            self.assertEqual(self.lib.test_image_frees(), 0)
            self.assertEqual(self.lib.test_frees(), 1)

    def test_restormer_tile_values_are_forwarded_unchanged(self):
        for tile, overlap in ((0, 0), (-1, -2), (64, 8), (1, 1)):
            model = self.create(BINDING.CrispRestormer)
            model.process(np.zeros(18, dtype=np.uint8), 3, 2, tile_size=tile, tile_overlap=overlap)
            self.assertEqual((self.lib.test_tile(), self.lib.test_overlap()), (tile, overlap))


if __name__ == "__main__":
    unittest.main()
