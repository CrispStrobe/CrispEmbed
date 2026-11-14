#!/usr/bin/env python3
"""Actual DAT tiling/output and Python ownership with a nearest-neighbor oracle."""

import ctypes as C
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from test_preprocess_bindings import BINDING

import numpy as np

from test_cc_detect_ownership import extract_function

ROOT = Path(__file__).resolve().parents[1]
GATE = "CRISPEMBED_DAT_POSITIVE_TILE_BLEND"


class DatTileBlendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.temp = tempfile.TemporaryDirectory(prefix="dat-tile-oracle-")
        cls.addClassCleanup(cls.temp.cleanup)
        native = (ROOT / "src/dat_sr.cpp").read_text()
        public = (ROOT / "src/crispembed.cpp").read_text()
        helpers = "\n".join(extract_function(native, prefix) for prefix in (
            "static float dat_tile_axis_weight(", "static int dat_round_extent("))
        process = extract_function(native, "int dat_sr_process(")
        process = process.replace("    // Normalize and convert to uint8", """
    zero_weights=small_weights=0;
    min_weight=*std::min_element(weight.begin(),weight.end());
    for (float value:weight) {
        if (value==0) ++zero_weights;
        if (value>0 && value<1e-6f) ++small_weights;
    }
    // Normalize and convert to uint8""", 1)
        harness = r'''
#include "crispembed.h"
#include "dat_sr.h"
#include "core/env_gate.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <utility>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
struct dat_sr_context { bool bench=false; int upscale=2; int split_size[2]={8,32}; };
static dat_sr_context ctx;
static int forward_calls=0, zero_weights=0, small_weights=0;
static float min_weight=0;
static std::set<void *> output_live;
static int output_allocs=0, output_frees=0, image_free_calls=0, context_frees=0;
static bool fail_alloc=false;
static int ce_resolve_threads(int n) { return n; }
dat_sr_context * dat_sr_init(const char *,int) { return &ctx; }
void dat_sr_free(dat_sr_context *) { ++context_frees; }
static void * test_alloc(size_t count) {
    if (fail_alloc) return nullptr;
    void * p=std::malloc(count); if (!p) std::abort();
    output_live.insert(p); ++output_allocs; return p;
}
static void test_release(void * p) {
    if (!p) return;
    if (output_live.erase(p)!=1) std::abort();
    ++output_frees; std::memset(p,255,3); std::free(p);
}
static void dat_forward(dat_sr_context * c,const float * in,int h,int w,float * out) {
    ++forward_calls;
    int r=c->upscale;
    for(int ch=0;ch<3;ch++) for(int y=0;y<h*r;y++) for(int x=0;x<w*r;x++)
        out[ch*h*r*w*r+y*w*r+x]=in[ch*h*w+(y/r)*w+x/r];
}
extern "C" void * test_context(int scale,int split) {
    if (!output_live.empty()) std::abort();
    output_allocs=output_frees=image_free_calls=context_frees=0; fail_alloc=false;
    ctx.upscale=scale; ctx.split_size[0]=std::min(8,split); ctx.split_size[1]=split;
    forward_calls=zero_weights=small_weights=0; min_weight=0; return &ctx;
}
extern "C" int test_allocations() { return output_allocs; }
extern "C" int test_frees() { return output_frees; }
extern "C" int test_live() { return int(output_live.size()); }
extern "C" int test_image_frees() { return image_free_calls; }
extern "C" int test_context_frees() { return context_frees; }
extern "C" void test_fail_alloc(int value) { fail_alloc=value; }
extern "C" int test_calls() { return forward_calls; }
extern "C" int test_zero_weights() { return zero_weights; }
extern "C" int test_small_weights() { return small_weights; }
extern "C" float test_min_weight() { return min_weight; }
#define malloc test_alloc
#define free test_release
'''
        probes = r'''
extern "C" float test_axis(int i,int n,int origin,int image,int overlap) {
    return dat_tile_axis_weight(i,n,origin,image,overlap);
}
extern "C" float test_legacy_axis(int i,int n) {
    return 0.5f*(1.0f-cosf(2.0f*(float)M_PI*i/(n-1)));
}
extern "C" int test_round(int64_t value,int split) { return dat_round_extent(value,split); }
'''
        wrappers = "\n".join(extract_function(public, prefix) for prefix in (
            'extern "C" void * crispembed_dat_sr_init(',
            'extern "C" void crispembed_dat_sr_free(',
            'extern "C" int crispembed_dat_sr_process(',
            'extern "C" void crispembed_dat_sr_free_image('))
        wrappers = wrappers.replace('void crispembed_dat_sr_free_image(uint8_t * pixels) {',
                                    'void crispembed_dat_sr_free_image(uint8_t * pixels) { ++image_free_calls;')
        cpp, library = Path(cls.temp.name)/"native.cpp", Path(cls.temp.name)/"native.so"
        cpp.write_text(harness+helpers+process+extract_function(native, "void dat_sr_free_image(")+"\n#undef malloc\n#undef free\n"+probes+wrappers)
        subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(ROOT / "src"),
                        str(cpp), "-o", str(library)], check=True, timeout=30)
        cls.lib = C.CDLL(str(library))
        cls.lib.test_fail_alloc.argtypes = [C.c_int]
        cls.lib.test_context.argtypes = [C.c_int, C.c_int]
        cls.lib.test_context.restype = C.c_void_p
        cls.lib.test_min_weight.restype = C.c_float
        cls.lib.test_axis.argtypes = [C.c_int]*5
        cls.lib.test_axis.restype = C.c_float
        cls.lib.test_legacy_axis.argtypes = [C.c_int, C.c_int]
        cls.lib.test_legacy_axis.restype = C.c_float
        cls.lib.test_round.argtypes = [C.c_int64, C.c_int]
        cls.lib.crispembed_dat_sr_process.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int,
            C.c_int, C.c_int, C.POINTER(C.POINTER(C.c_uint8)), C.POINTER(C.c_int), C.POINTER(C.c_int)]
        cls.lib.crispembed_dat_sr_free_image.argtypes = [C.POINTER(C.c_uint8)]

    def tearDown(self):
        self.assertEqual(self.lib.test_live(), 0)
        self.assertEqual(self.lib.test_allocations(), self.lib.test_frees())

    def create_python(self, scale=2):
        self.lib.test_context(scale, 32)
        with patch.object(BINDING, "_load_library", return_value=self.lib):
            model = BINDING.CrispDatSr("fixture.gguf")
        self.addCleanup(model.__del__)
        return model

    def run_image(self, image, scale=2, split=32, tile=(0, 0), gate="1"):
        height, width, _ = image.shape
        context = self.lib.test_context(scale, split)
        out = C.POINTER(C.c_uint8)()
        ow, oh = C.c_int(), C.c_int()
        with patch.dict(os.environ):
            os.environ.pop(GATE, None)
            if gate is not None:
                os.environ[GATE] = gate
            rc = self.lib.crispembed_dat_sr_process(context, image.ctypes.data_as(C.POINTER(C.c_uint8)),
                width, height, *tile, C.byref(out), C.byref(ow), C.byref(oh))
        try:
            result = None if not out else np.ctypeslib.as_array(out, shape=(oh.value*ow.value*3,)).copy().reshape(
                oh.value, ow.value, 3)
            return rc, result
        finally:
            if out:
                self.lib.crispembed_dat_sr_free_image(out)

    def test_python_rgb_prefix_dtype_strides_and_returned_scale(self):
        base = np.arange(36, dtype=np.int16).reshape(2,6,3)-4
        arrays = (np.arange(18, dtype=np.uint8), np.arange(23, dtype=np.uint8),
                  np.arange(24, dtype=np.uint8).reshape(2,3,4), base[:,::2], base[::-1,::-2])
        for scale in (1,2,3,4):
            for pixels in arrays:
                with self.subTest(scale=scale, shape=pixels.shape, strides=pixels.strides):
                    model = self.create_python(scale)
                    before = pixels.copy()
                    out, width, height = model.process(pixels,3,2)
                    expected = pixels.astype(np.uint8).flatten()[:18].reshape(2,3,3)
                    np.testing.assert_array_equal(out, expected.repeat(scale,0).repeat(scale,1))
                    self.assertEqual((width,height,out.dtype), (3*scale,2*scale,np.uint8))
                    self.assertEqual(self.lib.test_image_frees(),1)
                    np.testing.assert_array_equal(pixels,before)

    def test_python_invalid_input_precedes_conversion_and_native_call(self):
        model = self.create_python()
        invalid = ((0,1),(1,0),(-1,1),(1,-1),(-1,-1),(2**31,1),(1,2**31),
                   (2**30,1),(65536,65536),(1.5,1),(1,1.5))
        with patch.object(self.lib,"crispembed_dat_sr_process", side_effect=AssertionError("native before guard")):
            for width,height in invalid:
                with patch.object(BINDING.np,"asarray",side_effect=AssertionError("conversion before guard")):
                    with self.assertRaises(ValueError):
                        model.process([],width,height)
            for count in (0,1,17):
                with self.assertRaises(ValueError):
                    model.process(np.zeros(count,dtype=np.uint8),3,2)
        self.assertEqual(self.lib.test_calls(),0)

    def test_python_native_errors_null_and_malformed_output_dimensions(self):
        for dimensions in ((0,4),(-1,-1),(6,0),(2**31-1,2),(1,2**31-1)):
            model = self.create_python()
            native = self.lib.crispembed_dat_sr_process
            def malformed(*args):
                self.assertEqual(native(*args),0)
                C.cast(args[-2],C.POINTER(C.c_int))[0] = dimensions[0]
                C.cast(args[-1],C.POINTER(C.c_int))[0] = dimensions[1]
                return 0
            with patch.object(self.lib,"crispembed_dat_sr_process",malformed), patch.object(
                    np.ctypeslib,"as_array",side_effect=AssertionError("view before geometry guard")) as read:
                with self.assertRaisesRegex(RuntimeError,"invalid output dimensions"):
                    model.process(np.zeros(18,dtype=np.uint8),3,2)
                read.assert_not_called()
            self.assertEqual(self.lib.test_image_frees(),1)
        model = self.create_python()
        native = self.lib.crispembed_dat_sr_process
        def allocated_error(*args):
            self.assertEqual(native(*args),0)
            return -1
        with patch.object(self.lib,"crispembed_dat_sr_process",allocated_error):
            with self.assertRaisesRegex(RuntimeError,"DAT SR processing failed"):
                model.process(np.zeros(18,dtype=np.uint8),3,2)
        self.assertEqual(self.lib.test_image_frees(),1)
        model = self.create_python()
        self.lib.test_fail_alloc(1)
        with self.assertRaises(RuntimeError):
            model.process(np.zeros(18,dtype=np.uint8),3,2)
        self.assertEqual(self.lib.test_image_frees(),0)
        model = self.create_python()
        with patch.object(self.lib,"crispembed_dat_sr_process",return_value=0):
            with self.assertRaises(RuntimeError):
                model.process(np.zeros(18,dtype=np.uint8),3,2)
        self.assertEqual(self.lib.test_image_frees(),0)

    def test_python_matching_free_on_view_copy_and_reshape_errors(self):
        for stage in ("view","copy","reshape"):
            model = self.create_python()
            if stage=="view":
                args = {"side_effect":MemoryError("view")}
            elif stage=="copy":
                args = {"return_value":SimpleNamespace(copy=Mock(side_effect=MemoryError("copy")))}
            else:
                args = {"return_value":SimpleNamespace(copy=lambda:SimpleNamespace(
                    reshape=Mock(side_effect=MemoryError("reshape"))))}
            with patch.object(np.ctypeslib,"as_array",**args):
                with self.assertRaises(MemoryError):
                    model.process(np.zeros(18,dtype=np.uint8),3,2)
            self.assertEqual(self.lib.test_image_frees(),1)
            self.assertEqual(self.lib.test_live(),0)

    def test_python_tile_arguments_and_idempotent_context_free(self):
        for tile in ((0,0),(-1,-2),(32,64),(1,1)):
            model = self.create_python()
            native = self.lib.crispembed_dat_sr_process
            with patch.object(self.lib,"crispembed_dat_sr_process",wraps=native) as call:
                model.process(np.zeros(18,dtype=np.uint8),3,2,tile_w=tile[0],tile_h=tile[1])
                self.assertEqual(call.call_args.args[4:6],tile)
            model.__del__(); model.__del__()
            self.assertEqual(self.lib.test_context_frees(),1)

    def test_python_preserves_gated_tiled_output(self):
        pixels = np.full((64,96,3),200,dtype=np.uint8)
        for gate,errors in ((None,1920),("0",1920),("1",0)):
            model = self.create_python()
            with patch.dict(os.environ):
                os.environ.pop(GATE,None)
                if gate is not None:
                    os.environ[GATE] = gate
                out,width,height = model.process(pixels,96,64)
            self.assertEqual((width,height),(192,128))
            self.assertEqual(np.count_nonzero(out!=200),errors)
            self.assertEqual(self.lib.test_image_frees(),1)

    def test_legacy_default_and_off_values_preserve_reproduced_failure(self):
        image = np.full((64, 96, 3), 200, dtype=np.uint8)
        rc, baseline = self.run_image(image, gate=None)
        self.assertEqual(rc, 0)
        self.assertEqual(np.count_nonzero(baseline != 200), 1920)
        self.assertGreater(self.lib.test_zero_weights(), 0)
        self.assertGreater(self.lib.test_small_weights(), 0)
        for value in ("", "0", "false", "OFF", "no"):
            rc, out = self.run_image(image, gate=value)
            self.assertEqual(rc, 0)
            np.testing.assert_array_equal(out, baseline)
        rc, out = self.run_image(image, gate="1")
        self.assertEqual(rc, 0)
        np.testing.assert_array_equal(out, np.full_like(out, 200))
        self.assertEqual(self.lib.test_zero_weights(), 0)
        self.assertGreater(self.lib.test_min_weight(), 0)

    def test_corrected_tiling_preserves_constant_and_nonuniform_oracle(self):
        geometries = ((96,64,(0,0),32), (65,33,(32,32),32), (17,97,(64,32),32),
                      (97,17,(32,64),32), (1,65,(1,1),32), (65,1,(1,1),32),
                      (9,5,(8,8),1), (5,9,(4,4),1), (3,2,(1,1),1))
        for width, height, tile, split in geometries:
            for scale in (1, 2, 3, 4):
                for constant in (True, False):
                    with self.subTest(width=width, height=height, tile=tile, scale=scale, constant=constant):
                        y, x, c = np.indices((height, width, 3))
                        image = np.full((height,width,3), 200, dtype=np.uint8) if constant else (
                            (x*7+y*13+c*53) % 256).astype(np.uint8)
                        before = image.copy()
                        rc, out = self.run_image(image, scale, split, tile)
                        self.assertEqual(rc, 0)
                        np.testing.assert_array_equal(out, image.repeat(scale, 0).repeat(scale, 1))
                        np.testing.assert_array_equal(image, before)
                        self.assertEqual(self.lib.test_zero_weights(), 0)
                        self.assertGreater(self.lib.test_min_weight(), 0)

    def test_single_tile_stays_identical_with_both_modes_and_tiny_axes(self):
        for width, height, split, tile in ((1,1,1,(1,1)), (1,7,32,(0,0)), (17,23,32,(64,64))):
            image = np.arange(height*width*3, dtype=np.uint8).reshape(height,width,3)
            for gate in (None, "1"):
                rc, out = self.run_image(image, split=split, tile=tile, gate=gate)
                self.assertEqual(rc, 0)
                self.assertEqual(self.lib.test_calls(), 1)
                np.testing.assert_array_equal(out, image.repeat(2,0).repeat(2,1))

    def test_axis_weights_preserve_positive_interior_and_cover_degenerate_cases(self):
        for length in (2, 16, 64, 128, 50000):
            for index in sorted({0, 1, length//2, length-2, length-1}):
                legacy = self.lib.test_legacy_axis(index, length)
                weight = self.lib.test_axis(index, length, 0, length, 8)
                self.assertTrue(np.isfinite(weight))
                self.assertGreater(weight, 0)
                if 0 < index < length-1 and legacy > 0:
                    self.assertEqual(weight, legacy)
            self.assertEqual(self.lib.test_axis(0,length,5,length+10,8), 0)
            self.assertEqual(self.lib.test_axis(length-1,length,5,length+10,8), 0)
        for overlap in (0, 1):
            self.assertEqual(self.lib.test_axis(0,1,0,1,overlap), 1)
            self.assertEqual(self.lib.test_axis(4,8,3,20,overlap), 1)

    def test_unsafe_rounding_geometry_and_legacy_zero_step_reject_before_forward(self):
        self.assertEqual(self.lib.test_round(2**31-32,32), 2**31-32)
        self.assertEqual(self.lib.test_round(2**31-31,32), 0)
        pixel = (C.c_uint8*3)()
        for gate in ("0", "1"):
            for width,height,scale,split,tw,th in ((2**31-1,1,1,32,64,64),
                    (32768,32768,1,32,64,64), (1,1,2**31-1,32,64,64),
                    (1,1,2,32,2**31-1,64), (1,1,0,32,64,64), (1,1,2,0,64,64)):
                context = self.lib.test_context(scale,split)
                out = C.POINTER(C.c_uint8)(); ow, oh = C.c_int(), C.c_int()
                with patch.dict(os.environ, {GATE:gate}):
                    rc = self.lib.crispembed_dat_sr_process(context,pixel,width,height,tw,th,
                        C.byref(out),C.byref(ow),C.byref(oh))
                self.assertEqual(rc, -1)
                self.assertFalse(out)
                self.assertEqual(self.lib.test_calls(), 0)
        rc, out = self.run_image(np.zeros((5,9,3),dtype=np.uint8), split=1, tile=(8,8), gate="0")
        self.assertEqual((rc,out), (-1,None))
        self.assertEqual(self.lib.test_calls(), 0)


if __name__ == "__main__":
    unittest.main()
