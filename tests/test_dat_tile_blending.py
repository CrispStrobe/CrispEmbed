#!/usr/bin/env python3
"""Actual DAT tiling/preprocess/output with a model-free nearest-neighbor oracle."""

import ctypes as C
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

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
#include <utility>
#include <vector>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
struct dat_sr_context { bool bench=false; int upscale=2; int split_size[2]={8,32}; };
static dat_sr_context ctx;
static int forward_calls=0, zero_weights=0, small_weights=0;
static float min_weight=0;
static void dat_forward(dat_sr_context * c,const float * in,int h,int w,float * out) {
    ++forward_calls;
    int r=c->upscale;
    for(int ch=0;ch<3;ch++) for(int y=0;y<h*r;y++) for(int x=0;x<w*r;x++)
        out[ch*h*r*w*r+y*w*r+x]=in[ch*h*w+(y/r)*w+x/r];
}
extern "C" void * test_context(int scale,int split) {
    ctx.upscale=scale; ctx.split_size[0]=std::min(8,split); ctx.split_size[1]=split;
    forward_calls=zero_weights=small_weights=0; min_weight=0; return &ctx;
}
extern "C" int test_calls() { return forward_calls; }
extern "C" int test_zero_weights() { return zero_weights; }
extern "C" int test_small_weights() { return small_weights; }
extern "C" float test_min_weight() { return min_weight; }
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
            'extern "C" int crispembed_dat_sr_process(',
            'extern "C" void crispembed_dat_sr_free_image('))
        cpp, library = Path(cls.temp.name)/"native.cpp", Path(cls.temp.name)/"native.so"
        cpp.write_text(harness+helpers+process+extract_function(native, "void dat_sr_free_image(")+probes+wrappers)
        subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(ROOT / "src"),
                        str(cpp), "-o", str(library)], check=True, timeout=30)
        cls.lib = C.CDLL(str(library))
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
