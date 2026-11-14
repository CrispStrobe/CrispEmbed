#!/usr/bin/env python3
"""Real table detector with injected NULL-return morphology allocations."""

import ctypes as C
from pathlib import Path
import tempfile
import unittest

import test_table_parse_binding as table_fixture
from test_cc_detect_ownership import extract_function

ROOT = Path(__file__).resolve().parents[1]


def detector_support():
    source = (ROOT / "src/table_parse.cpp").read_text()
    projections = "\n".join(extract_function(source, prefix).replace(name + "(", name + "_impl(", 1)
                            for prefix, name in (("static std::vector<int> h_projection(", "h_projection"),
                                                 ("static std::vector<int> v_projection(", "v_projection")))
    morphology = (ROOT / "src/morph_fast.cpp").read_text()
    dilation = extract_function(morphology, "uint32_t * morph_dilate_brick(")
    morphology = morphology.replace(dilation, dilation.replace(
        "{", "{\n    if (hsize==5 && vsize==3) live_at_dilate=int(morph_live.size());", 1), 1)
    prelude = r'''
#include "morph_fast.h"
#include <memory>
#include <new>
#include <set>
static std::set<uint32_t *> morph_live;
static int allocation_call=0, fail_allocation=0, mask_frees=0;
static int projection_calls=0, null_projections=0, peak_masks=0, live_at_dilate=-1, throw_projection=0;
static Grid detect_grid(const uint8_t *,int,int);
namespace core_env { static bool on(const char *) { return false; } }
static uint8_t otsu_threshold(const uint8_t *,int,int) { ++calls; return 127; }
extern "C" void test_morph_reset(int fail,int,int throw_at_projection) {
    if (!morph_live.empty()) std::abort();
    allocation_call=mask_frees=projection_calls=null_projections=peak_masks=0;
    fail_allocation=fail; live_at_dilate=-1; throw_projection=throw_at_projection;
}
extern "C" int test_morph_live() { return int(morph_live.size()); }
extern "C" int test_morph_frees() { return mask_frees; }
extern "C" int test_morph_allocations() { return allocation_call; }
extern "C" int test_morph_projections() { return projection_calls; }
extern "C" int test_morph_null_projections() { return null_projections; }
extern "C" int test_morph_peak() { return peak_masks; }
extern "C" int test_morph_live_at_dilate() { return live_at_dilate; }
static void * test_calloc(size_t count,size_t size) {
    ++allocation_call;
    if (allocation_call==fail_allocation) return nullptr;
    auto * pointer=static_cast<uint32_t *>(std::calloc(count,size));
    if (!pointer) std::abort();
    morph_live.insert(pointer); peak_masks=std::max(peak_masks,int(morph_live.size()));
    return pointer;
}
static void test_mask_free(void * pointer) {
    if (!pointer) return;
    if (morph_live.erase(static_cast<uint32_t *>(pointer))!=1) std::abort();
    ++mask_frees; std::free(pointer);
}
#define calloc test_calloc
#define free test_mask_free
'''
    suffix = r'''
static std::vector<int> h_projection(const uint32_t * pointer,int w,int h,int wpl) {
    ++projection_calls;
    if (throw_projection) throw std::bad_alloc();
    if (!pointer) { ++null_projections; return std::vector<int>(h,0); }
    return h_projection_impl(pointer,w,h,wpl);
}
static std::vector<int> v_projection(const uint32_t * pointer,int w,int h,int wpl) {
    ++projection_calls;
    if (!pointer) { ++null_projections; return std::vector<int>(w,0); }
    return v_projection_impl(pointer,w,h,wpl);
}
extern "C" int test_catch_projection_failure() {
    uint8_t pixels[4096]={}; pixels[0]=255;
    try { detect_grid(pixels,64,64); } catch (const std::bad_alloc &) { return 1; }
    return 0;
}
'''
    otsu = extract_function((ROOT / "src/core/cpu_ops.h").read_text(),
                            "static inline uint8_t otsu_threshold(")
    prelude = prelude.replace(
        "static uint8_t otsu_threshold(const uint8_t *,int,int) { ++calls; return 127; }",
        otsu.replace("otsu_threshold(", "scalar_otsu_threshold(", 1) +
        "\nstatic uint8_t otsu_threshold(const uint8_t * p,int w,int h) { ++calls; return scalar_otsu_threshold(p,w*h); }")
    return prelude + morphology + '\n#undef calloc\n#undef free\n' + projections + suffix

class TableParseMorphFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="table-morph-failures-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.lib = C.CDLL(str(table_fixture.TableParseBindingTests.build_fixture(
            cls.temp.name, actual_html=True, detector_support=detector_support())))
        cls.lib.test_reset.argtypes = [C.c_int, C.c_int]
        cls.lib.test_morph_reset.argtypes = [C.c_int, C.c_int, C.c_int]
        cls.lib.crispembed_table_parse_init.argtypes = [C.c_char_p, C.c_int]
        cls.lib.crispembed_table_parse_init.restype = C.c_void_p
        cls.lib.crispembed_table_parse_free.argtypes = [C.c_void_p]
        cls.ctx = cls.lib.crispembed_table_parse_init(None, 4)
        cls.addClassCleanup(cls.lib.crispembed_table_parse_free, cls.ctx)
        cls.lib.crispembed_table_parse_detect_grid.argtypes = [C.POINTER(C.c_uint8), C.c_int, C.c_int,
                                                             C.POINTER(C.c_int), C.POINTER(C.c_int)]
        cls.lib.crispembed_table_parse_to_html.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int]
        cls.lib.crispembed_table_parse_to_html.restype = C.c_void_p
        cls.lib.crispembed_table_parse_free_string.argtypes = [C.c_void_p]

    def setUp(self):
        self.pixels = (C.c_uint8 * 4096)()
        self.reset()

    def reset(self, fail=0, pattern=0, throw=0):
        self.lib.test_reset(4096, 0)
        self.lib.test_morph_reset(fail, pattern, throw)
        for y in range(64):
            for x in range(64):
                ink = (x in (0, 32, 63) or y in (0, 32, 63)) if pattern == 0 else (
                    (8 <= x < 24 or 40 <= x < 56) and (8 <= y < 24 or 40 <= y < 56))
                self.pixels[y*64+x] = 0 if ink else 255

    def tearDown(self):
        self.assertEqual(self.lib.test_morph_live(), 0)
        self.assertEqual(self.lib.test_live(), 0)

    def grid(self, width=64, height=64):
        rows, cols = C.c_int(99), C.c_int(88)
        status = self.lib.crispembed_table_parse_detect_grid(self.pixels, width, height, C.byref(rows), C.byref(cols))
        return status, rows.value, cols.value

    def test_every_actual_calloc_failure_returns_error_and_releases_successful_allocations(self):
        for pattern in (0, 1):
            self.reset(pattern=pattern)
            self.assertEqual(self.grid()[0], 0)
            attempts = self.lib.test_morph_allocations()
            self.assertGreater(attempts, 0)
            for fail in range(1, attempts + 1):
                for html in (False, True):
                    with self.subTest(pattern=pattern, failed_allocation=fail, html=html):
                        self.reset(fail, pattern=pattern)
                        if html:
                            self.assertFalse(self.lib.crispembed_table_parse_to_html(self.ctx, self.pixels, 64, 64))
                        else:
                            self.assertEqual(self.grid(), (-1, 0, 0))
                        # A primitive may try its second buffer after its first fails.
                        self.assertEqual(self.lib.test_morph_frees(), self.lib.test_morph_allocations()-1)
                        self.assertEqual(self.lib.test_morph_null_projections(), 0)
                        self.assertEqual(self.lib.test_morph_projections(),
                                         2 if self.lib.test_morph_live_at_dilate() >= 0 else 0)
                        self.assertEqual(self.lib.test_morph_live(), 0)

    def test_ruled_and_borderless_results_keep_geometry_and_peak_lifetime(self):
        for pattern, size in ((0, 2), (1, 4)):
            with self.subTest(pattern=pattern):
                self.reset(pattern=pattern)
                self.assertEqual(self.grid(), (0, size, size))
                allocations = self.lib.test_morph_allocations()
                self.assertEqual(self.lib.test_morph_frees(), allocations)
                self.assertLessEqual(self.lib.test_morph_peak(), 5)
                self.assertEqual(self.lib.test_morph_live_at_dilate(), 1 if pattern else -1)
                self.reset(pattern=pattern)
                pointer = self.lib.crispembed_table_parse_to_html(self.ctx, self.pixels, 64, 64)
                self.assertTrue(pointer)
                try:
                    self.assertEqual(C.string_at(pointer).decode().count("cell &amp; &lt;"), size*size)
                finally:
                    self.lib.crispembed_table_parse_free_string(pointer)
                self.assertEqual(self.lib.test_morph_live(), 0)
                self.assertEqual(self.lib.test_morph_frees(), allocations)

    def test_uniform_pages_return_no_grid_or_html(self):
        for level in (0, 128, 190, 191, 192, 254, 255):
            for width, height in ((1, 1), (1, 64), (64, 1), (6, 6), (64, 64)):
                with self.subTest(level=level, width=width, height=height):
                    for html in (False, True):
                        self.reset()
                        for i in range(4096):
                            self.pixels[i] = level
                        if html:
                            self.assertFalse(self.lib.crispembed_table_parse_to_html(
                                self.ctx, self.pixels, width, height))
                        else:
                            self.assertEqual(self.grid(width, height), (-1, 0, 0))
                        self.assertEqual(self.lib.test_morph_allocations(), 0)
                        self.assertEqual(self.lib.test_morph_frees(), 0)
                        self.assertEqual(self.lib.test_morph_projections(), 0)

    def test_nonuniform_empty_foreground_returns_no_grid_or_html(self):
        for html in (False, True):
            self.reset()
            for i in range(4096):
                self.pixels[i] = 254 + i % 2
            if html:
                self.assertFalse(self.lib.crispembed_table_parse_to_html(self.ctx, self.pixels, 64, 64))
            else:
                self.assertEqual(self.grid(), (-1, 0, 0))
            self.assertEqual(self.lib.test_morph_allocations(), 1)
            self.assertEqual(self.lib.test_morph_frees(), 1)
            self.assertEqual(self.lib.test_morph_projections(), 0)

    def test_faint_nonempty_ruled_and_borderless_content_is_preserved(self):
        for level in (225, 250):
            for pattern, size in ((0, 2), (1, 4)):
                with self.subTest(level=level, pattern=pattern):
                    self.reset(pattern=pattern)
                    for i in range(4096):
                        if self.pixels[i] == 0:
                            self.pixels[i] = level
                    self.assertEqual(self.grid(), (0, size, size))
                    pointer = self.lib.crispembed_table_parse_to_html(self.ctx, self.pixels, 64, 64)
                    self.assertTrue(pointer)
                    try:
                        self.assertEqual(C.string_at(pointer).count(b"cell &amp; &lt;"), size*size)
                    finally:
                        self.lib.crispembed_table_parse_free_string(pointer)

    def test_single_faint_component_is_not_suppressed(self):
        for blob in (False, True):
            self.reset()
            for i in range(4096):
                self.pixels[i] = 255
            for y in range(30, 34) if blob else (32,):
                for x in range(30, 34) if blob else (32,):
                    self.pixels[y*64+x] = 250
            status, rows, cols = self.grid()
            self.assertEqual(status, 0)
            self.assertGreater(rows, 0)
            self.assertGreater(cols, 0)
            pointer = self.lib.crispembed_table_parse_to_html(self.ctx, self.pixels, 64, 64)
            self.assertTrue(pointer)
            self.lib.crispembed_table_parse_free_string(pointer)

    def test_projection_exception_releases_buffers_already_returned_to_table_code(self):
        # This checks table-owned buffers only, not exception safety inside morphology.
        self.reset(throw=1)
        self.assertEqual(self.lib.test_catch_projection_failure(), 1)
        self.assertEqual(self.lib.test_morph_frees(), self.lib.test_morph_allocations())
        self.assertEqual(self.lib.test_morph_live(), 0)


if __name__ == "__main__":
    unittest.main()
