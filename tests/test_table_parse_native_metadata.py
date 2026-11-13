#!/usr/bin/env python3
"""Direct native TableParse guards and real entry arithmetic without model builds."""

import ctypes as C
import tempfile
import unittest

import test_table_parse_binding as table_fixture


class TableParseNativeMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="table-native-metadata-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.lib = C.CDLL(str(table_fixture.TableParseBindingTests.build_fixture(cls.temp.name, actual_html=True)))
        cls.lib.test_reset.argtypes = [C.c_int, C.c_int]
        cls.lib.crispembed_table_parse_init.argtypes = [C.c_char_p, C.c_int]
        cls.lib.crispembed_table_parse_init.restype = C.c_void_p
        cls.lib.crispembed_table_parse_free.argtypes = [C.c_void_p]
        cls.lib.crispembed_table_parse_free_string.argtypes = [C.c_void_p]
        cls.ctx = cls.lib.crispembed_table_parse_init(None, 4)
        cls.addClassCleanup(cls.lib.crispembed_table_parse_free, cls.ctx)
        for prefix in ("table_parse", "crispembed_table_parse"):
            html = getattr(cls.lib, prefix + "_to_html")
            html.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.c_int]
            html.restype = C.c_void_p
            grid = getattr(cls.lib, prefix + "_detect_grid")
            grid.argtypes = [C.POINTER(C.c_uint8), C.c_int, C.c_int,
                             C.POINTER(C.c_int), C.POINTER(C.c_int)]
            grid.restype = C.c_int
        cls.lib.test_metadata_valid.argtypes = [C.POINTER(C.c_uint8), C.c_int, C.c_int]
        cls.lib.test_midpoint.argtypes = [C.c_int, C.c_int]

    def setUp(self):
        self.lib.test_reset(12, 0)
        self.pixels = (C.c_uint8 * 12)(*range(12))

    def tearDown(self):
        self.assertEqual(self.lib.test_live(), 0, "native HTML allocation leaked")

    def test_internal_and_public_metadata_fail_before_detector_and_clear_counts(self):
        invalid = ((None, 4, 3), (self.pixels, 0, 3), (self.pixels, 4, 0),
                   (self.pixels, -1, 3), (self.pixels, -1, -1),
                   (self.pixels, 2**31-1, 1), (self.pixels, 2**31-1-30, 1),
                   (self.pixels, 65536, 65536), (self.pixels, 1, 2**31-1),
                   (self.pixels, 1, 1908874355))
        for pointer, w, h in invalid:
            for prefix in ("table_parse", "crispembed_table_parse"):
                with self.subTest(null_pixels=pointer is None, width=w, height=h, prefix=prefix):
                    self.assertFalse(getattr(self.lib, prefix + "_to_html")(self.ctx, pointer, w, h))
                    rows, cols = C.c_int(99), C.c_int(88)
                    status = getattr(self.lib, prefix + "_detect_grid")(
                        pointer, w, h, C.byref(rows), C.byref(cols))
                    self.assertEqual((status, rows.value, cols.value), (-1, 0, 0))
                    self.assertEqual(self.lib.test_calls(), 0)
        self.assertFalse(self.lib.crispembed_table_parse_to_html(None, self.pixels, 4, 3))
        self.assertEqual(self.lib.test_calls(), 0)

    def test_valid_results_and_optional_counts_preserve_native_status_and_html(self):
        for prefix in ("table_parse", "crispembed_table_parse"):
            self.lib.test_reset(12, 0)
            rows, cols = C.c_int(), C.c_int()
            grid = getattr(self.lib, prefix + "_detect_grid")
            self.assertEqual(grid(self.pixels, 4, 3, C.byref(rows), C.byref(cols)), 0)
            self.assertEqual((rows.value, cols.value), (2, 3))
            self.assertEqual(grid(self.pixels, 4, 3, None, None), 0)
            pointer = getattr(self.lib, prefix + "_to_html")(self.ctx, self.pixels, 4, 3)
            self.assertTrue(pointer)
            try:
                html = C.string_at(pointer).decode()
                self.assertEqual(html.count("cell &amp; &lt;"), 6)
                self.assertEqual(html.count("<th>"), 3)
                self.assertEqual(html.count("<td>"), 3)
            finally:
                self.lib.crispembed_table_parse_free_string(pointer)
            self.assertEqual(self.lib.test_frees(), 1)

    def test_no_grid_and_partial_grid_clear_every_supplied_count(self):
        for mode in (1, 4):
            for rows_present, cols_present in ((True, True), (True, False), (False, True), (False, False)):
                with self.subTest(mode=mode, rows=rows_present, cols=cols_present):
                    self.lib.test_reset(12, mode)
                    rows, cols = C.c_int(99), C.c_int(88)
                    status = self.lib.crispembed_table_parse_detect_grid(
                        self.pixels, 4, 3, C.byref(rows) if rows_present else None,
                        C.byref(cols) if cols_present else None)
                    self.assertEqual(status, -1)
                    if rows_present:
                        self.assertEqual(rows.value, 0)
                    if cols_present:
                        self.assertEqual(cols.value, 0)

    def test_scalar_index_edges_need_no_large_allocations(self):
        limit = 2**31 - 1
        # The exact final-row-plus-radius boundary for the derived vertical SE.
        self.assertEqual(self.lib.test_metadata_valid(self.pixels, 1, 1908874354), 1)
        self.assertEqual(self.lib.test_metadata_valid(self.pixels, 1, 1908874355), 0)
        self.assertEqual(self.lib.test_metadata_valid(self.pixels, limit-31, 1), 1)
        self.assertEqual(self.lib.test_metadata_valid(self.pixels, limit-30, 1), 0)
        for begin, end in ((0, 0), (0, 1), (1, 2), (limit-7, limit), (limit, limit)):
            self.assertEqual(self.lib.test_midpoint(begin, end), (begin+end)//2)
        self.assertEqual(self.lib.test_projection_helpers(), 1)


if __name__ == "__main__":
    unittest.main()
