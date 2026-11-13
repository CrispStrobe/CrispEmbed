#!/usr/bin/env python3
"""Exercise model-free grayscale preprocessing calls through typed native ABIs."""

import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "preprocess_binding_under_test", ROOT / "python/crispembed/_binding.py")
BINDING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BINDING)
U8P = ctypes.POINTER(ctypes.c_uint8)
INT = ctypes.c_int
INTP = ctypes.POINTER(INT)
FLOATP = ctypes.POINTER(ctypes.c_float)
METHODS = ("dewarp", "tps_dewarp", "find_skew", "adaptive_binarize",
           "background_norm", "despeckle", "cc_detect")


class NativePreprocess:
    def __init__(self, rc=0, geometry=None):
        self.lib = SimpleNamespace()
        self.calls = []
        self.rc = rc
        self.geometry = geometry

        def export(name, result, args, callback):
            setattr(self.lib, "crispembed_" + name,
                    ctypes.CFUNCTYPE(result, *args)(callback))

        def copy(name, gray, w, h, out):
            self.calls.append((name, w, h, list(gray[:w * h])))
            ctypes.memmove(out, gray, w * h)

        def dewarp(gray, w, h, out, ow, oh):
            copy("dewarp", gray, w, h, out)
            ow[0], oh[0] = self.geometry or (w, h)
            return self.rc

        def tps(gray, w, h, model, out):
            copy("tps_dewarp", gray, w, h, out)
            self.model_path = model
            return self.rc

        def skew(gray, w, h, angle, confidence):
            self.calls.append(("find_skew", w, h, list(gray[:w * h])))
            angle[0], confidence[0] = -1.5, 0.75
            return self.rc

        def despeckle(gray, w, h, max_w, max_h, out):
            copy("despeckle", gray, w, h, out)
            self.limits = (max_w, max_h)

        def cc(gray, w, h, count):
            self.calls.append(("cc_detect", w, h, list(gray[:w * h])))
            count[0] = 0
            return None

        export("dewarp", INT, [U8P, INT, INT, U8P, INTP, INTP], dewarp)
        export("tps_auto_dewarp", INT, [U8P, INT, INT, ctypes.c_char_p, U8P], tps)
        export("find_skew", INT, [U8P, INT, INT, FLOATP, FLOATP], skew)
        export("adaptive_binarize", None, [U8P, INT, INT, U8P],
               lambda gray, w, h, out: copy("adaptive_binarize", gray, w, h, out))
        export("background_norm", None, [U8P, INT, INT, U8P],
               lambda gray, w, h, out: copy("background_norm", gray, w, h, out))
        export("despeckle", None, [U8P, INT, INT, INT, INT, U8P], despeckle)
        export("cc_detect", ctypes.c_void_p, [U8P, INT, INT, INTP], cc)
        export("pdf_page_dpi", INT, [ctypes.c_char_p, INT, FLOATP, INTP],
               lambda *args: 0)
        export("ocr_render", ctypes.c_void_p,
               [ctypes.c_void_p, INT, INT, INT, ctypes.c_char_p], lambda *args: None)


class PreprocessBindingTests(unittest.TestCase):
    def create(self, **kwargs):
        native = NativePreprocess(**kwargs)
        with patch.object(BINDING, "_load_library", return_value=native.lib):
            pp = BINDING.CrispPreprocess(lib_path="selected-library")
        return pp, native

    def invoke(self, pp, method, gray, w, h):
        if method == "tps_dewarp":
            return pp.tps_dewarp(gray, w, h, "模型.gguf")
        if method == "despeckle":
            return pp.despeckle(gray, w, h, max_w=2, max_h=3)
        return getattr(pp, method)(gray, w, h)

    def test_strided_grayscale_views_send_logical_pixels(self):
        base = np.arange(48, dtype=np.uint8).reshape(6, 8)
        for image in (base[:, ::2], base.T, base[::-1, ::-1]):
            expected = image.flatten().tolist()
            for method in METHODS:
                with self.subTest(method=method, shape=image.shape):
                    pp, native = self.create()
                    before = base.copy()
                    h, w = image.shape
                    result = self.invoke(pp, method, image, w, h)
                    self.assertEqual(native.calls[-1], (method, w, h, expected))
                    np.testing.assert_array_equal(base, before)
                    if method == "find_skew":
                        self.assertEqual(result, (-1.5, 0.75))
                    elif method == "cc_detect":
                        self.assertEqual(result, [])
                    else:
                        self.assertEqual((result.shape, result.dtype), (image.shape, np.uint8))
                        np.testing.assert_array_equal(result, image)
                        self.assertFalse(np.shares_memory(result, base))
                    if method == "tps_dewarp":
                        self.assertEqual(native.model_path, "模型.gguf".encode())
                    elif method == "despeckle":
                        self.assertEqual(native.limits, (2, 3))

    def test_flat_and_converted_uint8_input(self):
        for image in (np.arange(12, dtype=np.uint8),
                      np.arange(12, dtype=np.int16).reshape(3, 4)):
            for method in METHODS:
                with self.subTest(method=method, dtype=image.dtype):
                    pp, native = self.create()
                    result = self.invoke(pp, method, image, 4, 3)
                    self.assertEqual(native.calls[-1][-1], list(range(12)))
                    if isinstance(result, np.ndarray):
                        np.testing.assert_array_equal(result, np.arange(12).reshape(3, 4))
                        self.assertEqual(result.dtype, np.uint8)

    def test_invalid_buffers_are_rejected_before_native_reads(self):
        for method in METHODS:
            for w, h, count in ((0, 1, 0), (1, 0, 0), (-1, 1, 0), (1, -1, 0),
                                (2**31, 1, 0), (1, 2**31, 0), (1.5, 1, 3),
                                (1, 1.5, 3), (2, 2, 3), (2, 2, 5)):
                with self.subTest(method=method, width=w, height=h, count=count):
                    pp, native = self.create()
                    # Invalid-input tests must never execute an unsafe pointer read,
                    # including when a future regression removes Python validation.
                    symbol = "tps_auto_dewarp" if method == "tps_dewarp" else method
                    reject = Mock(side_effect=AssertionError("native read before validation"))
                    setattr(native.lib, "crispembed_" + symbol, reject)
                    with self.assertRaises(ValueError):
                        self.invoke(pp, method, np.zeros(count, dtype=np.uint8), w, h)
                    reject.assert_not_called()
                    self.assertEqual(native.calls, [])

    def test_dewarp_failure_falls_back_to_owned_normalized_pixels(self):
        image = np.arange(48, dtype=np.int16).reshape(6, 8)[:, ::2]
        for method in ("dewarp", "tps_dewarp"):
            with self.subTest(method=method):
                pp, native = self.create(rc=1)
                result = self.invoke(pp, method, image, 4, 6)
                self.assertEqual(result.dtype, np.uint8)
                np.testing.assert_array_equal(result, image)
                self.assertFalse(np.shares_memory(result, image))

    def test_dewarp_output_geometry_fits_preallocated_buffer(self):
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        for geometry in ((0, 3), (-2, -2), (5, 3)):
            with self.subTest(geometry=geometry):
                pp, native = self.create(geometry=geometry)
                with self.assertRaisesRegex(RuntimeError, "output dimensions"):
                    pp.dewarp(image, 4, 3)
        pp, native = self.create(geometry=(3, 2))
        np.testing.assert_array_equal(pp.dewarp(image, 4, 3), np.arange(6).reshape(2, 3))


if __name__ == "__main__":
    unittest.main()
