#!/usr/bin/env python3
"""Production Python scan cleanup against the shared tiny C interface fixture."""

import ctypes
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_preprocess_bindings import BINDING
from test_rust_scan_cleanup_safety import native_fixture


class ScanCleanupBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="python-scan-cleanup-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.lib = ctypes.CDLL(str(native_fixture(cls.temp.name)))
        cls.lib.test_reset.argtypes = [ctypes.c_int, ctypes.c_int]
        cls.lib.test_param_float.restype = ctypes.c_float

    def setUp(self):
        self.lib.test_reset(0, 0)
        with patch.object(BINDING, "_load_library", return_value=self.lib):
            self.cleanup = BINDING.CrispScanCleanup()
        self.addCleanup(self.cleanup.__del__)

    def tearDown(self):
        self.assertEqual(self.lib.test_live(), 0, "native image leaked")

    def test_supported_images_keep_logical_pixels_dtype_and_source_unchanged(self):
        for channels in (1, 3, 4):
            base = np.arange(6 * channels, dtype=np.int16).reshape(2, 3, channels)
            images = (base[::-1, ::-1], base[:, :, 0] if channels == 1 else base)
            for image in images:
                with self.subTest(channels=channels, shape=image.shape, strides=image.strides):
                    expected = image.flatten().astype(np.uint8).tolist()
                    before = base.copy()
                    self.lib.test_reset(len(expected), 0)
                    for method, symbol in (("process", "process"),
                                           ("detect_page_split", "detect_page_split"),
                                           ("content_bbox", "content_bbox")):
                        name = "crispembed_scan_cleanup_" + symbol
                        real = getattr(self.lib, name)
                        seen = []
                        def capture(*args):
                            offset = 1 if method == "process" else 0
                            pointer, w, h, ch = args[offset:offset + 4]
                            seen.append((w, h, ch, list(pointer[:w*h*ch])))
                            return real(*args)
                        with patch.object(self.lib, name, side_effect=capture):
                            result = getattr(self.cleanup, method)(image)
                        self.assertEqual(seen, [(3, 2, channels, expected)])
                        if method == "process":
                            np.testing.assert_array_equal(result.flatten(), np.arange(18, dtype=np.uint8))
                            self.assertFalse(np.shares_memory(result, base))
                        elif method == "detect_page_split":
                            self.assertEqual(result, 1)
                        else:
                            self.assertEqual(result, (0, 0, 3, 2))
                    np.testing.assert_array_equal(base, before)
                    self.assertEqual(self.lib.test_image_frees(), 1)

    def test_invalid_shape_channels_and_empty_images_never_call_native(self):
        for shape in ((), (4,), (2, 3, 2), (2, 3, 5), (2, 3, 1, 1), (0, 3), (2, 0, 3)):
            for method in ("process", "detect_page_split", "content_bbox"):
                with self.subTest(shape=shape, method=method):
                    self.lib.test_reset(0, 0)
                    with self.assertRaises(ValueError):
                        getattr(self.cleanup, method)(np.zeros(shape, dtype=np.uint8))
                    self.assertEqual(self.lib.test_calls(), 0)
                    self.assertEqual(self.lib.test_defaults_calls(), 0)

    def test_large_metadata_rejected_before_pixel_copy_or_native_call(self):
        for shape in ((50000, 50000, 1), (2**31, 1, 3)):
            copy = Mock(side_effect=AssertionError("large image copy before validation"))
            metadata = SimpleNamespace(ndim=3, shape=shape, flatten=copy)
            for method in ("process", "detect_page_split", "content_bbox"):
                with self.subTest(shape=shape, method=method):
                    with patch.object(BINDING.np, "asarray", return_value=metadata):
                        with self.assertRaises(ValueError):
                            getattr(self.cleanup, method)(object())
                    copy.assert_not_called()
                    self.assertEqual(self.lib.test_calls(), 0)

    def test_error_and_invalid_output_geometry_free_without_reading(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        for mode in (1, 2, 3, 4, 5, 7, 8):
            with self.subTest(mode=mode):
                self.lib.test_reset(18, mode)
                with patch.object(np.ctypeslib, "as_array") as read:
                    with self.assertRaises(RuntimeError):
                        self.cleanup.process(image)
                    read.assert_not_called()
                self.assertEqual(self.lib.test_image_frees(), 0 if mode in (1, 7) else 1)

    def test_copy_exception_and_larger_output_keep_matching_cleanup(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        self.lib.test_reset(18, 0)
        with patch.object(np.ctypeslib, "as_array", side_effect=MemoryError):
            with self.assertRaises(MemoryError):
                self.cleanup.process(image)
        self.assertEqual(self.lib.test_image_frees(), 1)
        self.lib.test_reset(18, 6)
        out = self.cleanup.process(image, deskew=False, sauvola_k=0.375)
        self.assertEqual(out.shape, (4, 5, 3))
        np.testing.assert_array_equal(out.flatten(), np.arange(60, dtype=np.uint8))
        self.assertEqual(self.lib.test_image_frees(), 1)
        self.assertEqual(self.lib.test_param(), 0)
        self.assertEqual(self.lib.test_param_float(), 0.375)

    def test_model_path_nul_rejected_before_library_load_and_tier1_preserved(self):
        for path in ("bad\0path", b"bad\0path"):
            with self.subTest(path=path):
                with patch.object(BINDING, "_load_library", side_effect=AssertionError("native load")):
                    with self.assertRaises(ValueError):
                        BINDING.CrispScanCleanup(path)
        for path in (None, "", Path("model.gguf")):
            with self.subTest(path=path):
                self.lib.test_reset(0, 0)
                with patch.object(BINDING, "_load_library", return_value=self.lib):
                    cleanup = BINDING.CrispScanCleanup(path)
                self.addCleanup(cleanup.__del__)
                self.assertEqual(self.lib.test_model_null(), int(path in (None, "")))

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "optional Pillow image dependency unavailable")
    def test_path_and_pil_inputs_use_rgb_file_loading(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix="scan-path-") as directory:
            path = Path(directory) / "scan.png"
            image = Image.fromarray(np.arange(6, dtype=np.uint8).reshape(2, 3))
            image.save(path)
            for source in (path, str(path), image):
                with self.subTest(source_type=type(source).__name__):
                    self.lib.test_reset(18, 0)
                    self.assertEqual(self.cleanup.process(source).shape, (2, 3, 3))
                    self.assertEqual(self.cleanup.detect_page_split(source), 1)
                    self.assertEqual(self.cleanup.content_bbox(source), (0, 0, 3, 2))


if __name__ == "__main__":
    unittest.main()
