#!/usr/bin/env python3
"""Model-free tests of review snapshot copying and optional native availability."""

import ctypes
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ocr_review_binding_under_test", ROOT / "python/crispembed/_binding.py")
BINDING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BINDING)


class ReviewSnapshotTests(unittest.TestCase):
    def pipeline(self, callback=None):
        pipeline = BINDING.CrispOcrOrchestrator.__new__(BINDING.CrispOcrOrchestrator)
        pipeline._ctx = 1
        pipeline._lib = SimpleNamespace(crispembed_ocr_pipeline_free=lambda ctx: None)
        if callback is not None:
            pipeline._lib.crispembed_ocr_pipeline_review_json = callback
        return pipeline

    def test_older_library_has_explicit_absence(self):
        self.assertIsNone(self.pipeline().review_snapshot())

    def test_before_run_has_no_snapshot(self):
        def native(ctx, length):
            ctypes.cast(length, ctypes.POINTER(ctypes.c_int)).contents.value = 0
            return None
        self.assertIsNone(self.pipeline(native).review_snapshot())

    def test_result_owns_decoded_unicode_and_embedded_nul(self):
        expected = {"schema_version": 1, "text": "日本語 €\nA\0B",
                    "regions": [{"confidence": None}], "confidence_calibrated": False}
        payload = json.dumps(expected, ensure_ascii=False).encode("utf-8")
        storage = ctypes.create_string_buffer(payload)

        def native(ctx, length):
            self.assertEqual(ctx, 1)
            ctypes.cast(length, ctypes.POINTER(ctypes.c_int)).contents.value = len(payload)
            return ctypes.cast(storage, ctypes.POINTER(ctypes.c_char))

        snapshot = self.pipeline(native).review_snapshot()
        ctypes.memset(storage, 0, len(payload))
        self.assertEqual(snapshot, expected)

    def test_disposed_context_does_not_call_native(self):
        def native(ctx, length):
            self.fail("disposed context must not call native")
        pipeline = self.pipeline(native)
        pipeline._ctx = None
        self.assertIsNone(pipeline.review_snapshot())


if __name__ == "__main__":
    unittest.main()
