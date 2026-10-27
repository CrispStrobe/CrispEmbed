"""Stdlib tests for the Unlimited-OCR-2 precision-sweep metric helpers (no model, no network)."""
import importlib.util
import pathlib
import unittest

_P = pathlib.Path(__file__).resolve().parents[1] / "tools/kaggle/unlimited-ocr2-precision/precision_worker.py"
_spec = importlib.util.spec_from_file_location("precision_worker", _P)
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)


class Metrics(unittest.TestCase):
    A = "<|ref|>Title<|/ref|><|det|>[[10, 20, 300, 40]]<|/det|>\nHello"
    B = "<|ref|>Title<|/ref|><|det|>[[12, 20, 297, 40]]<|/det|>\nHello"

    def test_boxes(self):
        self.assertEqual(w.strip_boxes(self.A), w.strip_boxes(self.B))
        self.assertEqual(w.max_box_delta(self.A, self.B), 3.0)
        self.assertIsNone(w.max_box_delta(self.A, "Hello"))
        self.assertEqual(w.max_box_delta("x", "y"), 0.0)

    def test_cer_and_diff(self):
        self.assertEqual(w.cer("abc", "abc"), 0.0)
        self.assertAlmostEqual(w.cer("abd", "abc"), 1 / 3)
        self.assertEqual(w.cer("", ""), 0.0)
        self.assertEqual(w.first_diff([1, 2, 3], [1, 2, 3]), None)
        self.assertEqual(w.first_diff([1, 2, 3], [1, 5, 3]), 1)
        self.assertEqual(w.first_diff([1, 2], [1, 2, 3]), 2)
        self.assertIsNone(w.first_diff(None, [1]))

    def test_recipes(self):
        names = [p[0] for p in w.PRECISIONS]
        self.assertEqual(names[:4], ["f32", "f16", "q8_0", "q4_k"])
        allowed = {"f32", "f16", "q8_0", "q6_k", "q5_k"}  # tools/quantize.cpp --tensor-type
        for _, kind, qtype, ov in w.PRECISIONS:
            for o in ov:
                self.assertIn(o.rsplit("=", 1)[1], allowed)
            if kind == "quantize":
                self.assertIn(qtype, {"q8_0", "q4_k"})


if __name__ == "__main__":
    unittest.main()
