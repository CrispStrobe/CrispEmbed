"""Stdlib tests for tools/ci-heavy/unlimited_ocr2_publish.py (no network, no weights, no Hub)."""
import hashlib
import importlib.util
import io
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("uocr2_publish", ROOT / "tools/ci-heavy/unlimited_ocr2_publish.py")
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)


class Protocol(unittest.TestCase):
    def setUp(self):
        self.cfg, self.source = p.load()

    def test_pins_and_license(self):
        up = self.cfg["upstream"]
        self.assertEqual(self.cfg["destination"], "cstr/Unlimited-OCR-2-GGUF")
        self.assertEqual(up["revision"], self.source["revision"])
        self.assertEqual(up["license"], "mit")
        self.assertIn(up["license_file"]["commit"], up["license_file"]["url"])

    def test_variants(self):
        self.assertEqual(p.parse_variants("", self.cfg), ["F16", "Q8_0"])
        self.assertEqual(p.parse_variants("q8_0, f16", self.cfg), ["Q8_0", "F16"])
        for bad in ("Q3_K", "F16,F16"):
            with self.assertRaises(ValueError):
                p.parse_variants(bad, self.cfg)

    def test_plan_orders_conversions_first(self):
        steps, drop = p.plan(self.cfg, ["Q8_0", "F16"])
        self.assertEqual([(s[0], s[1]) for s in steps], [("convert", "F32"), ("convert", "F16"), ("quantize", "Q8_0")])
        self.assertEqual(steps[1][3], ["--fp16"])
        self.assertEqual(steps[2][3], ["q8_0"])
        self.assertTrue(drop)
        steps, drop = p.plan(self.cfg, ["F16"])
        self.assertEqual([s[1] for s in steps], ["F16"])
        self.assertFalse(drop)
        steps, drop = p.plan(self.cfg, ["F32", "Q4_K"])
        self.assertEqual([s[1] for s in steps], ["F32", "Q4_K"])
        self.assertFalse(drop)

    def test_receipts_summary(self):
        r = p.receipts(self.cfg)
        self.assertTrue(r["untiled"]["ids_exact"])
        self.assertEqual(r["tiled"]["status"], "experimental")
        self.assertLessEqual(r["tiled"]["max_box_coordinate_difference"], 3)

    def test_receipts_refuse_regression(self):
        bad = dict(self.cfg, parity_limits={"tiled_max_box_coordinate_difference": 2, "box_scale": 999})
        with self.assertRaises(ValueError):
            p.receipts(bad)

    def test_card(self):
        files = [{"variant": "F16", "filename": "unlimited-ocr-2-f16.gguf", "bytes": 6_700_000_000, "sha256": "a" * 64}]
        text = p.card(self.cfg, files, p.receipts(self.cfg), "b" * 40)
        self.assertTrue(text.startswith("---\nlicense: mit\n"))
        self.assertIn("base_model: baidu/Unlimited-OCR-2\n", text)
        self.assertIn(self.cfg["upstream"]["revision"], text)
        self.assertIn("a" * 64, text)
        self.assertIn("not validated", text)
        self.assertIn("Copyright (c) 2026 Baidu", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("/mnt/", text)

    def test_notices_verify_hashes(self):
        up = self.cfg["upstream"]

        def fake(content):
            return lambda url, timeout: io.BytesIO(content)
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            p.notices(self.cfg, d, opener=fake(b"tampered"))
        good = b"x" * up["license_file"]["bytes"]
        expected = dict(bytes=len(good), sha256=hashlib.sha256(good).hexdigest())
        self.assertEqual(p.fetch("u", expected, opener=fake(good)), good)


if __name__ == "__main__":
    unittest.main()
