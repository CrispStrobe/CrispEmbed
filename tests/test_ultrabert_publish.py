#!/usr/bin/env python3
"""Offline controls for tools/ci-heavy/ultrabert_publish.py (no network, no credential)."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/ci-heavy"))
import ultrabert_publish as up  # noqa: E402

V = "Q4_K-LAST-FOUR-FFN"


class T(unittest.TestCase):
    def setUp(self):
        self.cfg = up.load()

    def test_evidence_joins_receipts(self):
        ev = up.evidence(self.cfg, V)
        self.assertGreater(ev["normalized"]["min_row_cosine"], 0.94)
        self.assertEqual(ev["rows"], 5483)
        self.assertEqual(ev["quality_run"], 40871150002)

    def test_unknown_variant_and_pin_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            up.evidence(self.cfg, "Q8_0")
        bad = copy.deepcopy(self.cfg)
        bad["variants"][V]["model"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            up.evidence(bad, V)

    def test_extract_selects_only_exact_member(self):
        data = b"gguf-bytes"
        cfg = copy.deepcopy(self.cfg)
        cfg["variants"][V]["model"] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as d:
            z = Path(d) / "a.zip"
            with zipfile.ZipFile(z, "w") as w:
                w.writestr("decoy.gguf", b"x" * len(data))
                w.writestr("model/q4.gguf", data)
            self.assertEqual(up.extract(cfg, V, z, Path(d) / "o"), "model/q4.gguf")
            cfg["variants"][V]["model"]["sha256"] = "1" * 64
            with self.assertRaises(ValueError):
                up.extract(cfg, V, z, Path(d) / "o2")

    def test_notice_pin_enforced(self):
        opener = lambda url, timeout: io.BytesIO(b"tampered")
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            up.notices(self.cfg, d, opener)

    def test_card_has_warning_table_and_no_host_paths(self):
        f = {"filename": self.cfg["variants"][V]["filename"], "bytes": 576997440, "sha256": "a" * 64}
        text = up.card(self.cfg, V, f, up.evidence(self.cfg, V), "b" * 40)
        self.assertIn("Do not mix Q4_K vectors", text)
        self.assertIn("not measured", text)
        self.assertIn("EXPERIMENTAL", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("/tmp/", text)

    def test_publish_requires_exact_commit(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            up.publish(self.cfg, V, d, d, Path(d), "main")


if __name__ == "__main__":
    unittest.main()
