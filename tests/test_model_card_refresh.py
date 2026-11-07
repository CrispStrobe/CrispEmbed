#!/usr/bin/env python3
"""Offline controls for tools/ci-heavy/model_card_refresh.py (no network, no credential)."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "ci-heavy"))
import model_card_refresh as m  # noqa: E402

SPEC = "docs/model-card-refresh-v1.json"


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_hub(files, tree):
    def opener(url, timeout=60):
        if "/api/models/" in url:
            return Resp(json.dumps([{"type": "file", "path": n, "size": v["bytes"],
                                     "lfs": {"oid": v["sha256"]} if n.endswith(".gguf") else None}
                                    for n, v in tree.items()]).encode())
        return Resp(files[url.rsplit("/", 1)[1]])
    return opener


class Controls(unittest.TestCase):
    def entry(self, **kw):
        e = {"repo": "cstr/OXR-1.0-GGUF", "parent": "a" * 40, "mode": "insert", "anchor": "## Usage",
             "card_file": "docs/model-cards/oxr-quality.md", "date": "2026-11-07", "commit_message": "x",
             "weights": {"w.gguf": {"bytes": 10, "sha256": "b" * 64}}, "pins": {}, "_id": "t", "_spec": SPEC}
        e.update(kw)
        return e

    def test_spec_entries_load(self):
        spec = json.loads((ROOT / SPEC).read_text())
        for ident in spec["entries"]:
            m.load(SPEC, ident)

    def test_insert_then_replace_is_idempotent(self):
        old = b"# t\n\n## Files\n\n## Usage\n\nrun\n"
        e = self.entry()
        once = m.render_readme(e, old)
        self.assertIn(m.MARK_OPEN.encode(), once)
        self.assertEqual(m.render_readme(e, once), once)
        self.assertTrue(once.decode().index(m.MARK_CLOSE) < once.decode().index("## Usage"))

    def test_missing_anchor_rejects(self):
        with self.assertRaises(ValueError):
            m.render_readme(self.entry(anchor="## Nope"), b"# t\n## Usage\n")

    def test_weight_drift_rejects(self):
        e = self.entry()
        readme = b"# t\n\n## Usage\n"
        e["base_readme_sha256"] = hashlib.sha256(readme).hexdigest()
        tree = {"README.md": {"bytes": len(readme), "sha256": None}, "w.gguf": {"bytes": 10, "sha256": "c" * 64}}
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            m.plan(e, d, "f" * 40, fake_hub({"README.md": readme}, tree))

    def test_readme_base_drift_rejects(self):
        e = self.entry(base_readme_sha256="0" * 64)
        tree = {"README.md": {"bytes": 3, "sha256": None}, "w.gguf": {"bytes": 10, "sha256": "b" * 64}}
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            m.plan(e, d, "f" * 40, fake_hub({"README.md": b"abc"}, tree))

    def test_plan_only_small_files(self):
        e = self.entry()
        readme = b"# t\n\n## Usage\n"
        e["base_readme_sha256"] = hashlib.sha256(readme).hexdigest()
        man = json.dumps({"files": [1], "readme_sha256": "x"}).encode()
        tree = {"README.md": {"bytes": 1, "sha256": None}, "manifest.json": {"bytes": 1, "sha256": None},
                "w.gguf": {"bytes": 10, "sha256": "b" * 64}}
        with tempfile.TemporaryDirectory() as d:
            payload, _ = m.plan(e, d, "f" * 40, fake_hub({"README.md": readme, "manifest.json": man}, tree))
        self.assertEqual(set(payload), {"README.md", "manifest.json"})
        out = json.loads(payload["manifest.json"])
        self.assertEqual(out["files"], [1])
        self.assertEqual(out["readme_sha256"], hashlib.sha256(payload["README.md"]).hexdigest())


if __name__ == "__main__":
    unittest.main()
