"""Model-free controls for tools/ci-heavy/inpaint_gguf_publish.py (no network)."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pub", ROOT / "tools/ci-heavy/inpaint_gguf_publish.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)
PROTOCOL = "docs/inpaint-publication-v1.json"
REPOS = ("cstr/MI-GAN-512-GGUF", "cstr/Moebius-inpaint-GGUF", "cstr/PixelHacker-GGUF", "cstr/SDXL-VAE-inpaint-GGUF")


class InpaintPublishTests(unittest.TestCase):
    def test_every_destination_loads_with_pinned_variants(self):
        for repo in REPOS:
            cfg, entry = pub.load(PROTOCOL, repo)
            self.assertTrue(entry["variants"])
            for v in entry["variants"].values():
                self.assertRegex(v["sha256"], "^[0-9a-f]{64}$")
                self.assertTrue(v["source_repo"].endswith("-private"))

    def test_only_approved_variants(self):
        names = {repo: set(pub.load(PROTOCOL, repo)[1]["variants"]) for repo in REPOS}
        self.assertEqual(names["cstr/Moebius-inpaint-GGUF"], {"F32", "F16", "Q8_0", "Q4_K-mix"})
        self.assertEqual(names["cstr/MI-GAN-512-GGUF"], {"F32", "F16", "Q8_0", "Q4_K"})
        self.assertEqual(names["cstr/PixelHacker-GGUF"], {"F32", "F16", "Q8_0", "Q4_K"})

    def test_unknown_destination_and_protocol_rejected(self):
        with self.assertRaises(ValueError):
            pub.load(PROTOCOL, "cstr/crispembed-moebius-private")
        with self.assertRaises(ValueError):
            pub.load("docs/teleocr-publication-v1.json", REPOS[0])

    def test_unapproved_protocol_rejected(self):
        cfg = json.loads((ROOT / PROTOCOL).read_text())
        cfg["approved_by_owner"] = False
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "docs").mkdir()
            (Path(tmp) / PROTOCOL).write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                pub.load(PROTOCOL, REPOS[0], root=Path(tmp))

    def test_cards_carry_licences_bars_and_caveats(self):
        cfg, entry = pub.load(PROTOCOL, "cstr/MI-GAN-512-GGUF")
        text = pub.card(cfg, "cstr/MI-GAN-512-GGUF", entry, "0" * 40)
        self.assertIn("0.9019", text)
        self.assertIn("DBNet", text)
        self.assertIn("license: mit", text)
        cfg, entry = pub.load(PROTOCOL, "cstr/Moebius-inpaint-GGUF")
        text = pub.card(cfg, "cstr/Moebius-inpaint-GGUF", entry, "0" * 40)
        self.assertIn("Apache-2.0", text)
        self.assertIn("MIT", text)
        self.assertIn("No dual licence is claimed", text)
        self.assertNotIn("| Q4_K |", text)
        for repo in REPOS:
            cfg, entry = pub.load(PROTOCOL, repo)
            text = pub.card(cfg, repo, entry, "0" * 40)
            self.assertNotRegex(text.lower(), r"(?<!l)lama\b")
            for v in entry["variants"].values():
                self.assertIn(v["sha256"], text)

    def test_repo_licence_files_are_pinned(self):
        for repo in REPOS:
            cfg, entry = pub.load(PROTOCOL, repo)
            fake = lambda src: b"x"
            local = copy.deepcopy(entry)
            local["license_files"] = [lf for lf in entry["license_files"] if "from_repo_path" in lf]
            files = pub.small_files(cfg, repo, local, "0" * 40)
            self.assertIn("README.md", files)
            self.assertIn("manifest.json", files)
            bad = copy.deepcopy(entry)
            bad["license_files"] = [dict(lf, sha256="0" * 64) for lf in local["license_files"]][:1]
            if bad["license_files"]:
                with self.assertRaises(ValueError):
                    pub.small_files(cfg, repo, bad, "0" * 40, fetch=fake)

    def test_publish_requires_exact_source_commit(self):
        cfg, entry = pub.load(PROTOCOL, REPOS[0])
        with self.assertRaises(ValueError):
            pub.publish(cfg, REPOS[0], entry, "/nonexistent", Path(tempfile.gettempdir()), "main")


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1] + sys.argv[1:])
