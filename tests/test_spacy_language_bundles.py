#!/usr/bin/env python3
"""Consistency of tools/spacy/language_bundles.json and the bundle/German model cards (stdlib only)."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "tools/spacy/language_bundles.json").read_text(encoding="utf-8"))
RECEIPTS = {"en_core_web_sm": "linux-en-run-44363961502.json", "fr_core_news_sm": "linux-fr-run-44363955799.json"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def converter_wheels():
    """WHEELS pins from the converter source, without importing gguf/spacy."""
    source = (ROOT / "models/convert-spacy-pipeline-to-gguf.py").read_text(encoding="utf-8")
    start = source.index("WHEELS = {")
    namespace = {}
    exec(source[start:source.index("}\n", start) + 2], namespace)
    return namespace["WHEELS"]


class LanguageBundleCatalog(unittest.TestCase):
    def test_wheel_pins_match_converter(self):
        wheels = converter_wheels()
        for model, entry in CATALOG["bundles"].items():
            self.assertEqual(wheels[entry["wheel"]["name"]], (entry["wheel"]["sha256"], entry["wheel"]["bytes"]))
            self.assertTrue(entry["wheel"]["name"].startswith(model + "-" + entry["version"]))

    def test_file_pins_match_hosted_receipts(self):
        for model, name in RECEIPTS.items():
            receipt = json.loads((ROOT / "docs/validation/native-spacy-v2-20261118" / name).read_text(encoding="utf-8"))
            self.assertTrue(receipt["passed"])
            expected = CATALOG["bundles"][model]["expected_files"]
            self.assertEqual({k: {"sha256": v["sha256"], "bytes": v["size_bytes"]}
                              for k, v in receipt["bundle_files"].items()}, expected)

    def test_treebank_pins_match_ab_lane(self):
        source = (ROOT / "tools/ci-heavy/spacy_v2_ab.py").read_text(encoding="utf-8")
        for entry in CATALOG["bundles"].values():
            self.assertIn(entry["treebank"]["sha256"], source)
            self.assertIn(entry["treebank"]["file"], source)

    def test_repositories_are_distinct(self):
        repos = [entry["repository"] for entry in CATALOG["bundles"].values()]
        self.assertEqual(len(set(repos + [CATALOG["library_repository"]])), 3)


class Cards(unittest.TestCase):
    def catalog(self, model, libraries):
        entry = CATALOG["bundles"][model]
        files = {f"bundle/{k}": v for k, v in entry["expected_files"].items()}
        files["bundle/manifest.json"] = {"sha256": "a" * 64, "bytes": 1}
        return {
            "schema": "crispembed.spacy.language_bundle.v1", "repository": entry["repository"],
            "library_repository": CATALOG["library_repository"],
            "model": {**{k: entry[k] for k in ("language", "language_name", "version", "wheel", "license",
                                               "components", "entity_labels", "treebank", "qualification")},
                      "name": model},
            "bundle": {"revision": "b" * 40, "source_commit": "c" * 40, "run": "1", "files": files},
            "libraries": libraries, "not_qualified": ["Grammar checking."],
        }

    def test_bundle_card_lists_files_licenses_and_failures(self):
        card = load("bundle_card", "tools/spacy/bundle_card.py")
        ok = {"platform": "linux-x86_64", "library_sha256": "d" * 64, "capsule_path": "packages/x/" + "e" * 40 + "/c.tar.gz",
              "capsule_revision": "f" * 40, "source_commit": "e" * 40, "run": "2", "controls_passed": True,
              "comparison": {"cases": 3, "exact_cases": 3, "tokens": 9, "entities": {"oracle": 1, "matched": 1}},
              "passed": True, "failure": None}
        bad = {"platform": "windows-x86_64", "library_sha256": None, "capsule_path": None, "capsule_revision": None,
               "source_commit": None, "run": "2", "controls_passed": False, "comparison": None, "passed": False,
               "failure": "AssertionError: No published v2-capable capsule"}
        fr = card.render(self.catalog("fr_core_news_sm", [ok, bad]))
        self.assertIn("license: other", fr)
        self.assertIn("license_name: lgpl-lr", fr)
        self.assertIn("LGPL-LR", fr)
        self.assertIn("loads and matches", fr)
        self.assertIn("FAILED (AssertionError: No published v2-capable capsule)", fr)
        self.assertIn("8b8f9ab4aa322191faa1735d1b7d53e0e744e18f4025290fe37242faa9a8e53e", fr)
        self.assertIn("/resolve/" + "b" * 40 + "/bundle/model.gguf", fr)
        en = card.render(self.catalog("en_core_web_sm", []))
        self.assertIn("license: mit", en)
        self.assertNotIn("license_name", en)
        self.assertIn("none recorded yet", en)
        self.assertIn("WORK_OF_ART", en)

    def test_german_card_renders_language_bundles(self):
        card = load("model_card", "tools/spacy/model_card.py")
        catalog = json.loads((ROOT / "tools/spacy/hub_catalog.json").read_text(encoding="utf-8"))
        self.assertNotIn("Other languages", card.render(catalog))
        catalog["language_bundles"] = [{"language": "en", "model": "en_core_web_sm",
                                        "repository": "cstr/x", "revision": "1" * 40, "license": "MIT",
                                        "claims": "Analysis only"}]
        self.assertIn("Other languages (same native library)", card.render(catalog))


if __name__ == "__main__":
    unittest.main()
