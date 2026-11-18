"""Model-free controls for v2 platform capsules (synthetic fixture bytes only).

Synthetic bundle/native bytes replace module pins inside each test. These are
packaging controls, not native inference or linguistic evidence.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/spacy/platform_distribution.py"
SPEC = importlib.util.spec_from_file_location("spacy_platform_distribution", MODULE_PATH)
pd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pd)

PLATFORM = "macos-arm64" if sys.platform == "darwin" else "linux-x86_64"
LIBRARY = pd.PLATFORMS[PLATFORM]["library"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def writable(root):
    for current, _, _ in os.walk(root):
        os.chmod(current, 0o700)


class PlatformCapsuleControls(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix=".spacy-platform-", dir=Path.cwd()))
        self.addCleanup(lambda: (writable(self.root), shutil.rmtree(self.root)))
        bundle_files = {name: ("bundle fixture: " + name).encode() for name in pd.v1.BUNDLE_FILES}
        pins = {name: {"sha256": digest(data), "size_bytes": len(data)}
                for name, data in bundle_files.items()}
        patches = [mock.patch.object(pd.v1, "BUNDLE_FILES", pins),
                   mock.patch.object(pd.v1, "BUNDLE_MANIFEST_SHA256", pins["manifest.json"]["sha256"])]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        for name, data in bundle_files.items():
            (self.bundle / name).write_bytes(data)
        self.prefix = self.root / "install"
        for name in pd.REQUIRED_NOTICES + ["include/spacy_pipeline.h", "lib/" + LIBRARY]:
            path = self.prefix / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("native fixture: " + name).encode())

    def build(self, name="capsule.tar.gz"):
        return pd.build(PLATFORM, self.prefix, self.bundle, self.root / name,
                        source_commit="0" * 40, build_info={"fixture": True},
                        system_dependencies={"bundled": False})

    def test_build_verify_install_round_trip(self):
        built = self.build()
        library_sha = digest(("native fixture: lib/" + LIBRARY).encode())
        self.assertEqual(built["entrypoints"]["library"], f"native/{library_sha}/{LIBRARY}")
        archive = self.root / "capsule.tar.gz"
        verified = pd.verify_archive(archive, built["archive_sha256"])
        self.assertEqual(verified["payload_file_count"], built["payload_file_count"])
        cache = self.root / "cache"
        result = pd.install(archive, built["archive_sha256"], cache)
        self.assertTrue(result["installed"])
        tree = cache / built["archive_sha256"]
        self.assertEqual((tree / built["entrypoints"]["library"]).read_bytes(),
                         ("native fixture: lib/" + LIBRARY).encode())
        self.assertEqual(stat.S_IMODE(tree.lstat().st_mode), 0o555)
        again = pd.install(archive, built["archive_sha256"], cache)
        self.assertFalse(again["installed"])
        self.assertEqual([p.name for p in cache.iterdir()], [built["archive_sha256"]])

    def test_build_is_deterministic_and_never_replaces(self):
        first = self.build("a.tar.gz")
        second = self.build("b.tar.gz")
        self.assertEqual(first["archive_sha256"], second["archive_sha256"])
        with self.assertRaises(FileExistsError):
            self.build("a.tar.gz")

    def test_untrusted_digest_is_rejected_before_parsing(self):
        self.build()
        with self.assertRaises(pd.DistributionError):
            pd.verify_archive(self.root / "capsule.tar.gz", "f" * 64)

    def test_missing_notice_or_extra_member_is_rejected(self):
        (self.prefix / pd.REQUIRED_NOTICES[0]).unlink()
        with self.assertRaises(pd.DistributionError):
            self.build()
        (self.prefix / pd.REQUIRED_NOTICES[0]).write_bytes(b"restored")
        (self.prefix / "bin").mkdir()
        (self.prefix / "bin/tool").write_bytes(b"x")
        with self.assertRaises(pd.DistributionError):
            self.build()

    def test_manifest_cannot_change_bundle_pins(self):
        built = self.build()
        raw = gzip.decompress((self.root / "capsule.tar.gz").read_bytes())
        files = {}
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for entry in archive:
                files[entry.name] = archive.extractfile(entry).read()
        data = json.loads(files[pd.MANIFEST_NAME])
        data["files"]["bundle/model.gguf"]["sha256"] = "e" * 64
        files[pd.MANIFEST_NAME] = pd._manifest_bytes(data)
        with self.assertRaises(pd.DistributionError):
            pd._read_capsule(pd.v1._pack(files))
        self.assertTrue(built["archive_sha256"])

    def test_modified_installed_file_is_detected(self):
        built = self.build()
        cache = self.root / "cache"
        pd.install(self.root / "capsule.tar.gz", built["archive_sha256"], cache)
        tree = cache / built["archive_sha256"]
        target = tree / "bundle/LICENSE"
        os.chmod(target.parent, 0o755)
        os.chmod(target, 0o644)
        target.write_bytes(b"changed")
        os.chmod(target, 0o444)
        os.chmod(target.parent, 0o555)
        with self.assertRaises(pd.DistributionError):
            pd.verify_tree(tree, built["archive_sha256"])

    def test_v2_manifest_keeps_consumer_contract(self):
        built = self.build()
        _, files = pd._read_capsule((self.root / "capsule.tar.gz").read_bytes())
        data = json.loads(files[pd.MANIFEST_NAME])
        # Fields read by the Quds Dart installer.
        self.assertTrue(data["schema"].startswith("crispembed.spacy.distribution."))
        self.assertEqual(data["entrypoints"]["bundle"], "bundle")
        self.assertEqual(data["bundle_manifest_sha256"], pd.v1.BUNDLE_MANIFEST_SHA256)
        self.assertEqual(data["files"][built["entrypoints"]["library"]]["sha256"],
                         built["library_sha256"])
        self.assertNotIn(pd.MANIFEST_NAME, data["files"])


class HubCatalogControls(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "tools/ci-heavy"))
        self.addCleanup(sys.path.remove, str(root / "tools/ci-heavy"))
        spec = importlib.util.spec_from_file_location(
            "spacy_platform_package", root / "tools/ci-heavy/spacy_w7_platform_package.py")
        self.package = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.package)
        spec = importlib.util.spec_from_file_location("spacy_model_card", root / "tools/spacy/model_card.py")
        self.card = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.card)
        self.catalog = json.loads((root / "tools/spacy/hub_catalog.json").read_text())

    def test_card_lists_every_pinned_file_and_honest_status(self):
        text = self.card.render(self.catalog)
        for pin in self.catalog["bundle"]["files"].values():
            self.assertIn(pin["sha256"], text)
        self.assertIn(pd.v1.BUNDLE_MANIFEST_SHA256, text)
        linux = next(p for p in self.catalog["packages"] if p["platform"] == "linux-x86_64")
        self.assertEqual(linux["library_sha256"], pd.v1.LIBRARY_SHA256)
        self.assertIn(linux["sha256"], text)
        self.assertIn("Not qualified", text)
        self.assertNotIn("/resolve/main/", text)

    def test_new_platform_entry_replaces_and_never_claims_grammar(self):
        entry = {"platform": "macos-arm64", "status": "published", "revision": "a" * 40,
                 "path": "packages/macos-arm64/x/c.tar.gz", "sha256": "b" * 64, "bytes": 1,
                 "library_sha256": "c" * 64, "analysis": "component", "grammar": "Not qualified for this build",
                 "receipt": "packages/macos-arm64/x/receipt.json"}
        merged = self.package.catalog_with(entry, self.catalog)
        platforms = [p["platform"] for p in merged["packages"]]
        self.assertEqual(platforms.count("macos-arm64"), 1)
        self.assertIn("resolve/" + "a" * 40 + "/packages/macos-arm64/x/receipt.json", self.card.render(merged))
        self.assertEqual(json.loads(json.dumps(self.catalog)), self.catalog)


if __name__ == "__main__":
    unittest.main()
