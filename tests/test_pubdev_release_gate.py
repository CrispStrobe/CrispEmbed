#!/usr/bin/env python3
"""Offline contract tests for the pub.dev release preflight."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "tools/ci-heavy/pubdev_release_gate.py"
ASSETS = {
    "linux-x86_64": "crispembed-linux-x86_64.tar.gz",
    "linux-arm64": "crispembed-linux-arm64.tar.gz",
    "macos-arm64": "crispembed-macos-arm64.tar.gz",
    "windows-x86_64": "crispembed-windows-x86_64.zip",
    "android-arm64-v8a": "crispembed-android-arm64-v8a.tar.gz",
    "android-armeabi-v7a": "crispembed-android-armeabi-v7a.tar.gz",
    "ios-arm64": "crispembed-ios-arm64.tar.gz",
}


class PubdevReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workflow = self.root / "workflow"
        self.downloaded = self.root / "downloaded"
        self.downloaded.mkdir()
        self.pubspec = self.root / "pubspec.yaml"
        self.version = self.root / "VERSION"
        self.changelog = self.root / "CHANGELOG.md"
        self.pubspec.write_text("name: crispembed\nversion: 0.17.14\n")
        self.version.write_text("0.17.14\n")
        self.changelog.write_text("## 0.17.14\n\n- Release.\n")
        self.release = {"tag_name": "v0.17.14", "draft": False, "prerelease": False, "assets": []}
        for artifact, filename in ASSETS.items():
            directory = self.workflow / artifact
            directory.mkdir(parents=True)
            content = ("archive bytes: " + filename).encode()
            (directory / filename).write_bytes(content)
            (self.downloaded / filename).write_bytes(content)
            self.release["assets"].append({"name": filename, "state": "uploaded"})

    def run_gate(self, tag="v0.17.14"):
        release_json = self.root / "release.json"
        release_json.write_text(json.dumps(self.release))
        return subprocess.run(
            [sys.executable, str(GATE), "--tag", tag, "--version-file", str(self.version),
             "--pubspec", str(self.pubspec), "--changelog", str(self.changelog),
             "--release-json", str(release_json), "--workflow-assets", str(self.workflow),
             "--downloaded-assets", str(self.downloaded)],
            text=True, capture_output=True, check=False)

    def test_matching_stable_release_and_exact_assets_pass(self):
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mismatched_or_prerelease_tag_rejected(self):
        for tag in ("v0.17.13", "v0.17.14-rc.1", "crispembed-v0.17.14"):
            with self.subTest(tag=tag):
                self.assertNotEqual(self.run_gate(tag).returncode, 0)

    def test_draft_and_missing_release_asset_rejected(self):
        self.release["draft"] = True
        self.assertNotEqual(self.run_gate().returncode, 0)
        self.release["draft"] = False
        self.release["assets"].pop()
        self.assertNotEqual(self.run_gate().returncode, 0)

    def test_modified_downloaded_archive_rejected(self):
        (self.downloaded / ASSETS["ios-arm64"]).write_bytes(b"wrong bytes")
        self.assertNotEqual(self.run_gate().returncode, 0)

    def test_missing_current_changelog_rejected(self):
        self.changelog.write_text("## 0.17.9\n")
        self.assertNotEqual(self.run_gate().returncode, 0)

    def test_fields_and_changelog_heading_must_not_span_lines(self):
        for pubspec in ("name:\ncrispembed\nversion: 0.17.14\n",
                        "name: crispembed\nversion:\n0.17.14\n"):
            with self.subTest(pubspec=pubspec):
                self.pubspec.write_text(pubspec)
                self.assertNotEqual(self.run_gate().returncode, 0)
        self.pubspec.write_text("name: crispembed\nversion: 0.17.14\n")
        self.changelog.write_text("##\n0.17.14\n")
        self.assertNotEqual(self.run_gate().returncode, 0)

    def test_android_release_version_is_derived_from_package_pubspec(self):
        gradle = (ROOT / "flutter/crispembed/android/build.gradle").read_text()
        self.assertIn("pubspec.yaml", gradle)
        self.assertIn("version = pubspecVersion", gradle)
        self.assertNotIn("version '0.16.0'", gradle)


if __name__ == "__main__":
    unittest.main()
