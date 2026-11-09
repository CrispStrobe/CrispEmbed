#!/usr/bin/env python3
"""Exercise the actual Cargo build-script directives without native binaries.

The empty library files below intentionally test configuration selection only.
They are not architecture, native-link, inference or mobile-device evidence.
The build-script executable is compiled by hosted CI before this test runs.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class PrebuiltConfigTests(unittest.TestCase):
    def setUp(self):
        self.binary = Path(os.environ["CRISPEMBED_BUILD_SCRIPT_TEST_BIN"]).resolve()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.libs = self.root / "lib"
        self.libs.mkdir()

    def run_script(self, target, target_os, filename=None, explicit=True, host=None):
        if filename:
            library = self.libs / filename
            library.parent.mkdir(parents=True, exist_ok=True)
            library.touch()
        environment = os.environ.copy()
        environment.pop("CRISPEMBED_SYS_LIB_DIR", None)
        environment.update(
            HOST=host or "x86_64-unknown-linux-gnu",
            TARGET=target,
            CARGO_CFG_TARGET_OS=target_os,
            OUT_DIR=str(self.root / "out"),
            # If a rejected cross configuration reaches CMake, fail the test
            # with a different error rather than running any source build.
            PATH=str(self.root / "empty-path"),
        )
        if explicit:
            environment["CRISPEMBED_SYS_LIB_DIR"] = str(self.libs)
        return subprocess.run(
            [str(self.binary)], env=environment, capture_output=True, text=True, check=False
        )

    def test_ios_uses_archive_and_frameworks_without_host_rpath(self):
        result = self.run_script("aarch64-apple-ios-sim", "ios", "libcrispembed.a")
        self.assertEqual(result.returncode, 0, result.stderr)
        links = {line for line in result.stdout.splitlines() if "rustc-link-lib=" in line}
        self.assertEqual(
            links,
            {
                "cargo:rustc-link-lib=static=crispembed",
                "cargo:rustc-link-lib=dylib=c++",
                "cargo:rustc-link-lib=framework=Foundation",
                "cargo:rustc-link-lib=framework=Metal",
                "cargo:rustc-link-lib=framework=MetalKit",
                "cargo:rustc-link-lib=framework=Accelerate",
            },
        )
        self.assertNotIn("rpath", result.stdout)
        self.assertIn(f"cargo:LIB_DIR={self.libs}", result.stdout)

    def test_android_uses_only_verified_shared_closure(self):
        result = self.run_script("aarch64-linux-android", "android", "libcrispembed.so")
        self.assertEqual(result.returncode, 0, result.stderr)
        links = [line for line in result.stdout.splitlines() if "rustc-link-lib=" in line]
        self.assertEqual(links, ["cargo:rustc-link-lib=dylib=crispembed"])
        self.assertNotIn("rpath", result.stdout)

    def test_native_desktop_retains_shared_runtime_and_rpath(self):
        host = "x86_64-unknown-linux-gnu"
        result = self.run_script(host, "linux", "libcrispembed.so", host=host)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cargo:rustc-link-lib=dylib=crispembed", result.stdout)
        self.assertIn("cargo:rustc-link-lib=dylib=stdc++", result.stdout)
        self.assertIn("-Wl,-rpath,$ORIGIN", result.stdout)

    def test_missing_or_wrong_cross_library_fails_before_host_fallback(self):
        for target, target_os, wrong_file in [
            ("aarch64-apple-ios", "ios", "libcrispembed.dylib"),
            ("aarch64-linux-android", "android", "libcrispembed.a"),
            ("aarch64-apple-darwin", "macos", "libcrispembed.so"),
        ]:
            for explicit in [False, True]:
                with self.subTest(target=target, explicit=explicit):
                    result = self.run_script(target, target_os, wrong_file, explicit=explicit)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("CRISPEMBED_SYS_LIB_DIR", result.stderr)
                    self.assertIn("disabled for mobile/cross builds", result.stderr)
                    self.assertNotIn("failed to start", result.stderr)
                    self.assertNotIn("sources not found", result.stderr)
                    self.assertNotIn("cargo:rustc-link-lib=", result.stdout)
                    for library in self.libs.iterdir():
                        library.unlink()


if __name__ == "__main__":
    unittest.main(verbosity=2)
