#!/usr/bin/env python3
"""Small stdlib controls of hosted admission, downloads and evidence retention.

All network, subprocess and heavyweight model operations in these controls are
mocked. No native build, model download, framework import or forward is run.
"""

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("latent_hosted", ROOT / "tools/ci-heavy/latent_inpaint.py")
hosted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hosted)


class Response(io.BytesIO):
    status = 200
    url = "https://cdn.example.invalid/pinned"

    def __init__(self, data, advertised=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data) if advertised is None else advertised)}


def arguments(root, phase="build-only"):
    return argparse.Namespace(output=root / "evidence", scratch=root / "scratch", phase=phase,
                              model="moebius", cases="subtitle_bar", reference_python="", deadline_minutes=5)


class HostedControls(unittest.TestCase):
    def test_actual_cli_output_name_and_duplicate_binary_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run = hosted.Run(arguments(Path(temp)))
            build = run.scratch / "build"
            build.mkdir()
            actual = build / "crispembed"
            actual.write_bytes(b"not executed")
            self.assertEqual(run.binary("crispembed-cli"), actual)
            (build / "bin").mkdir()
            (build / "bin/crispembed").write_bytes(b"not executed either")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                run.binary("crispembed-cli")

    def test_build_only_never_calls_heavy_lane(self):
        with tempfile.TemporaryDirectory() as temp:
            run = hosted.Run(arguments(Path(temp)))
            with mock.patch.object(run, "provenance"), mock.patch.object(run, "build"), mock.patch.object(run, "heavy") as heavy:
                self.assertEqual(run.execute(), 0)
            heavy.assert_not_called()
            report = json.loads((run.output / "report.json").read_text())
            self.assertTrue(report["passed"])
            self.assertFalse(report["native_qualified"])
            self.assertIn("skipped", report["original_reference"])
            self.assertIn("skipped", report["native_replay"])

    def test_failure_retains_truthful_incomplete_report(self):
        with tempfile.TemporaryDirectory() as temp:
            run = hosted.Run(arguments(Path(temp), "original-native"))
            def failure():
                run.report["original_reference"] = "running"
                raise RuntimeError("original CUDA operator failed")
            with mock.patch.object(run, "provenance"), mock.patch.object(run, "build"), mock.patch.object(run, "heavy", side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    run.execute()
            report = json.loads((run.output / "report.json").read_text())
            self.assertFalse(report["complete"])
            self.assertFalse(report["passed"])
            self.assertIn("failed", report["original_reference"])
            self.assertEqual(report["native_replay"], "not-run")
            self.assertIn("CUDA", report["errors"][0])

    def test_timeout_kills_owned_process_group_and_retains_status(self):
        with tempfile.TemporaryDirectory() as temp:
            run = hosted.Run(arguments(Path(temp)))
            child = mock.MagicMock()
            child.__enter__.return_value = child
            child.pid = 12345
            child.wait.side_effect = [subprocess.TimeoutExpired(["not-executed"], 0.1), -9]
            with mock.patch.object(hosted.subprocess, "Popen", return_value=child), mock.patch.object(hosted.os, "killpg") as killpg:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run.command("timeout-control", ["not-executed"])
            killpg.assert_called_once_with(12345, hosted.signal.SIGKILL)
            self.assertEqual(run.report["commands"][0]["state"], "timed-out")
            self.assertTrue((run.output / "timeout-control.log").is_file())

    def test_download_hash_count_and_https_before_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            run = hosted.Run(arguments(Path(temp)))
            for index, (data, advertised, expected_count, expected_hash) in enumerate([
                    (b"abc", 3, 3, "0" * 64),
                    (b"ab", 3, 3, hashlib.sha256(b"abc").hexdigest()),
                    (b"abcd", 3, 3, hashlib.sha256(b"abc").hexdigest()),
                    (b"abc", 4, 3, hashlib.sha256(b"abc").hexdigest())]):
                target = run.scratch / (str(index) + ".bin")
                with mock.patch.object(hosted.urllib.request, "urlopen", return_value=Response(data, advertised)):
                    with self.assertRaisesRegex(ValueError, "differ|exceeded"):
                        run.download("https://example.invalid/pinned", target, expected_count, expected_hash)
                self.assertFalse(target.exists())
            with mock.patch.object(hosted.urllib.request, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "HTTPS"):
                    run.download("http://example.invalid/pinned", run.scratch / "unsafe", 3, "0" * 64)
                network.assert_not_called()

    def test_successful_and_empty_source_downloads_are_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            run = hosted.Run(arguments(Path(temp)))
            for index, data in enumerate((b"pinned source\n", b"")):
                destination = run.scratch / (str(index) + ".py")
                with mock.patch.object(hosted.urllib.request, "urlopen", return_value=Response(data)):
                    run.download("https://example.invalid/pinned", destination, len(data), hashlib.sha256(data).hexdigest())
                self.assertEqual(destination.read_bytes(), data)
                self.assertEqual(run.report["downloads"][-1]["state"], "verified")

    def test_source_path_escape_and_existing_outputs_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in ("/tmp/file", "../file", "part/../file", "part//file", "./file", "part/", "part\\file", "a\0b"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    hosted.contained_file(root, relative)
            self.assertEqual(hosted.contained_file(root, "layers/λ/module.py"), root / "layers/λ/module.py")
            first = hosted.Run(arguments(root))
            with self.assertRaisesRegex(ValueError, "new output"):
                hosted.Run(arguments(root))
            self.assertTrue((first.output / "report.json").is_file())

    def test_original_requires_preprovisioned_interpreter(self):
        with tempfile.TemporaryDirectory() as temp:
            run = hosted.Run(arguments(Path(temp), "original-native"))
            with mock.patch.object(run, "command") as command:
                with self.assertRaisesRegex(ValueError, "preprovisioned"):
                    run.original_preflight(run.scratch)
                command.assert_not_called()

    def test_observer_symbols_cannot_leak_individually(self):
        symbols = ["latent_inpaint::run_reference(", "latent_inpaint::set_observer(", "latent_inpaint::Engine::set_observer("]
        hosted.observer_symbols("ordinary production entry points", False)
        hosted.observer_symbols("\n".join(symbols), True)
        for symbol in symbols:
            with self.assertRaisesRegex(ValueError, "isolation"):
                hosted.observer_symbols(symbol, False)
        with self.assertRaisesRegex(ValueError, "isolation"):
            hosted.observer_symbols("\n".join(symbols[:2]), True)

    def test_png_control_contains_exact_owned_rgb_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            raw = bytes((i * 3) % 256 for i in range(512 * 512 * 3))
            hosted.png(path, raw, channels=3)
            data = path.read_bytes()
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            pos, compressed = 8, b""
            while pos < len(data):
                size = struct.unpack_from(">I", data, pos)[0]
                name, payload = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + size]
                self.assertEqual(struct.unpack_from(">I", data, pos + 8 + size)[0], zlib.crc32(name + payload))
                if name == b"IDAT":
                    compressed += payload
                pos += size + 12
            decoded = zlib.decompress(compressed)
            self.assertEqual(b"".join(decoded[y * 1537 + 1:(y + 1) * 1537] for y in range(512)), raw)


if __name__ == "__main__":
    unittest.main()
