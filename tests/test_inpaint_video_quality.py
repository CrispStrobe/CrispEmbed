#!/usr/bin/env python3
"""Model-free controls for the owned video fixtures, metrics and evidence writer."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import inpaint_video_quality
from inpaint_video_quality import CASES, FRAME_COUNT, HEIGHT, WIDTH, Clip, compare, digest, fixture, metrics, run_panel, run_process


class VideoQualityTests(unittest.TestCase):
    def test_protocol_v1_fixture_bytes_are_frozen(self):
        expected = {
            "static": "116303a08cbeb2cc04670b9e6ccdcdfe9a2fe12f92a59af881e3f8f0fe237248",
            "fade": "072cf8f261f531d4f6ead4fe95976a3bff72d6cd419bb04c7e65f8f16ba57035",
            "moving-fine-texture": "6de3c4de4e006ea50780db84e989e82b7e488b2dba23d22ed2f8c0f3f231a3a8",
            "subtitle-change": "e8ad313368390eec7aad5abd7ad5b3d93534a50b7bc090cfbfa8a891fb21d22c",
            "scene-cut": "135218960b184f5e979c9209e1acfc3796da03b86f2234aecf859ceaf38e969d",
        }
        self.assertEqual((WIDTH, HEIGHT, FRAME_COUNT), (96, 64, 6))
        self.assertEqual(set(CASES), set(expected))
        for name, sha256 in expected.items():
            with self.subTest(case=name):
                clip = fixture(name)
                self.assertEqual(digest(clip.records()), sha256)
                self.assertEqual(len(clip.records()), WIDTH * HEIGHT * 4 * FRAME_COUNT)
                self.assertEqual(len(clip.clean), FRAME_COUNT)

    def test_burned_input_only_changes_masked_pixels(self):
        for name in CASES:
            clip = fixture(name)
            records = clip.records()
            for t, (clean, mask) in enumerate(zip(clip.clean, clip.masks)):
                record = records[t * WIDTH * HEIGHT * 4:(t + 1) * WIDTH * HEIGHT * 4]
                self.assertEqual(record[WIDTH * HEIGHT * 3:], mask)
                for i, value in enumerate(mask):
                    self.assertEqual(record[3 * i:3 * i + 3],
                                     b"\xff\xff\xff" if value >= 128 else clean[3 * i:3 * i + 3])

    def test_cases_contain_the_declared_changes(self):
        static, moving, subtitle, cut = (fixture(name) for name in
                                        ("static", "moving-fine-texture", "subtitle-change", "scene-cut"))
        self.assertEqual(static.clean[0], static.clean[-1])
        self.assertFalse(any(static.masks[0]))
        self.assertFalse(any(static.masks[-1]))
        self.assertNotEqual(moving.clean[0], moving.clean[1])
        self.assertEqual(moving.clean[0], moving.clean[2])
        self.assertNotEqual(subtitle.masks[2], subtitle.masks[3])
        self.assertEqual(cut.clean[0], cut.clean[2])
        self.assertNotEqual(cut.clean[2], cut.clean[3])
        fade = fixture("fade")
        self.assertEqual(fade.clean[1][0] - fade.clean[0][0], 2)

    def test_perfect_output_has_zero_error_and_serializable_psnr(self):
        clip = fixture("fade")
        measured = metrics(clip, b"".join(clip.clean))
        self.assertEqual(measured["masked_mse"], 0)
        self.assertIsNone(measured["masked_psnr_db"])
        self.assertTrue(measured["masked_exact"])
        self.assertTrue(measured["known_rgb_exact"])
        self.assertEqual(measured["temporal_error_mae"], 0)
        json.dumps(measured, allow_nan=False)

    def test_metrics_match_an_independently_calculated_two_frame_vector(self):
        clip = Clip("vector", 2, 1, (bytes(6), bytes(6)), (bytes((255, 0)), bytes((255, 0))))
        measured = metrics(clip, bytes((1, 2, 3, 4, 0, 0, 2, 4, 6, 0, 0, 0)))
        # Masked errors [1,2,3,2,4,6]: squares total70 over6 channels.
        # Error change [1,2,3]: absolute total6 and squares14 over3.
        self.assertAlmostEqual(measured["masked_mse"], 70 / 6)
        self.assertAlmostEqual(measured["temporal_error_mae"], 2)
        self.assertAlmostEqual(measured["temporal_error_mse"], 14 / 3)
        self.assertEqual(measured["known_channel_errors"], 1)
        self.assertEqual(measured["known_max_abs"], 4)
        self.assertFalse(measured["known_rgb_exact"])

    def test_temporal_metric_includes_mask_appearance_and_disappearance(self):
        clip = Clip("mask-change", 1, 1, (bytes(3),) * 3,
                    (bytes((0,)), bytes((128,)), bytes((0,))))
        measured = metrics(clip, bytes(3) + bytes((3, 6, 9)) + bytes(3))
        self.assertEqual(measured["temporal_channels"], 6)
        self.assertEqual(measured["temporal_error_mae"], 6)
        self.assertTrue(measured["known_rgb_exact"])

    def test_mask_threshold_and_wrong_output_size(self):
        clip = Clip("threshold", 2, 1, (bytes(6),), (bytes((127, 128)),))
        measured = metrics(clip, bytes((1, 0, 0, 2, 0, 0)))
        self.assertEqual(measured["masked_channels"], 3)
        self.assertEqual(measured["masked_mse"], 4 / 3)
        self.assertEqual(measured["known_channel_errors"], 1)
        for bad in (bytes(5), bytes(7), bytes(12)):
            with self.subTest(length=len(bad)), self.assertRaises(ValueError):
                metrics(clip, bad)

    def test_regressions_are_retained_and_declared_bounds_fail(self):
        baseline = {"masked_mse": 5, "temporal_error_mae": 2}
        worse = {"masked_mse": 7, "temporal_error_mae": 3}
        diagnostic = compare(baseline, worse)
        self.assertTrue(diagnostic["masked_mse_regressed"])
        self.assertTrue(diagnostic["temporal_error_mae_regressed"])
        self.assertEqual(diagnostic["declared_bound_violations"], [])
        gated = compare(baseline, worse, max_mse_increase=1, max_temporal_increase=0.5)
        self.assertEqual(len(gated["declared_bound_violations"]), 2)
        self.assertEqual(compare(baseline, baseline, 0, 0)["declared_bound_violations"], [])

    def test_process_failure_keeps_stdout_stderr_and_exit_code(self):
        with tempfile.TemporaryDirectory(prefix="inpaint-video-test-") as directory:
            root = Path(directory)
            source, output, errors = root / "input", root / "output", root / "stderr"
            source.write_bytes(b"retained bytes")
            code = "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()); sys.stderr.write('failure'); sys.exit(7)"
            result = run_process([sys.executable, "-c", code], source, output, errors, os.environ.copy(), 5)
            self.assertEqual(result["returncode"], 7)
            self.assertFalse(result["timed_out"])
            self.assertEqual(output.read_bytes(), b"retained bytes")
            self.assertEqual(errors.read_text(), "failure")
            if sys.platform in ("linux", "darwin"):
                self.assertGreater(result["peak_rss_bytes"], 0)

    def test_process_timeout_is_a_recorded_failure(self):
        with tempfile.TemporaryDirectory(prefix="inpaint-video-test-") as directory:
            root = Path(directory)
            source, output, errors = root / "input", root / "output", root / "stderr"
            source.write_bytes(b"")
            result = run_process([sys.executable, "-c", "import time; time.sleep(2)"],
                                 source, output, errors, os.environ.copy(), 0.05)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["returncode"], 0)
            self.assertLess(result["elapsed_seconds"], 2)

    @unittest.skipUnless(hasattr(os, "wait4"), "wait4-specific exit/timeout race")
    def test_exit_between_poll_and_timeout_signal_keeps_one_reaper(self):
        # The child exits after the first empty wait4 poll, before the timeout
        # signal. Popen.kill would poll/reap it and make the next wait4 fail.
        for signal_error in (None, ProcessLookupError()):
            with self.subTest(signal_error=type(signal_error).__name__), \
                    tempfile.TemporaryDirectory(prefix="inpaint-video-test-") as directory:
                root = Path(directory)
                source, output, errors = root / "input", root / "output", root / "stderr"
                source.write_bytes(b"")
                process = SimpleNamespace(pid=12345, returncode=None,
                    kill=Mock(side_effect=AssertionError("Popen.kill must not poll/reap the child")))
                usage = SimpleNamespace(ru_maxrss=123)
                with patch.object(inpaint_video_quality.subprocess, "Popen", return_value=process), \
                        patch.object(inpaint_video_quality.os, "wait4",
                                     side_effect=[(0, 0, None), (process.pid, 0, usage)]) as wait4, \
                        patch.object(inpaint_video_quality.os, "kill", side_effect=signal_error) as kill, \
                        patch.object(inpaint_video_quality.time, "monotonic", side_effect=[0, 2, 3]), \
                        patch.object(inpaint_video_quality.time, "sleep"):
                    result = run_process(["test-child"], source, output, errors, {}, 1)
                self.assertTrue(result["timed_out"])
                self.assertEqual(result["returncode"], 0)
                self.assertEqual(wait4.call_count, 2)
                kill.assert_called_once_with(process.pid, signal.SIGKILL)
                process.kill.assert_not_called()
                if sys.platform in ("linux", "darwin"):
                    self.assertEqual(result["peak_rss_bytes"], 123 * (1 if sys.platform == "darwin" else 1024))

    def test_preflight_failure_writes_report_and_never_overwrites_it(self):
        with tempfile.TemporaryDirectory(prefix="inpaint-video-test-") as directory:
            root = Path(directory)
            args = argparse.Namespace(output_dir=root / "evidence", binary=root / "missing-binary",
                model=None, fake_engine=True, source_sha="test-source", window=2, threads=2, timeout=1,
                expected_model_sha256=None, max_mse_increase=None, max_temporal_mae_increase=None)
            report = run_panel(args)
            self.assertFalse(report["passed"])
            self.assertFalse(report["model_quality_qualified"])
            self.assertIn("error", report)
            saved = args.output_dir / "report.json"
            before = saved.read_bytes()
            self.assertEqual(json.loads(before)["schema"], "crispembed-inpaint-video-quality-v1")
            with self.assertRaises(FileExistsError):
                run_panel(args)
            self.assertEqual(saved.read_bytes(), before)

    def test_real_controls_are_recorded_and_fake_controls_are_cleared(self):
        controls = {"CRISPEMBED_MIGAN_CPU_KERNELS": "1", "CRISPEMBED_MIGAN_FAST_MATMUL": "0",
                    "CRISPEMBED_MIGAN_BATCH": "3", "CRISPEMBED_MIGAN_DEQUANT_WEIGHTS": "1",
                    "CRISPEMBED_MIGAN_QUANT": "1", "CRISPEMBED_INPAINT_SHA256_ACCEL": "1"}
        threads = {"OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "5"}
        declared = {**controls, **threads, "CRISPEMBED_MIGAN": "1", "CRISPEMBED_INPAINT_STREAM": "1"}
        with tempfile.TemporaryDirectory(prefix="inpaint-video-test-") as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"test identity only; subprocess is mocked")
            for fake_engine in (False, True):
                observed = []

                def complete_arm(command, input_path, output_path, stderr_path, environment, timeout):
                    observed.append(dict(environment))
                    output_path.write_bytes((input_path.parent / "clean.rgb").read_bytes())
                    stderr_path.write_text(f"inpaint: {FRAME_COUNT} frames, {FRAME_COUNT} masked, 0 engine;\n")
                    return {"returncode": 0, "timed_out": False, "elapsed_seconds": 0.01,
                            "peak_rss_bytes": None, "peak_rss_method": None}

                args = argparse.Namespace(output_dir=root / ("fake" if fake_engine else "real"),
                    binary=Path(sys.executable), model=model, fake_engine=fake_engine, source_sha="a" * 40,
                    window=2, threads=2, timeout=1, expected_model_sha256=None,
                    max_mse_increase=None, max_temporal_mae_increase=None)
                with self.subTest(fake_engine=fake_engine), patch.dict(os.environ, declared), \
                        patch.object(inpaint_video_quality, "CASES", ("fade",)), \
                        patch.object(inpaint_video_quality, "run_process", side_effect=complete_arm):
                    report = run_panel(args)
                self.assertTrue(report["passed"], report)
                self.assertFalse(report["model_quality_qualified"])
                self.assertEqual(report["runtime_environment"], threads)
                self.assertEqual(len(observed), 2)
                for key, value in controls.items():
                    self.assertEqual(report["gates"][key], None if fake_engine else value)
                    for environment in observed:
                        self.assertEqual(environment.get(key), None if fake_engine else value)
                for environment in observed:
                    self.assertEqual({key: environment.get(key) for key in threads}, threads)
                    self.assertEqual(environment["CRISPEMBED_MIGAN"], "1")
                    self.assertEqual(environment["CRISPEMBED_INPAINT_STREAM"], "1")


if __name__ == "__main__":
    unittest.main()
