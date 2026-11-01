#!/usr/bin/env python3
"""Bounded stdlib controls; no compiler, GPU, model or network process runs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inpaint_graph_hosted", ROOT / "tools/ci-heavy/inpaint_graphs.py")
hosted = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hosted)


class HostedGraphControls(unittest.TestCase):
    def test_migan_candidate_requires_observed_kernel_selection(self):
        for selected in (False, True):
            evidence = {"native_records": [{"kind": "synthetic-controls", "cpu_kernels": selected}]}
            hosted.migan_kernel_evidence(evidence, selected)
            with self.assertRaises(RuntimeError):
                hosted.migan_kernel_evidence(evidence, not selected)
        for evidence in ({"native_records": []}, {"native_records": [{"kind": "synthetic-controls"}]},
                         {"native_records": [{"requested_cpu_kernels": True}]}):
            with self.assertRaises(RuntimeError):
                hosted.migan_kernel_evidence(evidence, True)

    def test_cpu_candidates_and_attention_diagnostic_are_separate(self):
        profiles = hosted.component_profiles("cpu")
        self.assertEqual([row[0] for row in profiles], ["portable", "cpu-optimized", "cpu-attention-attribution"])
        self.assertTrue(all(value == "0" for value in profiles[0][2].values()))
        self.assertEqual(profiles[1][2]["CRISPEMBED_INPAINT_CPU_ATTENTION_TILES"], "0")
        self.assertEqual(profiles[1][2]["CRISPEMBED_MIGAN_CPU_KERNELS"], "1")
        self.assertEqual(profiles[1][2]["CRISPEMBED_INPAINT_CPU_FFT"], "1")
        self.assertEqual(profiles[2][1], ("test-inpaint-graph",))
        self.assertEqual({name for name, value in profiles[2][2].items() if value == "1"},
                         {"CRISPEMBED_INPAINT_CPU_ATTENTION_TILES"})

    def test_gpu_controls_never_select_cpu_candidates(self):
        for backend in ("cuda", "metal"):
            profiles = hosted.component_profiles(backend)
            self.assertEqual(len(profiles), 1)
            self.assertTrue(all(value == "0" for value in profiles[0][2].values()))
        with self.assertRaises(ValueError):
            hosted.component_profiles("auto")

    def test_per_command_environment_is_recorded_without_mutating_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = {name: "0" for name in hosted.CPU_GATES}
            candidate = dict(baseline, CRISPEMBED_INPAINT_CPU_FFT="1")
            runner = hosted.Runbook(directory, time.monotonic() + 20, {"commands": []}, baseline)
            process = Mock(pid=12345, returncode=0)
            process.wait.return_value = 0
            with patch.object(hosted.subprocess, "Popen", return_value=process) as launch:
                record = runner.run("candidate", ["synthetic-only"], environment=candidate)
            self.assertEqual(launch.call_args.kwargs["env"], candidate)
            self.assertEqual(record["cpu_candidate_gates"]["CRISPEMBED_INPAINT_CPU_FFT"], "1")
            self.assertEqual(runner.environment["CRISPEMBED_INPAINT_CPU_FFT"], "0")

    def test_output_and_scratch_cannot_nest_or_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hosted.separate_output_trees(root / "evidence", root / "work")
            for first, second in ((root, root), (root, root / "evidence"), (root / "evidence", root)):
                with self.assertRaisesRegex(ValueError, "separate"):
                    hosted.separate_output_trees(first, second)

    def test_uninitialized_or_wrong_ggml_cannot_claim_build_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commit = "a" * 40
            gitlink = "160000 commit " + commit + "\tggml"
            hosted.validate_ggml_identity(root, str(root / "ggml"), commit, gitlink)
            with self.assertRaisesRegex(ValueError, "own initialized"):
                hosted.validate_ggml_identity(root, str(root), commit, gitlink)
            with self.assertRaisesRegex(ValueError, "gitlink"):
                hosted.validate_ggml_identity(root, str(root / "ggml"), "b" * 40, gitlink)

    def test_any_observer_hook_is_rejected_in_each_production_archive(self):
        all_tokens = [token for values in hosted.OBSERVER_SYMBOLS.values() for token in values]
        for family in hosted.OBSERVER_SYMBOLS:
            hosted.assert_observer_symbols("ordinary runtime symbol", family, False)
            for token in all_tokens:
                with self.assertRaises(RuntimeError):
                    hosted.assert_observer_symbols(token, family, False)

    def test_testing_archives_require_every_expected_hook(self):
        for family, tokens in hosted.OBSERVER_SYMBOLS.items():
            hosted.assert_observer_symbols("\n".join(tokens), family, True)
            for omitted in range(len(tokens)):
                with self.assertRaises(RuntimeError):
                    hosted.assert_observer_symbols("\n".join(token for i, token in enumerate(tokens) if i != omitted),
                                                   family, True)

    def test_explicit_device_receipt_cannot_claim_cpu_fallback(self):
        evidence = hosted.device_evidence("PASS backend=Metal precision=f32\n", "metal")
        self.assertEqual(evidence["selected_devices"], ["Metal"])
        for text in ("PASS backend=CPU\n", "PASS backend=Vulkan0\n", "no backend receipt"):
            with self.assertRaises(RuntimeError):
                hosted.device_evidence(text, "metal")
        with self.assertRaises(RuntimeError):
            hosted.device_evidence("PASS backend=CUDA0\nPASS backend=CPU\n", "cuda")

    def test_requested_backend_text_is_not_device_evidence(self):
        with self.assertRaises(RuntimeError):
            hosted.device_evidence("requested_backend=cuda; computation complete", "cuda")
        evidence = hosted.device_evidence("backend=CUDA0; requested_backend=cuda; full_model_parity=not_run", "cuda")
        self.assertEqual(evidence["selected_devices"], ["CUDA0"])

    def test_full_model_requires_reuse_and_observer_gates(self):
        rows = [{"kind": "legacy-cpu", "backend": "CPU"},
                {"kind": "production", "backend": "CUDA0"},
                {"selected_backend": "CUDA0", "passed": True, "warm_reuse_verified": True,
                 "repeat_bit_exact": True, "observer_isolated": True, "observer_bit_exact": True}]
        text = "\n".join("MIGAN_GRAPH_STATS " + json.dumps(row) for row in rows)
        self.assertEqual(hosted.device_evidence(text, "cuda", full_model=True)["selected_devices"], ["CUDA0"])
        for key in ("passed", "warm_reuse_verified", "repeat_bit_exact", "observer_isolated", "observer_bit_exact"):
            broken = dict(rows[-1], **{key: False})
            with self.assertRaises(RuntimeError):
                hosted.device_evidence("MIGAN_GRAPH_BENCHMARK " + json.dumps(broken), "cuda", full_model=True)
        with self.assertRaises(RuntimeError):
            hosted.device_evidence("MIGAN_GRAPH_STATS " + json.dumps(rows[1]), "cuda", full_model=True)

    def test_nonfinite_json_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            hosted.device_evidence('MIGAN_GRAPH_STATS {"backend":"CPU","time":NaN}', "cpu")

    def test_actual_cli_filename_and_ambiguous_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "Release").mkdir()
            actual = build / "Release/crispembed.exe"
            actual.write_bytes(b"synthetic")
            self.assertEqual(hosted.binary_path(build, "crispembed-cli"), actual)
            (build / "crispembed").write_bytes(b"duplicate")
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                hosted.binary_path(build, "crispembed-cli")
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                hosted.binary_path(build, "test-migan-graph")

    def test_timeout_preserves_running_command_and_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = {"commands": []}
            runner = hosted.Runbook(output, time.monotonic() + 20, report)
            process = Mock(pid=12345, returncode=-15)
            process.wait.side_effect = subprocess.TimeoutExpired(["synthetic-command"], 1)

            def launch(command, **kwargs):
                receipt = json.loads((output / "graph-acceptance.json").read_text())
                self.assertEqual(receipt["commands"][0]["status"], "starting")
                self.assertEqual(receipt["commands"][0]["command"], ["synthetic-command"])
                kwargs["stdout"].write("partial compiler diagnostic\n")
                return process

            with patch.object(hosted.subprocess, "Popen", side_effect=launch), patch.object(hosted, "terminate_tree") as stop:
                with self.assertRaises(subprocess.TimeoutExpired):
                    runner.run("compile", ["synthetic-command"], timeout=1)
                stop.assert_called_once_with(process)
            receipt = json.loads((output / "graph-acceptance.json").read_text())
            record = receipt["commands"][0]
            self.assertEqual(record["status"], "timeout")
            self.assertEqual(record["returncode"], -15)
            self.assertEqual(record["pid"], 12345)
            self.assertEqual(record["sha256"], hosted.sha256(output / "compile.log"))
            self.assertIn("partial compiler diagnostic", (output / "compile.log").read_text())

    def test_cancellation_retains_interrupted_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runner = hosted.Runbook(output, time.monotonic() + 20, {"commands": []})
            process = Mock(pid=12345, returncode=-15)
            process.wait.side_effect = InterruptedError("cancelled by workflow")
            with patch.object(hosted.subprocess, "Popen", return_value=process), patch.object(hosted, "terminate_tree") as stop:
                with self.assertRaises(InterruptedError):
                    runner.run("native", ["synthetic-native"])
                stop.assert_called_once_with(process)
            receipt = json.loads((output / "graph-acceptance.json").read_text())
            self.assertEqual(receipt["commands"][0]["status"], "interrupted")

    def test_expired_deadline_cannot_launch_command(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            runner = hosted.Runbook(output, time.monotonic() - 1, {"commands": []})
            with patch.object(hosted.subprocess, "Popen") as launch:
                with self.assertRaises(TimeoutError):
                    runner.run("late", ["must-not-run"])
                launch.assert_not_called()
            receipt = json.loads((output / "graph-acceptance.json").read_text())
            self.assertEqual(receipt["commands"][0]["status"], "deadline-exceeded-before-start")

    @unittest.skipUnless(hosted.os.name == "posix", "POSIX process-group control")
    def test_compiler_descendants_are_killed_after_parent_exits(self):
        process = Mock(pid=12345)
        process.wait.return_value = -signal.SIGTERM
        with patch.object(hosted.os, "killpg") as kill:
            hosted.terminate_tree(process)
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(12345, signal.SIGTERM), (12345, signal.SIGKILL)])

    def test_full_model_requires_both_pins_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "tiny.gguf"
            payload = b"GGUF" + bytes(60)
            model.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            record = hosted.model_identity(model, digest, "ab" * 32)
            self.assertEqual(record["gguf_bytes"], 64)
            with self.assertRaises(ValueError):
                hosted.model_identity(model, digest, None)
            model.write_bytes(payload + b"changed")
            with self.assertRaisesRegex(ValueError, "differs"):
                hosted.model_identity(model, digest, "ab" * 32)

    def test_build_only_enables_selected_backend_and_optimized_cpu_kernels(self):
        for backend in ("cpu", "cuda", "metal"):
            command = hosted.configure_command("synthetic-build", backend)
            self.assertIn("-DGGML_LLAMAFILE=ON", command)
            self.assertIn("-DGGML_NATIVE=OFF", command)
            self.assertIn("-DCRISPEMBED_NATIVE=OFF", command)
            self.assertIn("-DGGML_CUDA=" + ("ON" if backend == "cuda" else "OFF"), command)
            self.assertIn("-DGGML_METAL=" + ("ON" if backend == "metal" else "OFF"), command)
        with self.assertRaises(ValueError):
            hosted.configure_command("synthetic-build", "auto")


if __name__ == "__main__":
    unittest.main()
