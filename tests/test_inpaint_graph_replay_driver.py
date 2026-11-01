#!/usr/bin/env python3
"""Bounded stdlib controls for the hosted replay orchestrator.

Every native invocation is mocked. Tiny byte fixtures stand in for models and
captures; these tests neither execute model code nor claim numerical parity.
"""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inpaint_graph_replay", ROOT / "tools/ci-heavy/inpaint_graph_replay.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fixture(root, engine="moebius", backend="cuda", precision="f32", compare_eager=False, cases=None):
    inputs = root / "prepared"
    inputs.mkdir()
    binary = inputs / ("test-lama-diff" if engine == "lama" else "test-latent-inpaint")
    binary.write_bytes(b"mocked native executable; never run\n")
    binary.chmod(0o755)
    model = inputs / "model.gguf"
    model.write_bytes(b"tiny mocked converted model\n")
    model_sha = driver.sha256(model)
    write_json(model.with_suffix(".gguf.json"), {"gguf_sha256": model_sha, "gguf_bytes": model.stat().st_size})
    reference = inputs / "reference"
    reference.mkdir()
    case_rows = [{"id": value} for value in (cases or ["subtitle_bar"])]
    if engine == "lama":
        for case in case_rows:
            directory = reference / case["id"]
            directory.mkdir()
            file = directory / "reference.gguf"
            file.write_bytes(("mock original " + case["id"]).encode())
            case["reference_sha256"] = driver.sha256(file)
    manifest = reference / ("reference.json" if engine == "lama" else "report.json")
    write_json(manifest, {"cases": case_rows, "producer": {"github_sha": "c" * 40, "github_run_id": "987"}})
    argv = ["--engine", engine, "--backend", backend, "--precision", precision, "--binary", str(binary),
            "--model", str(model), "--model-sha256", model_sha, "--reference-dir", str(reference),
            "--reference-manifest-sha256", driver.sha256(manifest), "--output", str(root / "evidence"),
            "--scratch", str(root / "scratch"), "--deadline-minutes", "1"]
    if engine != "lama":
        vae = inputs / "vae.gguf"
        vae.write_bytes(b"tiny mocked VAE\n")
        write_json(vae.with_suffix(".gguf.json"), {"gguf_sha256": driver.sha256(vae), "gguf_bytes": vae.stat().st_size})
        argv += ["--vae", str(vae), "--vae-sha256", driver.sha256(vae)]
    if compare_eager:
        argv.append("--compare-eager")
    return driver.arguments(argv)


def source_provenance(run):
    run.report["source"] = {"native_commit": "a" * 40, "ggml_commit": "b" * 40,
                             "source_tree_clean": True, "expected_github_sha": None}


class NativeStub:
    """Emit comparator-shaped evidence without interpreting any model bytes."""
    def __init__(self, *, failed_cases=(), failed_arms=(), profile=True, actual_backend=None,
                 actual_precision=None, returncode_override=None, marker=True):
        self.failed_cases, self.failed_arms = set(failed_cases), set(failed_arms)
        self.profile, self.actual_backend, self.actual_precision = profile, actual_backend, actual_precision
        self.returncode_override, self.marker = returncode_override, marker
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        env = kwargs["env"]
        arm = "graph" if env["CRISPEMBED_INPAINT_GRAPH"] == "1" else "eager"
        lama = Path(argv[0]).name == "test-lama-diff"
        output = Path(argv[argv.index("--output" if lama else "--out") + 1])
        output.mkdir(parents=True)
        case = output.name
        passed = case not in self.failed_cases and arm not in self.failed_arms
        kwargs["stdout"].write(b"unaltered numerical bytes: \xff\x00\n")
        kwargs["stderr"].write(b"unaltered diagnostic bytes: \xfe\n")
        if lama:
            final = {"case": case, "stages": 227, "failed_stages": 0 if passed else 1, "passed": passed}
            prefix = b"LAMA_FINAL "
        else:
            native = {"schema": "crispembed-latent-inpaint-native-v1", "model": argv[argv.index("--model") + 1],
                      "complete": True, "synthetic_parity_pass": passed, "full_synthetic_panel_pass": False,
                      "native_qualified": False, "errors": [] if passed else ["fixed budget exceeded"],
                      "budgets": {"fixed_native_sentinel": 0.002}}
            native["original_report_sha256"] = driver.sha256(Path(argv[argv.index("--reference-dir") + 1]) / "report.json")
            write_json(output / "report.json", native)
            final = {"passed": passed, "report": str(output / "report.json")}
            prefix = b"LATENT_INPAINT_PARITY "
        if self.marker:
            kwargs["stdout"].write(prefix + json.dumps(final).encode() + b"\n")
        if arm == "graph" and self.profile:
            backend = self.actual_backend or {"cpu": "CPU", "cuda": "CUDA0", "metal": "MTL0"}[env["CRISPEMBED_INPAINT_BACKEND"]]
            precision = self.actual_precision or ("mixed-f16" if env["CRISPEMBED_INPAINT_FAST"] == "1" else "f32")
            line = (f"inpaint_graph label=actual-model backend={backend} precision={precision} nodes=12 "
                    "allocation_bytes=1024 runs=2 upload_bytes=100 download_bytes=50 compute_ms=1.250\n")
            kwargs["stderr"].write(line.encode())
        code = (0 if passed else 1) if self.returncode_override is None else self.returncode_override
        return mock.Mock(pid=12345, returncode=code, wait=mock.Mock(return_value=code))


def execute(run, stub):
    with mock.patch.object(run, "provenance", side_effect=lambda: source_provenance(run)), \
            mock.patch.object(driver.subprocess, "Popen", side_effect=stub), mock.patch("builtins.print"):
        return run.execute()


class ReplayDriverControls(unittest.TestCase):
    def test_same_native_budgets_and_inputs_for_eager_and_mixed_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp), engine="pixelhacker", backend="metal", precision="mixed", compare_eager=True)
            run = driver.Run(args)
            stub = NativeStub()
            with mock.patch.dict(os.environ, {"CRISPEMBED_INPAINT_GRAPH": "0", "CRISPEMBED_INPAINT_FAST": "0",
                                               "CRISPEMBED_MOEBIUS": "1", "CRISPEMBED_INPAINT_UNKNOWN": "1"}):
                self.assertEqual(execute(run, stub), 0)
            self.assertEqual(len(stub.calls), 2)
            eager, graph = stub.calls
            for option in ("--model", "--denoiser", "--vae", "--model-sha256", "--vae-sha256", "--reference-dir"):
                self.assertEqual(eager[0][eager[0].index(option) + 1], graph[0][graph[0].index(option) + 1])
            self.assertEqual(eager[1]["env"]["CRISPEMBED_INPAINT_BACKEND"], "cpu")
            self.assertEqual(eager[1]["env"]["CRISPEMBED_INPAINT_FAST"], "0")
            self.assertEqual(graph[1]["env"]["CRISPEMBED_INPAINT_BACKEND"], "metal")
            self.assertEqual(graph[1]["env"]["CRISPEMBED_INPAINT_FAST"], "1")
            self.assertEqual(graph[1]["env"]["CRISPEMBED_MOEBIUS"], "0")
            self.assertEqual(graph[1]["env"]["CRISPEMBED_PIXELHACKER"], "1")
            self.assertNotIn("CRISPEMBED_INPAINT_UNKNOWN", graph[1]["env"])
            self.assertNotIn("--atol", graph[0])
            self.assertTrue(run.report["passed"])
            self.assertFalse(run.report["native_qualified"])
            self.assertFalse(run.report["build_provenance_qualified"])
            self.assertFalse(run.report["framework_imported"])
            for record in run.report["commands"]:
                log = run.output / record["stdout"]["file"]
                self.assertTrue(log.read_bytes().startswith(b"unaltered numerical bytes: \xff\x00\n"))
                self.assertEqual(driver.sha256(log), record["stdout"]["sha256"])
                native = driver.read_json(run.output / record["native_report"]["file"])
                self.assertEqual(native["budgets"], {"fixed_native_sentinel": 0.002})

    def test_graph_failure_is_nonzero_even_with_passing_eager_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            run = driver.Run(fixture(Path(temp), compare_eager=True))
            self.assertEqual(execute(run, NativeStub(failed_arms={"graph"})), 1)
            self.assertTrue(run.report["complete"])
            self.assertTrue(run.report["eager_parity_pass"])
            self.assertFalse(run.report["graph_parity_pass"])
            self.assertFalse(run.report["passed"])
            self.assertEqual(run.report["commands"][-1]["returncode"], 1)
            self.assertIn("unchanged", run.report["commands"][-1]["errors"][0])

    def test_failing_eager_does_not_skip_requested_graph_or_become_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            run = driver.Run(fixture(Path(temp), compare_eager=True))
            stub = NativeStub(failed_arms={"eager"})
            self.assertEqual(execute(run, stub), 1)
            self.assertEqual(len(stub.calls), 2)
            self.assertTrue(run.report["graph_parity_pass"])
            self.assertFalse(run.report["eager_parity_pass"])
            self.assertFalse(run.report["passed"])

    def test_no_profile_cpu_fallback_and_wrong_precision_cannot_claim_graph_pass(self):
        for kwargs in ({"profile": False}, {"actual_backend": "CPU"}, {"actual_backend": "cuda-fallback-cpu"},
                       {"actual_precision": "f32"}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temp:
                run = driver.Run(fixture(Path(temp), backend="cuda", precision="mixed"))
                self.assertEqual(execute(run, NativeStub(**kwargs)), 1)
                record = run.report["commands"][0]
                self.assertEqual(record["returncode"], 0)
                self.assertFalse(record["backend_precision_verified"])
                self.assertFalse(record["passed"])
                self.assertTrue(record["errors"])

    def test_exit_code_and_native_final_are_both_required(self):
        for kwargs in ({"returncode_override": 7}, {"marker": False}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temp:
                run = driver.Run(fixture(Path(temp)))
                self.assertEqual(execute(run, NativeStub(**kwargs)), 1)
                self.assertFalse(run.report["graph_parity_pass"])

    def test_lama_runs_every_original_case_with_its_pinned_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            run = driver.Run(fixture(Path(temp), engine="lama", backend="cpu", cases=["subtitle_bar", "edge_touch", "wide"]))
            stub = NativeStub(failed_cases={"edge_touch"})
            self.assertEqual(execute(run, stub), 1)
            self.assertEqual(len(stub.calls), 3)
            self.assertEqual([row["case"] for row in run.report["commands"]], ["subtitle_bar", "edge_touch", "wide"])
            for (argv, _), case in zip(stub.calls, run.cases):
                self.assertEqual(argv[argv.index("--reference-sha256") + 1], case["reference_sha256"])
                self.assertTrue(argv[argv.index("--reference") + 1].endswith(case["id"] + "/reference.gguf"))
                self.assertNotIn("--threads", argv)
            self.assertTrue(run.report["complete"])
            self.assertFalse(run.report["passed"])

    def test_hash_and_receipt_mismatches_fail_before_native_launch_with_report(self):
        for mismatch in ("manifest", "model", "vae"):
            with self.subTest(mismatch=mismatch), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp))
                if mismatch == "manifest":
                    args.reference_manifest_sha256 = "0" * 64
                else:
                    path = args.model_receipt if mismatch == "model" else args.vae_receipt
                    value = driver.read_json(path)
                    value["gguf_sha256"] = "0" * 64
                    write_json(path, value)
                run = driver.Run(args)
                stub = NativeStub()
                self.assertEqual(execute(run, stub), 1)
                self.assertEqual(stub.calls, [])
                saved = driver.read_json(run.output / "report.json")
                self.assertFalse(saved["complete"])
                self.assertFalse(saved["passed"])
                self.assertTrue(saved["errors"])

    def test_new_output_separate_scratch_and_protected_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = fixture(root)
            original_output = args.output
            args.output = args.reference_dir / "evidence"
            with self.assertRaisesRegex(ValueError, "protected"):
                driver.Run(args)
            self.assertFalse(args.output.exists())
            args.output = original_output
            original_scratch = args.scratch
            args.scratch = args.output / "scratch"
            with self.assertRaisesRegex(ValueError, "separate"):
                driver.Run(args)
            args.scratch = original_scratch
            run = driver.Run(args)
            before = (run.output / "report.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "new output"):
                driver.Run(args)
            self.assertEqual((run.output / "report.json").read_bytes(), before)

    def test_reference_manifest_cannot_change_after_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            run = driver.Run(args)
            stub = NativeStub()
            def mutate_after_manifest_pin():
                source_provenance(run)
                manifest = args.reference_dir / "report.json"
                manifest.write_bytes(manifest.read_bytes() + b" ")
            with mock.patch.object(run, "provenance", side_effect=mutate_after_manifest_pin), \
                    mock.patch.object(driver.subprocess, "Popen", side_effect=stub), mock.patch("builtins.print"):
                self.assertEqual(run.execute(), 1)
            self.assertEqual(stub.calls, [])
            self.assertFalse(run.report["passed"])
            self.assertIn("changed", run.report["errors"][0])

    def test_binary_mutation_between_arms_cannot_keep_attested_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp), compare_eager=True)
            run = driver.Run(args)
            stub = NativeStub()
            command = run.command
            def mutate_between_arms(arm, case):
                if arm == "graph":
                    args.binary.write_bytes(args.binary.read_bytes() + b"changed executable\n")
                    # Exercise invalidation of an already qualified receipt.
                    run.report["build_provenance_qualified"] = True
                return command(arm, case)
            with mock.patch.object(run, "provenance", side_effect=lambda: source_provenance(run)), \
                    mock.patch.object(run, "command", side_effect=mutate_between_arms), \
                    mock.patch.object(driver.subprocess, "Popen", side_effect=stub), mock.patch("builtins.print"):
                self.assertEqual(run.execute(), 1)
            self.assertEqual(len(stub.calls), 1)
            self.assertFalse(run.report["build_provenance_qualified"])
            self.assertFalse(run.report["passed"])
            self.assertIn("executable changed", run.report["errors"][0])

    def test_mixed_cpu_and_unsafe_case_routing_are_rejected_before_launch(self):
        for control in ("cpu", "traversal", "duplicate"):
            with self.subTest(control=control), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp), engine="lama", backend="cpu", precision="mixed" if control == "cpu" else "f32")
                if control != "cpu":
                    manifest = args.reference_dir / "reference.json"
                    doc = driver.read_json(manifest)
                    if control == "traversal":
                        doc["cases"][0]["id"] = "../outside"
                    else:
                        doc["cases"].append(dict(doc["cases"][0]))
                    write_json(manifest, doc)
                    args.reference_manifest_sha256 = driver.sha256(manifest)
                run = driver.Run(args)
                stub = NativeStub()
                self.assertEqual(execute(run, stub), 1)
                self.assertEqual(stub.calls, [])

    @unittest.skipUnless(os.name == "posix", "hosted process groups use POSIX")
    def test_timeout_stops_entire_group_and_keeps_partial_raw_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            run = driver.Run(fixture(Path(temp)))
            child = mock.Mock(pid=12345, returncode=-9)
            child.wait.side_effect = [subprocess.TimeoutExpired("mocked comparator", 0.1), -15, -9]
            def launch(_argv, **kwargs):
                kwargs["stdout"].write(b"partial original/native error magnitudes\n")
                kwargs["stderr"].write(b"partial diagnostic\n")
                return child
            with mock.patch.object(run, "provenance", side_effect=lambda: source_provenance(run)), \
                    mock.patch.object(driver.subprocess, "Popen", side_effect=launch), \
                    mock.patch.object(driver.os, "killpg") as kill, mock.patch("builtins.print"):
                self.assertEqual(run.execute(), 124)
            self.assertEqual(kill.call_args_list, [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)])
            saved = driver.read_json(run.output / "report.json")
            self.assertEqual(saved["status"], "timed-out")
            self.assertFalse(saved["complete"])
            record = saved["commands"][0]
            self.assertEqual(record["returncode"], -9)
            self.assertEqual(record["state"], "timed-out")
            self.assertEqual((run.output / record["stdout"]["file"]).read_bytes(), b"partial original/native error magnitudes\n")

    def test_exhausted_deadline_preserves_report_without_starting_native(self):
        with tempfile.TemporaryDirectory() as temp:
            run = driver.Run(fixture(Path(temp)))
            run.deadline = 0
            stub = NativeStub()
            self.assertEqual(execute(run, stub), 124)
            self.assertEqual(stub.calls, [])
            self.assertFalse(run.report["complete"])
            self.assertEqual(run.report["status"], "timed-out")
            self.assertEqual(run.report["commands"][0]["stdout"]["bytes"], 0)

    def test_build_receipt_binds_exact_binary_and_producer_revisions(self):
        for mismatched in (False, True):
            with self.subTest(mismatched=mismatched), tempfile.TemporaryDirectory() as temp:
                args = fixture(Path(temp))
                args.build_receipt = Path(temp) / "prepared/build-receipt.json"
                receipt = {"schema": "crispembed-inpaint-graph-build-v1", "native_commit": "a" * 40,
                           "ggml_commit": "b" * 40, "source_tree_clean": True,
                           "producer": {"github_sha": "a" * 40, "github_run_id": "123456"},
                           "binaries": [{"name": "test-latent-inpaint", "path": "executables/test-latent-inpaint",
                                         "sha256": "0" * 64 if mismatched else driver.sha256(args.binary),
                                         "bytes": args.binary.stat().st_size}]}
                write_json(args.build_receipt, receipt)
                run = driver.Run(args)
                stub = NativeStub()
                self.assertEqual(execute(run, stub), 1 if mismatched else 0)
                self.assertEqual(run.report["build_provenance_qualified"], not mismatched)
                self.assertEqual(len(stub.calls), 0 if mismatched else 1)
                self.assertFalse(run.report["native_qualified"])

    def test_source_archive_is_recorded_without_inventing_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            run = driver.Run(fixture(Path(temp)))
            with mock.patch.object(run, "git", return_value=None), mock.patch.dict(os.environ, {"GITHUB_SHA": ""}):
                run.provenance()
            self.assertTrue(run.report["source"]["archive_without_git"])
            self.assertIsNone(run.report["source"]["native_commit"])
            self.assertIsNone(run.report["source"]["ggml_commit"])
            self.assertFalse(run.report["source"]["binary_revision_assumed_from_checkout"])

    def test_github_sha_mismatch_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            run = driver.Run(fixture(Path(temp)))
            with mock.patch.object(run, "git", return_value="a" * 40), mock.patch.dict(os.environ, {"GITHUB_SHA": "c" * 40}):
                with self.assertRaisesRegex(ValueError, "GITHUB_SHA"):
                    run.provenance()

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ambiguous.json"
            path.write_text('{"gguf_sha256":"a","gguf_sha256":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                driver.read_json(path)


if __name__ == "__main__":
    unittest.main()
