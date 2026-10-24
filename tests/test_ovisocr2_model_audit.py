"""Standard-library contracts for the hosted Ovis model audit driver."""

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest import mock


PATH = Path(__file__).resolve().parents[1] / "tools/ci-heavy/ovisocr2_model_audit.py"
SPEC = importlib.util.spec_from_file_location("ovisocr2_model_audit", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def good_receipt():
    return {"producer": {"status": "completed bounded original vLLM T4 reference",
                         "worker_sha256": audit.WORKER_SHA,
                         "pins": {"model": audit.MODEL, "revision": audit.REVISION,
                                  "checkpoint_sha256": audit.policy().CHECKPOINT_SHA256,
                                  "vllm_version": "0.22.1", "vllm_commit": audit.VLLM_COMMIT,
                                  "worker_sha256": audit.WORKER_SHA,
                                  "original_mirror_revision": audit.MIRROR_REVISION,
                                  "original_mirror_zip_sha256": audit.MIRROR_SHA},
                         "worker": {"status": "completed bounded original vLLM reference",
                                    "gpu_control": {"passed": True},
                                    "resolved_vllm_dtype": "torch.float16",
                                    "resolved_decoder_attention_backend": "TRITON_ATTN",
                                    "attention_backend_override": {"readme_auto_backend_equivalence_qualified": False},
                                    "runtime_stop_probe": {"all_stop_ids": [248044, 248046]},
                                    "generations": [{"case": "square"}, {"case": "readable"}],
                                    "owned_readable": {"case": "readable", "rgb_sha256": "a" * 64}}},
            "independent_review": {"genuine_gpu_forward_completed": True}}


class HostedAuditTest(unittest.TestCase):
    def test_cpu_runtime_attribution_records_only_model_and_thread_controls(self):
        with mock.patch.object(audit.platform, "machine", return_value="x86_64"), \
             mock.patch.object(audit.os, "cpu_count", return_value=8), \
             mock.patch.object(audit.os, "sched_getaffinity", return_value={0, 1}), \
             mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                                          "OPENBLAS_NUM_THREADS": "2"}):
            result = audit.cpu_runtime_attribution()
        self.assertEqual(result["machine"], "x86_64")
        self.assertEqual(result["logical_cpu_count"], 8)
        self.assertEqual(result["affinity_cpu_count"], 2)
        self.assertEqual(result["thread_env"], {"OMP_NUM_THREADS": "2",
                                                "MKL_NUM_THREADS": "2",
                                                "OPENBLAS_NUM_THREADS": "2"})
        self.assertEqual(set(result), {"cpu_model", "machine", "logical_cpu_count",
                                       "affinity_cpu_count", "thread_env"})

    def test_prior_failed_run_service_requires_exact_source_and_artifact(self):
        run = {"id": audit.PRIOR_REFERENCE_RUN, "status": "completed", "conclusion": "failure",
               "head_sha": audit.PRIOR_REFERENCE_HEAD,
               "path": ".github/workflows/ovisocr2-metadata.yml"}
        artifact = {"id": audit.PRIOR_REFERENCE_ARTIFACT_ID,
                    "name": f"ovisocr2-model-reference-{audit.PRIOR_REFERENCE_RUN}",
                    "expired": False, "size_in_bytes": audit.PRIOR_REFERENCE_ARTIFACT_BYTES,
                    "digest": "sha256:" + audit.PRIOR_REFERENCE_ARTIFACT_SHA,
                    "workflow_run": {"id": audit.PRIOR_REFERENCE_RUN,
                                     "head_sha": audit.PRIOR_REFERENCE_HEAD}}
        def call(argv):
            return json.dumps(artifact if "/artifacts/" in argv[-1] else run).encode()
        self.assertEqual(audit.prior_reference_service(call)["conclusion"], "failure")
        for changed in ({**run, "conclusion": "success"}, {**run, "head_sha": "0" * 40}):
            with self.assertRaisesRegex(ValueError, "prior failed-run"):
                audit.prior_reference_service(lambda argv: json.dumps(
                    artifact if "/artifacts/" in argv[-1] else changed).encode())
        artifact["digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "prior failed-run"):
            audit.prior_reference_service(call)

    def test_prior_npz_members_and_archive_hash_path_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            npz = root / "small.npz"
            with zipfile.ZipFile(npz, "w") as archive:
                archive.writestr("alpha.npy", b"\x93NUMPYheader-and-payload")
            stages = {"alpha": [1]}
            members = audit.npz_member_manifest(npz, stages)
            self.assertEqual(set(members), {"alpha"})
            self.assertEqual(members["alpha"]["sha256"], hashlib.sha256(
                b"\x93NUMPYheader-and-payload").hexdigest())
            with self.assertRaisesRegex(ValueError, "stage members"):
                audit.npz_member_manifest(npz, {"alpha": [1], "missing": [1]})
            with zipfile.ZipFile(npz, "w") as archive:
                archive.writestr("../escape.npy", b"\x93NUMPYx")
            with self.assertRaisesRegex(ValueError, "stage members"):
                audit.npz_member_manifest(npz, {"../escape": [1]})

            with zipfile.ZipFile(npz, "w") as archive:
                archive.writestr("alpha.npy", b"\x93NUMPYx")
            archive_path = root / "prior.zip"
            prior_json = b'{"stages":{"alpha":[1]}}'
            def write_archive(json_name="ovisocr2-original.json"):
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(json_name, prior_json)
                    archive.write(npz, "ovisocr2-original.npz")
                return archive_path.stat().st_size, audit.sha(archive_path)
            size, digest = write_archive()
            with mock.patch.object(audit, "PRIOR_REFERENCE_ARTIFACT_BYTES", size), \
                 mock.patch.object(audit, "PRIOR_REFERENCE_ARTIFACT_SHA", digest), \
                 mock.patch.object(audit, "PRIOR_REFERENCE_JSON_SHA", hashlib.sha256(prior_json).hexdigest()), \
                 mock.patch.object(audit, "prior_reference_json", return_value={"stages": stages}):
                data, old_npz, manifest, path = audit.extract_prior_reference(
                    archive_path, root / "extracted", root, {}, {}, root / "readable.png")
                self.assertEqual(data["stages"], stages)
                self.assertEqual(manifest["stage_count"], 1)
                self.assertTrue(old_npz.is_file())
                self.assertTrue(path.is_file())
                self.assertFalse(archive_path.exists())
                write_archive()
                with mock.patch.object(audit, "PRIOR_REFERENCE_ARTIFACT_SHA", "0" * 64):
                    with self.assertRaisesRegex(ValueError, "ZIP size/SHA"):
                        audit.extract_prior_reference(archive_path, root / "bad", root,
                                                      {}, {}, root / "readable.png")
            size, digest = write_archive("../ovisocr2-original.json")
            with mock.patch.object(audit, "PRIOR_REFERENCE_ARTIFACT_BYTES", size), \
                 mock.patch.object(audit, "PRIOR_REFERENCE_ARTIFACT_SHA", digest):
                with self.assertRaisesRegex(ValueError, "members/path"):
                    audit.extract_prior_reference(archive_path, root / "bad", root,
                                                  {}, {}, root / "readable.png")

    def test_prior_member_comparison_detects_first_drift_extra_and_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_npz, fresh_npz = root / "old.npz", root / "fresh.npz"
            original = {"alpha": b"\x93NUMPYalpha", "beta": b"\x93NUMPYbeta"}
            def write(path, members):
                with zipfile.ZipFile(path, "w") as archive:
                    for name, value in members.items():
                        archive.writestr(name + ".npy", value)
            write(old_npz, original)
            old_data = {"stages": {"alpha": [1], "beta": [1]},
                        "generated_ids": [[248046]], "decoded": ["x"], "raw_decoded": ["x<eos>"]}
            manifest = {"members": audit.npz_member_manifest(old_npz, old_data["stages"])}
            fresh_data = {**old_data, "stages": {**old_data["stages"], "new.cache": [1]},
                          "missing_stages": []}
            with mock.patch.object(audit, "PRIOR_MISSING_CACHE", ["new.cache"]), \
                 mock.patch.object(audit, "PRIOR_REFERENCE_STAGES", 2):
                write(fresh_npz, {**original, "new.cache": b"\x93NUMPYnew"})
                result = audit.compare_prior_reference(old_data, fresh_data, old_npz,
                                                       fresh_npz, manifest)
                self.assertTrue(result["passed"])
                self.assertEqual(result["unchanged_prior_members"], 2)
                self.assertEqual(result["added_cache_members"], 1)
                write(fresh_npz, {**original, "alpha": b"\x93NUMPYchanged",
                                  "new.cache": b"\x93NUMPYnew"})
                diagnostic = {"max_abs": 2.0, "old_l2_norm": 1.0,
                              "fresh_l2_norm": 3.0, "diff_l2_norm": 2.0}
                result = audit.compare_prior_reference(
                    old_data, fresh_data, old_npz, fresh_npz, manifest,
                    diagnose=lambda *_: diagnostic)
                self.assertFalse(result["passed"])
                self.assertEqual(result["first_failure"], "alpha")
                self.assertEqual(result["first_changed_member"]["numeric"], diagnostic)
                self.assertEqual(result["prior_generated_ids"], [[248046]])
                self.assertEqual(result["fresh_decoded"], ["x"])
                for stages, expected in (({"alpha": [1], "beta": [1]}, "new.cache"),
                                         ({**fresh_data["stages"], "extra": [1]}, "extra"),
                                         ({**fresh_data["stages"], "alpha": [2]}, "alpha")):
                    failed = audit.compare_prior_reference(
                        old_data, {**fresh_data, "stages": stages}, old_npz,
                        fresh_npz, manifest, diagnose=lambda *_: diagnostic)
                    self.assertFalse(failed["passed"])
                    self.assertEqual(failed["first_failure"], expected)
                write(fresh_npz, {"alpha": b"\x93NUMPYchanged-alpha",
                                  "beta": b"\x93NUMPYchanged-beta", "new.cache": b"\x93NUMPYnew"})
                observed_beta_first = {**old_data, "stages": {"beta": [1], "alpha": [1]}}
                ordered = audit.compare_prior_reference(
                    observed_beta_first, fresh_data, old_npz, fresh_npz, manifest,
                    diagnose=lambda *_: diagnostic)
                self.assertEqual(ordered["first_failure"], "beta")
                write(fresh_npz, {**original, "new.cache": b"\x93NUMPYnew"})
                changed_text = {**fresh_data, "decoded": ["other"]}
                self.assertFalse(audit.compare_prior_reference(
                    old_data, changed_text, old_npz, fresh_npz, manifest)["passed"])

    def test_missing_gpu_receipt_stops_before_heavy_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(audit, "GPU_RECEIPT", Path(temp) / "absent.json"), \
                 mock.patch.object(audit, "check_versions", side_effect=AssertionError("heavy deps touched")), \
                 mock.patch.dict(os.environ, {"HEAVY_OUT": temp, "HEAVY_SCRATCH": temp}):
                with self.assertRaisesRegex(ValueError, "before heavy imports"):
                    audit.run_main(preflight_only=True)
            report = json.loads((Path(temp) / "ovisocr2-model-audit.json").read_text())
            self.assertEqual(report["status"], "failed hosted model audit")
            self.assertNotIn("versions", report)

    def test_gpu_receipt_requires_actual_worker_and_independent_review(self):
        receipt = good_receipt()
        self.assertEqual(audit.require_gpu_receipt(receipt)["gpu_dtype"], "torch.float16")
        for path, replacement in (("review", False), ("control", False), ("dtype", "torch.bfloat16"),
                                  ("stop", [248046]), ("cases", [{"case": "square"}])):
            changed = copy.deepcopy(receipt)
            if path == "review":
                changed["independent_review"]["genuine_gpu_forward_completed"] = replacement
            elif path == "control":
                changed["producer"]["worker"]["gpu_control"]["passed"] = replacement
            elif path == "dtype":
                changed["producer"]["worker"]["resolved_vllm_dtype"] = replacement
            elif path == "stop":
                changed["producer"]["worker"]["runtime_stop_probe"]["all_stop_ids"] = replacement
            else:
                changed["producer"]["worker"]["generations"] = replacement
            with self.assertRaises(ValueError):
                audit.require_gpu_receipt(changed)

    def test_service_gate_checks_exact_head_workflow_and_success(self):
        expected = audit.SERVICES["frontend"]
        payload = {"id": expected[0], "head_sha": expected[1], "path": expected[2],
                   "status": "completed", "conclusion": "success"}
        call = lambda args: json.dumps(payload).encode()
        self.assertEqual(audit.service_gate("frontend", expected, call)["head_sha"], expected[1])
        payload["head_sha"] = "0" * 40
        with self.assertRaises(ValueError):
            audit.service_gate("frontend", expected, call)

    def test_native_matrix_and_log_failure_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            f32 = audit.native_arms("f32", root, root / "model", root / "readable.png", root / "ref.gguf")
            f16 = audit.native_arms("f16", root, root / "model", root / "readable.png", root / "ref.gguf")
            self.assertEqual(len(f32), 7)
            self.assertEqual(len(f16), 7)
            self.assertEqual(f32["f32-context"][-1], "--one-context")
            self.assertNotIn("--one-context", f16["f16-context"])
            report = {}
            self.assertFalse(audit.command("empty", [sys.executable, "-c", "pass"], root, report))
            self.assertFalse(audit.command("false-pass", [sys.executable, "-c", "print('FAIL decoded')"], root, report))
            self.assertEqual(report["commands"]["false-pass"]["first_failure"], "FAIL decoded")

    def test_raw_diff_manifest_requires_all_hidden_and_score_vectors(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            stages = {f"{phase}.decoder.layer.{layer}": [1] for phase in
                      ("prefill", "cached", "full_prefix") for layer in range(24)}
            stages["generated.scores.0"] = [1]
            for stage in stages:
                (folder / (stage + ".f32")).write_bytes(b"\0" * 4)
            lines = ["PASS prefill.decoder.layer.0: cos_min=1 max_abs=0",
                     "PASS generated.scores.0: cos_min=1 max_abs=0"]
            receipt = audit.raw_diff_manifest(folder, stages, lines, "f16-decoder-native-frontend")
            self.assertTrue(receipt["coverage_complete"])
            self.assertEqual(receipt["file_count"], 73)
            self.assertEqual(receipt["files"][0]["original_source_shape"], [1])
            (folder / "cached.decoder.layer.3.f32").unlink()
            receipt = audit.raw_diff_manifest(folder, stages, lines, "f16-decoder-native-frontend")
            self.assertFalse(receipt["coverage_complete"])
            self.assertIn("cached.decoder.layer.3", receipt["missing_stages"])
            (folder / "cached.decoder.layer.3.f32").write_bytes(b"\0" * 8)
            receipt = audit.raw_diff_manifest(folder, stages, lines, "f16-decoder-native-frontend")
            self.assertFalse(receipt["coverage_complete"])
            self.assertIn("cached.decoder.layer.3: raw F32 byte count differs", receipt["invalid_files"])
            self.assertFalse(audit.raw_diff_manifest(folder, stages, [],
                                                      "f16-decoder-native-frontend")["coverage_complete"])

    def test_export_requires_qualified_file_and_manifest_sha(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.gguf"
            manifest = root / "model.gguf.manifest.json"
            model.write_bytes(b"frozen-bytes")
            manifest.write_text(json.dumps({"dtype": "f32", "source_model": audit.MODEL,
                "source_revision": audit.REVISION, "keep_f32": [],
                "checkpoint_tensor_count": 473, "export_tensor_count": 473,
                "records": [{}] * 473}))
            expected = {"model_sha256": audit.sha(model), "manifest_sha256": audit.sha(manifest),
                        "bytes": model.stat().st_size}
            evidence = {"producer": {"exports": {"f32": expected}},
                        "independent_review": {"exports": {"f32": {
                            "manifest_sha256": expected["manifest_sha256"],
                            "bytes_reported": expected["bytes"],
                            "model_sha256_reported": expected["model_sha256"]}}}}
            self.assertEqual(audit.verify_export(model, manifest, "f32", evidence)["tensor_count"], 473)
            model.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs"):
                audit.verify_export(model, manifest, "f32", evidence)

    def test_capture_gate_requires_every_score_and_strict_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reference.json"
            profile = {"checkpoint_sha256": "a" * 64,
                       "qualification": {"loaded_source_sha256": {"model": "b" * 64}}}
            payload = {"source_model": audit.MODEL, "source_revision": audit.REVISION,
                       "checkpoint_sha256": profile["checkpoint_sha256"],
                       "loaded_source_sha256": profile["qualification"]["loaded_source_sha256"],
                       "vision_layers": 12, "decoder_layers": 24, "missing_stages": [],
                       "observational_control": {"prefill_logits_exact": True, "greedy_ids_exact": True},
                       "cached_vs_full_prefix_same_argmax": True, "cached_vs_full_prefix_max_abs": 0.001,
                       "generated_ids": [[17, 248044]],
                       "generation_result": {"max_new_tokens": 12, "eos_ids": [248044, 248046],
                                             "generated_token_count": 2, "score_count": 2,
                                             "reached_eos": True, "stop_id": 248044,
                                             "length_limited": False, "generated_id_shape": [1, 2]},
                       "stages": {"generated.scores.0": [1, 248320],
                                  "generated.scores.1": [1, 248320],
                                  "backend_off.generated.ids": [1, 2]},
                       "decoded": ["text"], "raw_decoded": ["text"]}
            names = [f"vision.layer.{index}" for index in range(12)]
            names += [f"vision.{name}" for name in (
                "patch_embed", "position", "embedded", "rotary_cos", "rotary_sin",
                "merger.norm", "merger.up", "merger.activation", "merger", "output")]
            names += [f"vision.probe.{index}.{field}" for index in (5, 11)
                      for field in ("norm1", "qkv", "pooled", "attention",
                                    "norm2", "up", "activation", "down")]
            names += ["decoder.norm", "logits", "argmax_id"]
            magnitudes = {}
            for label, baseline in (("backend_off", "prefill"),
                                    ("backend_off_full_prefix", "full_prefix")):
                for name in names:
                    payload["stages"][f"{label}.{name}"] = [1, 2]
                    payload["stages"][f"{baseline}.{name}"] = [1, 2]
                    magnitudes[f"{label}.{name}"] = 0.0
            payload["backend_off_control"] = {
                "mkldnn_available": True, "enabled_before": True,
                "enabled_inside": False, "enabled_after": True,
                "generated_ids": [[17, 248044]], "decoded": ["text"],
                "raw_decoded": ["text"], "stage_max_abs": magnitudes}
            path.write_text(json.dumps(payload))
            self.assertEqual(audit.verify_original_capture(path, profile, {})["generated_ids"], [[17, 248044]])
            prior_only = copy.deepcopy(payload)
            prior_only["missing_stages"] = ["cached.cache.0.conv_states"]
            path.write_text(json.dumps(prior_only))
            with self.assertRaisesRegex(ValueError, "capture"):
                audit.verify_original_capture(path, profile, {})
            self.assertEqual(audit.verify_original_capture(
                path, profile, {}, expected_missing=prior_only["missing_stages"])["missing_stages"],
                prior_only["missing_stages"])
            path.write_text(json.dumps(payload))
            for change in ("missing_position", "missing_probe", "mismatched_shape", "nan_magnitude",
                           "changed_ids", "not_disabled"):
                changed = copy.deepcopy(payload)
                key = "backend_off.vision.position"
                if change == "missing_position":
                    del changed["stages"][key]
                elif change == "missing_probe":
                    del changed["stages"]["backend_off_full_prefix.vision.probe.11.pooled"]
                elif change == "mismatched_shape":
                    changed["stages"][key] = [2, 1]
                elif change == "nan_magnitude":
                    changed["backend_off_control"]["stage_max_abs"][key] = math.nan
                elif change == "changed_ids":
                    changed["backend_off_control"]["generated_ids"] = [[248046]]
                else:
                    changed["backend_off_control"]["enabled_inside"] = True
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "capture"):
                    audit.verify_original_capture(path, profile, {})
            path.write_text(json.dumps(payload))
            del payload["stages"]["generated.scores.1"]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "capture"):
                audit.verify_original_capture(path, profile, {})
            payload["stages"]["generated.scores.1"] = [1, 248320]
            payload["cached_vs_full_prefix_max_abs"] = 0.003
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "cache"):
                audit.verify_original_capture(path, profile, {})


if __name__ == "__main__":
    unittest.main()
