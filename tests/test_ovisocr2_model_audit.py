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
