"""Pure preflight tests; no model, native library or framework imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ovis_public", ROOT/"tools/ci-heavy/ovisocr2_public_surfaces.py")
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
HASH = "a" * 64
HEAD = "b" * 40


def receipt():
    arms = ("f16-delta", "f16-vision", "f16-decoder-prepared", "f16-decoder-native-vision",
            "f16-decoder-native-positions", "f16-decoder-native-frontend", "f16-context")
    return {"producer": {"status": "completed diagnostic audit", "failures": [],
             "source_commit": HEAD,
             "commands": {name: {"exit_code": 0, "passed": True, "comparison_coverage": True,
                                  "sha256": HASH} for name in arms},
             "exports": {"f16": {"model_sha256": HASH, "tensor_count": 473}},
             "readable": {"png_sha256": HASH, "raw_rgb_sha256": HASH},
             "original_capture": {"decoded": ["Hello OCR\n12345"],
                                  "generated_ids": [[1, 2, 248044]],
                                  "generation_result": {"max_new_tokens": 12,
                                                        "generated_token_count": 3,
                                                        "eos_ids": [248044, 248046]},
                                  "observational_control": {"prefill_logits_exact": True,
                                                            "greedy_ids_exact": True},
                                  "cached_vs_full_prefix_same_argmax": True,
                                  "cached_vs_full_prefix_max_abs": .001}},
            "independent_review": {"f16_positive": True}}


class PublicSurfacePreflight(unittest.TestCase):
    def test_reviewed_receipt_has_exact_contract(self):
        pins = subject.require_receipt(receipt())
        self.assertEqual(pins["expected_text"], "Hello OCR\n12345")
        self.assertEqual(pins["model_sha256"], HASH)

    def test_rejects_absent_or_weakened_review(self):
        for mutation in (lambda d: d.pop("independent_review"),
                         lambda d: d["independent_review"].update(f16_positive=False),
                         lambda d: d["producer"]["commands"]["f16-context"].update(passed=False),
                         lambda d: d["producer"]["commands"]["f16-vision"].update(exit_code=1),
                         lambda d: d["producer"]["commands"]["f16-delta"].update(comparison_coverage=False),
                         lambda d: d["producer"]["commands"]["f16-decoder-native-frontend"].update(comparison_coverage=False),
                         lambda d: d["producer"]["original_capture"]["generation_result"].update(max_new_tokens=13),
                         lambda d: d["producer"]["original_capture"].update(cached_vs_full_prefix_max_abs=.0021),
                         lambda d: d["producer"].update(status="unreviewed")):
            data = receipt()
            mutation(data)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                subject.require_receipt(data)

    def test_run_head_and_workflow_are_bound(self):
        correct = {"id": 123, "status": "completed", "conclusion": "success",
                   "head_sha": HEAD, "path": ".github/workflows/ovisocr2-metadata.yml"}
        self.assertEqual(subject.service_gate("123", HEAD, lambda _: json.dumps(correct).encode())["id"], 123)
        # An independently reviewed positive F16 arm may coexist with a failed
        # F32 arm; the service conclusion is retained, not substituted.
        failed_f32 = {**correct, "conclusion": "failure"}
        self.assertEqual(subject.service_gate("123", HEAD, lambda _: json.dumps(failed_f32).encode())["conclusion"], "failure")
        for changed in ({**correct, "head_sha": "c"*40},
                        {**correct, "conclusion": "cancelled"},
                        {**correct, "path": ".github/workflows/other.yml"}):
            with self.assertRaises(ValueError):
                subject.service_gate("123", HEAD, lambda _: json.dumps(changed).encode())

    def test_artifacts_require_unique_name_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/"ovisocr2-f16.gguf").write_bytes(b"model")
            (folder/"owned-readable.png").write_bytes(b"image")
            pins = {"model_sha256": subject.sha(folder/"ovisocr2-f16.gguf"),
                    "image_sha256": subject.sha(folder/"owned-readable.png")}
            self.assertEqual(set(subject.require_artifacts(folder, pins)), {"model", "image"})
            (folder/"owned-readable.png").write_bytes(b"wrong")
            with self.assertRaises(ValueError): subject.require_artifacts(folder, pins)

    def test_wrong_profile_requires_qualified_lighton_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"lighton.gguf"
            path.write_bytes(b"unqualified")
            with self.assertRaises(ValueError):
                subject.require_lighton_artifact(path)

    def test_public_score_comparison_catches_missing_binding_or_drift(self):
        arms = {"c-abi": {"passed": True, "confidence_count": 3, "mean_confidence": .75},
                "python": {"passed": True, "confidence_count": 3, "mean_confidence": .75},
                "cli-stage21": {"passed": True, "mean_confidence": .750},
                "http-stage21": {"passed": True, "mean_confidence": .7500},
                "rust-run": {"passed": True, "metric_lines":
                             ["PASS Rust unified RGB cap=12 prompt-rejection text_bytes=3 confidence_count=3 mean=0.750000000"]},
                "dart-run": {"passed": True, "metric_lines":
                             ["PASS Dart unified RGB cap=12 prompt-rejection text_bytes=3 confidence_count=3 mean=0.75"]}}
        self.assertTrue(subject.compare_confidences(arms)["confidence_count"] == 3)
        arms["dart-run"]["metric_lines"] = []
        with self.assertRaises(ValueError): subject.compare_confidences(arms)

    def test_compiled_python_dart_abi_requires_exact_stage_layout(self):
        native = ("ABI stage_size=168 cleanup_size=40 params_size=88 "
                  "vlm_max_tokens_offset=136 vlm_prompt_offset=144\n")
        dart = ["ABI stage_size=168 cleanup_size=40 params_size=88"]
        python = {"stage_size": 168, "cleanup_size": 40, "params_size": 88,
                  "vlm_max_tokens_offset": 136, "vlm_prompt_offset": 144}
        self.assertEqual(subject.compare_abi(native, dart, python)["dart"]["stage_size"], 168)
        with self.assertRaises(ValueError):
            subject.compare_abi(native, ["ABI stage_size=160 cleanup_size=40 params_size=88"], python)
        with self.assertRaises(ValueError):
            subject.compare_abi(native, dart, {**python, "vlm_prompt_offset": 140})
        with self.assertRaises(ValueError):
            subject.compare_abi("no ABI record", dart, python)

    def test_two_stage_cache_probe_requires_second_model_load_failure(self):
        source = Path("/hosted/ovisocr2-metadata-complete-truncated.gguf")
        trace = {"stages_tried": 2, "metrics": [
            {"index": 1, "engine": "ovisocr2", "accepted": False, "text_chars": 15},
            {"index": 2, "engine": "ovisocr2", "accepted": False, "text_chars": 0}],
            "fallback_text": "Hello OCR\n12345"}
        log = ("qwen35-ocr: truncated/corrupt GGUF '" + str(source) + "' — tensor data past EOF\n"
               "ocr_orchestrator: qwen35/ovis load failed\n"
               "RESULT stage-cache " + json.dumps(trace) + "\n")
        result = subject.validate_stage_cache_probe({"exit_code": 0}, log,
                                                     "Hello OCR\n12345", source)
        self.assertEqual(result["metrics"][1]["text_chars"], 0)
        for changed in (
                log.replace('"text_chars": 0', '"text_chars": 15'),
                log.replace("truncated/corrupt GGUF", "metadata rejected"),
                log.replace("ocr_orchestrator: qwen35/ovis load failed", "load reused"),
                log.replace('"stages_tried": 2', '"stages_tried": 1')):
            with self.assertRaises(ValueError):
                subject.validate_stage_cache_probe({"exit_code": 0}, changed,
                                                   "Hello OCR\n12345", source)

    def test_truncated_second_file_keeps_exact_metadata_boundary(self):
        class Field:
            def __init__(self, value): self.value = value
            def contents(self): return self.value
        class Reader:
            def __init__(self, _):
                self.tensors = [None] * 473
                self.fields = {"qwen35_ocr.format_version": Field(2),
                               "qwen35_ocr.profile": Field("ovisocr2")}
                self.data_offset = 64
        with tempfile.TemporaryDirectory() as tmp:
            original, truncated = Path(tmp)/"original.gguf", Path(tmp)/"truncated.gguf"
            original.write_bytes(bytes(range(256)) * 8)
            with mock.patch.dict("sys.modules", {"gguf": types.SimpleNamespace(GGUFReader=Reader)}):
                evidence = subject.metadata_complete_truncation(original, truncated)
            self.assertEqual(truncated.read_bytes(), original.read_bytes()[:128])
            self.assertEqual(evidence["metadata_bytes"], 64)
            self.assertEqual(evidence["retained_tensor_bytes"], 64)


if __name__ == "__main__":
    unittest.main()
