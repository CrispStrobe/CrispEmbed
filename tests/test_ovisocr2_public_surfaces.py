"""Pure preflight tests; no model, native library or framework imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
