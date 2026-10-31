"""Retained-evidence checks for the hosted Xiaomi F32 export control."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tests/results/xiaomi-ocr-0"


def load(name):
    return json.loads((RESULTS / name).read_text())


class XiaomiExportEvidenceTest(unittest.TestCase):
    def test_current_export_control_and_source_closure(self):
        import hashlib
        report = load("kaggle-export-v2-report.json")
        manifest = load("kaggle-export-v2-manifest.json")
        closure = load("kaggle-export-v2-source-closure.json")
        self.assertEqual(report["phase"], "complete")
        self.assertEqual(report["source_closure"], closure)
        self.assertEqual(report["roundtrip_tensors"], 473)
        self.assertEqual(report["checkpoint_equal_and_reloaded_tensors"], 473)
        self.assertEqual(manifest["checkpoint_tensor_count"], 488)
        self.assertEqual(manifest["main_tensor_count"], 473)
        self.assertEqual(len(manifest["excluded_source_tensors"]), 15)
        self.assertEqual(report["export"]["sha256"], load("kaggle-export-v1-report.json")["export"]["sha256"])
        for name, digest in closure["files_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)
        self.assertEqual([c["name"] for c in report["gpu_controls"]],
                         ["qianziwen", "receipt", "dense-equations"])
        self.assertTrue(all(c["ids_equal"] and c["text_equal"] for c in report["gpu_controls"]))
        self.assertTrue(report["gpu_controls"][-1]["capped"])

    def test_report_and_manifest_describe_a_finished_control(self):
        report = load("kaggle-export-v1-report.json")
        self.assertEqual(report["phase"], "complete")
        self.assertIn("no native parity", report["status"])
        self.assertIs(report["native_alias_qualified"], False)
        self.assertEqual(report["roundtrip_tensors"], 473)
        self.assertEqual(report["checkpoint_equal_and_reloaded_tensors"], 473)
        self.assertEqual([c["name"] for c in report["gpu_controls"]],
                         ["qianziwen", "receipt", "dense-equations"])
        for control in report["gpu_controls"]:
            with self.subTest(case=control["name"]):
                self.assertTrue(control["ids_equal"])
                self.assertTrue(control["text_equal"])
        self.assertTrue(report["gpu_controls"][-1]["capped"])

    def test_manifest_covers_every_checkpoint_declaration(self):
        manifest = load("kaggle-export-v1-manifest.json")
        header = load("checkpoint-header.json")
        names = [row[0] for row in header["tensors"]]
        sources = [row["source"] for row in manifest["records"]]
        auxiliary = sorted(name for name in names if name.startswith("mtp."))
        self.assertEqual(len(names), 488)
        self.assertEqual(len(auxiliary), 15)
        self.assertEqual(len(sources), 473)
        self.assertEqual(len(set(sources)), 473)
        self.assertEqual(sorted(set(names) - set(sources)), auxiliary)
        self.assertNotIn("lm_head.weight", sources)
        self.assertTrue(manifest["tie_word_embeddings"])
        self.assertEqual(manifest["export_tensor_count"], 473)

    def test_every_exported_record_is_f32_with_distinct_name_and_hash(self):
        manifest = load("kaggle-export-v1-manifest.json")
        records = manifest["records"]
        self.assertEqual(len({row["name"] for row in records}), 473)
        for row in records:
            with self.subTest(name=row["name"]):
                self.assertEqual(row["dtype"], "float32")
                self.assertEqual(len(row["sha256"]), 64)
                self.assertEqual(len(row["shape"]), len(row["source_shape"]) if row["name"] != "v.patch_embed.proj.weight" else 2)

    def test_closure_is_the_bytes_the_hosted_run_actually_executed(self):
        import hashlib
        report = load("kaggle-export-v1-report.json")
        closure = load("kaggle-export-v1-source-closure.json")
        self.assertEqual(report["source_closure"], closure)
        self.assertEqual(hashlib.sha256((ROOT / "models/qwen35_xiaomi_profile.py").read_bytes())
                         .hexdigest(), closure["files_sha256"]["models/qwen35_xiaomi_profile.py"])
        self.assertEqual(hashlib.sha256((ROOT / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json")
                                        .read_bytes()).hexdigest(),
                         closure["files_sha256"]["tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json"])
        # The converter changed after the export; the recorded digest is the
        # executed one, and must not silently match the edited file.
        self.assertNotEqual(hashlib.sha256((ROOT / "models/convert-qwen35-ocr-to-gguf.py").read_bytes())
                            .hexdigest(),
                            closure["files_sha256"]["models/convert-qwen35-ocr-to-gguf.py"])

    def test_recorded_export_size_is_the_f32_payload_plus_container(self):
        report = load("kaggle-export-v1-report.json")
        manifest = load("kaggle-export-v1-manifest.json")
        self.assertEqual(report["export"]["bytes"], manifest["bytes"])
        self.assertGreater(manifest["bytes"], sum(
            __import__("math").prod(row["shape"]) * 4 for row in manifest["records"]))


if __name__ == "__main__":
    unittest.main()
