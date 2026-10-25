"""Pure controls for the frozen original OvisOCR3 GPU receipt."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ovisocr3_vllm_review_test", ROOT / "tools/ci-heavy/ovisocr3_vllm_review.py")
reviewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reviewer)


class VllmReviewControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr3-vllm-reference.json").read_text())
        cls.metadata = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr3-metadata.json").read_text())
        cls.logs = {row["log"]: b"Using AttentionBackendEnum.TRITON_ATTN backend.\n"
                    b"Using AttentionBackendEnum.TORCH_SDPA for MMEncoderAttention\n"
                    b"Using Triton/FLA GDN prefill kernel (requested=triton\n"
                    for row in cls.receipt["producer"]["commands"]}
        for row in cls.receipt["producer"]["commands"]:
            row["log_sha256"] = reviewer.sha(cls.logs[row["log"]])

    def reviewed(self, first=None, second=None, metadata=None, logs=None):
        return reviewer.review(
            first if first is not None else copy.deepcopy(self.receipt["initial_attempt"]["producer"]),
            second if second is not None else copy.deepcopy(self.receipt["producer"]),
            metadata if metadata is not None else copy.deepcopy(self.metadata),
            logs if logs is not None else self.logs,
        )

    def test_positive_report_and_public_scope(self):
        receipt = self.reviewed()
        self.assertTrue(receipt["independent_review"]["genuine_gpu_forward_completed"])
        self.assertFalse(receipt["initial_attempt"]["independent_review"]["genuine_gpu_forward_completed"])
        self.assertEqual([g["raw_text"] for g in receipt["independent_review"]["generations"]],
                         ["The quick brown fox jumps over the lazy dog.", "Hello OCR\n12345"])
        self.assertEqual(receipt["producer_report_sha256"], reviewer.SECOND_REPORT_SHA256)
        self.assertNotIn("kaggle.com/code/", json.dumps(receipt))

    def test_initial_attempt_cannot_be_presented_as_gpu_forward(self):
        first = copy.deepcopy(self.receipt["initial_attempt"]["producer"])
        first["commands"] = [{"name": "original-vllm-worker"}]
        with self.assertRaisesRegex(ValueError, "Initial hardware preflight"):
            self.reviewed(first=first)

    def test_missing_generated_score_and_wrong_prompt_rejected(self):
        second = copy.deepcopy(self.receipt["producer"])
        second["worker"]["generations"][1]["generated_scores"].pop()
        with self.assertRaisesRegex(ValueError, "Generated prompt/IDs/EOS/cap"):
            self.reviewed(second=second)
        second = copy.deepcopy(self.receipt["producer"])
        second["worker"]["generations"][1]["prompt_token_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "Generated prompt/IDs/EOS/cap"):
            self.reviewed(second=second)

    def test_source_metadata_backend_and_private_path_rejected(self):
        second = copy.deepcopy(self.receipt["producer"])
        second["worker"]["source_files_sha256"]["config.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "metadata/source/checkpoint"):
            self.reviewed(second=second)
        second = copy.deepcopy(self.receipt["producer"])
        empty_log = b"no backend selection"
        second["commands"][-1]["log_sha256"] = reviewer.sha(empty_log)
        with self.assertRaisesRegex(ValueError, "backend log"):
            self.reviewed(second=second, logs={**self.logs, "original-vllm-worker.log": empty_log})
        second = copy.deepcopy(self.receipt["producer"])
        second["private_locator"] = "https://www.kaggle.com/code/account/kernel/2"
        with self.assertRaisesRegex(ValueError, "private run locator"):
            self.reviewed(second=second)

    def test_no_framework_import(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("vllm", sys.modules)
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main()
