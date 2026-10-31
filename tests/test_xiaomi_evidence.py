"""Safety checks for the experimental Xiaomi original-capture gate."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("xiaomi_profile", ROOT / "models/qwen35_xiaomi_profile.py")
assert spec is not None and spec.loader is not None
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class XiaomiEvidenceTest(unittest.TestCase):
    def test_unfiltered_hub_listing_records_missing_assets(self):
        listing = json.loads((ROOT / "tests/results/xiaomi-ocr-0/hub-revision-file-list.json").read_text())
        self.assertEqual(listing["model"], profile.MODEL)
        self.assertEqual(listing["revision"], profile.REVISION)
        names = listing["siblings"]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("model.safetensors", names)
        self.assertIn("processor_config.json", names)
        self.assertEqual(set(listing["absent"]),
                         {"generation_config.json", "preprocessor_config.json",
                          "model.safetensors.index.json", "LICENSE"})
        self.assertFalse(set(listing["absent"]) & set(names))

    def test_default_is_completed_v3_capture(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(profile.evidence_path(), ROOT / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json")
            self.assertEqual(profile._evidence()["original_prefill"]["status"], "complete")

    def test_v3_source_or_alias_changes_are_rejected_even_if_rehashed(self):
        original = json.loads((ROOT / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json").read_text())
        for field in ("source", "alias"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                evidence = json.loads(json.dumps(original))
                if field == "source":
                    evidence["loaded_implementations"]["model"]["normalized_sha256"] = "0" * 64
                else:
                    evidence["tied_head_observation"]["same_storage"] = False
                evidence["evidence_canonical_sha256"] = profile._canonical(
                    {k: v for k, v in evidence.items() if k != "evidence_canonical_sha256"})
                path = Path(directory) / "evidence.json"
                path.write_text(json.dumps(evidence))
                with patch.dict(os.environ, {"CRISPEMBED_XIAOMI_EVIDENCE": str(path)}), \
                     patch.object(profile, "EVIDENCE_CANONICAL_SHA256", evidence["evidence_canonical_sha256"]):
                    with self.assertRaisesRegex(ValueError, "loaded implementation, original alias"):
                        profile._evidence()

    def test_v3_qualification_includes_loaded_model_hash(self):
        evidence_path = ROOT / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json"
        evidence = json.loads(evidence_path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in profile.metadata_files():
                (source / name).write_bytes(b"isolated hash-verifier fixture")
            (source / "config.json").write_bytes(
                (ROOT / "tests/results/xiaomi-ocr-0/source-config.json").read_bytes())
            def verified_hash(path):
                return evidence["checkpoint_files_sha256"][Path(path).name]
            with patch.dict(os.environ, {"CRISPEMBED_XIAOMI_EVIDENCE": str(evidence_path)}), \
                 patch.object(profile, "_sha256_file", side_effect=verified_hash):
                result = profile.load_profile(source, profile.MODEL, profile.REVISION)
            self.assertEqual(result["qualification"]["loaded_source_sha256"]["model"],
                             "641da34484f6b9769e28615c680cc4e3202afbf5bdbf1da7d56d5a3d678918c7")

    def test_generation_stop_is_the_observed_single_eos(self):
        path = ROOT / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json"
        with patch.dict(os.environ, {"CRISPEMBED_XIAOMI_EVIDENCE": str(path)}):
            evidence = profile._evidence()
        self.assertEqual(profile.generation_stop_ids(evidence), [248046])
        self.assertIn(248044, evidence["stop_ids"])  # recorded padding/source union

    def test_completed_captured_evidence_uses_its_actual_config_key(self):
        path = ROOT / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json"
        with patch.dict(os.environ, {"CRISPEMBED_XIAOMI_EVIDENCE": str(path)}):
            evidence = profile._evidence()
        self.assertEqual(evidence["config_canonical_sha256"],
                         "1367ff440a3d165b7b5547c68ea609e409ad5d2053c638f79af6511a321ec845")

    def test_failed_original_capture_is_not_qualification(self):
        evidence = json.loads((ROOT / "tests/results/xiaomi-ocr-0/kaggle-v1-evidence.json").read_text())
        self.assertEqual(evidence["status"], "failed blueprint capture")

        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            evidence_path.write_text(json.dumps(evidence))
            with patch.dict(os.environ, {"CRISPEMBED_XIAOMI_EVIDENCE": str(evidence_path)}):
                with self.assertRaisesRegex(ValueError, "completed"):
                    profile._evidence()

    def test_self_hashed_changed_prompt_does_not_qualify(self):
        evidence = json.loads((ROOT / "tests/results/xiaomi-ocr-0/kaggle-v1-evidence.json").read_text())
        # A synthetic complete result, with its changed prompt correctly hashed,
        # must not be mistaken for the actual original prompt.
        evidence["status"] = "completed original blueprint capture; no native parity claimed"

        evidence["rendered_prompt"] += "changed"
        evidence["rendered_prompt_sha256"] = hashlib.sha256(evidence["rendered_prompt"].encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            evidence_path.write_text(json.dumps(evidence))
            with patch.dict(os.environ, {"CRISPEMBED_XIAOMI_EVIDENCE": str(evidence_path)}):
                with self.assertRaisesRegex(ValueError, "prompt"):
                    profile._evidence()

    def test_rehashed_changed_capture_does_not_qualify(self):
        evidence = json.loads((ROOT / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json").read_text())
        evidence["stop_ids"] = [42]
        payload = {k: v for k, v in evidence.items() if k != "evidence_canonical_sha256"}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        evidence["evidence_canonical_sha256"] = hashlib.sha256(encoded).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            evidence_path.write_text(json.dumps(evidence))
            with patch.dict(os.environ, {"CRISPEMBED_XIAOMI_EVIDENCE": str(evidence_path)}):
                with self.assertRaisesRegex(ValueError, "evidence"):
                    profile._evidence()

    def test_mutated_source_assets_do_not_qualify(self):
        evidence_path = ROOT / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json"
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in profile.metadata_files():
                (source / name).write_bytes(b"not the audited original")
            (source / "config.json").write_bytes(
                (ROOT / "tests/results/xiaomi-ocr-0/source-config.json").read_bytes())
            with patch.dict(os.environ, {"CRISPEMBED_XIAOMI_EVIDENCE": str(evidence_path)}):
                with self.assertRaisesRegex(ValueError, "source file.*digest"):
                    profile.load_profile(source, profile.MODEL, profile.REVISION)

    def test_qualification_records_capture_canonical_digest(self):
        evidence_path = ROOT / "tests/results/xiaomi-ocr-0/kaggle-v3-evidence.json"
        evidence = json.loads(evidence_path.read_text())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for name in profile.metadata_files():
                (source / name).write_bytes(b"isolated hash-verifier fixture")
            (source / "config.json").write_bytes(
                (ROOT / "tests/results/xiaomi-ocr-0/source-config.json").read_bytes())
            def verified_hash(path):
                return evidence["checkpoint_files_sha256"][Path(path).name]
            with patch.dict(os.environ, {"CRISPEMBED_XIAOMI_EVIDENCE": str(evidence_path)}), \
                 patch.object(profile, "_sha256_file", side_effect=verified_hash):
                result = profile.load_profile(source, profile.MODEL, profile.REVISION)
            self.assertEqual(result["qualification"]["evidence_canonical_sha256"],
                             evidence["evidence_canonical_sha256"])


if __name__ == "__main__":
    unittest.main()
