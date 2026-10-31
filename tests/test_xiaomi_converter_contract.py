"""Focused routing checks for the experimental Xiaomi converter."""
import importlib.util
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qwen35_export", ROOT / "models/convert-qwen35-ocr-to-gguf.py")
assert spec is not None and spec.loader is not None
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class XiaomiConverterContractTest(unittest.TestCase):
    def test_manifest_accounts_for_original_auxiliary_tensors(self):
        actual = {f"main.{i}": ("checkpoint", [1]) for i in range(473)}
        excluded = [f"mtp.{i}" for i in range(15)]
        counts = converter.tensor_disposition(actual, excluded)
        self.assertEqual(counts["checkpoint_tensor_count"], 488)
        self.assertEqual(counts["main_tensor_count"], 473)
        self.assertEqual(counts["excluded_source_tensors"], excluded)
        self.assertEqual(converter.tensor_disposition(actual, [])["checkpoint_tensor_count"], 473)
        with self.assertRaisesRegex(ValueError, "tensor disposition"):
            converter.tensor_disposition(actual, ["main.0"])
        with self.assertRaisesRegex(ValueError, "tensor disposition"):
            converter.tensor_disposition(actual, ["mtp.0", "mtp.0"])

    def test_xiaomi_requires_exact_audited_tensor_counts(self):
        check = converter.validate_xiaomi_tensor_counts
        profile = {"name": "xiaomi", "main_tensors": 473, "auxiliary_tensors": 15}
        check(profile, 473, 15)
        for main, auxiliary in ((472, 15), (473, 14), (474, 15), (473, 16)):
            with self.subTest(main=main, auxiliary=auxiliary):
                with self.assertRaisesRegex(ValueError, "tensor count"):
                    check(profile, main, auxiliary)
        check({"name": "ovisocr2"}, 472, 0)

    def test_auxiliary_tensors_are_not_serialized(self):
        select = getattr(converter, "export_source_names", None)
        self.assertTrue(callable(select), "serialization must apply audited exclusions")
        actual = {"model.language_model.embed_tokens.weight": ("model.safetensors", [2, 2]),
                  "lm_head.weight": ("model.safetensors", [2, 2])}
        names = ["mtp.fc.weight", "lm_head.weight", "model.language_model.embed_tokens.weight"]
        self.assertEqual(list(select("model.safetensors", names, actual, ["mtp.fc.weight"], True)),
                         ["model.language_model.embed_tokens.weight"])

    def test_missing_validated_tensor_during_serialization_is_rejected(self):
        actual = {"model.language_model.embed_tokens.weight": ("model.safetensors", [2, 2])}
        with self.assertRaisesRegex(ValueError, "Tensor changed"):
            list(converter.export_source_names("model.safetensors", [], actual, [], True))

    def test_transformers_source_hash_is_required_for_pinned_profiles(self):
        check = getattr(converter, "validate_transformers_implementation", None)
        self.assertTrue(callable(check), "pinned implementation must be verified")
        source = b"qualified model source"
        with self.assertRaisesRegex(ValueError, "implementation"):
            check("5.8.1", source, {"transformers_version": "5.8.1"})
        expected = {"transformers_version": "5.8.1",
                    "loaded_source_sha256": {"model": hashlib.sha256(source).hexdigest()}}
        check("5.8.1", source, expected)
        with self.assertRaisesRegex(ValueError, "implementation"):
            check("5.8.1", b"different model source", expected)

    def test_xiaomi_config_uses_xiaomi_policy_not_ovis_policy(self):
        config = json.loads((ROOT / "tests/results/xiaomi-ocr-0/source-config.json").read_text())
        pinned = {"name": "xiaomi"}
        xiaomi = Mock()
        ovis = Mock()
        ovis.validate_config.side_effect = AssertionError("Xiaomi was sent to Ovis validation")
        with patch.object(converter, "_xiaomi_profile", return_value=xiaomi), \
             patch.object(converter, "_ovis_profiles", return_value=ovis):
            converter.validate_config(config, pinned)
        xiaomi.validate_config.assert_called_once_with(config, pinned)
        ovis.validate_config.assert_not_called()


if __name__ == "__main__":
    unittest.main()
