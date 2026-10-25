"""Source-only OvisOCR3 gates; no framework, weights, or model execution."""

import copy
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/ovisocr3-source"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load(ROOT / "models/qwen35_ovis3_profile.py", "qwen35_ovis3_profile_test")
converter = load(ROOT / "models/convert-qwen35-ocr-to-gguf.py", "qwen35_ovis3_converter_test")


class OvisOCR3ProfileTest(unittest.TestCase):
    def test_real_audit_and_frozen_source_files(self):
        report, shared, source_hashes = profile._evidence()
        self.assertEqual(len(report["prompt_files"]), 18)
        self.assertEqual(len(profile.metadata_files()), 31)
        self.assertEqual(report["checkpoint_header"]["tensor_count"], 473)
        self.assertEqual(source_hashes["model"], "788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263")
        self.assertEqual(shared["tokenizer.json"], "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42")
        for name in ("config.json", "generation_config.json", "prompts/plain_ocr.txt"):
            expected = (report["prompt_files"] if name.startswith("prompts/") else report["source_files"])[name]["sha256"]
            self.assertEqual(profile._file_sha256(FIXTURE / name), expected)

    def test_real_config_and_generation_profile(self):
        report, _, source_hashes = profile._evidence()
        config = json.loads((FIXTURE / "config.json").read_text())
        generation = json.loads((FIXTURE / "generation_config.json").read_text())
        prompt = (FIXTURE / "prompts/plain_ocr.txt").read_text()
        pinned = profile._expected_profile(report, source_hashes, generation, prompt)
        profile.validate_config(config, pinned)
        converter.validate_config(config, pinned)
        self.assertEqual(pinned["format_version"], 3)
        self.assertEqual(pinned["source_eos_ids"], [248044, 248046])
        self.assertFalse(pinned["effective_stops_qualified"])
        self.assertFalse(pinned["native_alias_qualified"])
        for field, value in (("name", "ovisocr2"), ("format_version", 2),
                             ("native_alias_qualified", True), ("task_prompt", "changed"),
                             ("rendered_prompt_qualified", True)):
            changed = copy.deepcopy(pinned)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                profile.validate_config(config, changed)
        changed = copy.deepcopy(config)
        changed["text_config"]["mtp_num_hidden_layers"] = 1
        with self.assertRaises(ValueError):
            profile.validate_config(changed, pinned)
        changed = copy.deepcopy(pinned)
        changed["generation"]["eos_token_id"] = 248046
        with self.assertRaises(ValueError):
            profile.validate_config(config, changed)

    def test_source_identity_missing_file_and_full_checkpoint_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "generation_config.json", "prompts/plain_ocr.txt"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(FIXTURE / name, target)
            with self.assertRaises(ValueError):
                profile.load_profile(root, "ATH-MaaS/OvisOCR2", profile.REVISION)
            with self.assertRaises(ValueError):
                profile.load_profile(root, profile.MODEL, "0" * 40)
            with self.assertRaises(FileNotFoundError):
                profile.load_profile(root, profile.MODEL, profile.REVISION)
            (root / "processor_config.json").write_text("{}")
            with self.assertRaises(ValueError):
                profile.load_profile(root, profile.MODEL, profile.REVISION)
            (root / "processor_config.json").unlink()
            (root / "model.safetensors.index.json").write_text("{}")
            with self.assertRaises(ValueError):
                profile.load_profile(root, profile.MODEL, profile.REVISION)
            (root / "model.safetensors.index.json").unlink()
            report, shared, _ = profile._evidence()
            hashes = {name: value["sha256"] for name, value in report["source_files"].items()}
            hashes.update({name: value["sha256"] for name, value in report["prompt_files"].items()})
            hashes.update(shared)
            real_hash = profile._file_sha256
            def audited_hash(path):
                path = Path(path)
                return real_hash(path) if path.exists() else hashes[path.relative_to(root).as_posix()]
            with patch.object(profile, "_file_sha256", side_effect=audited_hash):
                pinned = profile.load_profile(root, profile.MODEL, profile.REVISION)
            config = json.loads((root / "config.json").read_text())
            profile.validate_config(config, pinned)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                profile.verify_checkpoint(checkpoint, pinned)
            (root / "generation_config.json").write_text("{}")
            with patch.object(profile, "_file_sha256", side_effect=audited_hash):
                with self.assertRaises(ValueError):
                    profile.load_profile(root, profile.MODEL, profile.REVISION)

    def test_legacy_and_ovis2_dispatch_remain_separate(self):
        config = json.loads((FIXTURE / "config.json").read_text())
        with self.assertRaises(ValueError):
            converter.validate_config(config, {"name": "ovisocr2"})
        self.assertEqual(converter.validate_config(config)[0]["mtp_num_hidden_layers"], 0)
        self.assertEqual(converter.read_tokenizer_metadata.__defaults__, (None, False))


if __name__ == "__main__":
    unittest.main()
