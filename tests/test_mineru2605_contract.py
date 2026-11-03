#!/usr/bin/env python3
"""Small source/profile and receipt controls; no model load or native build."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import mineru2605_qualify as qualification
import mineru2605_validate as protocol

spec = importlib.util.spec_from_file_location("mineru2605_converter", ROOT / "models/convert-mineru25-pro-to-gguf.py")
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


class Writer:
    def __init__(self):
        self.values = {}

    def __getattr__(self, name):
        if name.startswith("add_"):
            return lambda key, value: self.values.__setitem__(key, value)
        raise AttributeError(name)


class SourceContract(unittest.TestCase):
    def test_distinct_profiles_preserve_2604_default(self):
        old, new = converter.PROFILES["2604"], converter.PROFILES["2605"]
        self.assertEqual(old.repository, converter.MODEL_ID)
        self.assertEqual(old.family, converter.FAMILY)
        self.assertNotEqual(old.revision, new.revision)
        self.assertNotEqual(converter.source_manifest(old)["files"]["model.safetensors"]["sha256"],
                            converter.source_manifest(new)["files"]["model.safetensors"]["sha256"])
        with self.assertRaisesRegex(ValueError, "model repository"):
            converter.resolve_model(old.repository, new)
        with self.assertRaisesRegex(ValueError, "model repository"):
            converter.resolve_model(new.repository, old)

    def test_actual_metadata_hashes_and_tied_geometry(self):
        profile = converter.PROFILES["2605"]
        manifest = converter.source_manifest(profile)
        fixtures = ROOT / "tests/fixtures/mineru2605"
        values = []
        for name in ("config.json", "preprocessor_config.json", "generation_config.json"):
            data = (fixtures / name).read_bytes()
            self.assertEqual(len(data), manifest["files"][name]["size"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), manifest["files"][name]["sha256"])
            values.append(json.loads(data))
        converter.validate_config(*values)
        self.assertTrue(values[0]["tie_word_embeddings"])
        self.assertEqual(values[0]["text_config"]["max_position_embeddings"], 8192)
        changed = copy.deepcopy(values)
        changed[0]["text_config"]["max_position_embeddings"] = 32768
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            converter.validate_config(*changed)
        writer = Writer()
        converter.write_metadata(writer, values[0], values[1], ["a"], [1], ["a b"], profile.revision,
                                 {"model.safetensors": manifest["files"]["model.safetensors"]["sha256"]}, "f32", profile)
        self.assertEqual(writer.values["qwen2vl.model_family"], "mineru2.5-pro-2605")
        self.assertEqual(writer.values["qwen2vl.source.expected_revision"], profile.revision)
        self.assertEqual(writer.values["qwen2vl.blueprint.revision"], "mineru-vl-utils==2.0.7")
        self.assertEqual(writer.values["qwen2vl.mineru.native_document_blueprint_revision"], converter.BLUEPRINT_REVISION)

    def test_new_profile_rejects_2604_byte_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            name = "config.json"
            (p / name).write_bytes((ROOT / "tests/fixtures/mineru25" / name).read_bytes())
            manifest = {"files": {name: converter.source_manifest(converter.PROFILES["2605"])["files"][name]}}
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                converter.verify_source_files(p, manifest)

    def test_helper_source_guard_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unmodified"):
                protocol.verify_blueprint(Path(directory))

    def test_portable_build_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CMakeCache.txt"
            path.write_text("GGML_NATIVE:BOOL=OFF\nCRISPEMBED_NATIVE:BOOL=OFF\n")
            qualification.portable_cache(path)
            path.write_text("GGML_NATIVE:BOOL=ON\nCRISPEMBED_NATIVE:BOOL=OFF\n")
            with self.assertRaisesRegex(ValueError, "CMakeCache"):
                qualification.portable_cache(path)


class PrecisionContract(unittest.TestCase):
    def setUp(self):
        manifest = converter.source_manifest(converter.PROFILES["2605"])
        self.ref = {"model": protocol.MODEL_ID, "model_revision": protocol.MODEL_REVISION,
                    "blueprint_revision": protocol.BLUEPRINT_REVISION, "dtype": "f32", "device": "cpu",
                    "engine": "original-python", "packages": {"transformers": "5.10.1"},
                    "source_sha256": {n: f["sha256"] for n, f in manifest["files"].items()},
                    "token_ids": [12, 151645], "input_ids": [151644, 151655], "eos": True, "capped": False,
                    "final_logits": [{"position": i, "file": "c0_prefill_logits_last.f32" if i == 0 else f"c0_decode{i}_logits.f32",
                                      "elements": 151936, "sha256": "a" * 64} for i in range(2)]}
        self.forced = {"token_ids": self.ref["token_ids"], "prompt_ids": self.ref["input_ids"]}
        self.rows = [{"position": i, "cosine": .985, "finite": True, "elements": 151936} for i in range(2)]
        self.stages = {"pass": False, "pass_v2": False, "first_failed_stage": "vis_layer_0", "stages": [
            {"name": f"stage_{i}", "observed": True, "finite": True, "cos_min": .98} for i in range(64)]}

    def result(self, tier="f16"):
        return qualification.precision_result(self.ref, self.forced, self.rows, self.stages, tier, "portable")

    def test_current_policy_retains_strict_failures_as_diagnostics(self):
        self.assertTrue(self.result()["precision_pass"])
        self.assertFalse(self.result()["strict_stage_pass"])
        self.assertFalse(self.result("f32")["precision_pass"])
        self.assertEqual(self.result()["worst_stage"]["cos_min"], .98)

    def test_exclusive_threshold(self):
        self.rows[0]["cosine"] = .98
        self.assertFalse(self.result()["precision_pass"])

    def test_missing_decode_or_changed_prefix_cannot_pass(self):
        self.rows.pop()
        self.assertFalse(self.result()["precision_pass"])
        self.rows.append({"position": 1, "cosine": .9999, "finite": True, "elements": 151936})
        self.forced["token_ids"] = [13, 151645]
        self.assertFalse(self.result()["precision_pass"])

    def test_dequantized_reference_cannot_pass_as_original(self):
        self.ref["vision_weights_gguf"] = {"sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "untouched original"):
            self.result()

    def test_source_relabel_cannot_pass(self):
        self.ref["model_revision"] = converter.MODEL_REVISION
        with self.assertRaisesRegex(ValueError, "identity"):
            self.result()

    def test_incomplete_reference_or_stages_rejected(self):
        self.ref["final_logits"].pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.result()
        self.setUp()
        self.stages["stages"].pop()
        with self.assertRaisesRegex(ValueError, "diagnostics"):
            self.result()


if __name__ == "__main__":
    unittest.main()
