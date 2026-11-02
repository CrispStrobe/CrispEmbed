#!/usr/bin/env python3
"""Weight-free contract checks against the actual checkpoint header/config."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/mineru25"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


converter = load_module("mineru25_converter", ROOT / "models/convert-mineru25-pro-to-gguf.py")
validator = load_module("mineru25_validator", ROOT / "tools/mineru25_validate.py")


def read(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class ConversionContract(unittest.TestCase):
    def setUp(self):
        self.config = read("config.json")
        self.pp = read("preprocessor_config.json")
        self.gen = read("generation_config.json")
        self.header = read("safetensors-header.json")
        self.header.pop("__metadata__", None)

    def test_original_metadata(self):
        converter.validate_config(self.config, self.pp, self.gen)

    def test_exact_upstream_tensor_inventory_and_shapes(self):
        specs = converter.validate_tensors(self.header)
        self.assertEqual(len(specs), 681)
        self.assertEqual(len({s.target for s in specs}), 681)
        self.assertNotIn("lm_head.weight", self.header)
        self.assertEqual(self.header["model.embed_tokens.weight"]["shape"], [151936, 896])

    def test_missing_and_unexpected_tensors_fail_closed(self):
        del self.header["visual.blocks.31.mlp.fc2.weight"]
        with self.assertRaisesRegex(ValueError, "inventory"):
            converter.validate_tensors(self.header)
        self.header = read("safetensors-header.json")
        self.header.pop("__metadata__", None)
        self.header["lm_head.weight"] = {"shape": [151936, 896]}
        with self.assertRaisesRegex(ValueError, "inventory"):
            converter.validate_tensors(self.header)

    def test_wrong_gqa_shape_is_rejected(self):
        self.header["model.layers.0.self_attn.k_proj.weight"]["shape"] = [896, 896]
        with self.assertRaisesRegex(ValueError, "k_proj.weight"):
            converter.validate_tensors(self.header)

    def test_top_level_context_does_not_override_nested_blueprint_limit(self):
        self.assertEqual(self.config["max_position_embeddings"], 32768)
        self.assertEqual(self.config["text_config"]["max_position_embeddings"], 8192)
        self.config["text_config"]["max_position_embeddings"] = 32768
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            converter.validate_config(self.config, self.pp, self.gen)

    def test_wrong_normalization_and_activation_fail(self):
        self.pp["min_pixels"] = 3136
        with self.assertRaisesRegex(ValueError, "min_pixels"):
            converter.validate_config(self.config, self.pp, self.gen)
        self.pp = read("preprocessor_config.json")
        self.config["vision_config"]["hidden_act"] = "gelu"
        with self.assertRaisesRegex(ValueError, "hidden_act"):
            converter.validate_config(self.config, self.pp, self.gen)

    def test_f16_recipe_keeps_sensitive_tensors_f32(self):
        specs = {s.target: s for s in converter.tensor_specs()}
        for name in ["l.embed_tokens.weight", "v.blk.0.norm1.weight", "l.blk.0.attn_q.bias"]:
            self.assertEqual(converter.storage_dtype(specs[name], "f16"), "float32")
        self.assertEqual(converter.storage_dtype(specs["v.patch_embed.weight"], "f16"), "float16")

    def test_metadata_identifies_native_profile_and_processor(self):
        values = {}
        class Writer:
            def __getattr__(self, name):
                if name.startswith("add_"):
                    def add(key, value):
                        if key in values:
                            raise AssertionError("duplicate metadata: " + key)
                        values[key] = value
                    return add
                raise AttributeError(name)
        converter.write_metadata(Writer(), self.config, self.pp, ["x"], [1], ["x y"], "local-unverified", {}, "f16")
        self.assertEqual(values["qwen2vl.model_family"], "mineru2.5-pro-2604")
        self.assertEqual(values["qwen2vl.mineru.processor"], "qwen2vl-fast")
        self.assertEqual(values["qwen2vl.rope_sections"], [8, 12, 12, 0])
        self.assertEqual(values["qwen2vl.vision.window_size"], 0)
        self.assertEqual(values["qwen2vl.vision.min_pixels"], 50176)
        self.assertEqual(values["qwen2vl.max_position_embeddings"], 8192)

    def test_bf16_loading_and_invalid_byte_ranges(self):
        import numpy as np
        header = {"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.safetensors"
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + struct.pack("<HH", 0x3f80, 0xc020))
            entries = converter.read_safetensors_header(path)
            np.testing.assert_array_equal(converter.read_tensor(path, entries["x"], (2,), "float32"), [1.0, -2.5])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "byte range"):
                converter.read_safetensors_header(path)

    def test_byte_identity_required_for_qualification(self):
        manifest = json.loads((ROOT / "models/mineru25-pro-source.json").read_text())
        manifest["files"] = {"config.json": manifest["files"]["config.json"]}
        verified = converter.verify_source_files(FIXTURES, manifest)
        self.assertEqual(verified["config.json"], manifest["files"]["config.json"]["sha256"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            raw = (FIXTURES / "config.json").read_bytes()
            path.write_bytes(raw[:-1] + b" ")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                converter.verify_source_files(Path(directory), manifest)

    def test_tokenizer_retains_layout_and_table_tokens(self):
        # Exercise dense IDs and token typing without distributing the full vocab.
        vocab = {f"token{i}": i for i in range(151643)}
        tokenizer = {"model": {"type": "BPE", "vocab": vocab, "merges": [["a", "b"]]},
                     "added_tokens": [{"id": i, "content": t} for i, t in converter.SPECIAL_TOKENS.items()]}
        tokens, types, merges = converter.tokenizer_payload(tokenizer, read("tokenizer_config.json"))
        self.assertEqual(len(tokens), 151674)
        self.assertEqual(tokens[151663], "<fcel>")
        self.assertEqual(tokens[151673], "<|paratext|>")
        self.assertEqual(types[151648], 3)
        self.assertEqual(merges, ["a b"])
        tokenizer["added_tokens"].pop()
        with self.assertRaisesRegex(ValueError, "dense IDs"):
            converter.tokenizer_payload(tokenizer, read("tokenizer_config.json"))


class ComparisonContract(unittest.TestCase):
    def report(self):
        report = {"task": "text", "prompt": "\nText Recognition:", "max_tokens": 8, "effective_max_tokens": 8,
                "grid_thw": [1, 16, 16], "input_ids": [151644, 1],
                "token_ids": [123, 151645], "eos": True, "capped": False,
                "text": "Hello", "raw_text": "Hello<|im_end|>", "device": "cpu", "threads": 4,
                "machine_fingerprint": "same-host", "timing": {"runs_ms": [100, 20, 22]}}
        report["runs"] = [copy.deepcopy(report) for _ in range(3)]
        return report

    def test_exact_text_cannot_mask_token_or_stop_mismatch(self):
        ref, native = self.report(), self.report()
        self.assertTrue(validator.comparison(ref, native, 1.0)["request_parity_pass"])
        native["token_ids"] = [123, 151643]
        result = validator.comparison(ref, native, 1.0)
        self.assertTrue(result["text_metrics"]["exact"])
        self.assertFalse(result["request_parity_pass"])
        self.assertFalse(result["speed_pass"])

    def test_capped_output_never_qualifies(self):
        ref, native = self.report(), self.report()
        for report in (ref, native):
            report["eos"], report["capped"] = False, True
        self.assertFalse(validator.comparison(ref, native, 1.0)["request_parity_pass"])

    def test_warm_state_drift_cannot_hide_behind_correct_cold_output(self):
        ref, native = self.report(), self.report()
        native["runs"][2]["token_ids"] = [42, 151645]
        self.assertFalse(validator.comparison(ref, native, 1.0)["request_parity_pass"])

    def test_unmatched_hardware_timing_is_diagnostic(self):
        ref, native = self.report(), self.report()
        native["machine_fingerprint"] = "different-host"
        result = validator.comparison(ref, native, 1.0)
        self.assertTrue(result["request_parity_pass"])
        self.assertFalse(result["speed_comparable"])
        self.assertFalse(result["speed_pass"])

    def test_metrics_unicode_empty_and_edits(self):
        self.assertEqual(validator.text_metrics("é猫", "é犬")["cer"], 0.5)
        self.assertEqual(validator.text_metrics("a b", "a c")["wer"], 0.5)
        self.assertEqual(validator.text_metrics("", "")["cer"], 0)
        self.assertEqual(validator.text_metrics("", "x")["character_errors"], 1)

    def test_nonfinite_or_missing_warm_times_are_rejected(self):
        for times in ([5], [5, float("nan")], [5, 0], [5, -1]):
            with self.assertRaises(ValueError):
                validator.median_warm({"timing": {"runs_ms": times}})


if __name__ == "__main__":
    unittest.main()
