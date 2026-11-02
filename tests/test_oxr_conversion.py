"""Weight-free OXR converter/protocol contracts; no neural parity is implied."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/oxr"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


converter = load_module("oxr_converter_tests", ROOT / "models/convert-oxr-to-gguf.py")
protocol = load_module("oxr_protocol_tests", ROOT / "tools/oxr_validate.py")


def read(name):
    return json.loads((FIXTURES / name).read_text())


class ConverterTests(unittest.TestCase):
    def test_real_752_tensor_inventory(self):
        actual = read("safetensors-header.json")
        actual.pop("__metadata__")
        specs = converter.validate_tensors(actual)
        self.assertEqual(len(specs), 752)
        self.assertEqual(sum(__import__("math").prod(s.shape) for s in specs), 1534378216)
        self.assertIn("visual_tokenizer.vit.vision_model.post_layernorm.weight", actual)
        written = [converter.gguf_name(s.name) for s in specs]
        self.assertEqual(len(set(written)), len(written))
        self.assertTrue(all(len(n.encode()) <= 63 for n in written), max(written, key=len))
        self.assertIn("vit.post_layernorm.weight", written)

    def test_missing_tensor_fails(self):
        header = read("safetensors-header.json")
        header.pop("__metadata__")
        header.pop("vte.weight")
        with self.assertRaisesRegex(ValueError, "inventory"):
            converter.validate_tensors(header)

    def test_wrong_projection_shape_fails(self):
        header = read("safetensors-header.json")
        header.pop("__metadata__")
        header["llm.model.layers.0.self_attn.q_proj.weight"]["shape"] = [1024, 1024]
        with self.assertRaisesRegex(ValueError, "q_proj"):
            converter.validate_tensors(header)

    def test_unknown_tensor_fails(self):
        header = read("safetensors-header.json")
        header.pop("__metadata__")
        header["extra.weight"] = {"shape": [1], "dtype": "BF16"}
        with self.assertRaisesRegex(ValueError, "inventory"):
            converter.validate_tensors(header)

    def test_actual_config_contract(self):
        converter.validate_config(read("config.json"), read("preprocessor_config.json"), read("generation_config.json"))

    def test_windowed_attention_substitution_fails(self):
        config = read("config.json")
        config["vit_config"]["fullatt_block_indexes"] = "3|7|11"
        with self.assertRaisesRegex(ValueError, "fullatt_block_indexes"):
            converter.validate_config(config, read("preprocessor_config.json"), read("generation_config.json"))

    def test_wrong_image_resampling_fails(self):
        processor = read("preprocessor_config.json")
        processor["resample"] = 3
        with self.assertRaisesRegex(ValueError, "resample"):
            converter.validate_config(read("config.json"), processor, read("generation_config.json"))

    def test_sensitive_f16_recipe(self):
        specs = {s.name: s for s in converter.tensor_specs()}
        for name in ["llm.lm_head.weight", "llm.model.embed_tokens.weight", "vte.weight",
                     "visual_tokenizer.head.0.weight", "llm.model.layers.0.self_attn.q_norm.weight",
                     "visual_tokenizer.vit.vision_model.embeddings.patch_embedding.weight"]:
            self.assertEqual(converter.storage_dtype(specs[name], "f16"), "float32", name)
        self.assertEqual(converter.storage_dtype(specs["llm.model.layers.0.self_attn.q_proj.weight"], "f16"), "float16")
        self.assertTrue(all(converter.storage_dtype(spec, "f32") == "float32" for spec in specs.values()))

    def test_authenticated_files_reject_same_size_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            path.write_bytes(b"abc")
            manifest = {"model_id": converter.MODEL_ID, "revision": converter.MODEL_REVISION,
                        "files": {"source.json": {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}}
            self.assertEqual(converter.verify_source_files(Path(directory), manifest)["source.json"], manifest["files"]["source.json"]["sha256"])
            path.write_bytes(b"abd")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                converter.verify_source_files(Path(directory), manifest)

    def test_safetensors_overlap_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mini.safetensors"
            header = {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x80?\x00@")
            entries = converter.read_safetensors_header(path)
            self.assertEqual(entries["a"]["absolute_offset"], 8 + len(raw))
            header["b"] = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
            raw = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 4)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                converter.read_safetensors_header(path)
            path.write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "truncated"):
                converter.read_safetensors_header(path)

    def test_bf16_expansion_and_nonfinite_rejection(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(struct.pack("<HHHH", 0x0000, 0x3F80, 0xC020, 0x7F80))
            entry = {"dtype": "BF16", "shape": [3], "absolute_offset": 0}
            np.testing.assert_array_equal(converter.read_tensor(path, entry, "float32"), [0., 1., -2.5])
            entry["shape"] = [4]
            with self.assertRaisesRegex(ValueError, "non-finite"):
                converter.read_tensor(path, entry, "float32")

    def test_tied_matrices_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"abcdabcd")
            entries = {"llm.model.embed_tokens.weight": {"shape": [2], "dtype": "BF16", "data_offsets": [0, 4], "absolute_offset": 0},
                       "llm.lm_head.weight": {"shape": [2], "dtype": "BF16", "data_offsets": [4, 8], "absolute_offset": 4}}
            converter.verify_tied_weights(path, entries)
            path.write_bytes(b"abcdabce")
            with self.assertRaisesRegex(ValueError, "unequal tied"):
                converter.verify_tied_weights(path, entries)

    def test_pinned_source_manifest(self):
        manifest = converter.source_manifest()
        self.assertEqual(manifest["revision"], converter.MODEL_REVISION)
        self.assertEqual(manifest["files"]["model.safetensors"]["size"], 3068857520)
        self.assertEqual(manifest["files"]["model.safetensors"]["sha256"], "795bb2f1faac579f58829913f2c606202531a96c1203e19ef7a70b103fb7ea51")
        for name in ["config.json", "preprocessor_config.json", "generation_config.json", "tokenizer_config.json", "chat_template.jinja"]:
            self.assertEqual(converter.sha256(FIXTURES / name), manifest["files"][name]["sha256"])

    def test_actual_tokenizer_edge_receipt(self):
        actual = read("tokenizer-cases.json")
        self.assertFalse(actual["neural_inference"])
        self.assertEqual(actual["encoding"][0]["ids"], [963, 963])
        self.assertEqual(actual["encoding"][1]["ids"], list(range(16, 25)) + [15])
        self.assertEqual(actual["decoding"][0]["text"], '!"')
        self.assertEqual(actual["decoding"][1]["text"], "<tool_call><think></think>")


class TokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read("tokenizer_config.json")
        cls.tokenizer = {
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"pretokenizers": [
                {"pattern": {"Regex": converter.QWEN_PATTERN}},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}]},
            "model": {"type": "BPE", "vocab": {f"token{i}": i for i in range(151643)},
                      "merges": [["a", "b"]] * converter.MERGE_COUNT},
            "added_tokens": [{"id": i, "content": text, "special": i <= 151656}
                             for i, text in converter.SPECIAL_TOKENS.items()],
        }

    def test_all_added_tokens_preserved_with_distinct_special_flags(self):
        tokens, types, merges = converter.tokenizer_payload(self.tokenizer, self.config)
        self.assertEqual(len(tokens), 151669)
        self.assertEqual(len(merges), 151387)
        self.assertEqual(types.count(3), 14)
        self.assertEqual(types.count(4), 12)
        self.assertEqual(tokens[151667:], ["<think>", "</think>"])

    def test_nfc_required(self):
        payload = {**self.tokenizer, "normalizer": None}
        with self.assertRaisesRegex(ValueError, "normalizer"):
            converter.tokenizer_payload(payload, self.config)

    def test_non_special_added_token_cannot_be_stripped(self):
        added = [dict(value) for value in self.tokenizer["added_tokens"]]
        added[-1]["special"] = True
        with self.assertRaisesRegex(ValueError, "special flag"):
            converter.tokenizer_payload({**self.tokenizer, "added_tokens": added}, self.config)

    def test_missing_merges_rejected(self):
        payload = {**self.tokenizer, "model": {**self.tokenizer["model"], "merges": []}}
        with self.assertRaisesRegex(ValueError, "merge count"):
            converter.tokenizer_payload(payload, self.config)


class ProtocolTests(unittest.TestCase):
    def request(self):
        request = {"prompt": "Recognize the content of the image.", "max_tokens": 16, "effective_max_tokens": 16,
                   "min_pixels": 200704, "max_pixels": 921600, "repetition_penalty": 1.0, "hf_generation": True,
                   "grid_thw": [1, 28, 28], "input_ids": [151644, -301, -300, -302], "token_ids": [1, 151645],
                   "text": "a", "raw_text": "a<|im_end|>", "eos": True, "capped": False, "finish_reason": "stop",
                   "image_width": 448, "image_height": 448}
        return {**request, "device": "cpu", "threads": 4, "machine_fingerprint": "same-host",
                "dtype": "f32", "conversion_recipe": "f32",
                "cache_neutrality_pass": True, "confidence_observer_neutral": True, "confidence": .8,
                "timing": {"runs_ms": [100., 80., 90.]}, "runs": [dict(request) for _ in range(3)]}

    def test_stage_inventory_66(self):
        self.assertEqual(len(protocol.STAGE_NAMES), 66)
        self.assertEqual(len(set(protocol.STAGE_NAMES)), 66)

    def test_exact_same_host_request_passes(self):
        reference = self.request();native = copy.deepcopy(reference)
        native["timing"]["runs_ms"] = [100., 70., 75.]
        result = protocol.comparison(reference, native)
        self.assertTrue(result["request_parity_pass"])
        self.assertTrue(result["speed_pass"])

    def test_different_host_cannot_claim_speed(self):
        reference = self.request();native = copy.deepcopy(reference)
        native["machine_fingerprint"] = "different"
        result = protocol.comparison(reference, native)
        self.assertTrue(result["request_parity_pass"])
        self.assertFalse(result["speed_comparable"])
        self.assertFalse(result["speed_pass"])

    def test_mixed_precision_cannot_claim_primary_speed_parity(self):
        reference = self.request();native = copy.deepcopy(reference)
        native["conversion_recipe"] = "f16-f32-sensitive"
        result = protocol.comparison(reference, native)
        self.assertTrue(result["request_parity_pass"])
        self.assertFalse(result["primary_precision_profile_match"])
        self.assertFalse(result["speed_pass"])

    def test_capped_completion_does_not_pass_quality(self):
        reference = self.request();native = copy.deepcopy(reference)
        for report in [reference, native]:
            for item in [report, *report["runs"]]:
                item.update(eos=False, capped=True, finish_reason="length")
        self.assertFalse(protocol.comparison(reference, native)["request_parity_pass"])

    def test_resident_drift_rejected(self):
        reference = self.request();native = copy.deepcopy(reference)
        native["runs"][-1]["token_ids"] = [2, 151645]
        self.assertFalse(protocol.comparison(reference, native)["request_parity_pass"])

    def test_cache_neutrality_failure_rejected(self):
        reference = self.request();native = copy.deepcopy(reference)
        reference["cache_neutrality_pass"] = False
        self.assertFalse(protocol.comparison(reference, native)["request_parity_pass"])

    def test_c_float_penalty_contract(self):
        self.assertTrue(protocol.request_value_equal("repetition_penalty", 1.12, 1.12000000477))
        self.assertFalse(protocol.request_value_equal("repetition_penalty", 1.12, 1.121))
        self.assertFalse(protocol.request_value_equal("repetition_penalty", 1.12, float("nan")))

    def test_bad_timing_and_caps_rejected(self):
        with self.assertRaises(ValueError):
            protocol.median_warm({"timing": {"runs_ms": [10., float("nan")]}})
        for bad in ["0", "-1", "32769"]:
            with self.assertRaises(Exception):
                protocol.token_limit(bad)

    def test_text_metrics_use_true_distance(self):
        metrics = protocol.text_metrics("hello world", "hello word")
        self.assertEqual(metrics["character_errors"], 1)
        self.assertEqual(metrics["word_errors"], 1)
        self.assertFalse(metrics["exact"])


if __name__ == "__main__":
    unittest.main()
