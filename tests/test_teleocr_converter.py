#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Weight-free tests of TeleOCR's executable-code-free GGUF conversion.

Run with ``python -m unittest discover -s tests -p test_teleocr_converter.py``.
Round-trip tests require NumPy and gguf; no torch, transformers, network, or
model download is used. Small authored weights test tensor direction, BF16
decoding, quantization protection and failure handling.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("teleocr_converter", ROOT / "models/convert-teleocr-to-gguf.py")
CONV = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CONV
SPEC.loader.exec_module(CONV)
HAS_GGUF = importlib.util.find_spec("gguf") is not None


def tiny_config():
    return {
        "model_type": "qwen2_5_vl", "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 32, "vocab_size": 32, "max_position_embeddings": 1024,
        "tie_word_embeddings": True, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0,
        "rope_scaling": {"mrope_section": [4, 6, 6], "rope_type": "default"},
        "image_token_id": 3, "vision_start_token_id": 4, "vision_end_token_id": 5, "eos_token_id": 2,
        "vision_config": {
            "depth": 1, "hidden_size": 16, "intermediate_size": 20, "num_heads": 2,
            "out_hidden_size": 32, "patch_size": 2, "spatial_merge_size": 2,
            "temporal_patch_size": 2, "window_size": 8, "fullatt_block_indexes": [0],
        },
    }


def tiny_processor():
    return {"patch_size": 2, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711], "min_pixels": 16, "max_pixels": 4096}


def tiny_tokenizer():
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|image_pad|>",
                "<|vision_start|>", "<|vision_end|>"]
    return {"model": {"type": "BPE", "vocab": {"a": 6, "b": 7, "ab": 8}, "merges": [["a", "b"]]},
            "added_tokens": [{"id": i, "content": s, "special": True} for i, s in enumerate(specials)]}


def write_safetensors(path, arrays, dtype="BF16"):
    header, payloads, offset = {}, [], 0
    for name, array in arrays.items():
        array = np.ascontiguousarray(array, dtype=np.float32)
        if dtype == "BF16":
            payload = (array.view(np.uint32) >> 16).astype("<u2").tobytes()
        elif dtype == "F16":
            payload = array.astype("<f2").tobytes()
        else:
            payload = array.astype("<f4").tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(payload)]}
        offset += len(payload)
        payloads.append(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    with Path(path).open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for payload in payloads:
            stream.write(payload)


def make_model(path):
    config, processor, tokenizer = tiny_config(), tiny_processor(), tiny_tokenizer()
    for name, value in (("config.json", config), ("preprocessor_config.json", processor), ("tokenizer.json", tokenizer)):
        (path / name).write_text(json.dumps(value))
    profile = CONV.validate_config(config, processor)
    rng = np.random.default_rng(125)
    arrays = {name: rng.uniform(-0.4, 0.4, spec.shape).astype(np.float32)
              for name, spec in CONV.expected_tensors(profile).items()}
    arrays["lm_head.weight"] = arrays["model.embed_tokens.weight"].copy()
    write_safetensors(path / "model.safetensors", arrays)
    return profile, arrays


class TeleOCRContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.profile, self.arrays = make_model(self.path)

    def test_complete_schema_has_qk_norms_and_nonstandard_head_width(self):
        index = CONV.load_tensor_index(self.path)
        specs = CONV.validate_tensors(self.profile, index)
        self.assertEqual(len(specs), 31)
        by_name = {s.target: s for s in specs}
        self.assertEqual(by_name["l.blk.0.attn_q.weight"].shape, (64, 32))
        self.assertEqual(by_name["l.blk.0.attn_o.weight"].shape, (32, 64))
        self.assertEqual(by_name["l.blk.0.attn_q_norm.weight"].shape, (32,))
        self.assertEqual(by_name["v.patch_embed.weight"].output_shape, (16, 24))
        self.assertEqual(by_name["v.merger.fc2.weight"].shape, (32, 64))

    def test_missing_qk_norm_is_rejected(self):
        del self.arrays["model.layers.0.self_attn.k_norm.weight"]
        write_safetensors(self.path / "model.safetensors", self.arrays)
        with self.assertRaisesRegex(ValueError, "missing=.*k_norm"):
            CONV.validate_tensors(self.profile, CONV.load_tensor_index(self.path))

    def test_extra_tensor_is_not_silently_discarded(self):
        self.arrays["model.unexpected.weight"] = np.ones((2, 2))
        write_safetensors(self.path / "model.safetensors", self.arrays)
        with self.assertRaisesRegex(ValueError, "unexpected=.*unexpected.weight"):
            CONV.validate_tensors(self.profile, CONV.load_tensor_index(self.path))

    def test_projection_shape_cannot_be_inferred_from_hidden_divided_by_heads(self):
        self.arrays["model.layers.0.self_attn.q_proj.weight"] = np.zeros((32, 32))
        write_safetensors(self.path / "model.safetensors", self.arrays)
        with self.assertRaisesRegex(ValueError, "q_proj.weight: expected"):
            CONV.validate_tensors(self.profile, CONV.load_tensor_index(self.path))

    def test_conflicting_tied_head_is_rejected(self):
        self.arrays["lm_head.weight"][0, 0] = 8
        write_safetensors(self.path / "model.safetensors", self.arrays)
        with self.assertRaisesRegex(ValueError, "not byte-identical"):
            CONV.validate_tensors(self.profile, CONV.load_tensor_index(self.path))

    def test_missing_duplicate_head_is_valid_when_tied(self):
        del self.arrays["lm_head.weight"]
        write_safetensors(self.path / "model.safetensors", self.arrays)
        self.assertEqual(len(CONV.validate_tensors(self.profile, CONV.load_tensor_index(self.path))), 31)

    def test_nested_config_and_mrope_validation(self):
        config = tiny_config()
        config["text_config"] = {"head_dim": 16}
        with self.assertRaisesRegex(ValueError, "mRoPE"):
            CONV.validate_config(config, tiny_processor())
        config["text_config"] = {"head_dim": 32}
        self.assertEqual(CONV.validate_config(config, tiny_processor()).text["head_dim"], 32)

    def test_wrong_preprocessing_is_rejected(self):
        processor = tiny_processor()
        processor["patch_size"] = 14
        with self.assertRaisesRegex(ValueError, "disagrees"):
            CONV.validate_config(tiny_config(), processor)
        processor = tiny_processor()
        processor["image_std"][1] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            CONV.validate_config(tiny_config(), processor)

    def test_tokenizer_preserves_reserved_rows_and_special_ids(self):
        tokens, types, merges = CONV.tokenizer_arrays(tiny_tokenizer(), self.profile)
        self.assertEqual((len(tokens), tokens[2], types[2], types[31]), (32, "<|im_end|>", 3, 5))
        self.assertEqual(tokens[31], "")
        self.assertEqual(merges, ["a b"])

    def test_tokenizer_duplicate_id_and_missing_merge_fail(self):
        tokenizer = tiny_tokenizer()
        tokenizer["model"]["vocab"]["x"] = 6
        with self.assertRaisesRegex(ValueError, "duplicate tokenizer ID"):
            CONV.tokenizer_arrays(tokenizer, self.profile)
        tokenizer = tiny_tokenizer()
        tokenizer["model"]["merges"] = [["a", "missing"]]
        with self.assertRaisesRegex(ValueError, "missing token"):
            CONV.tokenizer_arrays(tokenizer, self.profile)

    def test_nonfinite_weights_fail(self):
        self.arrays["model.norm.weight"][0] = np.nan
        write_safetensors(self.path / "model.safetensors", self.arrays)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            CONV.tensor_f32(CONV.load_tensor_index(self.path)["model.norm.weight"])

    def test_truncated_payload_rejected_before_conversion(self):
        path = self.path / "model.safetensors"
        with path.open("r+b") as stream:
            stream.truncate(path.stat().st_size - 1)
        with self.assertRaisesRegex(ValueError, "file size disagree"):
            CONV.load_tensor_index(self.path)

    def test_duplicate_header_key_and_shard_traversal_rejected(self):
        encoded = b'{"duplicate":{},"duplicate":{}}'
        (self.path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            CONV.load_tensor_index(self.path)
        (self.path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../x.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "within the model directory"):
            CONV.load_tensor_index(self.path)

    def test_q8_recipe_keeps_vision_and_output_projection_unquantized(self):
        by_target = {s.target: CONV.tensor_type(s, "q8_0") for s in CONV.expected_tensors(self.profile).values()}
        self.assertEqual(sum(x == "Q8_0" for x in by_target.values()), 7)
        self.assertEqual(by_target["l.embed_tokens.weight"], "F16")
        self.assertEqual(by_target["v.blk.0.attn_qkv.weight"], "F16")
        self.assertEqual(by_target["v.patch_embed.weight"], "F16")
        self.assertEqual(by_target["v.merger.fc1.weight"], "F16")
        self.assertEqual(by_target["l.blk.0.attn_q_norm.weight"], "F32")


@unittest.skipUnless(HAS_GGUF, "install gguf for binary round-trip tests")
class TeleOCRRoundTripTests(unittest.TestCase):
    setUp = TeleOCRContractTests.setUp
    def _roundtrip(self, dtype):
        import gguf

        path = self.path / (dtype + ".gguf")
        report = CONV.convert(self.path, path, dtype, revision="unit-test")
        reader = gguf.GGUFReader(path)
        tensors = {t.name: t for t in reader.tensors}
        self.assertEqual(len(tensors), 31)
        self.assertEqual(report["source_tensor_count"], 32)
        self.assertTrue(report["tied_head_omitted"])
        self.assertEqual(report["output_sha256"], CONV.file_sha256(path))
        self.assertEqual(reader.fields["general.architecture"].contents(), "teleocr")
        self.assertEqual(reader.fields["qwen2vl.attention.head_dim"].contents(), 32)
        self.assertTrue(reader.fields["qwen2vl.has_qk_norm"].contents())
        self.assertEqual(reader.fields["tokenizer.ggml.eos_token_id"].contents(), 2)
        original = CONV.load_tensor_index(self.path)
        for spec in CONV.expected_tensors(self.profile).values():
            tensor = tensors[spec.target]
            self.assertEqual(tuple(tensor.shape), tuple(reversed(spec.output_shape)))
            expected = CONV.tensor_f32(original[spec.source]).reshape(spec.output_shape)
            actual = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(spec.output_shape)
            if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
                self.assertLess(float(np.max(np.abs(actual - expected))), 0.004)
            else:
                # Source values are BF16 in the [-0.4,0.4] interval, hence
                # exactly representable by both F16 and F32 destinations.
                np.testing.assert_array_equal(actual, expected)
        return path

    def test_f32_roundtrip(self):
        self._roundtrip("f32")

    def test_f16_roundtrip(self):
        self._roundtrip("f16")

    def test_q8_roundtrip(self):
        self._roundtrip("q8_0")

    def test_existing_output_is_preserved(self):
        path = self.path / "existing.gguf"
        path.write_bytes(b"existing bytes")
        with self.assertRaises(FileExistsError):
            CONV.convert(self.path, path)
        self.assertEqual(path.read_bytes(), b"existing bytes")

    def test_invalid_model_leaves_no_output(self):
        path = self.path / "invalid.gguf"
        del self.arrays["visual.merger.ln_q.weight"]
        write_safetensors(self.path / "model.safetensors", self.arrays)
        with self.assertRaisesRegex(ValueError, "tensor census"):
            CONV.convert(self.path, path)
        self.assertFalse(path.exists())
        self.assertFalse(Path(str(path) + ".json").exists())


if __name__ == "__main__":
    unittest.main()
