"""Model-free export controls; no weights, full tokenizer, GGUF or inference.

Post-admission metadata/CLI tests explicitly mock the full-tokenizer verifier.
Those tests prove dispatch and exact expected identities, never payload fidelity.
Real acquired config/header/prefix bytes supply the independent shape evidence.
"""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lighton4b_converter_test", ROOT / "models/convert-qwen35-ocr-to-gguf.py")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)
profile = converter._lighton4b_profile()
FIXTURE = ROOT / "tests/fixtures/lightonocr3-4b-source"


class MetadataWriter:
    def __init__(self):
        self.fields = {}
        self.types = {}

    def record(self, key, value, kind):
        if key in self.fields:
            raise AssertionError("Duplicate metadata key: " + key)
        self.fields[key] = value
        self.types[key] = kind

    def add_uint32(self, key, value):
        if type(value) is not int or not 0 <= value < 2 ** 32:
            raise AssertionError("Invalid uint32 metadata")
        self.record(key, value, "uint32")

    def add_float32(self, key, value):
        self.record(key, value, "float32")

    def add_string(self, key, value):
        if not isinstance(value, str):
            raise AssertionError("Invalid string metadata")
        self.record(key, value, "string")

    def add_bool(self, key, value):
        if type(value) is not bool:
            raise AssertionError("Invalid bool metadata")
        self.record(key, value, "bool")

    def add_array(self, key, value):
        self.record(key, value, "array")

    def add_key_value(self, key, value, kind, subtype):
        self.record(key, value, (kind, subtype))


class LightOnOCR3FourBExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((FIXTURE / "config.json").read_bytes())
        cls.header_bytes = (FIXTURE / "model.safetensors.header.json").read_bytes()
        cls.header = json.loads(cls.header_bytes)
        cls.source_schema = {name: item["shape"] for name, item in cls.header.items()
                             if name != "__metadata__"}

    def copy_metadata(self, destination):
        for name in profile.METADATA_FILES:
            shutil.copyfile(FIXTURE / name, destination / name)

    def simulated_tokenizer_admission(self, root=FIXTURE):
        # Deliberately substitute only the payload-verification result. The
        # expected path/length/hash and every actual small source byte are checked.
        with patch.object(converter, "_verify_pinned_file", return_value={"payload_verified": True}) as verify:
            pinned, config = converter.load_lighton4b_export_source(root, profile.MODEL, profile.REVISION)
        verify.assert_called_once_with(Path(root) / "tokenizer.json", 19989343,
                                       "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4")
        return pinned, config

    def test_source_identity_is_explicit_and_precedes_filesystem_access(self):
        self.assertEqual(converter.LIGHTON4B_MODEL, profile.MODEL)
        self.assertEqual(converter.LIGHTON4B_REVISION, profile.REVISION)
        self.assertEqual(converter.LIGHTON4B_FORMAT, 5)
        cases = ((profile.MODEL, profile.REVISION, None),
                 (profile.MODEL, "0" * 40, converter.LIGHTON4B_PROFILE),
                 (converter.LIGHTON_MODEL, profile.REVISION, converter.LIGHTON4B_PROFILE))
        with patch.object(Path, "exists", side_effect=AssertionError("filesystem access")):
            for model, revision, selected in cases:
                with self.subTest(model=model, revision=revision, selected=selected), self.assertRaises(ValueError):
                    converter.convert("absent", "unused.gguf", model, revision, profile=selected)
        self.assertFalse(converter.lighton4b_request(converter.LIGHTON_MODEL, converter.LIGHTON_REVISION, None))

    def test_historical_recipe_stays_exactly_twelve_layers(self):
        fixed = ["v.patch_embed.proj.weight", "v.pos_embed.weight", "v.merger.linear_fc1.weight",
                 "v.merger.linear_fc2.weight"]
        parts = ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")
        historical = fixed + [f"v.blocks.{layer}.{part}.weight" for layer in range(12) for part in parts]
        new = fixed + [f"v.blocks.{layer}.{part}.weight" for layer in range(24) for part in parts]
        self.assertEqual(converter.precision_recipe("vision-f32"), historical)
        self.assertEqual(len(historical), 52)
        self.assertEqual(converter.precision_recipe(converter.LIGHTON4B_VISION_RECIPE), new)
        self.assertEqual(len(new), 100)
        self.assertEqual(converter.precision_recipe("original-f16"), [])
        for model, revision, selected, recipe in (
                (profile.MODEL, profile.REVISION, converter.LIGHTON4B_PROFILE, "vision-f32"),
                (converter.LIGHTON_MODEL, converter.LIGHTON_REVISION, None, converter.LIGHTON4B_VISION_RECIPE)):
            with self.assertRaises(ValueError):
                converter.lighton4b_request(model, revision, selected, recipe)

    def test_real_small_files_are_verified_and_full_tokenizer_identity_is_required(self):
        pinned, config = self.simulated_tokenizer_admission()
        self.assertEqual(config, self.config)
        self.assertEqual(set(pinned["source_metadata_verified"]), set(profile.METADATA_FILES))
        self.assertEqual(pinned["source_preflight_name"], "lightonocr3-4b-source-only")
        self.assertEqual(pinned["format_version"], 5)
        self.assertEqual(pinned["tokenizer"]["source_metadata"]["normalizer"], {"type": "NFC"})
        self.assertFalse(pinned["checkpoint"]["payload_verified"])
        for key in ("native_supported", "native_qualified", "export_qualified", "framework_runtime_observed"):
            self.assertIs(pinned[key], False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_metadata(root)
            with self.assertRaises(FileNotFoundError):
                converter.load_lighton4b_export_source(root, profile.MODEL, profile.REVISION)
            (root / "tokenizer.json").write_bytes((FIXTURE / "LightOnOCR-3-4B-tokenizer.prefix.txt").read_bytes())
            with self.assertRaisesRegex(ValueError, "byte count"):
                converter.load_lighton4b_export_source(root, profile.MODEL, profile.REVISION)

    def test_each_changed_small_file_and_unexpected_sharding_fail_before_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_metadata(root)
            with patch.object(converter, "_verify_pinned_file") as verify:
                for name in profile.METADATA_FILES:
                    original = (root / name).read_bytes()
                    (root / name).write_bytes(bytes([original[0] ^ 1]) + original[1:])
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Source bytes changed"):
                        converter.load_lighton4b_export_source(root, profile.MODEL, profile.REVISION)
                    (root / name).write_bytes(original)
                (root / "model.safetensors.index.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "shard index"):
                    converter.load_lighton4b_export_source(root, profile.MODEL, profile.REVISION)
                verify.assert_not_called()

    def test_streamed_identity_checks_actual_tiny_bytes_not_declared_file_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny-source.bin"
            data = b"bounded source identity\x00\xff" * 17
            path.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            proof = converter._verify_pinned_file(path, len(data), digest)
            self.assertEqual(proof["sha256"], digest)
            self.assertTrue(proof["payload_verified"])
            for size, changed in ((len(data) + 1, digest), (len(data), "0" * 64), (0, digest)):
                with self.subTest(size=size, digest=changed), self.assertRaises(ValueError):
                    converter._verify_pinned_file(path, size, changed)

    def test_header_checks_precede_full_payload_hash_and_never_certify_header_only_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.safetensors"
            path.write_bytes(profile.HEADER_BYTES.to_bytes(8, "little") + self.header_bytes)
            with self.assertRaisesRegex(ValueError, "byte count"):
                converter.verify_lighton4b_checkpoint(path, self.config)
            with patch.object(Path, "stat", return_value=types.SimpleNamespace(st_size=profile.CHECKPOINT_BYTES)):
                with self.assertRaisesRegex(ValueError, "truncated"):
                    converter.verify_lighton4b_checkpoint(path, self.config)
                for raw in ((profile.HEADER_BYTES + 1).to_bytes(8, "little") + self.header_bytes,
                            profile.HEADER_BYTES.to_bytes(8, "little") + b"!" + self.header_bytes[1:]):
                    path.write_bytes(raw)
                    with patch.object(converter, "_verify_pinned_file") as verify, self.assertRaises(ValueError):
                        converter.verify_lighton4b_checkpoint(path, self.config)
                    verify.assert_not_called()

    def test_all_planned_shapes_match_actual_header_and_tied_head_is_not_duplicated(self):
        plan = converter.lighton4b_tensor_plan(self.config)
        records = {item["source"]: item for item in plan["records"]}
        self.assertEqual(set(records), set(self.source_schema))
        self.assertEqual(len(records), 723)
        self.assertNotIn("lm_head.weight", records)
        self.assertFalse(any("mtp" in name for name in records))
        self.assertEqual(plan["mtp_policy"], "no-mtp-tensors")
        self.assertTrue(plan["tied_head_stored_once"])
        for name, shape in self.source_schema.items():
            self.assertEqual(records[name]["source_shape"], shape, name)
        self.assertEqual(records["model.visual.patch_embed.proj.weight"]["source_shape"], [1024, 3, 2, 16, 16])
        self.assertEqual(records["model.visual.patch_embed.proj.weight"]["shape"], [1024, 1536])
        self.assertEqual(records["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]["shape"], [8192, 2560])
        self.assertEqual(records["model.language_model.layers.0.linear_attn.conv1d.weight"]["shape"], [8192, 1, 4])
        self.assertEqual(records["model.language_model.embed_tokens.weight"]["shape"], [248320, 2560])
        self.assertEqual(len({item["name"] for item in plan["records"]}), 723)

    def test_fourb_precision_census_and_recurrent_norm_protection(self):
        plain = converter.lighton4b_tensor_plan(self.config)
        vision = converter.lighton4b_tensor_plan(self.config, recipe=converter.LIGHTON4B_VISION_RECIPE)
        full = converter.lighton4b_tensor_plan(self.config, dtype="f32")
        self.assertEqual(Counter(item["dtype"] for item in plain["records"]), {"float16": 349, "float32": 374})
        self.assertEqual(Counter(item["dtype"] for item in vision["records"]), {"float16": 249, "float32": 474})
        self.assertEqual(Counter(item["dtype"] for item in full["records"]), {"float32": 723})
        self.assertEqual(len(vision["keep_f32"]), 100)
        for item in vision["records"]:
            if item["name"].startswith("v.") or "norm" in item["name"] or len(item["shape"]) != 2:
                self.assertEqual(item["dtype"], "float32", item["name"])
        for kwargs in ({"dtype": "q8_0"}, {"keep_f32": ["unknown.weight"]}, {"recipe": "vision-f32"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                converter.lighton4b_tensor_plan(self.config, **kwargs)

    def test_changed_architecture_never_becomes_a_generic_fourb_profile(self):
        for section, field, value in (("text_config", "linear_num_key_heads", 32),
                                      ("text_config", "mtp_num_hidden_layers", 1),
                                      ("text_config", "tie_word_embeddings", False),
                                      ("vision_config", "depth", 12)):
            changed = copy.deepcopy(self.config)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                converter.lighton4b_tensor_plan(changed)

    def test_meta_schema_allows_only_the_declared_tied_alias(self):
        meta = copy.deepcopy(self.source_schema)
        meta["lm_head.weight"] = [248320, 2560]
        result = converter.validate_lighton4b_meta_schema(meta, self.config)
        self.assertEqual(result["official_tensor_count"], 724)
        self.assertEqual(result["checkpoint_tensor_count"], 723)
        self.assertFalse(result["model_forward_observed"])
        alternatives = [copy.deepcopy(self.source_schema), copy.deepcopy(meta), copy.deepcopy(meta)]
        alternatives[1]["mtp.layers.0.norm.weight"] = [2560]
        alternatives[2]["lm_head.weight"] = [248320, 1024]
        for changed in alternatives:
            with self.assertRaisesRegex(ValueError, "meta schema differs"):
                converter.validate_lighton4b_meta_schema(changed, self.config)

    def test_meta_source_pins_are_separate_from_loaded_original_inference(self):
        result = converter.validate_lighton4b_meta_sources("5.5.4", dict(converter.LIGHTON4B_META_SOURCES))
        self.assertEqual(result["upstream_revision"], "75d3bdcd4b3ba70cee21287218d9764f33da41f0")
        self.assertIn("no model forward", result["scope"])
        for version, sources in (("5.16.1", converter.LIGHTON4B_META_SOURCES),
                                 ("5.5.4", {**converter.LIGHTON4B_META_SOURCES, "model": "0" * 64}),
                                 ("5.5.4", {"model": converter.LIGHTON4B_META_SOURCES["model"]})):
            with self.assertRaisesRegex(ValueError, "implementation differs"):
                converter.validate_lighton4b_meta_sources(version, sources)

    def test_emitted_geometry_tokenizer_and_qualification_metadata(self):
        pinned, config = self.simulated_tokenizer_admission()
        plan = converter.lighton4b_tensor_plan(config, recipe=converter.LIGHTON4B_VISION_RECIPE)
        writer = MetadataWriter()
        fake = types.ModuleType("gguf")
        fake.GGUFValueType = types.SimpleNamespace(ARRAY="ARRAY", INT32="INT32")
        with patch.dict(sys.modules, {"gguf": fake}):
            converter.write_lighton4b_metadata(writer, pinned, config, plan, {"scope": "synthetic meta control"})
        prefix = "qwen35_ocr."
        for key, value in {
                "context_length": 262144, "embedding_length": 2560, "feed_forward_length": 9216,
                "block_count": 32, "vocab_size": 248320, "attention.head_count": 16,
                "attention.head_count_kv": 4, "attention.key_length": 256, "attention.value_length": 256,
                "linear_attention.key_head_count": 16, "linear_attention.value_head_count": 32,
                "linear_attention.key_length": 128, "linear_attention.value_length": 128,
                "linear_attention.conv_kernel": 4, "vision.block_count": 24, "vision.embedding_length": 1024,
                "vision.feed_forward_length": 4096, "vision.attention.head_count": 16,
                "vision.patch_size": 16, "vision.temporal_patch_size": 2, "vision.spatial_merge_size": 2,
                "vision.position_embedding_count": 2304, "vision.projection_dim": 2560,
                "source_text_eos_id": 248044, "source_top_level_eos_id": 248046, "primary_eos_id": 248046,
                "image_min_pixels": 65536, "image_max_pixels": 16777216}.items():
            self.assertEqual(writer.fields[prefix + key], value, key)
            self.assertEqual(writer.types[prefix + key], "uint32", key)
        self.assertEqual(writer.fields["general.file_type"], 1)
        self.assertEqual(writer.fields[prefix + "rope.dimension_sections"], [11, 11, 10])
        self.assertEqual(writer.types[prefix + "stop_token_ids"], ("ARRAY", "INT32"))
        self.assertEqual(writer.fields[prefix + "stop_token_ids"], [248046])
        self.assertEqual(writer.fields[prefix + "layer_types"].count("full_attention"), 8)
        for key in ("native_supported", "native_qualified", "export_qualified", "native_alias_qualified",
                    "image_max_qualified", "full_document_qualified", "framework_runtime_observed",
                    "tokenizer_runtime_qualified", "processor_qualified", "effective_stops_qualified"):
            self.assertIs(writer.fields[prefix + key], False, key)
        tokenizer = json.loads(writer.fields[prefix + "tokenizer_source_metadata"])
        self.assertEqual(tokenizer["normalizer"], {"type": "NFC"})
        self.assertEqual(len(tokenizer["added_tokens"]), 33)
        self.assertIn(r"\p{M}", tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"])
        self.assertEqual(tokenizer["decoder"], {"type": "ByteLevel", "add_prefix_space": False,
                                              "trim_offsets": False, "use_regex": False})
        self.assertNotIn("temperature", writer.fields)
        self.assertEqual(converter.read_tokenizer_metadata.__defaults__, (None, False))

    def test_cli_rejects_wrong_identity_and_cross_model_recipe_before_download(self):
        fake = types.ModuleType("huggingface_hub")
        fake.snapshot_download = Mock()
        cases = [
            ["--model", profile.MODEL, "--revision", profile.REVISION],
            ["--model", profile.MODEL, "--revision", "main", "--profile", converter.LIGHTON4B_PROFILE],
            ["--model", profile.MODEL, "--revision", "0" * 40, "--profile", converter.LIGHTON4B_PROFILE],
            ["--model", profile.MODEL, "--revision", profile.REVISION, "--profile", converter.LIGHTON4B_PROFILE,
             "--precision-recipe", "vision-f32"],
            ["--model", converter.LIGHTON_MODEL, "--revision", converter.LIGHTON_REVISION,
             "--precision-recipe", converter.LIGHTON4B_VISION_RECIPE],
        ]
        with patch.dict(sys.modules, {"huggingface_hub": fake}), contextlib.redirect_stderr(io.StringIO()):
            for extra in cases:
                with self.subTest(extra=extra), patch.object(sys, "argv", ["convert", "--output", "unused", *extra]), \
                        self.assertRaises(SystemExit):
                    converter.main()
        fake.snapshot_download.assert_not_called()

    def test_cli_requires_source_and_tokenizer_before_requesting_checkpoint(self):
        fake = types.ModuleType("huggingface_hub")
        arguments = ["convert", "--output", "unused", "--model", profile.MODEL,
                     "--revision", profile.REVISION, "--profile", converter.LIGHTON4B_PROFILE]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_metadata(root)
            fake.snapshot_download = Mock(return_value=str(root))
            with patch.dict(sys.modules, {"huggingface_hub": fake}), patch.object(sys, "argv", arguments), \
                    self.assertRaises(FileNotFoundError):
                converter.main()
            fake.snapshot_download.assert_called_once()
            patterns = fake.snapshot_download.call_args.kwargs["allow_patterns"]
            self.assertEqual(set(patterns), set(profile.METADATA_FILES) | {"tokenizer.json"})
            self.assertFalse(any("safetensors" in pattern or "*" in pattern for pattern in patterns))

    def test_cli_export_route_is_explicit_after_simulated_metadata_admission(self):
        fake = types.ModuleType("huggingface_hub")
        fake.snapshot_download = Mock(return_value=str(FIXTURE))
        arguments = ["convert", "--output", "unused", "--model", profile.MODEL,
                     "--revision", profile.REVISION, "--profile", converter.LIGHTON4B_PROFILE,
                     "--precision-recipe", converter.LIGHTON4B_VISION_RECIPE]
        with patch.dict(sys.modules, {"huggingface_hub": fake}), patch.object(sys, "argv", arguments), \
                patch.object(converter, "_verify_pinned_file", return_value={"payload_verified": True}), \
                patch.object(converter, "convert", return_value={"records": [], "native_supported": False}) as convert, \
                contextlib.redirect_stdout(io.StringIO()):
            converter.main()
        self.assertEqual(fake.snapshot_download.call_count, 2)
        self.assertEqual(set(fake.snapshot_download.call_args.kwargs["allow_patterns"]),
                         set(profile.METADATA_FILES) | {"tokenizer.json", "model.safetensors"})
        convert.assert_called_once()
        self.assertEqual(convert.call_args.kwargs["profile"], converter.LIGHTON4B_PROFILE)
        self.assertEqual(convert.call_args.kwargs["recipe"], converter.LIGHTON4B_VISION_RECIPE)
        self.assertEqual(len(convert.call_args.args[5]), 100)

    def test_model_libraries_are_not_imported_by_controls(self):
        for name in ("torch", "transformers", "safetensors", "numpy", "gguf", "huggingface_hub"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
