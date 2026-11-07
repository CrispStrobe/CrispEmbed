"""Bounded source/header controls; no frameworks, weights, network or inference."""

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lightonocr3_4b_profile_test", ROOT / "models/lightonocr3_4b_profile.py")
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)
FIXTURE = ROOT / "tests/fixtures/lightonocr3-4b-source"


class LightOnOCR3FourBProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((FIXTURE / "config.json").read_text())
        cls.generation = json.loads((FIXTURE / "generation_config.json").read_text())
        cls.header_bytes = (FIXTURE / profile.HEADER_FILE).read_bytes()
        cls.header = json.loads(cls.header_bytes)

    def copy_metadata(self, root):
        for name in profile.METADATA_FILES + (profile.HEADER_FILE,):
            shutil.copyfile(FIXTURE / name, root / name)

    def test_actual_capture_and_full_independent_tensor_census(self):
        evidence = profile.evidence()
        self.assertEqual(evidence["repository"], profile.MODEL)
        self.assertEqual(evidence["revision"], profile.REVISION)
        schema = profile.expected_tensor_schema(self.config)
        self.assertEqual(len(schema), 723)
        self.assertNotIn("lm_head.weight", schema)
        self.assertFalse(any(name.startswith("mtp.") for name in schema))
        result = profile.validate_header_bytes(self.header_bytes, self.config)
        self.assertEqual(result["parameter_count"], 4539265536)
        self.assertEqual(result["payload_bytes_from_header"], 9078531072)
        self.assertFalse(result["payload_verified"])

    def test_unequal_gdn_heads_and_gated_full_attention_are_explicit(self):
        schema = profile.expected_tensor_schema(self.config)
        prefix = "model.language_model.layers.0.linear_attn."
        self.assertEqual(schema[prefix + "in_proj_qkv.weight"], [8192, 2560])
        self.assertEqual(schema[prefix + "in_proj_z.weight"], [4096, 2560])
        self.assertEqual(schema[prefix + "conv1d.weight"], [8192, 1, 4])
        self.assertEqual(schema[prefix + "A_log"], [32])
        self.assertEqual(schema[prefix + "norm.weight"], [128])
        attention = "model.language_model.layers.3.self_attn."
        self.assertEqual(schema[attention + "q_proj.weight"], [8192, 2560])
        self.assertEqual(schema[attention + "k_proj.weight"], [1024, 2560])
        self.assertEqual(schema[attention + "o_proj.weight"], [2560, 4096])

    def test_correct_preflight_never_claims_native_or_export_qualification(self):
        result = profile.load_profile()
        for key in ("native_supported", "native_qualified", "export_qualified",
                    "native_alias_qualified", "image_max_qualified", "full_document_qualified",
                    "framework_runtime_observed"):
            self.assertIs(result[key], False, key)
        self.assertFalse(result["checkpoint"]["payload_verified"])
        self.assertFalse(result["tokenizer"]["payload_verified"])
        self.assertFalse(result["tokenizer"]["runtime_qualified"])
        self.assertEqual(result["native_source_review_revision"], profile.AUDITED_NATIVE_REVISION)
        self.assertEqual(len(result["verified_source_files"]), 7)

    def test_generation_and_nested_text_eos_are_not_conflated(self):
        result = profile.load_profile()
        self.assertEqual(result["source_text_config_eos_id"], 248044)
        self.assertEqual(result["source_top_level_eos_id"], 248046)
        self.assertEqual(result["generation"]["eos_token_id"], 248046)
        self.assertEqual(result["generation"], self.generation)
        for key in ("temperature", "top_p", "do_sample", "max_new_tokens"):
            self.assertNotIn(key, result["generation"])
        changed = copy.deepcopy(self.generation)
        changed["eos_token_id"] = 248044
        with self.assertRaisesRegex(ValueError, "generation config differs"):
            profile.validate_generation(changed)

    def test_runtime_blockers_include_vocabulary_limit_and_frontend(self):
        blockers = {item["id"]: item for item in profile.native_blockers(self.config)}
        self.assertEqual(blockers["unequal_gdn_heads"]["qk_repeat_factor"], 2)
        self.assertEqual(blockers["unequal_gdn_heads"]["current_equal_head_channels"], 12288)
        vocabulary = blockers["vocabulary_projection_limit"]
        self.assertEqual(vocabulary["head_elements"], 635699200)
        self.assertGreater(vocabulary["head_elements"], vocabulary["current_projection_element_limit"])
        self.assertEqual(vocabulary["head_f32_bytes"], 2542796800)
        self.assertIn("distinct_frontend_and_recipe", blockers)
        self.assertIn("original_native_evidence_pending", blockers)

    def test_blocker_sources_are_pinned_evidence_urls(self):
        base = "https://github.com/CrispStrobe/CrispEmbed/blob/" + profile.AUDITED_NATIVE_REVISION + "/"
        for blocker in profile.native_blockers(self.config):
            for source in blocker.get("files", []):
                self.assertTrue(source.startswith(base), source)

    def test_original_capture_is_unchanged_by_separate_prefix_audit(self):
        self.assertEqual(hashlib.sha256((FIXTURE / "audit_manifest.json").read_bytes()).hexdigest(),
                         profile.AUDIT_SHA256)

    def test_prefix_comparison_is_derived_and_does_not_claim_full_tokenizers(self):
        result = profile.tokenizer_prefix_evidence()
        self.assertFalse(result["full_tokenizer_verified_by_prefix"])
        self.assertFalse(result["tokenizer_runtime_qualified"])
        for source in result["sources"].values():
            self.assertEqual(source["prefix_bytes"], 65536)
            self.assertFalse(source["complete_tokenizer"])
        comparison = result["comparison"]
        self.assertTrue(comparison["normalizer_equal"])
        self.assertTrue(comparison["added_tokens_equal"])
        self.assertEqual(comparison["added_token_count"], 33)
        differences = comparison["differing_top_level_metadata_fields"]
        self.assertEqual(set(differences), {"pre_tokenizer", "decoder"})
        four = differences["pre_tokenizer"]["LightOnOCR-3-4B"]["pretokenizers"]
        small = differences["pre_tokenizer"]["LightOnOCR-3-0.8B"]["pretokenizers"]
        self.assertIn(r"\p{M}", four[0]["pattern"]["Regex"])
        self.assertNotIn(r"\p{M}", small[0]["pattern"]["Regex"])
        self.assertFalse(four[1]["trim_offsets"])
        self.assertTrue(small[1]["trim_offsets"])
        for name in ("add_prefix_space", "trim_offsets", "use_regex"):
            self.assertFalse(differences["decoder"]["LightOnOCR-3-4B"][name])
            self.assertTrue(differences["decoder"]["LightOnOCR-3-0.8B"][name])
        license_source = result["official_repository_license"]
        self.assertEqual(license_source["repository"], "lightonai/LightOnOCR")
        self.assertEqual(license_source["revision"], profile.LICENSE_REVISION)
        self.assertEqual(license_source["sha256"], profile.LICENSE_SHA256)
        self.assertEqual(license_source["git_blob_sha1"], profile.LICENSE_BLOB)

    def test_prefix_is_incomplete_json_and_only_pre_model_fields_are_parsed(self):
        data = (FIXTURE / "LightOnOCR-3-4B-tokenizer.prefix.txt").read_bytes()
        with self.assertRaises(json.JSONDecodeError):
            json.loads(data)
        fields = profile.tokenizer_prefix_fields(data, 7191)
        self.assertNotIn("model", fields)
        self.assertEqual(fields["normalizer"], {"type": "NFC"})
        for raw, offset in ((data[:-1], 7191), (data, 0), (data, 7192), (data, len(data))):
            with self.subTest(length=len(raw), offset=offset), self.assertRaises(ValueError):
                profile.tokenizer_prefix_fields(raw, offset)
        changed = data.replace(b'"normalizer"', b'"normalizex"', 1)
        with self.assertRaisesRegex(ValueError, "field coverage"):
            profile.tokenizer_prefix_fields(changed, 7191)

    def test_prefix_and_license_byte_changes_fail_closed(self):
        names = ("tokenizer_prefix_audit.json", "LightOnOCR-3-4B-tokenizer.prefix.txt",
                 "LightOnOCR-3-0.8B-tokenizer.prefix.txt", "LightOnOCR-Apache-2.0-LICENSE.txt")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                shutil.copyfile(FIXTURE / name, root / name)
            profile.tokenizer_prefix_evidence(root)
            for name in names[1:]:
                path = root / name
                original = path.read_bytes()
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Source bytes changed"):
                    profile.tokenizer_prefix_evidence(root)
                path.write_bytes(original)

    def test_derived_prefix_report_is_pinned_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = json.loads((FIXTURE / "tokenizer_prefix_audit.json").read_text())
            report["derived_comparison"]["added_token_count"] = 32
            (root / "tokenizer_prefix_audit.json").write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                profile.tokenizer_prefix_evidence(root)

    def test_wrong_source_identity_rejects_before_filesystem_access(self):
        absent = FIXTURE / "not-present"
        for model, revision in (("lightonai/LightOnOCR-3-0.8B", profile.REVISION),
                                (profile.MODEL, "main"), (profile.MODEL, "0" * 40)):
            with self.subTest(model=model, revision=revision), self.assertRaisesRegex(ValueError, "identity"):
                profile.load_profile(absent, model, revision)

    def test_config_drift_does_not_gain_generic_model_admission(self):
        mutations = (
            ("text_config", "linear_num_value_heads", 16),
            ("text_config", "hidden_size", 1024),
            ("text_config", "mtp_num_hidden_layers", 1),
            ("text_config", "tie_word_embeddings", False),
            ("vision_config", "depth", 12),
            ("vision_config", "deepstack_visual_indexes", [5, 11, 17]),
        )
        for section, field, value in mutations:
            changed = copy.deepcopy(self.config)
            changed[section][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "pinned config"):
                profile.validate_config(changed)

    def test_equal_head_qkv_shortcut_fails_tensor_contract(self):
        changed = copy.deepcopy(self.header)
        name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        changed[name]["shape"] = [12288, 2560]
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            profile.validate_tensor_contract(changed, self.config)

    def test_missing_extra_head_and_mtp_tensors_fail(self):
        source = "model.language_model.embed_tokens.weight"
        changed = copy.deepcopy(self.header)
        del changed[source]
        with self.assertRaisesRegex(ValueError, "Tensor census"):
            profile.validate_tensor_contract(changed, self.config)
        for name in ("lm_head.weight", "mtp.layers.0.norm.weight", "model.visual.blocks.24.norm1.weight"):
            changed = copy.deepcopy(self.header)
            changed[name] = copy.deepcopy(self.header[source])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Tensor census"):
                profile.validate_tensor_contract(changed, self.config)

    def test_dtype_and_non_integer_shapes_fail(self):
        name = "model.language_model.norm.weight"
        for field, value in (("dtype", "F16"), ("shape", [2560.0]), ("shape", [True])):
            changed = copy.deepcopy(self.header)
            changed[name][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "shape/dtype"):
                profile.validate_tensor_contract(changed, self.config)

    def test_offset_overlap_gap_and_wrong_element_width_fail(self):
        name = "model.language_model.norm.weight"
        for start_delta, end_delta in ((2, 2), (-2, -2), (0, -2), (0, 2)):
            changed = copy.deepcopy(self.header)
            changed[name]["data_offsets"][0] += start_delta
            changed[name]["data_offsets"][1] += end_delta
            with self.subTest(deltas=(start_delta, end_delta)), self.assertRaisesRegex(ValueError, "byte interval"):
                profile.validate_tensor_contract(changed, self.config)

    def test_raw_header_byte_identity_is_preserved(self):
        for raw in (self.header_bytes + b"\n", self.header_bytes[:-1], b"{}"):
            with self.subTest(length=len(raw)), self.assertRaisesRegex(ValueError, "header bytes changed"):
                profile.validate_header_bytes(raw, self.config)

    def test_duplicate_json_fields_and_nonfinite_constants_reject(self):
        for data in (b'{"shape": [1], "shape": [2]}', b'{"shape": [NaN]}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                profile._json(data)

    def test_missing_or_changed_metadata_fails_without_weight_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_metadata(root)
            # This sentinel is intentionally invalid and must never be opened:
            # source preflight cannot certify a checkpoint payload.
            (root / "model.safetensors").write_bytes(b"not model weights")
            original_open = Path.open
            def forbid_weight_open(path, *args, **kwargs):
                if path.name == "model.safetensors":
                    raise AssertionError("Source preflight opened a model payload")
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", forbid_weight_open):
                self.assertFalse(profile.load_profile(root)["checkpoint"]["payload_verified"])
            path = root / "tokenizer_config.json"
            original = path.read_bytes()
            path.write_bytes(original.replace(b"TokenizersBackend", b"TokenizersBackenz"))
            with self.assertRaisesRegex(ValueError, "Source bytes changed"):
                profile.load_profile(root)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                profile.load_profile(root)

    def test_unexpected_sharded_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_metadata(root)
            (root / "model.safetensors.index.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "single file"):
                profile.load_profile(root)

    def test_tokenizer_verification_is_explicit_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.copy_metadata(root)
            with self.assertRaises(FileNotFoundError):
                profile.load_profile(root, verify_tokenizer=True)
            (root / "tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "byte length"):
                profile.load_profile(root, verify_tokenizer=True)

    def test_cli_reports_source_success_and_native_pending(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "models/lightonocr3_4b_profile.py")],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["checkpoint"]["tensor_contract"]["tensor_count"], 723)
        self.assertFalse(report["native_supported"])
        rejected = subprocess.run(
            [sys.executable, str(ROOT / "models/lightonocr3_4b_profile.py"), "--revision", "main"],
            text=True, capture_output=True, check=False,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(rejected.stdout)
        self.assertIn("source identity differs", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
