"""Stdlib controls for the hosted-only meta probe; no framework or model execution."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lighton4b_export_meta_test", ROOT / "tools/ci-heavy/lighton4b_export_meta.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
converter = probe.load_converter()
profile = converter._lighton4b_profile()
FIXTURE = ROOT / "tests/fixtures/lightonocr3-4b-source"


class MetaTensorControl:
    """Shape-only test object; cannot allocate or expose parameter values."""
    def __init__(self, shape, device="meta"):
        self.shape = tuple(shape)
        self.device = device
        self.is_meta = device == "meta"
        self.dtype = "torch.float32"


class MetaModelControl:
    def __init__(self, schema):
        self.parameters = {name: MetaTensorControl(shape) for name, shape in schema.items()}
        self.embedding = self.parameters["model.language_model.embed_tokens.weight"]
        self.head = self.embedding
        self.buffers = {"model.language_model.rotary_emb.inv_freq": MetaTensorControl([32])}

    def state_dict(self):
        return {**self.parameters, "lm_head.weight": self.head}

    def named_parameters(self):
        return iter(self.parameters.items())

    def named_buffers(self):
        return iter(self.buffers.items())

    def get_input_embeddings(self):
        return types.SimpleNamespace(weight=self.embedding)

    def get_output_embeddings(self):
        return types.SimpleNamespace(weight=self.head)


class LightOnOCR3FourBMetaProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((FIXTURE / "config.json").read_bytes())
        header = json.loads((FIXTURE / "model.safetensors.header.json").read_bytes())
        cls.source_schema = {name: item["shape"] for name, item in header.items() if name != "__metadata__"}

    def test_shape_control_validates_schema_and_actual_tied_object_ownership(self):
        model = MetaModelControl(self.source_schema)
        result = probe.inspect_meta_model(model, converter, self.config)
        self.assertEqual(result["state_dict_tensor_count"], 724)
        self.assertEqual(result["unique_parameter_tensor_count"], 723)
        self.assertEqual(result["unique_parameter_count"], 4539265536)
        self.assertEqual(result["state_dict_elements_including_tied_alias"], 5174964736)
        self.assertEqual(result["meta_parameter_dtype_counts"], {"torch.float32": 723})
        self.assertTrue(result["all_parameters_and_buffers_on_meta"])
        self.assertTrue(result["output_head_is_input_embedding"])
        self.assertFalse(result["model_forward_observed"])
        self.assertEqual(result["state_dict_shapes_sha256"], probe.canonical_sha256(result["state_dict_shapes"]))
        self.assertEqual(result["state_dict_shapes"]["lm_head.weight"], [248320, 2560])

    def test_untied_same_shape_head_rejects(self):
        model = MetaModelControl(self.source_schema)
        model.head = MetaTensorControl([248320, 2560])
        with self.assertRaisesRegex(ValueError, "does not tie"):
            probe.inspect_meta_model(model, converter, self.config)

    def test_any_materialized_parameter_or_buffer_rejects(self):
        for kind in ("parameter", "buffer"):
            model = MetaModelControl(self.source_schema)
            target = next(iter(model.parameters.values())) if kind == "parameter" else next(iter(model.buffers.values()))
            target.is_meta = False
            target.device = "cpu"
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Materialized tensor"):
                probe.inspect_meta_model(model, converter, self.config)

    def test_unknown_missing_or_misshaped_original_meta_keys_reject(self):
        for change in ("extra", "missing", "shape"):
            model = MetaModelControl(self.source_schema)
            if change == "extra":
                model.parameters["mtp.layers.0.norm.weight"] = MetaTensorControl([2560])
            elif change == "missing":
                del model.parameters["model.language_model.norm.weight"]
            else:
                model.parameters["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"].shape = (12288, 2560)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "meta schema differs"):
                probe.inspect_meta_model(model, converter, self.config)

    def test_duplicate_unique_parameter_enumeration_rejects(self):
        model = MetaModelControl(self.source_schema)
        ordinary = list(model.named_parameters())
        model.named_parameters = lambda: iter(ordinary + [("duplicate", ordinary[0][1])])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            probe.inspect_meta_model(model, converter, self.config)

    def test_parameter_enumeration_must_match_stored_source_names(self):
        model = MetaModelControl(self.source_schema)
        ordinary = list(model.named_parameters())
        model.named_parameters = lambda: iter(ordinary[1:])
        with self.assertRaisesRegex(ValueError, "unique parameters differ"):
            probe.inspect_meta_model(model, converter, self.config)

    def test_preceding_generic_converter_guards_accept_the_exact_fourb_config(self):
        text, vision = converter.validate_config(self.config)
        self.assertEqual(text["num_hidden_layers"], 32)
        self.assertEqual(text["linear_num_key_heads"], 16)
        self.assertEqual(text["linear_num_value_heads"], 32)
        self.assertEqual(vision["depth"], 24)
        self.assertEqual(text["mtp_num_hidden_layers"], 0)
        self.assertEqual(vision["deepstack_visual_indexes"], [])

    def test_export_plan_summaries_do_not_claim_payload_execution(self):
        plans = probe.export_plan_summaries(converter, self.config)
        self.assertEqual(set(plans), {"f32:original-f16", "f16:original-f16", "f16:lightonocr3-4b-vision-f32"})
        self.assertEqual(plans["f32:original-f16"]["planned_dtype_counts"], {"float32": 723})
        self.assertEqual(plans["f16:original-f16"]["planned_dtype_counts"], {"float16": 349, "float32": 374})
        selected = plans["f16:lightonocr3-4b-vision-f32"]
        self.assertEqual(selected["planned_dtype_counts"], {"float16": 249, "float32": 474})
        self.assertEqual(selected["retained_f32_matrix_count"], 100)
        for record in plans.values():
            self.assertEqual(record["format_version"], 5)
            self.assertEqual(record["tensor_count"], 723)
            self.assertTrue(record["metadata_only"])
            self.assertFalse(record["payload_export_executed"])
            self.assertRegex(record["plan_sha256"], r"^[0-9a-f]{64}$")

    def test_offline_guard_blocks_dns_and_connections_and_restores_environment(self):
        keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
        previous = {key: os.environ.get(key) for key in keys}
        with probe.offline_imports():
            self.assertTrue(all(os.environ[key] == "1" for key in keys))
            with self.assertRaisesRegex(RuntimeError, "Network access is forbidden"):
                socket.getaddrinfo("example.invalid", 80)
            with self.assertRaisesRegex(RuntimeError, "Network access is forbidden"):
                socket.create_connection(("127.0.0.1", 9))
            with socket.socket() as sock, self.assertRaisesRegex(RuntimeError, "Network access is forbidden"):
                sock.connect(("127.0.0.1", 9))
        self.assertEqual({key: os.environ.get(key) for key in keys}, previous)

    def test_source_provenance_requires_committed_unchanged_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {}
            for name in probe.CODE_PATHS:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = (name + "\n").encode()
                path.write_bytes(raw)
                expected[name] = hashlib.sha256(raw).hexdigest()
            with patch.object(probe.subprocess, "check_output", side_effect=["1" * 40, "tracked", ""]):
                proof = probe.source_provenance(root)
            self.assertEqual(proof["revision"], "1" * 40)
            self.assertEqual(proof["source_files_sha256"], expected)
            for responses, message in ((["main"], "immutable commit"),
                                       (["1" * 40, "tracked", " M " + probe.CONVERTER_PATH], "uncommitted")):
                with patch.object(probe.subprocess, "check_output", side_effect=responses), \
                        self.assertRaisesRegex(ValueError, message):
                    probe.source_provenance(root)

    def test_scoped_success_record_keeps_every_model_qualification_false(self):
        controlled_meta = {"scope": "shape-control substitution; no framework imported",
                           "model": probe.inspect_meta_model(MetaModelControl(self.source_schema), converter, self.config)}
        with patch.object(probe, "source_provenance", return_value={"revision": "1" * 40}), \
                patch.object(probe, "execute_official_meta", return_value=controlled_meta) as execute:
            result = probe.run_probe()
        execute.assert_called_once()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["official_meta_observed"])
        self.assertTrue(result["source"]["metadata_verified"])
        self.assertEqual(result["source"]["declared_checkpoint_dtype"], "BF16")
        for key in ("original_inference_observed", "checkpoint_payload_acquired", "full_tokenizer_acquired",
                    "payload_export_executed", "native_supported", "native_qualified", "export_qualified",
                    "native_alias_qualified", "image_max_qualified", "full_document_qualified"):
            self.assertIs(result[key], False, key)

    def test_failed_meta_probe_retains_negative_scoped_evidence(self):
        with patch.object(probe, "source_provenance", return_value={"revision": "1" * 40}), \
                patch.object(probe, "execute_official_meta", side_effect=ValueError("intentional schema rejection")):
            result = probe.run_probe()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "ValueError")
        self.assertEqual(result["error"], "intentional schema rejection")
        self.assertFalse(result["official_meta_observed"])
        self.assertFalse(result["export_qualified"])
        self.assertFalse(result["payload_export_executed"])

    def test_cli_writes_scoped_report_and_returns_status_without_framework_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "meta.json"
            for status, expected_exit in (("passed", 0), ("failed", 1)):
                controlled = {"status": status, "scope": "synthetic CLI control",
                              "official_meta_observed": status == "passed", "export_qualified": False}
                with patch.object(probe, "run_probe", return_value=copy.deepcopy(controlled)), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    code = probe.main(["--output", str(path)])
                self.assertEqual(code, expected_exit)
                self.assertEqual(json.loads(path.read_bytes()), controlled)
                self.assertFalse(path.with_name(path.name + ".partial").exists())
                summary = json.loads(output.getvalue())
                self.assertEqual(summary["report"], "meta.json")
                self.assertEqual(summary["report_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertFalse(summary["payload_export_executed"])
                self.assertFalse(summary["native_supported"])

    def test_control_suite_imports_no_model_libraries(self):
        for name in ("torch", "transformers", "safetensors", "numpy", "gguf", "huggingface_hub"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
