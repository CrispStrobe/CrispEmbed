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
import mineru2605_document as document
import mineru2605_resources as resources

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
    def candidate(self):
        manifest = converter.source_manifest(converter.PROFILES["2605"])
        metadata = {"general.architecture": "qwen2vl", "qwen2vl.model_family": "mineru2.5-pro-2605",
                    "qwen2vl.source.repository": manifest["repository"], "qwen2vl.source.revision": manifest["revision"],
                    "qwen2vl.source.expected_revision": manifest["revision"],
                    "qwen2vl.blueprint.revision": "mineru-vl-utils==2.0.7",
                    "qwen2vl.blueprint.artifact_sha256": manifest["blueprint"]["wheel_sha256"],
                    "qwen2vl.source.file_sha256": json.dumps({name: row["sha256"] for name, row in manifest["files"].items()})}
        return {"metadata": {key: {"value": value} for key, value in metadata.items()}, "bytes": 1,
                "tensors": {row.target: {"stored_shape": list(reversed((1280, 1176) if row.flatten_patch else row.shape)),
                                         "type": 0} for row in converter.tensor_specs()}}

    def test_candidate_cannot_relabel_checkpoint_or_precision(self):
        data = self.candidate()
        self.assertEqual(qualification.candidate_contract(data, "f32")["tensor_count"], 681)
        for precision in ("f16", "q8_0", "q4_k"):
            with self.subTest(precision=precision), self.assertRaisesRegex(ValueError, "storage"):
                qualification.candidate_contract(data, precision)
        for key in ("qwen2vl.model_family", "qwen2vl.source.revision", "qwen2vl.blueprint.artifact_sha256"):
            changed = copy.deepcopy(data); changed["metadata"][key]["value"] = "2604"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "2605"):
                qualification.candidate_contract(changed, "f32")
        changed = copy.deepcopy(data); changed["tensors"]["v.patch_embed.weight"]["stored_shape"].reverse()
        with self.assertRaisesRegex(ValueError, "shape"):
            qualification.candidate_contract(changed, "f32")

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
            {"name": name, "observed": True, "finite": True, "visits": 1, "elements": 10, "cos_min": .98}
            for name in qualification.REQUIRED_STAGES]}

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


    def test_duplicate_or_unknown_stage_name_cannot_count_as_coverage(self):
        for replacement in (self.stages["stages"][0]["name"], "invented_stage"):
            with self.subTest(name=replacement):
                stages = copy.deepcopy(self.stages)
                stages["stages"][-1]["name"] = replacement
                with self.assertRaisesRegex(ValueError, "every required name"):
                    qualification.precision_result(self.ref, self.forced, self.rows, stages, "f16", "portable")

    def test_empty_repeated_nonfinite_or_unobserved_stage_cannot_pass(self):
        for key, value in (("elements", 0), ("visits", 2), ("visits", True), ("observed", False),
                           ("observed", 1), ("finite", False), ("cos_min", float("nan"))):
            with self.subTest(field=key, value=value):
                stages = copy.deepcopy(self.stages)
                stages["stages"][0][key] = value
                with self.assertRaisesRegex(ValueError, "diagnostics"):
                    qualification.precision_result(self.ref, self.forced, self.rows, stages, "f16", "portable")


class DocumentHelperContract(unittest.TestCase):
    def setUp(self):
        def request(prompt, width, height):
            prediction = {"text": "same visible result", "token_ids": [23, 151645], "eos": True, "capped": False}
            return {"prompt": prompt, "width": width, "height": height, "result": prediction}
        self.trace = [request("\nLayout Detection:", 1036, 1036), request("\nText Recognition:", 128, 64)]
        predictions = [{"prompt": row["prompt"], "image_width": row["width"], "image_height": row["height"],
                        **copy.deepcopy(row["result"])} for row in self.trace]
        self.native = {"capped": False, "layout": predictions[0], "recognitions": [{"prediction": predictions[1]}]}

    def result(self):
        # Equal rendered outputs alone must never hide a trace/completion failure.
        return document.helper_result(self.trace, self.native, "[]", "[]", "same markdown", "same markdown")

    def test_identical_complete_layout_and_crop_requests_pass(self):
        result = self.result()
        self.assertTrue(result["helper_parity_pass"])
        self.assertTrue(result["native_reached_eos"])
        self.assertEqual(result["requests"]["native_requests"], 2)

    def test_native_truncation_cannot_pass_equal_rendered_output(self):
        for role in ("layout", "crop", "aggregate"):
            with self.subTest(role=role):
                self.setUp()
                if role == "aggregate":
                    self.native["capped"] = True
                else:
                    row = self.native["layout"] if role == "layout" else self.native["recognitions"][0]["prediction"]
                    row.update(eos=False, capped=True, token_ids=[23, 24])
                result = self.result()
                self.assertFalse(result["helper_parity_pass"])
                self.assertTrue(result["reference_reached_eos"])
                self.assertFalse(result["native_reached_eos"])

    def test_both_truncated_arms_still_fail(self):
        self.trace[0]["result"].update(eos=False, capped=True, token_ids=[23, 24])
        self.native["layout"].update(eos=False, capped=True, token_ids=[23, 24])
        self.native["capped"] = True
        result = self.result()
        self.assertTrue(result["requests"]["requests_exact"])
        self.assertFalse(result["helper_parity_pass"])

    def test_prompt_crop_tokens_and_text_are_checked_even_if_renderings_match(self):
        for field, value in (("prompt", "different task"), ("image_width", 129), ("image_height", 65),
                             ("token_ids", [24, 151645]), ("text", "changed ignored content")):
            with self.subTest(field=field):
                self.setUp()
                self.native["recognitions"][0]["prediction"][field] = value
                result = self.result()
                self.assertFalse(result["helper_parity_pass"])
                self.assertFalse(result["requests"]["requests_exact"])

    def test_missing_extra_and_malformed_requests_fail_closed(self):
        for mutation in ("empty_official", "missing_crop", "extra_crop", "bad_dimensions", "missing_cap", "false_eos"):
            with self.subTest(mutation=mutation):
                self.setUp()
                if mutation == "empty_official":
                    self.trace = []
                elif mutation == "missing_crop":
                    self.native["recognitions"] = []
                elif mutation == "extra_crop":
                    self.native["recognitions"].append(copy.deepcopy(self.native["recognitions"][0]))
                elif mutation == "bad_dimensions":
                    self.native["layout"]["image_width"] = True
                elif mutation == "missing_cap":
                    del self.native["layout"]["capped"]
                else:
                    self.native["layout"]["token_ids"] = [23, 24]
                self.assertFalse(self.result()["helper_parity_pass"])


class HostedResourceContract(unittest.TestCase):
    def test_one_case_dispatch_preserves_token_limits(self):
        for case_id in ("all", "receipt,table", "missing"):
            with self.assertRaisesRegex(ValueError, "one frozen case"):
                resources.selection(case_id, "f32,f16,q8_0,q4_k", 2048, 3, "1")
        _, tiers = resources.selection("receipt", "f32,f16,q8_0,q4_k", 2048, 3, "1")
        plan = resources.resource_plan(tiers, 2048)
        self.assertEqual(plan["logit_bytes_upper"], 5 * 2048 * 151936 * 4)
        self.assertEqual(resources.resource_plan(tiers, 0)["positions_upper"], 8192)
        self.assertFalse(plan["qualification"])

    def test_shared_filesystem_requires_combined_space(self):
        plan = {"scratch_bytes_required": 8, "output_bytes_required": 6}
        shared = resources.disk_capacity(plan, {"scratch": {"device": 1, "free_bytes": 10},
                                                "output": {"device": 1, "free_bytes": 10}})
        self.assertEqual(shared[0]["required_bytes"], 14)
        self.assertFalse(shared[0]["sufficient"])
        separate = resources.disk_capacity(plan, {"scratch": {"device": 1, "free_bytes": 10},
                                                  "output": {"device": 2, "free_bytes": 10}})
        self.assertTrue(all(row["sufficient"] for row in separate))


if __name__ == "__main__":
    unittest.main()
