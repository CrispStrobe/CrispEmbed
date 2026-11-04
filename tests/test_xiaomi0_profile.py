"""No weights/framework/network: exact source/config, exclusion and gate controls."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/xiaomi0-source"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = load("xiaomi0_policy_test", ROOT / "models/qwen35_xiaomi0_profile.py")
converter = load("xiaomi0_converter_test", ROOT / "models/convert-qwen35-ocr-to-gguf.py")


class XiaomiSourceTest(unittest.TestCase):
    def image_processor(self):
        # Original pinned checkpoint JSON, with the exact 5.17.0 runtime
        # standardization in BaseImageProcessor._standardize_kwargs.
        settings = json.loads((FIXTURE / "processor_config.json").read_text())["image_processor"]
        settings["image_mean"] = tuple(settings["image_mean"])
        settings["image_std"] = tuple(settings["image_std"])
        return SimpleNamespace(**settings)

    def test_original_processor_tuple_normalization_preserves_exact_values(self):
        image = self.image_processor()
        self.assertNotEqual(image.image_mean, [0.5] * 3)  # Initial hosted guard defect.
        result = policy.validate_image_processor(image)
        expected = json.loads((FIXTURE / "processor_config.json").read_text())["image_processor"]
        expected.pop("image_processor_type")
        self.assertEqual(result["parameters"], expected)
        self.assertEqual(result["runtime_types"]["image_mean"], "tuple")
        self.assertEqual(result["runtime_types"]["image_std"], "tuple")
        # Validation never rewrites the original processor's runtime state.
        self.assertIsInstance(image.image_mean, tuple)
        image.image_mean = list(image.image_mean)
        image.image_std = list(image.image_std)
        self.assertEqual(policy.validate_image_processor(image)["parameters"], expected)

    def test_processor_guard_rejects_numeric_geometry_and_flag_drift(self):
        changes = (
            ("image_mean", (0.5, 0.5, 0.5000000001)),
            ("image_std", (0.5, 0.5)), ("image_mean", "0.5,0.5,0.5"),
            ("size", {"shortest_edge": 65537, "longest_edge": 16777216}),
            ("size", {"shortest_edge": 65536, "longest_edge": 16777216, "height": 32}),
            ("size", None), ("patch_size", 14), ("temporal_patch_size", 1), ("merge_size", 1),
            ("resample", 2), ("resample", 3.5), ("rescale_factor", 1 / 256),
            ("do_resize", False), ("do_rescale", False), ("do_normalize", False),
            ("do_convert_rgb", False), ("do_resize", "true"), ("do_resize", 1),
        )
        for name, value in changes:
            image = self.image_processor()
            setattr(image, name, value)
            with self.subTest(field=name, value=value), self.assertRaisesRegex(ValueError, name):
                policy.validate_image_processor(image)

    def test_processor_failure_retains_exact_fields_and_runtime_containers(self):
        image = self.image_processor()
        image.image_std = (0.5, 0.5, 0.25)
        with self.assertRaises(ValueError) as error:
            policy.validate_image_processor(image)
        details = json.loads(str(error.exception).split(": ", 1)[1])
        self.assertEqual(details["fields"], ["image_std"])
        self.assertEqual(details["parameters"]["image_std"], [0.5, 0.5, 0.25])
        self.assertEqual(details["expected"]["image_std"], [0.5] * 3)
        self.assertEqual(details["runtime_types"]["image_std"], "tuple")

    def test_every_meta_parameter_shape_is_checked_not_only_the_count(self):
        source = json.loads(policy.MAIN_SCHEMA.read_text())
        actual = source["tensors"]
        actual["lm_head.weight"] = actual["model.language_model.embed_tokens.weight"]
        self.assertEqual(len(actual), 474)
        self.assertEqual(policy.validate_main_schema(actual), policy.MAIN_SCHEMA_SHA256)
        changed = {name: list(shape) for name, shape in actual.items()}
        name = next(iter(changed))
        changed[name][-1] += 1
        with self.assertRaises(ValueError):
            policy.validate_main_schema(changed)
        changed = dict(actual)
        changed["not_a_parameter"] = changed.pop(name)
        self.assertEqual(len(changed), 474)
        with self.assertRaises(ValueError):
            policy.validate_main_schema(changed)

    def test_audit_attribution_and_unqualified_scope(self):
        report = policy.evidence()
        self.assertEqual(report["license"]["base_model"], "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(report["license"]["base_card_license"], "apache-2.0")
        self.assertEqual(report["layout"]["model"], "PaddlePaddle/PP-DocLayoutV3_safetensors")
        self.assertFalse(report["layout"]["native_graph_implemented"])
        self.assertFalse(any(report["qualification"].values()))
        self.assertNotEqual(report["tokenizer"]["serialized_regex"], report["tokenizer"]["config_regex"])
        for name in ("config.json", "processor_config.json", "tokenizer_config.json", "chat_template.jinja"):
            self.assertEqual(policy.file_sha256(FIXTURE / name), report["source_files"][name]["sha256"])

    def test_config_identity_and_legacy_policy(self):
        config = json.loads((FIXTURE / "config.json").read_text())
        pinned = policy.expected_profile()
        self.assertEqual(converter.validate_config(config, pinned)[0]["mtp_num_hidden_layers"], 1)
        with self.assertRaises(ValueError):
            converter.validate_config(config)  # Existing unprofiled MTP refusal remains.
        for path, value in (("mtp_num_hidden_layers", 0), ("num_hidden_layers", 25), ("hidden_size", 2048)):
            changed = copy.deepcopy(config)
            changed["text_config"][path] = value
            with self.subTest(path=path), self.assertRaises(ValueError):
                policy.validate_config(changed, pinned)
        for name, value in (("native_alias_qualified", True), ("format_version", 1),
                            ("image_max_qualified", True), ("stop_ids", [248044, 248046])):
            changed = copy.deepcopy(pinned)
            changed[name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                policy.check_profile(changed)
        self.assertEqual(converter.read_tokenizer_metadata.__defaults__, (None, False))

    def test_exact_mtp_exclusion_rejects_unknown_weights(self):
        exclusions = policy.evidence()["checkpoint"]["mtp_exclusions"]
        # Synthetic main names test exclusion policy, never impersonate a checkpoint schema.
        tensors = {"model.synthetic.%d" % i: ("model.safetensors", [1]) for i in range(473)}
        tensors.update({name: ("model.safetensors", data["shape"]) for name, data in exclusions.items()})
        self.assertEqual(len(policy.non_speculative_tensors(tensors, policy.expected_profile())), 473)
        name = next(iter(exclusions))
        for mutate in (lambda d: d.pop(name), lambda d: d.update({"mtp.unreviewed.weight": ("x", [1])}),
                       lambda d: d.update({name: ("x", [1])})):
            changed = copy.deepcopy(tensors)
            mutate(changed)
            with self.assertRaises(ValueError):
                policy.non_speculative_tensors(changed, policy.expected_profile())
        with self.assertRaises(ValueError):
            converter.tensor_name(name)  # No broad ignore added to generic exporter.

    def test_source_pin_and_missing_file_guards(self):
        report = policy.evidence()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "processor_config.json", "tokenizer_config.json", "chat_template.jinja"):
                shutil.copyfile(FIXTURE / name, root / name)
            with self.assertRaises(ValueError):
                policy.load_profile(root, policy.MODEL, "0" * 40)
            real = policy.file_sha256
            # Only large/unvendored metadata files are mocked; pinned fixtures are read.
            def digest(path):
                return real(path) if Path(path).exists() else report["source_files"][Path(path).name]["sha256"]
            with patch.object(policy, "file_sha256", side_effect=digest):
                self.assertEqual(policy.load_profile(root, policy.MODEL, policy.REVISION), policy.expected_profile())
                for name in policy.MISSING_FILES + ["model.safetensors.index.json"]:
                    (root / name).write_text("{}")
                    with self.assertRaises(ValueError):
                        policy.load_profile(root, policy.MODEL, policy.REVISION)
                    (root / name).unlink()
                (root / "config.json").write_text("{}")
                with self.assertRaises(ValueError):
                    policy.load_profile(root, policy.MODEL, policy.REVISION)
            (root / "model.safetensors").write_bytes(b"not a checkpoint")
            with self.assertRaises(ValueError):
                policy.verify_checkpoint(root / "model.safetensors", policy.expected_profile())

    def test_task_prompts_do_not_conflate_hf_and_vllm(self):
        for task in policy.evidence()["prompts"]:
            image_first = policy.rendered_prompt(task)
            text_first = policy.rendered_prompt(task, order="text-first")
            self.assertNotEqual(image_first, text_first)
            self.assertEqual(image_first.count("<|image_pad|>"), 1)
            self.assertTrue(image_first.endswith("<think>\n\n</think>\n\n"))
        with self.assertRaises(ValueError):
            policy.rendered_prompt("custom-vqa")
        with self.assertRaises(ValueError):
            policy.rendered_prompt(order="unknown")

    def test_source_diff_is_separate_and_default_off(self):
        cpp = (ROOT / "src/qwen35_tokenizer.cpp").read_text()
        native = (ROOT / "src/qwen35_ocr.cpp").read_text()
        self.assertIn('core_env::on("CRISPEMBED_XIAOMI_OCR0_FRONTEND")', cpp)
        self.assertIn('core_env::on("CRISPEMBED_XIAOMI_OCR0_INFERENCE")', native)
        self.assertIn('if (c.tokenizer.is_xiaomi0())', native)
        self.assertIn('item.second->type != GGML_TYPE_F32 && item.second->type != GGML_TYPE_F16', native)
        header = (ROOT / "src/qwen35_xiaomi0.h").read_text()
        for value in (policy.MODEL, policy.REVISION, policy.CHECKPOINT_SHA256, policy.AUDIT_SHA256):
            self.assertIn(value, header)
        self.assertIn('if (c.logit_observer)', native)
        self.assertIn('if (c.stage_observer)', native)


if __name__ == "__main__":
    unittest.main()
