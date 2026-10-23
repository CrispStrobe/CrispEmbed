"""Standard-library guards for the pinned original-reference capture adapter."""

import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import dump_ovisocr2_reference as ovis
import dump_qwen35_ocr_reference as dispatcher


class ReferenceAdapterTest(unittest.TestCase):
    def test_preflight_order_and_prompt_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text("{}")
            profile = {"rendered_prompt": "x" * 625}
            calls = []
            with (mock.patch.object(ovis.ovis_profile, "load_profile", side_effect=lambda *a: calls.append("metadata") or profile),
                  mock.patch.object(ovis.ovis_profile, "verify_checkpoint", side_effect=lambda *a: calls.append("checkpoint")),
                  mock.patch.object(ovis.ovis_profile, "validate_config", side_effect=lambda *a: calls.append("config"))):
                with self.assertRaisesRegex(ValueError, "custom prompt"):
                    ovis.validate_request(directory, "different")
                with self.assertRaisesRegex(ValueError, "custom prompt"):
                    ovis.validate_request(directory, profile["rendered_prompt"])
                self.assertEqual(calls, [])
                self.assertIs(ovis.validate_request(directory, ""), profile)
                self.assertEqual(calls, ["metadata", "checkpoint", "config"])

    def test_utf8_fields_preserve_empty_lengths_and_nonascii_bytes(self):
        self.assertEqual(ovis.utf8_field(""), ([0], 0))
        self.assertEqual(ovis.utf8_field("é"), ([0xC3, 0xA9], 2))
        self.assertEqual(ovis.utf8_field("🧪"), ([0xF0, 0x9F, 0xA7, 0xAA], 4))

    def test_version_and_source_rejection(self):
        profile = {"qualification": {"loaded_source_sha256": {"model": "expected"}}}
        processor = types.SimpleNamespace(image_processor=object(), tokenizer=object())
        modules = [types.SimpleNamespace(__version__=version) for version in
                   ("5.16.1", "0.23.3", "2.11.0+cpu")]
        with mock.patch.dict(sys.modules, {"transformers": types.SimpleNamespace(GenerationConfig=object)}), \
             mock.patch.object(ovis.inspect, "getfile", return_value=__file__), \
             mock.patch.object(ovis.ovis_profile, "_file_sha256", return_value="different"):
            with self.assertRaisesRegex(ValueError, "source hashes"):
                ovis.check_framework_sources(profile, *modules, processor, object)
            modules[1].__version__ = "0.22.2"
            with self.assertRaisesRegex(ValueError, "tokenizers==0.23.3"):
                ovis.check_framework_sources(profile, *modules, processor, object)
            modules[1].__version__ = "0.23.3"
            modules[2].__version__ = "2.14.0+cpu"
            with self.assertRaisesRegex(ValueError, "torch==2.11.0"):
                ovis.check_framework_sources(profile, *modules, processor, object)

    def test_processor_contract_matches_json_arrays_and_rejects_value_changes(self):
        class ImageProcessor:
            backend = "torchvision"

            def to_dict(self):
                return {"image_mean": (0.5, 0.5, 0.5), "patch_size": 16}

        class Processor:
            image_processor = ImageProcessor()
            tokenizer = types.SimpleNamespace(eos_token_id=248046)

        processor = Processor()
        profile = {"primary_eos": 248046, "processor": {
            "class_name": type(processor).__module__ + "." + type(processor).__name__,
            "image_class": type(processor.image_processor).__module__ + "." + type(processor.image_processor).__name__,
            "image_backend": "torchvision",
            "image_config": {"image_mean": [0.5, 0.5, 0.5], "patch_size": 16}}}
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            report = ovis.processor_contract(processor, profile)
            self.assertTrue(all(report["checks"].values()))
            self.assertFalse(report["image_config_native_equal"])
            self.assertEqual(report["image_config_python_types"]["image_mean"], "tuple")
            for key, changed in (("image_mean", [0.500001, 0.5, 0.5]), ("patch_size", 14)):
                with mock.patch.object(processor.image_processor, "to_dict", return_value={
                        **profile["processor"]["image_config"], key: changed}):
                    with self.assertRaisesRegex(ValueError, "image_config"):
                        ovis.processor_contract(processor, profile)
            processor.tokenizer.eos_token_id = 248044
            with self.assertRaisesRegex(ValueError, "primary_eos"):
                ovis.processor_contract(processor, profile)
            processor.tokenizer.eos_token_id = 248046
            processor.image_processor.backend = "pil"
            with self.assertRaisesRegex(ValueError, "image_backend"):
                ovis.processor_contract(processor, profile)

    def test_generation_bound_reports_both_eos_ids_and_length_cap(self):
        eos_ids = [248044, 248046]
        for stop_id in eos_ids:
            report = ovis.generation_limit_report([17, stop_id], 12, eos_ids)
            self.assertEqual(report["stop_id"], stop_id)
            self.assertTrue(report["reached_eos"])
            self.assertFalse(report["length_limited"])
            self.assertEqual(report["eos_ids"], eos_ids)
        capped = ovis.generation_limit_report([17, 18], 2, eos_ids)
        self.assertEqual(capped["generated_token_count"], 2)
        self.assertTrue(capped["length_limited"])
        self.assertFalse(capped["reached_eos"])
        self.assertIsNone(capped["stop_id"])
        self.assertFalse(ovis.generation_limit_report([17], 2, eos_ids)["length_limited"])
        with self.assertRaisesRegex(ValueError, "exceeded"):
            ovis.generation_limit_report([1, 2, 3], 2, eos_ids)

    def test_actual_visual_frame_local_only(self):
        class Visual:
            def forward(self):
                pos_embeds = [1, 2]
                return pos_embeds

        visual = Visual()
        saved = []
        prior = sys.getprofile()
        try:
            sys.setprofile(lambda frame, event, arg: ovis.observed_visual_positions(
                frame, event, arg, visual, lambda *x: saved.append(x), ["prefill"]))
            visual.forward()
        finally:
            sys.setprofile(prior)
        self.assertEqual(saved, [("prefill.vision.position", [1, 2])])

    def test_hook_and_global_restoration(self):
        original = object()
        module = types.SimpleNamespace(torch_chunk_gated_delta_rule="wrapped")
        handles = [types.SimpleNamespace(remove=mock.Mock())]
        prior = sys.getprofile()
        sys.setprofile(lambda *args: None)
        try:
            ovis.restore_observers(handles, module, {"chunk_gated_delta_rule": original}, prior)
            self.assertIs(sys.getprofile(), prior)
            self.assertIs(module.torch_chunk_gated_delta_rule, original)
            handles[0].remove.assert_called_once()
        finally:
            sys.setprofile(prior)

    def test_required_stage_contract_has_all_layers_and_real_delta_boundary(self):
        config = types.SimpleNamespace(
            vision_config=types.SimpleNamespace(depth=12),
            text_config=types.SimpleNamespace(
                num_hidden_layers=24,
                layer_types=["linear_attention"] * 23 + ["full_attention"]))
        stages = ovis.required_stages(config, 3)
        for name in ("prefill.vision.layer.11", "prefill.vision.position",
                     "prefill.decoder.layer.23", "cached.cache.0.conv_states",
                     "cached.delta.initial_state", "generated.2.decoder.position_ids",
                     "generated.scores.0", "generated.scores.2",
                     "backend_off.vision.layer.11", "backend_off.vision.position",
                     "backend_off.vision.probe.5.qkv",
                     "backend_off_full_prefix.vision.probe.11.pooled",
                     "backend_off.decoder.norm", "backend_off.logits",
                     "backend_off_full_prefix.vision.merger",
                     "backend_off_full_prefix.decoder.norm",
                     "backend_off.generated.ids",
                     "input.prompt_length", "input.rendered_prompt_length",
                     "input.decoded_length", "input.raw_decoded_length"):
            self.assertIn(name, stages)
        self.assertNotIn("prefill.delta.initial_state", stages)

    def test_backend_control_capture_filter_bounds_reference_size(self):
        for name in ("backend_off.vision.layer.5", "backend_off.vision.position",
                     "backend_off.vision.merger.norm", "backend_off.decoder.norm",
                     "backend_off.vision.probe.5.qkv",
                     "backend_off_full_prefix.vision.probe.11.pooled",
                     "backend_off.logits", "backend_off.argmax_id",
                     "backend_off.generated.ids",
                     "backend_off_full_prefix.vision.output"):
            self.assertTrue(ovis.backend_control_stage(name))
        for name in ("backend_off.decoder.layer.0", "backend_off.delta.output",
                     "backend_off.cache.0.conv_states", "backend_off.vision.probe.4.qkv",
                     "backend_off.vision.probe.5.extra",
                     "backend_off.vision.layer.12", "backend_off_full_prefix.linear.0.qkv"):
            self.assertFalse(ovis.backend_control_stage(name))

    def test_dispatch_preserves_default_and_explicit_profile(self):
        result = {"decoded": ["x"], "cached_vs_full_prefix_max_abs": 0.0,
                  "cached_vs_full_prefix_same_argmax": True}
        argv = ["dump", "--model-dir", "m", "--image", "i", "--output", "o"]
        with mock.patch.object(dispatcher, "dump_reference", return_value=result) as lighton, \
             mock.patch.object(ovis, "dump_reference", return_value=result) as original, \
             mock.patch.object(sys, "argv", argv), mock.patch("sys.stdout", new_callable=io.StringIO):
            dispatcher.main()
            lighton.assert_called_once()
            original.assert_not_called()
        with mock.patch.object(dispatcher, "dump_reference", return_value=result) as lighton, \
             mock.patch.object(ovis, "dump_reference", return_value=result) as original, \
             mock.patch.object(sys, "argv", argv + ["--profile", "ovisocr2"]), \
             mock.patch("sys.stdout", new_callable=io.StringIO):
            dispatcher.main()
            original.assert_called_once()
            lighton.assert_not_called()


if __name__ == "__main__":
    unittest.main()
