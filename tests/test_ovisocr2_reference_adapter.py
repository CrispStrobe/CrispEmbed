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
                   ("5.16.1", "0.23.3", "2.14.0+cpu")]
        with mock.patch.dict(sys.modules, {"transformers": types.SimpleNamespace(GenerationConfig=object)}), \
             mock.patch.object(ovis.inspect, "getfile", return_value=__file__), \
             mock.patch.object(ovis.ovis_profile, "_file_sha256", return_value="different"):
            with self.assertRaisesRegex(ValueError, "source hashes"):
                ovis.check_framework_sources(profile, *modules, processor, object)
            modules[1].__version__ = "0.22.2"
            with self.assertRaisesRegex(ValueError, "tokenizers==0.23.3"):
                ovis.check_framework_sources(profile, *modules, processor, object)

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
                     "input.prompt_length", "input.rendered_prompt_length",
                     "input.decoded_length", "input.raw_decoded_length"):
            self.assertIn(name, stages)
        self.assertNotIn("prefill.delta.initial_state", stages)

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
