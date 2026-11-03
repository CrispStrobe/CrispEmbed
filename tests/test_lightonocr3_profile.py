#!/usr/bin/env python3
"""Model-free LightOnOCR-3 input, parsing and acceptance controls."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "models"), str(ROOT / "tools")]
from lightonocr3_profile import manifest, validate_config, sha256, TRANSCRIPTION, GROUNDING
from lightonocr3_output import parse_grounding


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ROOT / "tests/fixtures/lightonocr3"
        self.config, self.processor, self.generation = [json.loads((self.fixture / n).read_text()) for n in
            ("config.json", "processor_config.json", "generation_config.json")]

    def test_official_files_are_immutable(self):
        pins = manifest()
        self.assertEqual(pins["revision"], "23d544085c886bbdace647bcfab91203299417e2")
        for path in self.fixture.iterdir():
            if path.name in pins["files"]:
                self.assertEqual(sha256(path), pins["files"][path.name], path.name)
        validate_config(self.config, self.processor, self.generation)

    def test_incompatible_backbones_rejected(self):
        for mutate in (
            lambda c: c["text_config"].update(model_type="qwen3_5"),
            lambda c: c["text_config"].update(tie_word_embeddings=False),
            lambda c: c["text_config"].update(use_sliding_window=True),
            lambda c: c["text_config"].update(num_hidden_layers=27),
            lambda c: c["vision_config"].update(patch_size=16),
            lambda c: c.update(spatial_merge_size=1),
            lambda c: c["text_config"].update(rope_parameters={"rope_theta": 1000000, "rope_type": "linear"}),
        ):
            c = copy.deepcopy(self.config)
            mutate(c)
            with self.assertRaises(ValueError):
                validate_config(c, self.processor, self.generation)

    def test_processor_and_stops_rejected(self):
        for key, value in (("resample", 2), ("patch_size", 28), ("size", {"longest_edge": 1024}),
                           ("image_mean", [0.5, 0.5, 0.5])):
            p = copy.deepcopy(self.processor)
            p["image_processor"][key] = value
            with self.assertRaises(ValueError):
                validate_config(self.config, p, self.generation)
        with self.assertRaises(ValueError):
            validate_config(self.config, self.processor, {"eos_token_id": [151645]})

    def test_observed_prompt_ids(self):
        ids = json.loads((self.fixture / "prompt-ids.json").read_text())
        self.assertEqual(ids["transcription"]["prefix"], [151644, 8948, 151645, 198, 151644, 872, 198])
        self.assertEqual(ids["grounding"]["suffix"], [1951, 287, 198, 151645, 198, 151644, 77091, 198])
        self.assertEqual(ids["transcription"]["suffix"], [151645, 198, 151644, 77091, 198])

    def test_prompt_contract(self):
        self.assertNotIn("OCR this", TRANSCRIPTION)
        self.assertEqual(TRANSCRIPTION.count("<|image_pad|>"), 1)
        self.assertTrue(TRANSCRIPTION.startswith("<|im_start|>system<|im_end|>\n"))
        self.assertEqual(GROUNDING.replace("grounding\n", ""), TRANSCRIPTION)

    def test_grounding_normalization_and_continuation(self):
        result = parse_grounding("![title](80,70,920,140)\n# Report\n\n![table+](0,150,1000,900)\n<table><tr><td>A</td></tr></table>")
        self.assertFalse(result["errors"])
        self.assertFalse(result["unlocated_text"])
        a, b = result["blocks"]
        self.assertEqual(a["bbox"], [.08, .07, .92, .14])
        self.assertEqual(a["text"], "# Report")
        self.assertTrue(b["continuation"])
        self.assertEqual(b["label"], "table")
        self.assertIn("<table>", b["text"])

    def test_grounding_rejects_bad_boxes_without_repair(self):
        for marker, reason in (("![text](1,2,1001,9)", "out_of_range_bbox"),
                               ("![text](20,2,10,9)", "empty_or_inverted_bbox"),
                               ("![text](-1,2,10,9)", "malformed_marker"),
                               ("![text](1,2,10)", "malformed_marker")):
            result = parse_grounding(marker + "\ncontent")
            self.assertFalse(result["blocks"])
            self.assertEqual(result["errors"][0]["reason"], reason)
            self.assertIn(marker, result["unlocated_text"])

    def test_inline_spacing_and_unknown_labels(self):
        report = parse_grounding("![section+]( 1, 2 , 10, 9 ) heading ![future-label](10,10,20,20) content")
        self.assertEqual(len(report["blocks"]), 2)
        self.assertEqual(report["blocks"][0]["text"], "heading")
        self.assertTrue(report["blocks"][0]["continuation"])
        self.assertEqual(report["blocks"][1]["label"], "future-label")
        self.assertEqual(report["errors"][0]["reason"], "unknown_label")

    def test_code_and_unlocated_content_preserved(self):
        result = parse_grounding("lead\n![code](0,0,1000,1000)\n```text\n![text](1,2,8,9)\n```")
        self.assertEqual(result["unlocated_text"], "lead")
        self.assertEqual(len(result["blocks"]), 1)
        self.assertIn("![text]", result["blocks"][0]["text"])

    def test_frozen_case_digests(self):
        data = json.loads((self.fixture / "qualification.json").read_text())
        self.assertEqual(len(data["cases"]), 3)
        self.assertEqual(data["max_new_tokens"], 128)
        for case in data["cases"]:
            self.assertEqual(sha256(ROOT / case["image"]), case["sha256"])


try:
    import numpy as np
    from compare_lightonocr3 import BARS, compare, statistics
except ImportError:
    np = None


@unittest.skipIf(np is None, "numpy required for numerical acceptance controls")
class AcceptanceTests(unittest.TestCase):
    def test_owner_threshold_is_strict(self):
        self.assertEqual(BARS, {"f32": .99, "f16": .98, "q8_0": .97, "q4_k": .94})
        self.assertFalse(.99 > BARS["f32"])
        self.assertGreater(statistics([1, 2], [1, 2])["cosine"], BARS["f32"])
        for values in ([float("nan"), 1], [float("inf"), 1], [1]):
            with self.assertRaises(ValueError): statistics([1, 2], values)

    def test_missing_layer_and_final_logit_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref, native, auto, plain = [root / n for n in ("ref", "native", "auto", "plain")]
            for d in (ref, native, auto, plain): d.mkdir()
            stages = {"pixel_values", "prompt_ids", "patch_conv", "vis_ln_pre", "proj_out", "decoder_embed", "llm_final_norm", "logits_step0"}
            stages |= {f"vis_layer_{i}" for i in range(24)} | {f"llm_layer_{i}" for i in range(28)}
            meta = {"arrays": {}, "token_ids": [151645], "mode": "transcription", "stop_ids": [151645, 151643], "hit_cap": False, "seconds": 1}
            for name in stages:
                np.save(ref / f"{name}.npy", np.array([1, 2], dtype=np.float32), allow_pickle=False)
                meta["arrays"][name] = {"sha256": sha256(ref / f"{name}.npy")}
                np.array([1, 2], dtype=np.float32).tofile(native / f"{name}.f32")
            (ref / "reference.json").write_text(json.dumps(meta))
            for d in (ref, auto, plain): (d / "decoded.txt").write_text("")
            (auto / "seconds.txt").write_text("1")
            np.array([151645], dtype=np.float32).tofile(auto / "generated_ids.f32")
            self.assertTrue(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])
            (native / "llm_layer_27.f32").unlink()
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])
            np.array([1, 2], dtype=np.float32).tofile(native / "llm_layer_27.f32")
            (native / "logits_step0.f32").unlink()
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])
            meta["arrays"].pop("logits_step0")
            (ref / "reference.json").write_text(json.dumps(meta))
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])


if __name__ == "__main__":
    unittest.main()
