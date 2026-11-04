#!/usr/bin/env python3
"""Model-free LightOnOCR-3 input, parsing and acceptance controls."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "models"), str(ROOT / "tools")]
from lightonocr3_profile import (manifest, validate_config, sha256, TRANSCRIPTION, GROUNDING,
                                MANIFEST, MODEL_ID, REVISION, REFERENCE_PACKAGES, trace_shapes)
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
    def capture(self, directory):
        root = Path(directory)
        ref, native, auto, plain = [root / n for n in ("ref", "native", "auto", "plain")]
        for d in (ref, native, auto, plain): d.mkdir()
        meta = {"schema_version": 2, "model_id": MODEL_ID, "revision": REVISION,
                "source_manifest_sha256": sha256(MANIFEST), "dtype": "float32", "device": "cpu",
                "attention": "eager", "do_sample": False, "use_cache": True, "versions": REFERENCE_PACKAGES,
                "image_size": [28, 28], "image_sha256": "a" * 64, "max_new_tokens": 4,
                "arrays": {}, "token_ids": [151645], "mode": "transcription", "stop_ids": [151645, 151643],
                "forced_eos_token_ids": [], "hit_cap": False, "seconds": 1}
        shapes, prompt = trace_shapes(meta["image_size"], meta["mode"], 1)
        for name, shape in shapes.items():
            a = np.ones(shape, dtype=np.float32)
            if name == "prompt_ids": a = np.array(prompt, dtype=np.float32)
            if name == "generated_ids": a = np.array(meta["token_ids"], dtype=np.float32)
            if name.startswith("logits_step"):
                a.fill(0)
                a[0, 151645] = 2
            np.save(ref / f"{name}.npy", a, allow_pickle=False)
            meta["arrays"][name] = {"sha256": sha256(ref / f"{name}.npy"), "shape": list(shape)}
            if name != "generated_ids": a.tofile(native / f"{name}.f32")
        for d in (ref, auto, plain): (d / "decoded.txt").write_text("", encoding="utf-8")
        (ref / "forced-ids.txt").write_text("151645\n")
        meta.update(decoded_sha256=sha256(ref / "decoded.txt"), forced_ids_sha256=sha256(ref / "forced-ids.txt"))
        (ref / "reference.json").write_text(json.dumps(meta))
        for arm, d in (("forced", native), ("autonomous", auto), ("plain", plain)):
            receipt = {"schema_version": 2, "profile": "lightonocr3-1b", "image_sha256": meta["image_sha256"],
                       "mode": meta["mode"], "max_new_tokens": 1 if arm == "forced" else 4,
                       "model_bytes": 1000, "observed": arm != "plain", "seconds": 1,
                       "forced_ids": meta["token_ids"] if arm == "forced" else [],
                       "generated_ids": meta["token_ids"], "argmax_ids": meta["token_ids"]}
            (d / "trace.json").write_text(json.dumps(receipt))
            execution = {"schema_version": 1, "build_mode": "portable", "precision": "f32", "arm": arm,
                         "git_commit": "b" * 40, "model_bytes": 1000, "model_sha256": "c" * 64,
                         "binary_sha256": "d" * 64, "trace_sha256": sha256(d / "trace.json"),
                         "process": {"exit": 0, "timed_out": False}}
            (d / "execution.json").write_text(json.dumps(execution))
        return ref, native, auto, plain, meta

    def change_native(self, directory, **changes):
        data = json.loads((directory / "trace.json").read_text())
        data.update(changes)
        (directory / "trace.json").write_text(json.dumps(data))
        execution = json.loads((directory / "execution.json").read_text())
        execution["trace_sha256"] = sha256(directory / "trace.json")
        (directory / "execution.json").write_text(json.dumps(execution))

    def test_owner_threshold_is_strict(self):
        self.assertEqual(BARS, {"f32": .99, "f16": .98, "q8_0": .97, "q4_k": .94})
        self.assertFalse(.99 > BARS["f32"])
        self.assertGreater(statistics([1, 2], [1, 2])["cosine"], BARS["f32"])
        for values in ([float("nan"), 1], [float("inf"), 1], [1]):
            with self.assertRaises(ValueError): statistics([1, 2], values)

    def test_missing_layer_and_final_logit_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, native, auto, plain, meta = self.capture(directory)
            self.assertTrue(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])
            (native / "llm_layer_27.f32").unlink()
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])
            np.load(ref / "llm_layer_27.npy").tofile(native / "llm_layer_27.f32")
            (native / "logits_step0.f32").unlink()
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])
            meta["arrays"].pop("logits_step0")
            (ref / "reference.json").write_text(json.dumps(meta))
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])

    def test_short_logits_cannot_masquerade_as_full_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, native, auto, plain, meta = self.capture(directory)
            a = np.array([[1, 2]], dtype=np.float32)
            np.save(ref / "logits_step0.npy", a)
            a.tofile(native / "logits_step0.f32")
            meta["arrays"]["logits_step0"] = {"sha256": sha256(ref / "logits_step0.npy"), "shape": [1, 2]}
            (ref / "reference.json").write_text(json.dumps(meta))
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])

    def test_reference_source_input_and_cap_binding(self):
        for key, value in (("model_id", "different/model"), ("revision", "f" * 40),
                           ("source_manifest_sha256", "f" * 64), ("image_sha256", "f" * 64),
                           ("mode", "grounding"), ("hit_cap", True), ("versions", {})):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                ref, native, auto, plain, meta = self.capture(directory)
                meta[key] = value
                (ref / "reference.json").write_text(json.dumps(meta))
                self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])

    def test_reference_decoded_and_forcing_files_are_bound(self):
        for name in ("decoded.txt", "forced-ids.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                ref, native, auto, plain, _ = self.capture(directory)
                (ref / name).write_text("changed")
                self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])

    def test_native_forcing_and_identity_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, native, auto, plain, _ = self.capture(directory)
            self.change_native(native, generated_ids=[151643])
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])
        with tempfile.TemporaryDirectory() as directory:
            ref, native, auto, plain, _ = self.capture(directory)
            receipt = json.loads((auto / "execution.json").read_text())
            receipt["model_sha256"] = "f" * 64
            (auto / "execution.json").write_text(json.dumps(receipt))
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])

    def test_observer_ids_must_match_even_when_decoded_text_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, native, auto, plain, _ = self.capture(directory)
            self.change_native(plain, generated_ids=[151643], argmax_ids=[151643])
            report = compare(ref, native, auto, plain, "f32")
            self.assertFalse(report["owner_bar_pass"])
            self.assertTrue(report["decoded"]["observer_text_exact"])
            self.assertFalse(report["decoded"]["observer_ids_exact"])

    def test_noninteger_native_ids_and_trailing_tensor_bytes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            ref, native, auto, plain, _ = self.capture(directory)
            self.change_native(auto, generated_ids=[151645.5], argmax_ids=[151645.5])
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])
        with tempfile.TemporaryDirectory() as directory:
            ref, native, auto, plain, _ = self.capture(directory)
            with (native / "logits_step0.f32").open("ab") as stream: stream.write(b"x")
            self.assertFalse(compare(ref, native, auto, plain, "f32")["owner_bar_pass"])


class StorageTests(unittest.TestCase):
    def test_actual_storage_selects_precision_bar(self):
        spec = importlib.util.spec_from_file_location("lighton_runner", ROOT / "tools/ci-heavy/lightonocr3_1b.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        def text(value):
            value = value.encode("utf-8")
            return struct.pack("<Q", len(value)) + value
        metadata = {"general.architecture": "lightonocr", "lightonocr.profile": "lightonocr3-1b",
                    "lightonocr.profile_version": 1, "lightonocr.source.model": MODEL_ID,
                    "lightonocr.source.revision": REVISION}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authored-f32.gguf"
            payload = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata))
            for key, value in metadata.items():
                payload += text(key) + (struct.pack("<II", 4, value) if type(value) is int else struct.pack("<I", 8) + text(value))
            payload += text("authored.weight") + struct.pack("<IQIQ", 1, 4, 0, 0)
            payload += b"\0" * (-len(payload) % 32) + struct.pack("<ffff", 1, 2, 3, 4)
            path.write_bytes(payload)
            self.assertEqual(runner.model_census(path, "f32")["tensor_type_counts"], {0: 1})
            for precision in ("f16", "q8_0", "q4_k"):
                with self.subTest(precision=precision), self.assertRaisesRegex(ValueError, "storage"):
                    runner.model_census(path, precision)


if __name__ == "__main__":
    unittest.main()
