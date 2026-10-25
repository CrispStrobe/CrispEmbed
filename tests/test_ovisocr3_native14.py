"""Pure source and schema gates for the pending hosted OvisOCR3 native audit."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ovisocr3_native14_test", ROOT / "tools/ci-heavy/ovisocr3_native14.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
vision_spec = importlib.util.spec_from_file_location(
    "ovisocr3_native_vision_env_test", ROOT / "tools/ci-heavy/ovisocr2_vision_env.py")
vision_env = importlib.util.module_from_spec(vision_spec)
vision_spec.loader.exec_module(vision_env)


class Native14Controls(unittest.TestCase):
    def test_selected_vision_preset_is_exact_recorded_candidate(self):
        producer = {"native_environment": audit.NATIVE_ENV,
                    "selected_vision_preset": audit.SELECTED_VISION_PRESET}
        chosen = vision_env.reviewed_selection(producer)
        self.assertEqual(chosen["name"], "both")
        self.assertEqual(chosen["values"], {
            "CRISPEMBED_QWEN35_VISION_PRECISE_NORM": "1",
            "CRISPEMBED_QWEN35_VISION_PRECISE_ATTN": "1",
            "CRISPEMBED_QWEN35_VISION_CASCADE_NORM": "0"})
        changed = copy.deepcopy(producer)
        changed["native_environment"]["CRISPEMBED_QWEN35_VISION_PRECISE_ATTN"] = "0"
        with self.assertRaises(ValueError):
            vision_env.reviewed_selection(changed)

    def test_absent_reviewed_original_stops_before_service_and_weights(self):
        with patch.object(audit, "profile") as profile:
            profile._reviewed_receipts.side_effect = ValueError("proof absent")
            with self.assertRaisesRegex(ValueError, "proof absent"):
                audit.service_and_review(lambda args: self.fail("service called"))

    def test_actual_metadata_schema_precision_and_mutations(self):
        evidence = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr3-metadata.json").read_text())
        schema = evidence["producer"]["schema"]["official_state_dict"]
        records = []
        for source_name, source_shape in schema.items():
            if source_name == "lm_head.weight":
                continue
            if source_name.startswith("model.language_model."):
                name = "l." + source_name.removeprefix("model.language_model.")
            elif source_name.startswith("model.visual."):
                name = "v." + source_name.removeprefix("model.visual.")
            else:
                self.fail("Unknown actual source prefix")
            shape = ([source_shape[0], source_shape[1] * source_shape[2] * source_shape[3] * source_shape[4]]
                     if name == "v.patch_embed.proj.weight" else source_shape)
            records.append({"source": source_name, "name": name, "source_shape": source_shape,
                            "shape": shape, "dtype": "float16" if len(shape) == 2 and "norm" not in name else "float32",
                            "sha256": "a" * 64})
        manifest = {"profile": {"name": "ovisocr3", "qualification_kind": "hf5.16.1-original-reference"},
                    "source_model": audit.source.MODEL, "source_revision": audit.source.REVISION,
                    "dtype": "f16", "keep_f32": [], "checkpoint_tensor_count": 473,
                    "export_tensor_count": 473, "records": records}
        self.assertEqual(len(records), 473)
        audit.validate_manifest(manifest, "f16", schema)
        for key, wrong in (("checkpoint_tensor_count", 472), ("dtype", "f32")):
            changed = copy.deepcopy(manifest)
            changed[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.validate_manifest(changed, "f16", schema)
        patch_record = next(row for row in records if row["name"] == "v.patch_embed.proj.weight")
        self.assertEqual(patch_record["shape"], [768, 1536])
        changed = copy.deepcopy(manifest)
        next(row for row in changed["records"] if row["name"] == "v.patch_embed.proj.weight")["shape"] = [768, 3, 2, 16, 16]
        with self.assertRaises(ValueError):
            audit.validate_manifest(changed, "f16", schema)
        changed = copy.deepcopy(manifest)
        changed["records"][0]["source"] = "unknown.weight"
        with self.assertRaises(ValueError):
            audit.validate_manifest(changed, "f16", schema)

    def test_exact_four_reference_members_reject_extra_or_changed(self):
        fake = {"producer": {"reference_artifacts": {
            "npz": {"sha256": "0" * 64}, "json": {"sha256": "0" * 64},
            "gguf": {"sha256": "0" * 64}},
            "owned_readable": {"file_sha256": "0" * 64}}}
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "small.zip"
            archive.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "archive SHA differs"):
                audit.checked_reference_archive(archive, fake, Path(directory) / "out")

    def test_raw_decoder_manifest_requires_all_scores_cache_and_trace(self):
        layers = {f"{phase}.decoder.layer.{number}" for phase in ("prefill", "cached", "full_prefix")
                  for number in range(24)}
        cache = {"prefill.cache.0.conv_states", "cached.cache.0.conv_states"}
        scores = {"generated.scores.0", "generated.scores.1"}
        trace = {"prefill.attention.0.input"}
        names = layers | cache | scores | trace
        log = [f"PASS {name}: cos_min=1" for name in sorted(names - trace)]
        log += ["TRACE prefill.attention.0.input max_abs=0 cos=1 norms=1/1"]
        shapes = {name: [1] for name in names}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in names:
                (folder / (name + ".f32")).write_bytes(b"\0" * 4)
            result = audit.raw_diff_manifest(folder, shapes, log, "f16-decoder-native-frontend", 2)
            self.assertEqual(result["raw_vector_count"], len(names))
            self.assertEqual(result["trace_stage_count"], 1)
            (folder / "generated.scores.1.f32").unlink()
            with self.assertRaises(ValueError):
                audit.raw_diff_manifest(folder, shapes, log, "f16-decoder-native-frontend", 2)
            (folder / "generated.scores.1.f32").write_bytes(b"\0" * 4)
            with self.assertRaises(ValueError):
                audit.raw_diff_manifest(folder, shapes,
                                        [line for line in log if not line.startswith("PASS generated.scores.1:")],
                                        "f16-decoder-native-frontend", 2)

    def test_native_vision_decoder_is_not_standalone_vision_and_requires_all_scores(self):
        layers = {f"{phase}.decoder.layer.{number}" for phase in ("prefill", "cached", "full_prefix")
                  for number in range(24)}
        cache = {"prefill.cache.0.keys", "cached.cache.0.recurrent_states"}
        scores = {f"generated.scores.{number}" for number in range(9)}
        names = layers | cache | scores
        log = [f"PASS {name}: cos_min=1" for name in sorted(names)]
        shapes = {name: [1] for name in names}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in names:
                (folder / (name + ".f32")).write_bytes(b"\0" * 4)
            result = audit.raw_diff_manifest(folder, shapes, log, "f16-decoder-native-vision", 9)
            self.assertEqual(result["raw_vector_count"], len(names))
            self.assertIsNone(result["dump_integrity_marker"])
            with self.assertRaisesRegex(ValueError, "all-layer/cache/all-token-score"):
                audit.raw_diff_manifest(folder, shapes, log, "f16-decoder-native-vision", 10)
            with self.assertRaisesRegex(ValueError, "Unknown OvisOCR3 numerical arm"):
                audit.raw_diff_manifest(folder, shapes, log, "f16-other-vision", 9)

    def test_original_fixed_task_source_rejects_wrong_task_and_rendering(self):
        metadata = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr3-metadata.json").read_text())
        rendered = metadata["rendered_prompt"]
        task = rendered.split("<|vision_end|>", 1)[1].split("<|im_end|>", 1)[0]
        self.assertEqual(hashlib.sha256(task.encode()).hexdigest(),
                         metadata["producer"]["task_prompt_sha256"])
        original = {"prompt": task, "rendered_prompt": rendered}
        audit.validate_fixed_task_source(original, metadata)
        with self.assertRaisesRegex(ValueError, "fixed-task prompt"):
            audit.validate_fixed_task_source({**original, "prompt": task + " "}, metadata)
        with self.assertRaisesRegex(ValueError, "fixed-task prompt"):
            audit.validate_fixed_task_source({**original, "rendered_prompt": rendered + " "}, metadata)


if __name__ == "__main__":
    unittest.main()
