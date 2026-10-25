"""Pure source and schema gates for the pending hosted OvisOCR3 native audit."""

import copy
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


class Native14Controls(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
