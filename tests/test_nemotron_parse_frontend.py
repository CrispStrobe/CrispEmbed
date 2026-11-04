#!/usr/bin/env python3
"""Weight-free receipt/metric controls, not original/native processor evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("nemotron_frontend", ROOT / "tools/nemotron_parse_frontend.py")
frontend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frontend)


def synthetic_receipt():
    # Synthetic metadata is used ONLY to exercise schema and negative controls.
    # Actual geometry/pixels must come from the pinned original capture tool.
    rows = []
    for case_id, (height, width, channels) in frontend.CASES.items():
        h, w, aspect = height, width, width / height
        if h > 2048:
            h, w = 2048, int(2048 * aspect)
        if w > 1664:
            w, h = 1664, int(1664 / aspect)
        ph, pw = (h + 15) // 16 * 16, (w + 63) // 64 * 64
        rows.append({"id": case_id, "original_shape": [height, width, channels],
                     "content_sizes": [h, w], "image_sizes": [ph, pw],
                     "patch_shape": [ph // 16 * (pw // 16), 768], "input_bytes": h * w * channels,
                     "reference_bytes": ph * pw * 3 * 4, "input_sha256": "1" * 64,
                     "reference_sha256": "2" * 64, "original_pixels_sha256": "3" * 64})
    return {"schema": "crispembed.nemotron-parse.frontend-reference.v1",
            "repo": frontend.REPOSITORY, "revision": frontend.REVISION,
            "source_sha256": dict(frontend.SOURCES), "packages": dict(frontend.PACKAGES),
            "case_order": list(frontend.CASES), "cases": rows,
            "negative_cases": [{"id": key, "original_shape": list(shape), "rejected": True}
                               for key, shape in frontend.NEGATIVE_CASES.items()],
            "max_abs_budget": frontend.MAX_ABS, "min_cosine_budget": frontend.MIN_COSINE,
            "native_resize_qualified": False, "model_execution": False, "model_qualified": False}


class ReferenceControls(unittest.TestCase):
    def test_complete_synthetic_metadata_schema(self):
        frontend.validate_reference(synthetic_receipt())

    def test_missing_duplicate_and_reordered_cases_fail(self):
        for edit in (lambda x: x["cases"].pop(),
                     lambda x: x["cases"].reverse(),
                     lambda x: x["cases"].__setitem__(1, copy.deepcopy(x["cases"][0]))):
            receipt = synthetic_receipt()
            edit(receipt)
            with self.assertRaises(ValueError):
                frontend.validate_reference(receipt)

    def test_source_revision_and_source_hash_drift_fail(self):
        for edit in (lambda x: x.update(revision="main"),
                     lambda x: x["source_sha256"].update({"hf_nemotron_parse_processor.py": "f" * 64})):
            receipt = synthetic_receipt()
            edit(receipt)
            with self.assertRaisesRegex(ValueError, "identity"):
                frontend.validate_reference(receipt)

    def test_malformed_sizes_bytes_hashes_fail(self):
        edits = [lambda x: x["cases"][0].update(content_sizes=[True, 1]),
                 lambda x: x["cases"][0].update(image_sizes=[336, 187]),
                 lambda x: x["cases"][0].update(patch_shape=[1, 768]),
                 lambda x: x["cases"][0].update(input_bytes=1),
                 lambda x: x["cases"][0].update(reference_sha256="g" * 64),
                 lambda x: x["cases"][4].update(original_shape=[True, 65, 1])]
        for edit in edits:
            receipt = synthetic_receipt()
            edit(receipt)
            with self.assertRaises(ValueError):
                frontend.validate_reference(receipt)

    def test_missing_or_changed_negative_controls_fail(self):
        for edit in (lambda x: x["negative_cases"].pop(),
                     lambda x: x["negative_cases"][0].update(rejected="true")):
            receipt = synthetic_receipt()
            edit(receipt)
            with self.assertRaises(ValueError):
                frontend.validate_reference(receipt)

    def test_budget_and_package_relabeling_fail(self):
        for edit in (lambda x: x.update(max_abs_budget=1),
                     lambda x: x["packages"].update(transformers="main"),
                     lambda x: x.update(packages=None)):
            receipt = synthetic_receipt()
            edit(receipt)
            with self.assertRaises(ValueError):
                frontend.validate_reference(receipt)

    def test_processor_evidence_cannot_claim_model_or_resize_qualification(self):
        for key in ("model_qualified", "model_execution", "native_resize_qualified"):
            receipt = synthetic_receipt()
            receipt[key] = True
            with self.assertRaisesRegex(ValueError, "unsupported qualification"):
                frontend.validate_reference(receipt)

    def test_corrupt_source_bytes_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hf_nemotron_parse_processor.py").write_text("raise Exception('must not execute')\n")
            with self.assertRaisesRegex(ValueError, "source drift"):
                frontend.verified_sources(directory)

    def test_strict_json_and_existing_output_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "receipt.json")
            for payload in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e309}'):
                path.write_text(payload)
                with self.assertRaises(ValueError):
                    frontend.load_json(path)
            with self.assertRaisesRegex(ValueError, "empty"):
                frontend.empty_directory(directory)


class NumericControls(unittest.TestCase):
    def test_exact_and_changed_channel_order(self):
        import numpy as np
        reference = np.asarray([-.7, .2, 1.3, 2.1, -.4, 1.8], dtype=np.float32)
        exact = frontend.array_metrics(reference, reference.copy())
        self.assertTrue(exact["packing_match"])
        self.assertEqual(exact["changed_values"], 0)
        self.assertEqual(exact["max_abs"], 0)
        self.assertFalse(frontend.array_metrics(reference, reference[::-1])["packing_match"])

    def test_rounding_difference_retained_with_frozen_budget(self):
        import numpy as np
        reference = np.asarray([1, 2, 3], dtype=np.float32)
        candidate = reference.copy()
        candidate[0] = np.nextafter(candidate[0], np.float32(2))
        result = frontend.array_metrics(reference, candidate)
        self.assertTrue(result["packing_match"])
        self.assertEqual(result["changed_values"], 1)
        self.assertEqual(result["first_changed_index"], 0)
        candidate[0] += .001
        self.assertFalse(frontend.array_metrics(reference, candidate)["packing_match"])

    def test_missing_nonfinite_and_zero_arrays_fail(self):
        import numpy as np
        reference = np.asarray([1, 2, 3], dtype=np.float32)
        for candidate in (reference[:0], reference[:2], np.zeros(3, dtype=np.float32),
                          np.asarray([1, float("nan"), 3]), np.asarray([1, float("inf"), 3])):
            with self.assertRaises(ValueError):
                frontend.array_metrics(reference, candidate)


if __name__ == "__main__":
    unittest.main()
