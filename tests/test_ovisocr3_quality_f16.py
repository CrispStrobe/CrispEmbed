"""Pure native OvisOCR3 F16 result and receipt prerequisites."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ovisocr3_quality_f16_test", ROOT / "tools/ci-heavy/ovisocr3_quality_f16.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class NativeF16Controls(unittest.TestCase):
    def test_preflight_binds_reviewed_native_vision_preset(self):
        native = {"native_environment": worker.native_audit.NATIVE_ENV,
                  "selected_vision_preset": worker.native_audit.SELECTED_VISION_PRESET}
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"HEAVY_OUT": directory, "HEAVY_SCRATCH": directory}), \
             patch.object(worker, "require_reviewed_original_quality", return_value={"status": "reviewed"}), \
             patch.object(worker, "require_reviewed_native", return_value=native), \
             patch.object(worker.original.reference, "metadata_service", return_value=({}, {})):
            report = worker.main(preflight=True)
        self.assertEqual(report["selected_vision_preset"]["name"], "both")
        self.assertEqual(report["status"], "preflight passed; no model/framework loaded")
        changed = dict(native, selected_vision_preset={"name": "baseline", "enabled_gates": []})
        with self.assertRaises(ValueError):
            worker.vision_env.reviewed_selection(changed)

    def test_missing_reviewed_original_and_fourteen_arm_receipts(self):
        with patch.object(worker, "ORIGINAL_QUALITY_RUN", 0), \
             self.assertRaisesRegex(ValueError, "Reviewed original OvisOCR3 eleven-case quality is absent"):
            worker.require_reviewed_original_quality()
        with patch.object(worker, "NATIVE_MODEL_RUN", 0), \
             self.assertRaisesRegex(ValueError, "Reviewed OvisOCR3 native fourteen-arm model proof is absent"):
            worker.require_reviewed_native()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_real_runner_json_contract_eos_at_cap_and_invalids(self):
        case = {"id": "dev-hello", "rgb_sha256": "a" * 64, "png_sha256": "b" * 64, "cap": 2}
        payload = {"raw_text_utf8_hex": "48656c6c6f", "token_ids": [100, 248046],
                   "confidences": [0.5, 0.75], "mean_confidence": 0.625,
                   "stop_kind": 1, "stop_id": 248046, "cap": 2,
                   "wall_seconds": 0.5, "init_seconds": 1.5}
        line = "OVIS3_QUALITY_RESULT " + json.dumps(payload)
        row = worker.parse_native_result(line, case)
        self.assertEqual(row["raw_text"], "Hello")
        self.assertEqual(row["stop_kind"], "eos")
        self.assertTrue(worker.contract.score_case("Hello", row, 2)["complete"])
        for mutation in ({"stop_id": None}, {"mean_confidence": 0.2},
                         {"token_ids": [100, 248046, 200]}, {"raw_text_utf8_hex": "ff"},
                         {"stop_kind": 2}):
            bad = dict(payload, **mutation)
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, UnicodeDecodeError)):
                worker.parse_native_result("OVIS3_QUALITY_RESULT " + json.dumps(bad), case)
        with self.assertRaises(ValueError):
            worker.parse_native_result(line + "\n" + line, case)


if __name__ == "__main__":
    unittest.main()
