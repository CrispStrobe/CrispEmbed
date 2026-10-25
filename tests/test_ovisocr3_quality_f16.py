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
    def test_reviewed_native_requires_shared_cmake_and_gitlink_closure(self):
        head = 'a' * 40
        producer = {'source_commit': head,
                    'checkpoint_sha256': worker.policy.CHECKPOINT_SHA256,
                    'native_environment': worker.native_audit.NATIVE_ENV,
                    'exports': {'f16': {'model_sha256': 'b' * 64}}}
        canonical = worker.policy._canonical_sha256(producer)
        receipt = {'run': 123, 'head_sha': head, 'producer': producer,
                   'independent_review': {'producer_canonical_sha256': canonical,
                                          'all_fourteen_native_arms_positive': True}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.json'
            path.write_text(json.dumps(receipt))
            with patch.object(worker, 'NATIVE_MODEL_RECEIPT', path), \
                 patch.object(worker, 'NATIVE_MODEL_RUN', 123), \
                 patch.object(worker, 'NATIVE_MODEL_HEAD', head), \
                 patch.object(worker, 'NATIVE_MODEL_PRODUCER_SHA256', canonical), \
                 patch.object(worker.source_closure, 'require_native_source_closure',
                              return_value={'file_count': 29}) as closure:
                self.assertEqual(worker.require_reviewed_native(), producer)
                closure.assert_called_once_with(producer, head, root=ROOT)
            with patch.object(worker, 'NATIVE_MODEL_RECEIPT', path), \
                 patch.object(worker, 'NATIVE_MODEL_RUN', 123), \
                 patch.object(worker, 'NATIVE_MODEL_HEAD', head), \
                 patch.object(worker, 'NATIVE_MODEL_PRODUCER_SHA256', canonical), \
                 patch.object(worker.source_closure, 'require_native_source_closure',
                              side_effect=ValueError('ggml gitlink differs')):
                with self.assertRaisesRegex(ValueError, 'ggml gitlink'):
                    worker.require_reviewed_native()
            changed = dict(producer, native_environment={**producer['native_environment'],
                         'CRISPEMBED_QWEN35_QUANT': '1'})
            changed_receipt = dict(receipt, producer=changed,
                                   independent_review={'producer_canonical_sha256':
                                       worker.policy._canonical_sha256(changed),
                                       'all_fourteen_native_arms_positive': True})
            path.write_text(json.dumps(changed_receipt))
            with patch.object(worker, 'NATIVE_MODEL_RECEIPT', path), \
                 patch.object(worker, 'NATIVE_MODEL_RUN', 123), \
                 patch.object(worker, 'NATIVE_MODEL_HEAD', head), \
                 patch.object(worker, 'NATIVE_MODEL_PRODUCER_SHA256',
                              worker.policy._canonical_sha256(changed)), \
                 self.assertRaisesRegex(ValueError, 'native fourteen-arm source/export'):
                worker.require_reviewed_native()

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

    def test_native_environment_overrides_all_ambient_reviewed_gates(self):
        producer = {"native_environment": worker.native_audit.NATIVE_ENV,
                    "selected_vision_preset": worker.native_audit.SELECTED_VISION_PRESET}
        selection = worker.vision_env.reviewed_selection(producer)
        poisoned = {name: "0" if value == "1" else "1"
                    for name, value in worker.native_audit.NATIVE_ENV.items()}
        poisoned["UNRELATED_TEST_SENTINEL"] = "preserved"
        env = worker.native_environment(selection, poisoned)
        self.assertEqual({name: env[name] for name in worker.native_audit.NATIVE_ENV},
                         worker.native_audit.NATIVE_ENV)
        self.assertEqual(env["UNRELATED_TEST_SENTINEL"], "preserved")
        self.assertEqual(env["CRISPEMBED_OVISOCR2_LARGE_IMAGES"], "0")
        self.assertEqual(env["CRISPEMBED_QWEN35_QUANT"], "0")

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
