#!/usr/bin/env python3
"""Weight-free Torch regression for immutable inference-mode captures."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from dump_ultrabert_reference import assert_masks, check_bos_negative


class InferenceCaptureTest(unittest.TestCase):
    def test_negative_mask_uses_independent_normal_clone(self):
        with torch.inference_mode():
            mask = torch.zeros(1, 1, 1, 1)
            captured = {f"layer_{i}.attention_mask": mask for i in range(24)}
        self.assertTrue(torch.is_inference(mask))
        encoded = {"input_ids": torch.zeros(1, 1, dtype=torch.int64),
                   "attention_mask": torch.ones(1, 1, dtype=torch.int64)}
        assert_masks(captured, encoded)
        check_bos_negative(captured, encoded)
        self.assertTrue(torch.is_inference(mask))
        self.assertEqual(mask.item(), 0)
        self.assertTrue(all(value is mask for value in captured.values()))
        assert_masks(captured, encoded)


if __name__ == "__main__":
    unittest.main()
