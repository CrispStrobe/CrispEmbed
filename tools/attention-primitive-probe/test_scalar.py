"""Hosted focused regression: raw all-key QK and separate causal scaled softmax."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from scalar import attention, expf


class ScalarStageTests(unittest.TestCase):
    def test_logits_are_unscaled_and_include_future_keys(self):
        q = np.zeros((4, 16, 256), dtype='<f4')
        k = np.zeros((4, 4, 256), dtype='<f4')
        v = np.ones_like(k)
        gate = np.zeros_like(q)
        q[:, :, 0] = 2
        for s in range(4):
            k[s, :, 0] = s + 1
        result = attention(q, k, v, gate)
        np.testing.assert_array_equal(result['logits'][0, 0, :4], [2, 4, 6, 8])
        self.assertTrue((result['logits'][:, :, 4:] == 0).all())
        np.testing.assert_array_equal(result['probs'][0, 0, :4], [1, 0, 0, 0])
        expected = np.float32(expf(np.float32(-.125)) / np.float32(expf(np.float32(-.125)) + np.float32(1)))
        self.assertEqual(result['probs'][0, 1, 0], expected)
        self.assertTrue((result['context'] == 1).all())
        self.assertTrue((result['gated'] == .5).all())


if __name__ == '__main__':
    unittest.main()
