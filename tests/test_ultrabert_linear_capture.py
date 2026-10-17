#!/usr/bin/env python3
"""Weight-free controls: added MLP probes cannot replace changed original anchors."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from dump_ultrabert_linear_reference import observe_mlp


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(3, 4, bias=False)
        self.up_proj = torch.nn.Linear(3, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 3, bias=False)
        self.act_fn = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LinearCaptureTest(unittest.TestCase):
    def test_neutral_capture_preserves_inputs_and_original_anchors(self):
        module = MLP().eval()
        x = torch.tensor([[[1., -2., 3.], [0., 4., -1.]]])
        before = x.clone()
        expected = dict(gate=module.gate_proj(x), up=module.up_proj(x), ffn_out=module(x))
        captured = observe_mlp(module, x, expected)
        self.assertEqual(set(captured), {"gate", "up", "silu", "product", "ffn_out"})
        self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.equal(captured['ffn_out'], expected['ffn_out']))
        self.assertFalse(module.act_fn._forward_hooks)
        self.assertFalse(module.down_proj._forward_pre_hooks)

    def test_signed_zero_anchor_drift_is_not_replaced(self):
        module = MLP().eval()
        with torch.no_grad():
            module.gate_proj.weight.fill_(1)
        x = torch.zeros(1, 1, 3)
        expected = dict(gate=module.gate_proj(x).detach().clone(), up=module.up_proj(x), ffn_out=module(x))
        expected['gate'][0, 0, 0] = -0.0
        self.assertTrue(torch.equal(expected['gate'], module.gate_proj(x)))
        with self.assertRaisesRegex(ValueError, 'Frozen MLP anchor drift: gate; max_abs=0.0'):
            observe_mlp(module, x, expected)

    def test_changed_old_anchor_rejected_and_hooks_removed(self):
        module = MLP().eval()
        x = torch.ones(1, 1, 3)
        expected = dict(gate=module.gate_proj(x) + 1, up=module.up_proj(x), ffn_out=module(x))
        with self.assertRaisesRegex(ValueError, 'Frozen MLP anchor drift: gate; max_abs='):
            observe_mlp(module, x, expected)
        self.assertFalse(module.gate_proj._forward_hooks)
        self.assertFalse(module.down_proj._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()
