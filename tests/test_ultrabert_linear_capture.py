#!/usr/bin/env python3
"""Weight-free controls: added MLP probes cannot replace changed original anchors."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from dump_ultrabert_linear_reference import FrozenForwardDrift, check_full_anchors, observe_local_operators, observe_mlp


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

    def test_full_forward_anchor_drift_keeps_first_stage_and_magnitude(self):
        import hashlib
        from types import SimpleNamespace
        old = torch.tensor([[0., 2.]])
        record = {"shape": [1, 2], "payload_sha256": hashlib.sha256(old.numpy().tobytes()).hexdigest()}
        records = {"first": record, "second": record}
        tensors = {name: SimpleNamespace(data=old.numpy()) for name in records}
        check_full_anchors({name: old for name in records}, records, tensors)
        changed = old.clone()
        changed[0, 1] = 3.
        with self.assertRaises(FrozenForwardDrift) as result:
            check_full_anchors({"first": changed, "second": old}, records, tensors)
        self.assertEqual(result.exception.drifts[0]["stage"], "first")
        self.assertEqual(result.exception.drifts[0]["max_abs_error"], 1.)
        self.assertEqual(result.exception.drifts[0]["original"], record)

    def test_local_operator_targets_call_original_modules_without_changing_inputs(self):
        module = MLP().eval()
        gate = torch.tensor([[[1., -2., 3., 0.]]])
        up = torch.tensor([[[2., 4., -1., 1.]]])
        before = gate.clone(), up.clone()
        calls = []
        handles = [module.act_fn.register_forward_hook(lambda *args: calls.append("act")),
                   module.down_proj.register_forward_hook(lambda *args: calls.append("down"))]
        try:
            values = observe_local_operators(module, gate, up)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, ["act", "down"])
        self.assertEqual(set(values), {"silu", "product", "local_down"})
        self.assertTrue(torch.equal(gate, before[0]) and torch.equal(up, before[1]))
        self.assertNotIn("ffn_out", values)

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
