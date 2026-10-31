"""Weight-free contracts for models/convert-oxr-dfine-to-gguf.py.

A synthetic state dict with the pinned checkpoint's key set drives the real
``build()``: every output tensor name must fit GGML_MAX_NAME (63 bytes + NUL),
the expected inventory and storage recipes must hold, and authentication must
reject tampered sources. No neural parity is implied.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GGML_MAX_NAME = 64  # ggml.h: names are stored in char[64] including the NUL


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


converter = load("oxr_dfine_converter_tests", ROOT / "models/convert-oxr-dfine-to-gguf.py")

try:
    import torch
except ImportError:  # pragma: no cover - torch is a converter requirement
    torch = None


class FakeState(dict):
    """Answers every key the converter asks for with a plausible tensor."""

    def __contains__(self, key):
        if key.endswith("num_batches_tracked"):
            return False
        if key.endswith(".lab.scale"):  # LAB exists on activated ConvBNAct only
            return not (key.split(".lab.")[0].endswith(".conv1") or ".downsample" in key)
        return True

    def __getitem__(self, key):
        if key == "decoder.anchors":
            return torch.zeros(1, 18900, 4)
        if key == "decoder.valid_mask":
            return torch.ones(1, 18900, 1, dtype=torch.bool)
        if key.endswith("reg_scale"):
            return torch.tensor([4])
        if key.endswith(".up"):
            return torch.tensor([0.5])
        if key.endswith("num_points_scale"):
            return torch.tensor([1 / n for n in (3, 6, 3) for _ in range(n)], dtype=torch.float32)
        if key.endswith("running_var"):
            return torch.ones(4)
        if ".bottlenecks." in key and key.endswith(".conv2.conv.weight"):
            return torch.zeros(4, 4, 1, 1)  # VGGBlock 1x1 branch
        if key.endswith("conv.weight"):
            return torch.zeros(4, 4, 3, 3)
        return torch.zeros(4)

    def __iter__(self):
        return iter(())


@unittest.skipIf(torch is None, "torch is required by the converter")
class DfineConverterTests(unittest.TestCase):
    def build(self, dtype="f32"):
        return converter.build(FakeState(), dtype)

    def test_names_fit_ggml_max_name(self):
        names = list(self.build().out)
        too_long = [n for n in names if len(n.encode("utf-8")) >= GGML_MAX_NAME]
        self.assertEqual(too_long, [])
        self.assertLessEqual(max(len(n.encode()) for n in names), 63)

    def test_inventory(self):
        out = self.build().out
        self.assertEqual(len(out), 533)
        for name in ("encoder.pos_embed2", "decoder.project", "decoder.anchors", "decoder.valid_mask",
                     "decoder.decoder.lqe_layers.3.reg_conf.layers.1.weight", "decoder.dec_score_head.3.weight",
                     "backbone.stem.stem1.lab.scale", "encoder.fpn_blocks.0.cv2.0.bottlenecks.1.fused.weight"):
            self.assertIn(name, out)
        self.assertEqual(tuple(out["encoder.pos_embed2"][0].shape), (900, 256))
        self.assertEqual(tuple(out["decoder.project"][0].shape), (33,))

    def test_put_rejects_long_names(self):
        builder = converter.Builder(FakeState(), "f32")
        with self.assertRaises(ValueError):
            builder.put("x" * 64, torch.zeros(1))
        builder.put("x" * 63, torch.zeros(1))

    def test_f16_recipe_keeps_vectors_f32(self):
        out = self.build("f16").out
        half = {n for n, (_, h) in out.items() if h}
        self.assertIn("backbone.stem.stem1.conv.weight", half)
        self.assertIn("decoder.decoder.layers.0.linear1.weight", half)
        for name in ("backbone.stem.stem1.bn.scale", "decoder.anchors", "decoder.project",
                     "decoder.enc_output.norm.weight", "decoder.decoder.lqe_layers.3.reg_conf.layers.0.weight"):
            self.assertNotIn(name, half)

    def test_project_matches_upstream_formula(self):
        project = converter.weighting_function_deploy(32, torch.tensor([0.5]), torch.tensor([4]))
        self.assertEqual(project.shape, (33,))
        self.assertEqual(float(project[0]), -4.0)
        self.assertEqual(float(project[16]), 0.0)
        self.assertEqual(float(project[-1]), 4.0)

    def test_authentication_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "layout_analysis").mkdir()
            (root / converter.CHECKPOINT).write_bytes(b"not the checkpoint")
            (root / converter.METADATA).write_text(json.dumps({"schema_version": 1}))
            with self.assertRaises(ValueError):
                converter.authenticate(root)

    def test_manifest_pins(self):
        entries = converter.manifest_entries()
        self.assertEqual(entries[converter.CHECKPOINT]["size"], 78791307)
        self.assertEqual(entries[converter.METADATA]["size"], 718)


if __name__ == "__main__":
    unittest.main()
