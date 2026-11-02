#!/usr/bin/env python3
"""Convert OXR-1.0's D-FINE layout detector (layout_analysis/last.pth) to GGUF.

The checkpoint is SoMarkAI/OXR-1.0 at the immutable revision recorded in
models/oxr-source.json. Its size and SHA-256 (and those of the companion
metadata.json) are verified before the file is deserialized, and it is loaded
with ``torch.load(weights_only=True)``: no pickled code is executed.

The output is a second architecture of CrispEmbed's ``layout_detect`` module
(``layout.arch = "dfine"``), consumed by src/layout_detect.cpp. Arithmetic
choices mirror the upstream service (services/layout_analyze/dfine_la in the
pinned OXR blueprint) after ``model.deploy()``:

* encoder ``ConvNormLayer_fuse`` and ``VGGBlock`` are BN-fused exactly as their
  ``convert_to_deploy`` does (torch float32 ops, same formula);
* backbone ``FrozenBatchNorm2d`` and encoder ``input_proj`` BatchNorm stay a
  separate per-channel scale/shift (the reference does not fuse them);
* the AIFI 2-D sin-cos position table and D-FINE's deploy-mode weighting
  function ``project`` are computed by the original formulas, since neither
  is a persistent checkpoint tensor.

F32 is the numerical baseline. ``--dtype f16`` stores conv/linear matrices in
F16 and keeps every vector, norm, table and head input F32; it is a separate,
unqualified candidate. The checkpoint is covered by the OXR License (Apache-2.0
with additional terms, licenses/oxr-LICENSE).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

MODEL_ID = "SoMarkAI/OXR-1.0"
MODEL_REVISION = "c708407f9d2b4df658f2543d9132b62bb7221df4"
BLUEPRINT_REVISION = "c1a1dcb312ee9f691404f127785510453c0d9350"
CHECKPOINT = "layout_analysis/last.pth"
METADATA = "layout_analysis/metadata.json"
LABELS = ["Caption", "Footnote", "Formula", "Text", "Page-footer", "Page-header",
          "Picture", "Title", "Table", "Code Block", "Stamp", "Chemical Structure"]
EXPECTED_METADATA = {
    "schema_version": 1, "model_type": "dfine", "model_version": "1.0.0",
    "architecture": "stage2_b2", "checkpoint": "last.pth", "num_classes": 12,
    "image_size": 960, "color_order": "rgb", "resize_backend": "opencv_lanczos4",
    "postprocess": "model_letterbox", "provides_reading_order": False,
    "id2label": {str(i): name for i, name in enumerate(LABELS)},
}
# HGNetv2-B2 stage configuration (in, mid, out, blocks, downsample, light, kernel, layers).
STAGES = [(32, 32, 96, 1, False, False, 3, 4), (96, 64, 384, 1, True, False, 3, 4),
          (384, 128, 768, 3, True, True, 5, 4), (768, 256, 1536, 1, True, True, 5, 4)]
NUM_QUERIES, NUM_LAYERS, REG_MAX, HIDDEN = 300, 4, 32, 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entries() -> dict:
    manifest = json.loads(Path(__file__).with_name("oxr-source.json").read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("models/oxr-source.json is not the pinned OXR-1.0 manifest")
    layout = manifest["optional_layout_model"]
    if layout.get("path") != CHECKPOINT:
        raise ValueError("unexpected optional_layout_model path in manifest")
    entries = {CHECKPOINT: {"size": layout["size"], "sha256": layout["sha256"]}}
    meta = layout.get("metadata")
    if not meta or meta.get("path") != METADATA:
        raise ValueError("manifest lacks the pinned layout metadata.json entry")
    entries[METADATA] = {"size": meta["size"], "sha256": meta["sha256"]}
    return entries


def resolve_source(model: str) -> Path:
    path = Path(model)
    if path.is_dir():
        return path
    if model != MODEL_ID:
        raise ValueError(f"--model must be {MODEL_ID} or a local directory")
    from huggingface_hub import hf_hub_download
    local = None
    for name in (CHECKPOINT, METADATA):
        local = hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION)
    return Path(local).parent.parent


def authenticate(root: Path) -> dict[str, str]:
    hashes = {}
    for name, expected in manifest_entries().items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise ValueError(f"pinned layout source size/missing-file mismatch: {name}")
        digest = sha256(path)
        if digest != expected["sha256"]:
            raise ValueError(f"pinned layout source SHA256 mismatch: {name}")
        hashes[name] = digest
    metadata = json.loads((root / METADATA).read_text(encoding="utf-8"))
    for key, value in EXPECTED_METADATA.items():
        if metadata.get(key) != value or type(metadata.get(key)) is not type(value):
            raise ValueError(f"unsupported layout metadata field: {key}")
    if metadata.get("checkpoint_sha256") != hashes[CHECKPOINT]:
        raise ValueError("metadata checkpoint_sha256 does not match the checkpoint")
    if metadata.get("checkpoint_size_bytes") != (root / CHECKPOINT).stat().st_size:
        raise ValueError("metadata checkpoint_size_bytes does not match the checkpoint")
    return hashes


def load_state(root: Path):
    import torch
    checkpoint = torch.load(str(root / CHECKPOINT), map_location="cpu", weights_only=True)
    if set(checkpoint) != {"model"}:
        raise ValueError(f"unexpected checkpoint top-level keys: {sorted(checkpoint)}")
    return checkpoint["model"]


class Builder:
    """Collect output tensors and check that every used source key exists."""

    def __init__(self, state, dtype: str):
        self.state, self.dtype, self.out, self.used = state, dtype, {}, set()

    def src(self, key):
        if key not in self.state:
            raise KeyError(f"checkpoint lacks {key}")
        self.used.add(key)
        return self.state[key].detach().clone()

    def put(self, name, tensor, matrix=False):
        import torch
        if len(name.encode("utf-8")) > 63:  # GGML_MAX_NAME = 64 including NUL
            raise ValueError(f"GGUF tensor name too long: {name}")
        if name in self.out:
            raise ValueError(f"duplicate tensor {name}")
        tensor = tensor.detach().to(torch.float32).contiguous()
        if not torch.isfinite(tensor).all() and name != "decoder.anchors":
            raise ValueError(f"non-finite values in {name}")
        if matrix and self.dtype == "f16-as-f32":
            # Diagnostic probe: the F16 recipe's exact weight rounding, stored as F32, so
            # native F32 code isolates F16 weight precision from the F16 compute path.
            tensor = tensor.to(torch.float16).to(torch.float32)
        self.out[name] = (tensor, matrix and self.dtype == "f16")

    # Backbone ConvBNAct: conv weight, FrozenBatchNorm2d scale/shift, optional LAB.
    def conv_bn_act(self, prefix, lab):
        self.put(prefix + ".conv.weight", self.src(prefix + ".conv.weight"), matrix=True)
        w, b = self.src(prefix + ".bn.weight"), self.src(prefix + ".bn.bias")
        rm, rv = self.src(prefix + ".bn.running_mean"), self.src(prefix + ".bn.running_var")
        scale = w * (rv + 1e-5).rsqrt()  # FrozenBatchNorm2d.forward, float32
        self.put(prefix + ".bn.scale", scale)
        self.put(prefix + ".bn.shift", b - rm * scale)
        for extra in ("bn.num_batches_tracked",):
            if prefix + "." + extra in self.state:
                self.used.add(prefix + "." + extra)
        if lab:
            self.put(prefix + ".lab.scale", self.src(prefix + ".lab.scale"))
            self.put(prefix + ".lab.bias", self.src(prefix + ".lab.bias"))
        elif prefix + ".lab.scale" in self.state:
            raise ValueError(f"unexpected LAB on {prefix}")

    def bn_tensors(self, prefix):
        keys = [prefix + s for s in (".weight", ".bias", ".running_mean", ".running_var")]
        vals = [self.src(k) for k in keys]
        if prefix + ".num_batches_tracked" in self.state:
            self.used.add(prefix + ".num_batches_tracked")
        return vals

    # Encoder ConvNormLayer_fuse.convert_to_deploy / VGGBlock._fuse_bn_tensor.
    def fused(self, prefix):
        kernel = self.src(prefix + ".conv.weight")
        gamma, beta, mean, var = self.bn_tensors(prefix + ".norm")
        std = (var + 1e-5).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - mean * gamma / std

    def conv_norm_fuse(self, prefix):
        kernel, bias = self.fused(prefix)
        self.put(prefix + ".fused.weight", kernel, matrix=True)
        self.put(prefix + ".fused.bias", bias)

    def vgg(self, prefix):
        import torch.nn.functional as F
        k3, b3 = self.fused(prefix + ".conv1")
        k1, b1 = self.fused(prefix + ".conv2")
        self.put(prefix + ".fused.weight", k3 + F.pad(k1, [1, 1, 1, 1]), matrix=True)
        self.put(prefix + ".fused.bias", b3 + b1)

    def batch_norm_affine(self, prefix, conv_prefix):
        """Plain eval BatchNorm2d after a bias-free conv (encoder.input_proj)."""
        import torch
        self.put(conv_prefix + ".weight", self.src(conv_prefix + ".weight"), matrix=True)
        gamma, beta, mean, var = self.bn_tensors(prefix)
        invstd = 1.0 / torch.sqrt(var.double() + 1e-5)  # ATen eval: accscalar double
        alpha = (gamma.double() * invstd)
        self.put(prefix + ".scale", alpha.float())
        self.put(prefix + ".shift", (beta.double() - mean.double() * alpha).float())

    def copy(self, key, matrix=False, name=None):
        self.put(name or key, self.src(key), matrix=matrix)

    def linear(self, prefix, matrix=True):
        self.copy(prefix + ".weight", matrix=matrix)
        self.copy(prefix + ".bias")


def sincos_position(w, h, dim=256, temperature=10000.0):
    """HybridEncoder.build_2d_sincos_position_embedding, verbatim."""
    import torch
    grid_w = torch.arange(int(w), dtype=torch.float32)
    grid_h = torch.arange(int(h), dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
    pos_dim = dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature ** omega)
    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]
    return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)


def weighting_function_deploy(reg_max, up, reg_scale):
    """dfine_utils.weighting_function(deploy=True), verbatim."""
    import torch
    upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
    upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
    values = ([-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values
              + [upper_bound2])
    return torch.tensor(values, dtype=up.dtype, device=up.device)


def build(state, dtype: str) -> Builder:
    import torch
    b = Builder(state, dtype)
    # ---- HGNetv2-B2 backbone ----
    for name in ("stem1", "stem2a", "stem2b", "stem3", "stem4"):
        b.conv_bn_act(f"backbone.stem.{name}", lab=True)
    for s, (cin, mid, cout, blocks, down, light, k, layers) in enumerate(STAGES):
        p = f"backbone.stages.{s}"
        if down:
            b.conv_bn_act(p + ".downsample", lab=False)
        for blk in range(blocks):
            q = f"{p}.blocks.{blk}"
            for i in range(layers):
                if light:
                    b.conv_bn_act(f"{q}.layers.{i}.conv1", lab=False)
                    b.conv_bn_act(f"{q}.layers.{i}.conv2", lab=True)
                else:
                    b.conv_bn_act(f"{q}.layers.{i}", lab=True)
            b.conv_bn_act(q + ".aggregation.0", lab=True)
            b.conv_bn_act(q + ".aggregation.1", lab=True)
    # ---- Hybrid encoder ----
    for i in range(3):
        b.batch_norm_affine(f"encoder.input_proj.{i}.norm", f"encoder.input_proj.{i}.conv")
    aifi = "encoder.encoder.0.layers.0"
    b.linear(aifi + ".self_attn.out_proj")
    b.copy(aifi + ".self_attn.in_proj_weight", matrix=True)
    b.copy(aifi + ".self_attn.in_proj_bias")
    b.linear(aifi + ".linear1")
    b.linear(aifi + ".linear2")
    for n in ("norm1", "norm2"):
        b.linear(f"{aifi}.{n}", matrix=False)
    b.put("encoder.pos_embed2", sincos_position(30, 30, HIDDEN))
    for i in range(2):
        b.conv_norm_fuse(f"encoder.lateral_convs.{i}")
        b.conv_norm_fuse(f"encoder.downsample_convs.{i}.0.cv1")
        b.conv_norm_fuse(f"encoder.downsample_convs.{i}.0.cv2")
        for kind in ("fpn_blocks", "pan_blocks"):
            p = f"encoder.{kind}.{i}"
            b.conv_norm_fuse(p + ".cv1")
            b.conv_norm_fuse(p + ".cv4")
            for cv in ("cv2", "cv3"):
                b.conv_norm_fuse(f"{p}.{cv}.0.conv1")
                b.conv_norm_fuse(f"{p}.{cv}.0.conv2")
                for j in range(2):
                    b.vgg(f"{p}.{cv}.0.bottlenecks.{j}")
                b.conv_norm_fuse(f"{p}.{cv}.1")
    # ---- D-FINE decoder ----
    anchors = b.src("decoder.anchors").reshape(-1, 4)
    valid = b.src("decoder.valid_mask").reshape(-1)
    if anchors.shape[0] != 18900 or valid.dtype != torch.bool:
        raise ValueError("unexpected anchors/valid_mask")
    b.put("decoder.anchors", anchors)
    b.put("decoder.valid_mask", valid.to(torch.float32))
    up, reg_scale = b.src("decoder.up"), b.src("decoder.reg_scale")
    for key in ("decoder.decoder.up", "decoder.decoder.reg_scale"):
        if not torch.equal(b.src(key), up if key.endswith("up") else reg_scale):
            raise ValueError(f"{key} differs from its decoder copy")
    b.put("decoder.up", up)
    b.put("decoder.reg_scale", reg_scale)
    b.put("decoder.project", weighting_function_deploy(REG_MAX, up, reg_scale))
    b.linear("decoder.enc_output.proj")
    b.linear("decoder.enc_output.norm", matrix=False)
    b.linear("decoder.enc_score_head")
    for i in range(3):
        b.linear(f"decoder.enc_bbox_head.layers.{i}")
        b.linear(f"decoder.pre_bbox_head.layers.{i}")
    for i in range(2):
        b.linear(f"decoder.query_pos_head.layers.{i}")
    b.linear(f"decoder.dec_score_head.{NUM_LAYERS - 1}")
    for li in range(NUM_LAYERS):
        for i in range(3):
            b.linear(f"decoder.dec_bbox_head.{li}.layers.{i}")
        p = f"decoder.decoder.layers.{li}"
        b.copy(p + ".self_attn.in_proj_weight", matrix=True)
        b.copy(p + ".self_attn.in_proj_bias")
        b.linear(p + ".self_attn.out_proj")
        b.linear(p + ".cross_attn.sampling_offsets")
        b.linear(p + ".cross_attn.attention_weights")
        nps = b.src(p + ".cross_attn.num_points_scale")
        expected = torch.tensor([1 / n for n in (3, 6, 3) for _ in range(n)], dtype=torch.float32)
        # The stored buffer is used verbatim: it carries fp16-rounded values
        # (e.g. 1/6 -> 0.16662598), which the reference loads as well.
        if nps.shape != expected.shape or not torch.allclose(nps, expected, atol=1e-3, rtol=0):
            raise ValueError("unexpected num_points_scale")
        b.put(p + ".cross_attn.num_points_scale", nps)
        b.linear(p + ".gateway.gate")
        b.linear(p + ".linear1")
        b.linear(p + ".linear2")
        for n in ("norm1", "gateway.norm", "norm3"):
            b.linear(f"{p}.{n}", matrix=False)
    lqe = f"decoder.decoder.lqe_layers.{NUM_LAYERS - 1}.reg_conf.layers"
    b.linear(lqe + ".0", matrix=False)
    b.linear(lqe + ".1", matrix=False)
    # Training-only / deploy-discarded parameters, listed so nothing is silently skipped.
    ignored = {k for k in state if k.startswith("decoder.denoising_class_embed")
               or (k.startswith("decoder.dec_score_head.") and not k.startswith(f"decoder.dec_score_head.{NUM_LAYERS - 1}."))
               or (k.startswith("decoder.decoder.lqe_layers.") and not k.startswith(f"decoder.decoder.lqe_layers.{NUM_LAYERS - 1}."))}
    unused = set(state) - b.used - ignored
    if unused:
        raise ValueError(f"unconverted checkpoint tensors: {sorted(unused)[:8]} ({len(unused)})")
    b.ignored = sorted(ignored)
    return b


def write(builder: Builder, output: Path, hashes: dict, dtype: str) -> dict:
    import gguf
    import numpy as np
    writer = gguf.GGUFWriter(str(output), arch="layout")
    meta = {
        "general.name": "oxr-1.0-layout-dfine", "general.license": "OXR-License (Apache-2.0 with additional terms)",
        "general.source": f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/{CHECKPOINT}",
        "layout.arch": "dfine", "layout.model_type": "dfine_hgnetv2_b2",
        "layout.license_gate": "oxr", "layout.source.model_id": MODEL_ID,
        "layout.source.revision": MODEL_REVISION, "layout.blueprint.revision": BLUEPRINT_REVISION,
        "layout.source.file_sha256": json.dumps(hashes, sort_keys=True),
        "layout.preprocess": "opencv_lanczos4_letterbox_fill114_rgb_div255",
        "layout.postprocess": "dfine_focal_topk_letterbox_clip",
        "layout.storage": "f32" if dtype == "f16-as-f32" else dtype,
    }
    for key, value in meta.items():
        writer.add_string(key, value)
    for key, value in {"layout.input_height": 960, "layout.input_width": 960, "layout.num_classes": 12,
                       "layout.num_queries": NUM_QUERIES, "layout.num_decoder_layers": NUM_LAYERS,
                       "layout.reg_max": REG_MAX, "layout.hidden_dim": HIDDEN, "layout.pad_value": 114}.items():
        writer.add_uint32(key, value)
    writer.add_array("layout.labels", LABELS)
    writer.add_array("layout.num_points", [3, 6, 3])
    storage = {}
    for name, (tensor, half) in builder.out.items():
        array = tensor.numpy().astype(np.float16 if half else np.float32)
        writer.add_tensor(name, array)
        storage[name] = {"dtype": "F16" if half else "F32", "shape": list(tensor.shape)}
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return storage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=MODEL_ID, help=f"{MODEL_ID} or a local checkout directory")
    parser.add_argument("--dtype", choices=("f32", "f16", "f16-as-f32"), default="f32",
                        help="f16-as-f32: diagnostic F16-rounded matrices stored as F32")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.check_only and args.output is None:
        parser.error("--output is required unless --check-only")
    root = resolve_source(args.model)
    hashes = authenticate(root)
    builder = build(load_state(root), args.dtype)
    print(f"authenticated {CHECKPOINT} ({hashes[CHECKPOINT]}); {len(builder.out)} output tensors, "
          f"{len(builder.ignored)} deploy-discarded", file=sys.stderr)
    if args.check_only:
        return 0
    receipt_path = Path(str(args.output) + ".json")
    if args.output.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing to replace output or receipt: {args.output}")
    storage = write(builder, args.output, hashes, args.dtype)
    receipt = {
        "schema": "crispembed.oxr-dfine.conversion.v1", "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "blueprint_revision": BLUEPRINT_REVISION, "source_sha256": hashes, "dtype": args.dtype,
        "output": args.output.name, "output_sha256": sha256(args.output), "output_bytes": args.output.stat().st_size,
        "tensors": len(storage), "deploy_discarded": builder.ignored, "storage": storage,
        "quality": "unverified until tests/results/oxr-dfine receipts", "speed": "unverified",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({receipt['output_bytes']} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
