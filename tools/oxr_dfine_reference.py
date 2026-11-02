#!/usr/bin/env python3
"""Original-PyTorch reference for OXR-1.0's D-FINE layout detector.

Runs the pinned upstream service code (OXR blueprint
services/layout_analyze/dfine_la) on the pinned checkpoint with the service's
own decode -> preprocess -> DfineWithPostProcessor path, and records:

* every intermediate stage the native port exposes (GGUF of float32 tensors,
  names shared with tests/oxr_dfine_diff.cpp), captured with forward hooks and
  wrappers that do not change the arithmetic;
* the exact /inference JSON body (``format_predictions`` with no confidence
  policy, i.e. what OXR's layout client receives);
* CPU timing of preprocessing and model forward (FP32, the service's CPU mode).

OpenCV's IPP Lanczos path (x86 wheels) differs from OpenCV's generic Lanczos4
kernel that the native port reproduces. The tool records both preprocessed
inputs; ``--resize generic`` disables IPP for the whole run (cv2.ipp), and
``--resize service`` keeps the shipped default.

  python tools/oxr_dfine_reference.py --blueprint OXR --checkpoint last.pth \
      --image page.png --dump ref.gguf --output ref.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

BLUEPRINT_REVISION = "c1a1dcb312ee9f691404f127785510453c0d9350"
CHECKPOINT_SHA256 = "dff78ccf18439955362ca44a69eb28fb0b40222a7d278860589682de0176dc17"


def service_digest(package: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(package.rglob("*.py")):
        digest.update(file.relative_to(package).as_posix().encode() + b"\0" + file.read_bytes() + b"\0")
    return digest.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True, type=Path, help="OXR checkout at the pinned revision")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path, help="lossless RGB PNG (decode JPEGs once with PIL)")
    ap.add_argument("--dump", type=Path, help="stage GGUF output")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--resize", choices=("service", "generic"), default="service")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--float64", action="store_true",
                    help="diagnostic: run the same model in float64 as an exact-arithmetic yardstick")
    args = ap.parse_args(argv)

    import cv2
    import numpy as np
    import torch

    service_root = args.blueprint / "services" / "layout_analyze"
    package = service_root / "dfine_la"
    if not (package / "model" / "decoder.py").is_file():
        raise SystemExit("blueprint lacks services/layout_analyze/dfine_la")
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint SHA256 mismatch: {digest}")
    sys.path.insert(0, str(service_root))
    from dfine_la import preprocess as prep
    from dfine_la.confidence import format_predictions
    from dfine_la.model import decoder as dec_mod
    from dfine_la.model import load_model

    torch.set_num_threads(args.threads)
    ipp_default = bool(cv2.ipp.useIPP())
    payload = args.image.read_bytes()

    def preprocess(use_ipp: bool):
        cv2.ipp.setUseIPP(use_ipp)
        try:
            images = prep.decode_images([payload])
            return images, prep.preprocess_images(images, 960)
        finally:
            cv2.ipp.setUseIPP(ipp_default)

    images, (batch_service, sizes) = preprocess(ipp_default)
    _, (batch_generic, _) = preprocess(False)
    ipp_input_diff = float((batch_service - batch_generic).abs().max())
    use_ipp = ipp_default and args.resize == "service"
    batch = batch_service if use_ipp else batch_generic

    model = load_model(args.checkpoint, torch.device("cpu"), num_classes=12, image_size=960)
    if args.float64:
        model = model.double()
        batch = batch.double()
    dfine = model.model
    stages: dict[str, np.ndarray] = {}

    def keep(name, tensor):
        stages[name] = tensor.detach().to(torch.float32).cpu().numpy().copy()

    hooks = []

    def hook(module, name, fn=None):
        hooks.append(module.register_forward_hook(
            lambda _m, _i, out: keep(name, fn(out) if fn else out)))

    bb, enc, dt = dfine.backbone, dfine.encoder, dfine.decoder
    hook(bb.stem, "bb.stem")
    for i, stage in enumerate(bb.stages):
        hook(stage, f"bb.stage{i}")
    for i, proj in enumerate(enc.input_proj):
        hook(proj, f"enc.proj{i}")
    hook(enc.encoder[0], "enc.aifi")
    for i in range(2):
        hook(enc.lateral_convs[i], f"enc.lat{i}")
        hook(enc.fpn_blocks[i], f"enc.fpn{i}")
        hook(enc.downsample_convs[i], f"enc.down{i}")
        hook(enc.pan_blocks[i], f"enc.pan{i}")
    hook(dt.enc_output, "dec.output_memory")
    hook(dt.enc_score_head, "dec.enc_logits")
    hook(dt.enc_bbox_head, "dec.enc_bbox_delta")
    layer_calls = {"qpos": 0}

    def qpos_hook(_m, _i, out):
        keep(f"dec.{layer_calls['qpos']}.qpos", out.clamp(min=-10, max=10))
        layer_calls["qpos"] += 1
    hooks.append(dt.query_pos_head.register_forward_hook(qpos_hook))
    for i, layer in enumerate(dt.decoder.layers):
        hook(layer.norm1, f"dec.{i}.self")
        hook(layer.cross_attn, f"dec.{i}.cross")
        hook(layer.gateway, f"dec.{i}.gate")
        hook(layer, f"dec.{i}.out")
    hook(dt.pre_bbox_head, "dec.pre_bbox_delta")
    hook(dt.decoder.lqe_layers[-1], "dec.logits")

    original_select = dt._select_topk

    def select_topk(memory, logits, anchors, topk):
        out = original_select(memory, logits, anchors, topk)
        _, ind = torch.topk(logits.max(-1).values, topk, dim=-1)
        keep("dec.topk_index", ind.to(torch.float32))
        keep("dec.topk_anchor", out[2])
        return out
    dt._select_topk = select_topk
    bbox_calls = {"n": 0}
    original_d2b = dec_mod.distance2bbox
    original_integral = dt.integral.forward

    def integral(x, project):
        keep(f"dec.{bbox_calls['n']}.corners", x)
        keep("dec.project", project)
        return original_integral(x, project)

    def distance2bbox(points, distance, reg_scale):
        out = original_d2b(points, distance, reg_scale)
        if bbox_calls["n"] == 0:
            keep("dec.ref_initial", points)
        keep(f"dec.{bbox_calls['n']}.bbox", out)
        bbox_calls["n"] += 1
        return out

    original_decoder_forward = dt.decoder.forward

    def decoder_forward(target, ref_points_unact, memory, *a, **k):
        keep("dec.memory", memory)
        keep("dec.content", target)
        keep("dec.ref_unact", ref_points_unact)
        return original_decoder_forward(target, ref_points_unact, memory, *a, **k)

    def run(record: bool):
        layer_calls["qpos"] = 0
        bbox_calls["n"] = 0
        if record:
            dt.integral.forward = integral
            dec_mod.distance2bbox = distance2bbox
            dt.decoder.forward = decoder_forward
        try:
            with torch.no_grad():
                return model(batch, sizes)
        finally:
            dt.integral.forward = original_integral
            dec_mod.distance2bbox = original_d2b
            dt.decoder.forward = original_decoder_forward

    labels, boxes, scores = run(True)
    for h in hooks:
        h.remove()
    dt._select_topk = original_select
    keep("input", batch)
    outputs = tuple(v.cpu().detach().numpy() for v in (labels, boxes, scores))
    keep("post.labels", labels.to(torch.float32))
    keep("post.boxes", boxes)
    keep("post.scores", scores)
    body = format_predictions(outputs)

    # Timing: preprocessing (decode + resize/pad + tensor) and model forward, separately.
    pre_ms, fwd_ms = [], []
    for _ in range(max(1, args.repeat)):
        t0 = time.perf_counter()
        preprocess(use_ipp)
        t1 = time.perf_counter()
        with torch.no_grad():
            again = model(batch, sizes)
        t2 = time.perf_counter()
        pre_ms.append((t1 - t0) * 1e3)
        fwd_ms.append((t2 - t1) * 1e3)
        if not all(torch.equal(a, b) for a, b in zip(again, (labels, boxes, scores))):
            raise SystemExit("reference forward is not deterministic across repeats")

    if args.dump:
        import gguf
        writer = gguf.GGUFWriter(str(args.dump), arch="oxr-dfine-reference")
        for name, array in sorted(stages.items()):
            writer.add_tensor(name, np.ascontiguousarray(array, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    result = {
        "schema": "crispembed.oxr-dfine.reference.v1",
        "blueprint_revision": BLUEPRINT_REVISION, "service_source_sha256": service_digest(package),
        "checkpoint_sha256": digest, "image": args.image.name,
        "image_sha256": hashlib.sha256(payload).hexdigest(),
        "image_width": int(images[0].shape[1]), "image_height": int(images[0].shape[0]),
        "resize": "opencv_ipp_lanczos" if use_ipp else "opencv_generic_lanczos4",
        "ipp_available": ipp_default, "ipp_vs_generic_input_max_abs": ipp_input_diff,
        "versions": {"torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__},
        "threads": args.threads, "precision": "fp64-diagnostic" if args.float64 else "fp32",
        "preprocess_ms": pre_ms, "forward_ms": fwd_ms,
        "preprocess_ms_median": statistics.median(pre_ms), "forward_ms_median": statistics.median(fwd_ms),
        "stages": {k: list(v.shape) for k, v in sorted(stages.items())},
        "response": body,
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"{args.image.name}: {len(body['data']['labels'][0])} detections, "
          f"forward {result['forward_ms_median']:.1f} ms, ipp diff {ipp_input_diff:.4g}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
