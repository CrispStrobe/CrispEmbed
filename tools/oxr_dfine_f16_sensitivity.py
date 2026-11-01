#!/usr/bin/env python3
"""Original-PyTorch F16 weight-rounding sensitivity probe for the OXR D-FINE layout model.

Diagnostic only. Runs the pinned service model (float32 arithmetic) with the
weights of one parameter group rounded to F16 (as the F16 recipe stores them),
and compares the final post-processed regions with the unrounded F32 model.
It separates model sensitivity from native runtime defects: if PyTorch itself
moves a selected box when F16-rounded weights are used, the loss is precision.

    python tools/oxr_dfine_f16_sensitivity.py --blueprint OXR --checkpoint last.pth \
        --image page.png [--image ...] --output sensitivity.json
"""
import argparse
import json
import sys
from pathlib import Path

GROUPS = {
    "all": lambda n: True,
    "backbone": lambda n: n.startswith("backbone."),
    "backbone.stem": lambda n: n.startswith("backbone.stem."),
    "backbone.stages.0": lambda n: n.startswith("backbone.stages.0."),
    "backbone.stages.1": lambda n: n.startswith("backbone.stages.1."),
    "backbone.stages.2": lambda n: n.startswith("backbone.stages.2."),
    "backbone.stages.3": lambda n: n.startswith("backbone.stages.3."),
    "encoder": lambda n: n.startswith("encoder."),
    "encoder.input_proj": lambda n: n.startswith("encoder.input_proj."),
    "encoder.aifi": lambda n: n.startswith("encoder.encoder."),
    "encoder.lateral+down": lambda n: n.startswith(("encoder.lateral_convs.", "encoder.downsample_convs.")),
    "encoder.fpn": lambda n: n.startswith("encoder.fpn_blocks."),
    "encoder.pan": lambda n: n.startswith("encoder.pan_blocks."),
    "encoder.pan.cv4": lambda n: n.startswith("encoder.pan_blocks.") and ".cv4." in n,
    "decoder": lambda n: n.startswith("decoder."),
    "decoder.enc_heads": lambda n: n.startswith(("decoder.enc_output.", "decoder.enc_score_head.", "decoder.enc_bbox_head.")),
    "decoder.layers": lambda n: n.startswith("decoder.decoder."),
    "decoder.box_heads": lambda n: n.startswith(("decoder.dec_bbox_head.", "decoder.pre_bbox_head.", "decoder.query_pos_head.")),
    "none": lambda n: False,
}


def select(labels, boxes, scores, thr=0.5, iou_thr=0.6):
    order = [i for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i)) if scores[i] >= thr]
    kept = []
    for i in order:
        b = boxes[i]
        if b[0] >= b[2] or b[1] >= b[3]:
            continue
        ok = True
        for k in kept:
            c = boxes[k]
            iw = max(0.0, min(b[2], c[2]) - max(b[0], c[0]))
            ih = max(0.0, min(b[3], c[3]) - max(b[1], c[1]))
            inter = iw * ih
            if inter and inter / ((b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - inter) > iou_thr:
                ok = False
                break
        if ok:
            kept.append(i)
    return [(int(labels[i]), [float(v) for v in boxes[i]], float(scores[i])) for i in kept]


def compare(ref, got):
    worst, unmatched = 0.0, 0
    for lab, box, _ in ref:
        cands = [g for g in got if g[0] == lab]
        if not cands:
            unmatched += 1
            continue
        d = min(max(abs(a - b) for a, b in zip(box, g[1])) for g in cands)
        worst = max(worst, d)
    return {"ref_count": len(ref), "count": len(got), "unmatched": unmatched, "max_box_px": worst,
            "count_equal": len(ref) == len(got)}


def cosine(a, b):
    import numpy as np
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path, action="append")
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    import cv2
    import torch
    root = args.blueprint / "services" / "layout_analyze"
    sys.path.insert(0, str(root))
    from dfine_la import preprocess as prep
    from dfine_la.model import load_model
    torch.set_num_threads(4)
    cv2.ipp.setUseIPP(False)
    model = load_model(args.checkpoint, torch.device("cpu"), num_classes=12, image_size=960)
    params = {n: p for n, p in model.model.named_parameters()}
    params.update({n: b for n, b in model.model.named_buffers() if b.dtype == torch.float32 and b.dim() >= 2})
    original = {n: p.detach().clone() for n, p in params.items()}
    matrix = [n for n, p in params.items() if p.dim() >= 2 and n not in ("decoder.anchors", "decoder.valid_mask")]
    batches = []
    for path in args.image:
        images = prep.decode_images([path.read_bytes()])
        batches.append((path.stem, prep.preprocess_images(images, 960)))

    def run():
        out = {}
        with torch.no_grad():
            for name, (batch, sizes) in batches:
                labels, boxes, scores = model(batch, sizes)
                out[name] = (labels[0].tolist(), boxes[0].tolist(), scores[0].tolist())
        return out

    base = run()
    report = {"schema": "crispembed.oxr-dfine.f16-sensitivity.v1", "pages": [n for n, _ in batches], "groups": {}}
    for g in args.groups.split(","):
        names = [n for n in matrix if GROUPS[g](n)]
        with torch.no_grad():
            for n in names:
                params[n].copy_(original[n].half().float())
        res = run()
        with torch.no_grad():
            for n in names:
                params[n].copy_(original[n])
        pages = {}
        for page in res:
            r = compare(select(*base[page]), select(*res[page]))
            r["boxes_cos"] = cosine(base[page][1], res[page][1])
            r["scores_cos"] = cosine(base[page][2], res[page][2])
            pages[page] = r
        report["groups"][g] = {"tensors": len(names), "pages": pages}
        print(g, len(names), {p: (v["max_box_px"], v["unmatched"], v["count_equal"]) for p, v in pages.items()},
              file=sys.stderr, flush=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
