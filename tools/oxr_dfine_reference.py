#!/usr/bin/env python3
"""Freeze original OXR D-FINE layout responses as test fixtures (test-only).

Runs the pinned upstream detector (SoMarkAI/OXR @ c1a1dcb3,
services/layout_analyze/dfine_la) in-process on CPU with the authenticated
checkpoint SoMarkAI/OXR-1.0 @ c708407f layout_analysis/{last.pth,metadata.json}.
Each page is PNG-encoded from BGR exactly as oxr.model.layout_client does,
then decoded/preprocessed/inferred/formatted by the original service code
with no confidence filtering (the document helper applies its own threshold).

    python tools/oxr_dfine_reference.py --oxr-source OXR --model-dir CKPT \
        --out tests/fixtures/oxr-layout name=path.png [...]

Writes <name>.json: {"response": <official batch response>, "provenance": {...}}.
Requires torch (CPU) in addition to the blueprint requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_ID = "SoMarkAI/OXR-1.0"
REVISION = "c708407f9d2b4df658f2543d9132b62bb7221df4"
BLUEPRINT = "c1a1dcb312ee9f691404f127785510453c0d9350"
CHECKPOINT_SHA256 = "dff78ccf18439955362ca44a69eb28fb0b40222a7d278860589682de0176dc17"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oxr-source", required=True, help="checkout of SoMarkAI/OXR at the pinned blueprint revision")
    ap.add_argument("--model-dir", help="directory containing layout_analysis/; downloaded when omitted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("images", nargs="+", help="name=path")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    source = Path(args.oxr_source).resolve()
    sys.path.insert(0, str(source / "services" / "layout_analyze"))
    from dfine_la.config import Settings
    from dfine_la.confidence import format_predictions
    from dfine_la.runtime import ModelRuntime

    model_dir = Path(args.model_dir) if args.model_dir else None
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(MODEL_ID, revision=REVISION,
                                           allow_patterns=["layout_analysis/last.pth", "layout_analysis/metadata.json"]))
    checkpoint = model_dir / "layout_analysis" / "last.pth"
    settings = Settings(model_path=checkpoint, device="cpu", inference_precision="fp32",
                        cuda_graph_enabled=False, parallel_preprocess=False)
    runtime = ModelRuntime(settings)  # validates metadata, size and SHA-256
    if runtime.settings.checkpoint_sha256 != CHECKPOINT_SHA256:
        raise SystemExit("unexpected D-FINE checkpoint")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for item in args.images:
        name, path = item.split("=", 1)
        with Image.open(path) as opened:
            rgb = np.ascontiguousarray(opened.convert("RGB"), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        ok, encoded = cv2.imencode(".png", bgr)  # oxr.model.layout_client.encode_images
        if not ok:
            raise SystemExit(f"cannot encode {path}")
        ready, _, _ = runtime.prepare([encoded.tobytes()])
        outputs, _ = runtime.infer(ready)
        response = format_predictions(outputs, 0.0, {}, {})
        record = {
            "response": response,
            "provenance": {
                "image": os.path.relpath(path), "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                "model": MODEL_ID, "revision": REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
                "blueprint_revision": BLUEPRINT, "torch": torch.__version__, "numpy": np.__version__,
                "opencv": cv2.__version__, "device": "cpu", "precision": "fp32",
                "candidates": len(response["data"]["labels"][0]),
            },
        }
        (out / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
        print(name, record["provenance"]["candidates"], "candidates")


if __name__ == "__main__":
    main()
