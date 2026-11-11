#!/usr/bin/env python3
"""Render the readiness kernel for one device into OUT_DIR, then push OUT_DIR with kpush.py.

    python3 tools/kaggle/ocr-gpu-readiness/render.py <full-sha> gpu|cpu OUT_DIR [engine,engine]
`cpu` gets its own kernel id (suffix -cpu) and enable_gpu=false; `gpu` keeps the committed metadata.
"""
import json, re, sys
from pathlib import Path

sha, device, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
if not re.fullmatch(r"[0-9a-f]{40}", sha) or device not in ("cpu", "gpu"):
    sys.exit(__doc__)
here = Path(__file__).resolve().parent
out.mkdir(parents=True, exist_ok=True)
(out / "kernel.py").write_text((here / "kernel.py").read_text().replace("__SHA__", sha).replace("__DEVICE__", device).replace("__ENGINES__", sys.argv[4] if len(sys.argv) > 4 else "__ENGINES__"))
meta = json.loads((here / "kernel-metadata.json").read_text())
if device == "cpu":
    meta.update(id=meta["id"] + "-cpu", title=meta["title"] + "-cpu", enable_gpu=False)
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
