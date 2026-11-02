#!/usr/bin/env python3
"""Summarize tools/metal/oxr_batch_ab.sh output.

    summarize_oxr_ab.py OUT_DIR [--base ARM] > summary.json

Per case and arm: decoded comparison with the original reference (.cmp.json)
and, when both arms have prefill stage dumps, per-stage max_abs / cosine of the
arm against --base (default cpu)."""
import json
import sys
from pathlib import Path

import numpy as np


def stages(d):
    idx = d / "index.json"
    if not idx.exists():
        return {}
    return {k: v for k, v in json.loads(idx.read_text()).items() if k != "first_token"}


def main():
    root = Path(sys.argv[1])
    base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "cpu"
    rows = []
    for case in sorted(p for p in root.iterdir() if p.is_dir()):
        for cmp in sorted(case.glob("*.cmp.json")):
            arm = cmp.name[: -len(".cmp.json")]
            try:
                row = json.loads(cmp.read_text())
            except Exception:
                row = {"label": f"{case.name}/{arm}", "error": "no comparison"}
            row.update(case=case.name, arm=arm)
            a, b = case / f"{arm}.stages", case / f"{base}.stages"
            if arm != base and a.is_dir() and b.is_dir():
                diffs = {}
                for name in stages(b):
                    fa, fb = a / f"{name}.f32", b / f"{name}.f32"
                    if not fa.exists():
                        continue
                    x = np.fromfile(fa, np.float32).astype(np.float64)
                    y = np.fromfile(fb, np.float32).astype(np.float64)
                    if x.shape != y.shape:
                        diffs[name] = {"shape_equal": False}
                        continue
                    diffs[name] = {"max_abs": float(np.abs(x - y).max()),
                                   "cosine": float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-300))}
                    if name == "llm.logits":
                        diffs[name]["argmax_equal"] = bool(x.argmax() == y.argmax())
                row["stages_vs_" + base] = diffs
            rows.append(row)
    json.dump(rows, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
