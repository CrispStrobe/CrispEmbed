#!/usr/bin/env python3
"""Compare two native OXR stage dumps (tests/oxr_stage_dump.cpp) stage by stage.

Reports max_abs, mean_abs, rel (= max_abs / max|ref|) and cosine per stage and
the earliest stage whose max_abs exceeds --max-abs (default 0.002, the CPU
qualification budget). Diagnostic A/B only; cosine alone is not parity.

    python tools/oxr_stage_compare.py REF_DIR TEST_DIR [--max-abs 0.002] [--output x.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("test", type=Path)
    ap.add_argument("--max-abs", type=float, default=0.002)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    ri = json.loads((a.ref / "index.json").read_text())
    ti = json.loads((a.test / "index.json").read_text())
    rows, earliest = [], None
    for name, shape in ri.items():
        if name == "first_token":
            continue
        if name not in ti:
            rows.append({"name": name, "found": False})
            earliest = earliest or name
            continue
        r = np.fromfile(a.ref / f"{name}.f32", np.float32).astype(np.float64)
        t = np.fromfile(a.test / f"{name}.f32", np.float32).astype(np.float64)
        if r.shape != t.shape:
            rows.append({"name": name, "found": True, "shape_equal": False})
            earliest = earliest or name
            continue
        d = np.abs(r - t)
        cos = float(r @ t / max(np.linalg.norm(r) * np.linalg.norm(t), 1e-300))
        row = {"name": name, "found": True, "shape_equal": True, "max_abs": float(d.max()),
               "mean_abs": float(d.mean()), "ref_max": float(np.abs(r).max()),
               "rel": float(d.max() / max(np.abs(r).max(), 1e-30)), "cosine": cos,
               "within_max_abs": bool(d.max() <= a.max_abs)}
        if name == "llm.logits":
            row["argmax_ref"], row["argmax_test"] = int(r.argmax()), int(t.argmax())
        rows.append(row)
        if not row["within_max_abs"] and earliest is None:
            earliest = name
    out = {"schema": "crispembed.oxr.stage-ab.v1", "ref": str(a.ref), "test": str(a.test), "max_abs_budget": a.max_abs,
           "first_token_ref": ri.get("first_token"), "first_token_test": ti.get("first_token"),
           "earliest_over_budget": earliest, "stages": rows}
    text = json.dumps(out, indent=1)
    if a.output:
        a.output.write_text(text + "\n")
    for r in rows:
        if r.get("found") and r.get("shape_equal"):
            print(f'{r["name"]:22s} max_abs={r["max_abs"]:.3e} rel={r["rel"]:.3e} cos={r["cosine"]:.9f}')
    print("earliest_over_budget:", earliest, "first tokens:", out["first_token_ref"], out["first_token_test"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
