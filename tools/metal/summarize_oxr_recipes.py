#!/usr/bin/env python3
"""Compare OXR recipe runs on one device (tools/metal/oxr_batch_ab.sh, one OUT/<recipe> each)
against a baseline recipe: decoded agreement with the original plus prefill stage cosine/max_abs.

    summarize_oxr_recipes.py OUT_DIR --base f16 [--arm metal] > summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
base = sys.argv[sys.argv.index("--base") + 1] if "--base" in sys.argv else "f16"
arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "metal"
rows = []
for recipe in sorted(p for p in root.iterdir() if p.is_dir()):
    for case in sorted(p for p in recipe.iterdir() if p.is_dir()):
        cmp = case / f"{arm}.cmp.json"
        row = {"recipe": recipe.name, "case": case.name}
        if cmp.exists() and cmp.stat().st_size:
            row.update(json.loads(cmp.read_text()))
        a, b = case / f"{arm}.stages", root / base / case.name / f"{arm}.stages"
        if recipe.name != base and a.is_dir() and b.is_dir():
            st = {}
            for f in sorted(b.glob("*.f32")):
                x = np.fromfile(a / f.name, np.float32).astype(np.float64) if (a / f.name).exists() else None
                y = np.fromfile(f, np.float32).astype(np.float64)
                if x is None or x.shape != y.shape:
                    continue
                st[f.stem] = {"cosine": float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-300)),
                              "max_abs": float(np.abs(x - y).max())}
            row["stages_vs_" + base] = st
        rows.append(row)
json.dump(rows, sys.stdout, indent=1)
print()
