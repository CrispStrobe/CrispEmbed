#!/usr/bin/env python3
"""Per-tensor F16 rounding audit for the OXR D-FINE converter's matrix tensors.

Diagnostic only (no GGUF is written). For every tensor the F16 recipe would store
in F16 it reports the relative Frobenius rounding error, the worst element
relative error, and the fraction of non-zero values in the F16 subnormal range
(|w| < 6.104e-5) or flushed to zero. Output is JSON, sorted by rounding error.

    python tools/oxr_dfine_f16_audit.py --model <layout checkout> --output audit.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_oxr_dfine", HERE.parent / "models" / "convert-oxr-dfine-to-gguf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    conv = load_converter()
    root = conv.resolve_source(args.model)
    conv.authenticate(root)
    b = conv.build(conv.load_state(root), "f16")
    rows = []
    for name, (t, half) in b.out.items():
        if not half:
            continue
        t = t.double()
        h = t.float().half().double()
        nz = t != 0
        mag = t.abs()
        rel = (h - t).norm() / max(t.norm(), 1e-30)
        elem = ((h - t).abs()[nz] / mag[nz]).max().item() if nz.any() else 0.0
        rows.append({
            "name": name, "shape": list(t.shape), "elements": t.numel(),
            "rel_fro": rel.item(), "max_elem_rel": elem,
            "subnormal_frac": ((mag < 6.103515625e-05) & nz).double().mean().item(),
            "flushed_frac": ((h == 0) & nz).double().mean().item(),
            "abs_max": mag.max().item(), "abs_median": mag[nz].median().item() if nz.any() else 0.0,
        })
    rows.sort(key=lambda r: -r["rel_fro"])
    args.output.write_text(json.dumps({"schema": "crispembed.oxr-dfine.f16-audit.v1", "tensors": rows}, indent=1) + "\n")
    for r in rows[:25]:
        print(f'{r["rel_fro"]:.3e} sub={r["subnormal_frac"]:.4f} flush={r["flushed_frac"]:.4f} '
              f'med={r["abs_median"]:.2e} {r["name"]} {r["shape"]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
