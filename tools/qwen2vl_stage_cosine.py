#!/usr/bin/env python3
"""Compare CRISPEMBED_QWEN2VL_STAGE_DUMP directories against an F32 reference.

    python tools/qwen2vl_stage_cosine.py --ref DUMP_F32 --cand DUMP_X \
        --ref-receipt f32-cpu.json --cand-receipt x-metal.json --width 896 --out cos.json

For each case subdirectory, compares the first call (c0_*) of:
  vis_merger_output, prefill_final_norm (row width = LLM hidden size),
  prefill_logits_last, and decode<k>_logits / decode<k>_final_norm.
Decode step k is compared only while the candidate's generated IDs equal the
reference's up to and including step k-1 (same inputs); later steps are
reported as not comparable rather than scored. Reports global cosine, worst
per-row cosine, max absolute error and the reference magnitude, plus the
argmax agreement and top-2 logit margin for logits.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load(path: Path):
    return np.fromfile(path, dtype=np.float32) if path.exists() else None


def stats(ref, cand, width=None):
    out = {"elements": int(ref.size)}
    if cand is None or cand.size != ref.size:
        out["error"] = "missing or shape mismatch"
        return out
    r = ref.astype(np.float64)
    c = cand.astype(np.float64)
    out["finite"] = bool(np.isfinite(c).all())
    denom = np.linalg.norm(r) * np.linalg.norm(c)
    out["cosine"] = float(r @ c / denom) if denom else 0.0
    out["max_abs"] = float(np.max(np.abs(r - c)))
    out["ref_max_abs"] = float(np.max(np.abs(r)))
    if width and ref.size % width == 0 and ref.size > width:
        rr = r.reshape(-1, width)
        cc = c.reshape(-1, width)
        num = (rr * cc).sum(1)
        den = np.linalg.norm(rr, axis=1) * np.linalg.norm(cc, axis=1)
        cos = np.where(den > 0, num / np.where(den > 0, den, 1), 1.0)
        out["rows"] = int(rr.shape[0])
        out["min_row_cosine"] = float(cos.min())
        out["worst_row"] = int(cos.argmin())
    return out


def logit_stats(ref, cand):
    out = stats(ref, cand)
    if "error" in out:
        return out
    top_ref = np.argsort(ref)[-2:]
    out["argmax_ref"] = int(top_ref[-1])
    out["argmax_cand"] = int(np.argmax(cand))
    out["argmax_equal"] = out["argmax_ref"] == out["argmax_cand"]
    out["ref_top2_margin"] = float(ref[top_ref[-1]] - ref[top_ref[-2]])
    return out


def ids_of(receipt, case):
    for c in receipt["cases"]:
        if c["id"] == case and c.get("runs"):
            return c["runs"][0]["ids"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--ref-receipt", required=True)
    ap.add_argument("--cand-receipt", required=True)
    ap.add_argument("--width", type=int, required=True, help="LLM hidden size")
    ap.add_argument("--vision-width", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bar", type=float, default=0.94)
    a = ap.parse_args()
    ref_receipt = json.loads(Path(a.ref_receipt).read_text())
    cand_receipt = json.loads(Path(a.cand_receipt).read_text())
    report = {"schema": "crispembed.qwen2vl.stage-cosine.v1", "reference": ref_receipt.get("precision") + "-" +
              ref_receipt.get("device"), "candidate": cand_receipt.get("precision") + "-" + cand_receipt.get("device"),
              "bar": a.bar, "cases": {}}
    worst = 1.0
    for case_dir in sorted(Path(a.ref).iterdir()):
        if not case_dir.is_dir():
            continue
        case = case_dir.name
        cdir = Path(a.cand) / case
        rids, cids = ids_of(ref_receipt, case), ids_of(cand_receipt, case)
        entry = {}
        v = load(case_dir / "c0_vis_merger_output.f32")
        if v is not None:
            entry["vis_merger_output"] = stats(v, load(cdir / "c0_vis_merger_output.f32"), a.width)
        h = load(case_dir / "c0_prefill_final_norm.f32")
        if h is not None:
            entry["prefill_final_norm"] = stats(h, load(cdir / "c0_prefill_final_norm.f32"), a.width)
        l = load(case_dir / "c0_prefill_logits_last.f32")
        if l is not None:
            entry["prefill_logits_last"] = logit_stats(l, load(cdir / "c0_prefill_logits_last.f32"))
        steps = []
        for f in sorted(case_dir.glob("c0_decode*_logits.f32"), key=lambda p: int(re.findall(r"\d+", p.name)[1])):
            k = int(re.findall(r"\d+", f.name)[1])
            same_inputs = rids is not None and cids is not None and rids[:k] == cids[:k]
            s = {"step": k, "same_inputs": same_inputs}
            if same_inputs:
                s["logits"] = logit_stats(load(f), load(cdir / f.name))
                fn = case_dir / f.name.replace("_logits", "_final_norm")
                if fn.exists():
                    s["final_norm"] = stats(load(fn), load(cdir / fn.name))
            steps.append(s)
        entry["decode"] = steps
        cos = []
        for key in ("prefill_final_norm", "prefill_logits_last"):
            if key in entry and "cosine" in entry[key]:
                cos.append(entry[key]["cosine"])
                cos.append(entry[key].get("min_row_cosine", 1.0))
        for s in steps:
            for key in ("logits", "final_norm"):
                if key in s and "cosine" in s[key]:
                    cos.append(s[key]["cosine"])
        entry["min_cosine_final"] = min(cos) if cos else None
        entry["passes_bar"] = bool(cos) and min(cos) > a.bar
        if cos:
            worst = min(worst, min(cos))
        report["cases"][case] = entry
    report["min_cosine_final"] = worst
    report["all_pass_bar"] = all(e["passes_bar"] for e in report["cases"].values())
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    for case, e in report["cases"].items():
        ph = e.get("prefill_final_norm", {})
        pl = e.get("prefill_logits_last", {})
        print(f"{case}: hidden cos={ph.get('cosine', float('nan')):.6f} minrow={ph.get('min_row_cosine', float('nan')):.6f} "
              f"logits cos={pl.get('cosine', float('nan')):.6f} min_final={e['min_cosine_final']} "
              f"decode_cmp={sum(1 for s in e['decode'] if s['same_inputs'])}")
    print("ALL PASS" if report["all_pass_bar"] else "SOME FAIL", worst)


if __name__ == "__main__":
    main()
