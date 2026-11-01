#!/usr/bin/env python3
"""Summarize qwen2vl_device_ab.py receipts (+ optional stage-cosine reports).

    python tools/qwen2vl_device_summary.py DIR [--cosine-dir DIR] --out DIR/summary.json

Prints one Markdown row per receipt: precision, device, label, exact cases,
first divergences, summed warm/cold milliseconds and the worst final-state
cosine against the reference when a cosine report with the same stem exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--cosine-dir", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(Path(a.dir).glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("schema") != "crispembed.qwen2vl.device-ab.v1":
            continue
        cases = d["cases"]
        exact = [c["id"] for c in cases if c.get("ids_exact_all_runs")]
        div = {c["id"]: c.get("first_divergence") for c in cases if not c.get("ids_exact_all_runs")}
        warm = sum(c.get("warm_ms_min", 0) for c in cases)
        cold = sum(c.get("cold_ms", 0) for c in cases)
        rss = max((c.get("max_rss_mib", 0) for c in cases), default=0)
        row = {"receipt": p.name, "precision": d["precision"], "device": d["device"],
               "label": p.stem[len(f"{d['precision']}-{d['device']}-"):]
               if p.stem.startswith(f"{d['precision']}-{d['device']}-") else "",
               "extra_env": d.get("extra_env", {}), "cases": len(cases), "exact": len(exact),
               "divergences": div, "consistent": all(c.get("runs_consistent") for c in cases),
               "warm_ms_sum": warm, "cold_ms_sum": cold, "max_rss_mib": rss,
               "per_case_warm_ms": {c["id"]: c.get("warm_ms_min") for c in cases}}
        if a.cosine_dir:
            cp = Path(a.cosine_dir) / f"{p.stem}.cosine.json"
            if cp.exists():
                cd = json.loads(cp.read_text())
                row["min_final_cosine"] = cd["min_cosine_final"]
                row["cosine_bar_pass"] = cd["all_pass_bar"]
                row["cosine_reference"] = cd["reference"]
        rows.append(row)
    Path(a.out).write_text(json.dumps({"schema": "crispembed.qwen2vl.device-summary.v1", "rows": rows},
                                      indent=1) + "\n")
    print("| precision | device | variant | exact | first divergence | warm ms (sum) | cold ms (sum) | peak RSS MiB | min final cosine vs F32 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        div = ", ".join(f"{k}@{v}" for k, v in r["divergences"].items()) or "-"
        cos = r.get("min_final_cosine")
        print(f"| {r['precision']} | {r['device']} | {r['label'] or 'default'} | {r['exact']}/{r['cases']} | {div} | "
              f"{r['warm_ms_sum']:.0f} | {r['cold_ms_sum']:.0f} | {r['max_rss_mib']:.0f} | "
              f"{'' if cos is None else f'{cos:.6f}'} |")


if __name__ == "__main__":
    main()
