#!/usr/bin/env python3
"""G6: byte-compare each F16-KV arm's transcripts against the f1 F32 baseline.

Baselines are the guard-on persistent arms in tests/results/f1/ (re-proven
byte-reproducible on current main in tests/results/g2/SUMMARY.md Gate A).
Also compares decode-time medians from runs.json (shared-box caveat applies).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
F1 = REPO.parents[1] if False else None  # baselines live in the MAIN checkout
BASE = Path("/Users/christianstrobele/code/CrispEmbed/tests/results/f1")
G6 = REPO / "tests/results/g6"

PAIRS = [
    ("m-kvf16-cc0", "m-guard-persist-cc0"),
    ("m-kvf16-synth", "m-guard-persist-synth"),
    ("c-kvf16-cc0", "c-guard-persist-cc0"),
    ("c-kvf16-synth", "c-guard-persist-synth"),
]


def med(vals):
    vals = sorted(vals)
    n = len(vals)
    return None if not n else (vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2)


report = {}
for arm, base in PAIRS:
    a_dir, b_dir = G6 / arm, BASE / base
    a_txt = sorted(p.name for p in a_dir.glob("*.txt"))
    b_txt = sorted(p.name for p in b_dir.glob("*.txt"))
    common = [n for n in a_txt if n in set(b_txt)]
    identical, differing = [], []
    for n in common:
        if (a_dir / n).read_bytes() == (b_dir / n).read_bytes():
            identical.append(n)
        else:
            differing.append(n)
    a_runs = json.loads((a_dir / "runs.json").read_text())
    b_runs = json.loads((b_dir / "runs.json").read_text())
    a_ok = sum(1 for r in a_runs["rows"] if r["ok"])
    b_ok = sum(1 for r in b_runs["rows"] if r["ok"])
    a_dec = [r["decode"] for r in a_runs["rows"] if r.get("ok") and "decode" in r]
    b_dec = [r["decode"] for r in b_runs["rows"] if r.get("ok") and "decode" in r]
    kv_vals = sorted({r.get("kv") for r in a_runs["rows"]})
    report[arm] = {
        "baseline": base,
        "files_arm": len(a_txt), "files_baseline": len(b_txt),
        "only_in_arm": sorted(set(a_txt) - set(b_txt)),
        "only_in_baseline": sorted(set(b_txt) - set(a_txt)),
        "identical": len(identical), "differing": differing,
        "ok_arm": f"{a_ok}/{len(a_runs['rows'])}",
        "ok_baseline": f"{b_ok}/{len(b_runs['rows'])}",
        "kv_field_arm": kv_vals,
        "decode_median_ms_arm": round(med(a_dec), 1) if a_dec else None,
        "decode_median_ms_baseline": round(med(b_dec), 1) if b_dec else None,
    }
    print(f"{arm:16s} vs {base:24s} identical={len(identical)}/{len(common)} "
          f"differing={differing} kv={kv_vals} "
          f"decode_med arm={report[arm]['decode_median_ms_arm']} "
          f"base={report[arm]['decode_median_ms_baseline']}")

(G6 / "compare_vs_f1.json").write_text(json.dumps(report, indent=2) + "\n")
