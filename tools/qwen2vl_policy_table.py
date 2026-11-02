#!/usr/bin/env python3
"""Markdown table of precision-policy scores from qwen2vl cosine/receipt JSONs.

    python tools/qwen2vl_policy_table.py teleocr DIR [DIR ...]
    python tools/qwen2vl_policy_table.py mineru DIR [DIR ...]

Each DIR holds <precision>-<device>.json receipts (free-running IDs) and
<precision>-<device>.cosine.json reports (teacher-forced, all positions).
Bars: docs/precision-acceptance.md.
"""

import json
import sys
from pathlib import Path

BARS = {"f32": 0.99, "f16": 0.98, "q8_0": 0.97}


def bar_for(precision):
    return BARS.get(precision, 0.94 if precision.startswith("q4") else None)


def main():
    model, dirs = sys.argv[1], sys.argv[2:]
    rows = []
    for d in map(Path, dirs):
        for cos in sorted(d.glob("*.cosine.json")):
            stem = cos.name[: -len(".cosine.json")]
            receipt = d / f"{stem}.json"
            if not receipt.exists():
                continue
            r = json.loads(receipt.read_text())
            c = json.loads(cos.read_text())
            if r.get("model") != model:
                continue
            p, dev = r["precision"], r["device"]
            bar = bar_for(p)
            fo = c.get("final_output_min_cosine")
            exact = sum(1 for x in r["cases"] if x.get("ids_exact_all_runs"))
            div = ", ".join(f"{x['id']}@{x['first_divergence']}" for x in r["cases"] if not x.get("ids_exact_all_runs"))
            worst_case = min(c["cases"].items(), key=lambda kv: kv[1].get("final_output_min_cosine") or 1)[0]
            ws = c.get("worst_stage") or {}
            rows.append((p, dev, fo, bar, worst_case, f"{exact}/{len(r['cases'])}", div or "-",
                         f"{ws.get('case', '')} {ws.get('stage', '')} {ws.get('cosine', float('nan')):.4f}",
                         c.get("final_output_complete")))
    print("| Precision | Device | Final-output cosine (worst case) | Bar | Pass | Decoded exact | First divergence | Worst stage |")
    print("|---|---|---|---|---|---|---|---|")
    for p, dev, fo, bar, wc, ex, div, ws, complete in sorted(rows):
        ok = fo is not None and bar is not None and fo > bar and complete
        print(f"| {p} | {dev} | {fo:.6f} ({wc}) | > {bar} | {'yes' if ok else 'no'} | {ex} | {div} | {ws} |")


if __name__ == "__main__":
    main()
