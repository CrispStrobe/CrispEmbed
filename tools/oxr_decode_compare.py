#!/usr/bin/env python3
"""Compare a native test-oxr-run JSON with the original Python reference (python.json).

Reports exact token IDs, first diverging token index, text equality, character
error rate (Levenshtein / reference length), EOS/cap and the native warm timing.
Diagnostic A/B for backend and precision arms; the hosted gate remains
tools/oxr_validate.py compare.

    python tools/oxr_decode_compare.py --reference python.json --native native.json [--label x]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    ref = json.loads(a.reference.read_text())
    nat = json.loads(a.native.read_text())
    rid, nid = ref["token_ids"], nat["token_ids"]
    first = next((i for i, (x, y) in enumerate(zip(rid, nid)) if x != y), None)
    if first is None and len(rid) != len(nid):
        first = min(len(rid), len(nid))
    runs = nat.get("timing", {}).get("runs_ms", [])
    out = {
        "label": a.label, "ids_exact": rid == nid, "first_divergence": first,
        "reference_tokens": len(rid), "native_tokens": len(nid),
        "text_exact": ref["text"] == nat["text"],
        "cer": levenshtein(ref["text"], nat["text"]) / max(1, len(ref["text"])),
        "eos": nat.get("eos"), "capped": nat.get("capped"),
        "runs_ms": runs, "warm_ms_median": statistics.median(runs[1:]) if len(runs) > 1 else (runs[0] if runs else None),
        "load_ms": nat.get("timing", {}).get("load_ms"),
        "runs_identical": all(r.get("token_ids") == nid for r in nat.get("runs", [])),
    }
    print(json.dumps(out))
    return 0 if out["ids_exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
