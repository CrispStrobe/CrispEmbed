#!/usr/bin/env python3
"""G6: interleaved F32-KV vs F16-KV A/B on one fixture (metal).

Mirrors tests/run_deepseek_ocr2_bench.py's interleave mode, but the two arms
are the KV cache dtype (both DS2_FAST_DECODE=1, guard on by default); the ONLY
delta is DS2_KV_F16=1.  Pair 0 is a discarded cold pair; pairs started at
1-min loadavg > max_load are recorded but excluded from the verdict (the box
is shared).  Untrimmed spread is reported alongside medians.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))
from run_deepseek_ocr2_bench import run_once  # noqa: E402

BIN = REPO / "build/crispembed"
MODEL = Path.home() / ".cache/crispembed-local/deepseek-ocr2-q4_k-stacked.gguf"
IMG = REPO / "tests/regression/images/cc0/receipt_historical.png"
OUT = REPO / "tests/results/g6"
PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAX_LOAD = 8.0

base = dict(os.environ, CRISPEMBED_DEEPSEEK_OCR2_BENCH="1", DS2_FAST_DECODE="1")
base.pop("DS2_KV_F16", None)
arms = {"kv_f32": dict(base), "kv_f16": dict(base, DS2_KV_F16="1")}

samples: dict[str, list[dict]] = {k: [] for k in arms}
texts: dict[str, set[str]] = {k: set() for k in arms}
pairs_out, dropped = [], []

for pair in range(PAIRS + 1):  # pair 0 = discarded cold pair
    load1 = float(os.popen("sysctl -n vm.loadavg").read().split()[1])
    rec_pair = {"pair": pair, "load_at_start": load1}
    for arm, env in arms.items():
        text, wall_ms, bench, rc, err = run_once(BIN, MODEL, IMG, env, "metal")
        ok = rc == 0 and bool(text) and "MTL0" in err
        texts[arm].add(text)
        rec = {"pair": pair, "ok": ok, "wall_ms": round(wall_ms, 1), **bench}
        rec_pair[arm] = rec
        if pair > 0 and ok and load1 <= MAX_LOAD:
            samples[arm].append(rec)
        print(f"pair {pair} {arm} ok={ok} decode={bench.get('decode')} "
              f"prefill={bench.get('prefill')} total={bench.get('total')} "
              f"kv={bench.get('kv')} chars={len(text)} load1={load1}", flush=True)
    if pair > 0 and all(rec_pair[a].get("ok") for a in arms):
        d32, d16 = rec_pair["kv_f32"].get("decode"), rec_pair["kv_f16"].get("decode")
        if d32 and d16:
            rec_pair["ratio_f16_over_f32"] = round(d16 / d32, 4)
    if pair > 0 and load1 > MAX_LOAD:
        dropped.append(pair)
        rec_pair["dropped_reason"] = f"load {load1} > {MAX_LOAD}"
    pairs_out.append(rec_pair)


def stats(vals):
    vals = sorted(vals)
    n = len(vals)
    if not n:
        return {}
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    spread = (vals[-1] - vals[0]) / med if med else 0.0
    return {"n": n, "median": round(med, 1), "min": round(vals[0], 1),
            "max": round(vals[-1], 1), "spread_frac": round(spread, 3)}


summary = {arm: {
    "decode_ms": stats([r["decode"] for r in rows if "decode" in r]),
    "prefill_ms": stats([r["prefill"] for r in rows if "prefill" in r]),
    "total_ms": stats([r["total"] for r in rows if "total" in r]),
    "wall_ms": stats([r["wall_ms"] for r in rows]),
    "distinct_texts": len(texts[arm]),
} for arm, rows in samples.items()}
ratios = [p["ratio_f16_over_f32"] for p in pairs_out
          if p["pair"] > 0 and "ratio_f16_over_f32" in p and p["load_at_start"] <= MAX_LOAD]
doc = {"fixture": IMG.name, "pairs": PAIRS, "max_load": MAX_LOAD,
       "gpu_backend": "metal", "samples": samples, "summary": summary,
       "per_pair": pairs_out, "dropped_pairs_high_load": dropped,
       "ratio_f16_over_f32": stats(ratios) if ratios else {},
       "text_identical_across_arms": len(texts["kv_f32"] | texts["kv_f16"]) == 1}
(OUT / "interleaved_kv_receipt_historical.json").write_text(json.dumps(doc, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(f"per-pair f16/f32 decode ratios (load-gated): {ratios}")
print(f"dropped for load: {dropped}")
print(f"text identical across arms: {doc['text_identical_across_arms']}")
