#!/usr/bin/env python3
"""Precision receipt for a qwen2vl Q4_K-class policy, in the publication schema.

    python tools/qwen2vl_q4k_receipt.py OUT.json --dtype teleocr_q4k_v1 \
        --arm cpu-hosted=DIR/teleocr_q4k_v1-cpu.json:DIR/teleocr_q4k_v1-cpu.cosine.json \
        --arm metal-m1=... --arm cuda-t4=...

Each arm pairs a free-running device-A/B receipt (decoded IDs against the committed
original-Python IDs) with the teacher-forced cosine report of the same GGUF
(tools/qwen2vl_stage_cosine.py). Cases carry `ids_exact` and `final_cos_min` (the
minimum final-logit cosine over every generated position), the field the cosine
gate of tools/ci-heavy/ocr_gguf_publish.py reads.
"""
import argparse
import json
from pathlib import Path


def arm(name, dtype, receipt, cosine):
    r = json.loads(Path(receipt).read_text())
    c = json.loads(Path(cosine).read_text())
    cases = []
    for case in r["cases"]:
        e = c["cases"][case["id"]]
        cases.append({"id": case["id"], "ids_exact": bool(case.get("ids_exact_all_runs")),
                      "first_divergence": case.get("first_divergence"),
                      "final_cos_min": e["final_output_min_cosine"],
                      "positions": e["final_output_positions"], "positions_total": e["final_output_positions_total"],
                      "worst_stage": e.get("worst_stage"), "worst_stage_cosine": e.get("worst_stage_cosine")})
    return {"name": name, "dtype": dtype, "device": r["device"], "gguf_sha256": r.get("gguf_sha256"),
            "commit": r.get("commit"), "extra_env": r.get("extra_env", {}),
            "final_output_complete": c.get("final_output_complete"), "cases": cases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reference", required=True, help="what the cosine is measured against")
    ap.add_argument("--arm", action="append", required=True, help="NAME=RECEIPT:COSINE")
    a = ap.parse_args()
    arms = []
    for spec in a.arm:
        name, _, pair = spec.partition("=")
        receipt, _, cosine = pair.partition(":")
        arms.append(arm(name, a.dtype, receipt, cosine))
    out = {"schema": "crispembed.qwen2vl.q4k-precision.v1", "model": a.model, "dtype": a.dtype,
           "metric": "minimum final-logit cosine over every generated position, teacher-forced on the committed "
                     "original-Python IDs (docs/qwen2vl-q4k-recipes.md)",
           "reference": a.reference, "bar": 0.94, "commit": arms[0]["commit"],
           "worst_final_cos": min(c["final_cos_min"] for x in arms for c in x["cases"]), "arms": arms}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(a.out, "worst", out["worst_final_cos"])


if __name__ == "__main__":
    main()
