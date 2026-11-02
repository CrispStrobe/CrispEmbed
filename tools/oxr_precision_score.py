#!/usr/bin/env python3
"""Score an OXR hosted qualification lane (tools/ci-heavy/oxr_qualify.sh output) under
docs/precision-acceptance.md: per case the final-logit cosine against the original
Python model (test-oxr-diff `llm.logits` cos_min), the worst intermediate-stage cosine,
the final hidden state (llm.norm) cosine, decoded agreement (token IDs, CER/WER) and EOS.

    python tools/oxr_precision_score.py LANE_DIR --bar 0.97 [--output score.json]
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lane", type=Path)
    ap.add_argument("--bar", type=float, required=True)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    cases, worst_final = {}, 1.0
    for d in sorted(p for p in a.lane.iterdir() if (p / "stages-default.json").exists()):
        st = json.loads((d / "stages-default.json").read_text())
        stages = {s["name"]: s for s in st["stages"] if "cos_min" in s}
        worst = min(stages.values(), key=lambda s: s["cos_min"])
        row = {"final_logits_cos": stages["llm.logits"]["cos_min"], "final_hidden_cos": stages["llm.norm"]["cos_min"],
               "worst_stage": [worst["name"], worst["cos_min"]], "first_token_exact": st.get("first_token_exact")}
        nd = d / "native-default.json"
        if nd.exists():
            n = json.loads(nd.read_text())
            row.update(ids_exact=n["checks"].get("token_ids"), text_exact=n["text_metrics"].get("exact"),
                       cer=n["text_metrics"].get("cer"), wer=n["text_metrics"].get("wer"), eos=n["checks"].get("eos"))
        cases[d.name] = row
        worst_final = min(worst_final, row["final_logits_cos"])
    out = {"schema": "crispembed.oxr.precision-score.v1", "lane": a.lane.name, "bar": a.bar,
           "min_final_logits_cos": worst_final, "pass": worst_final > a.bar and len(cases) == 5, "cases": cases}
    text = json.dumps(out, indent=1)
    if a.output:
        a.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
