#!/usr/bin/env python3
"""Aggregate the committed OXR D-FINE hosted lane into the publication receipt.

    python tools/oxr_dfine_receipts.py

Writes tests/results/oxr-dfine/hosted-parity.json in the publisher's `layout`
receipt kind (tools/ci-heavy/ocr_gguf_publish.py): per page, the F32 arm is
`regions_exact` only when OXR's own client selection (score >= 0.5, page clip,
per-class NMS at IoU 0.6) yields the same labels in the same order as the
original PyTorch service, with the recorded box and score differences.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/results/oxr-dfine"
LANE = "hosted-41251744740"
PAGES = ["receipt", "german", "scan", "table", "japanese"]
ARMS = {"f32": "diff-service"}  # native F32 against the shipped (cv2 default) service


def main():
    lane = OUT / LANE
    summary = json.loads((lane / "summary.json").read_text())
    commit = (lane / "provenance.txt").read_text().split()[0].split("=", 1)[1]
    cases = []
    for page in PAGES:
        if page not in summary:
            continue
        arms = {}
        for name, key in ARMS.items():
            sel = summary[page][key]["oxr_selection"]
            arms[name] = {"exit": 0, "regions_exact": sel["pass"] is True and sel["same_labels_in_order"] is True
                          and sel["native_count"] == sel["reference_count"],
                          "native_count": sel["native_count"], "reference_count": sel["reference_count"],
                          "box_max_abs_px": sel["box_max_abs_px"], "score_max_abs": sel["score_max_abs"],
                          "source_arm": key}
        cases.append({"id": page, "reference_regions": summary[page][ARMS["f32"]]["oxr_selection"]["reference_count"],
                      "arms": arms})
    receipt = {"schema": "crispembed.oxr-dfine.native-parity.v1", "kind": "layout", "commit": commit,
               "reference_revision": "c708407f9d2b4df658f2543d9132b62bb7221df4",
               "reference_versions": {"lane": LANE}, "max_new_tokens": 0,
               "completed": [c["id"] for c in cases] == PAGES,
               "summary_sha256": hashlib.sha256((lane / "summary.json").read_bytes()).hexdigest(), "cases": cases}
    (OUT / "hosted-parity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for c in cases:
        print(c["id"], c["reference_regions"], {a: v["regions_exact"] for a, v in c["arms"].items()})
    print("completed:", receipt["completed"])


if __name__ == "__main__":
    main()
