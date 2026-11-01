#!/usr/bin/env python3
"""Summarize tools/metal/oxr_dfine_metal_ab.sh output: one JSON row per page and arm."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for f in sorted(root.glob("*/diff-*.json")):
    try:
        j = json.loads(f.read_text())
    except Exception as e:  # keep failed runs on record
        rows.append({"page": f.parent.name, "arm": f.stem[5:], "error": str(e) or "empty"})
        continue
    st = {s["name"]: s for s in j.get("stages", [])}
    worst = max((s for s in j.get("stages", []) if s.get("budgeted")), key=lambda s: s["rel"], default={})
    sel = j.get("oxr_selection", {})
    rows.append({
        "page": f.parent.name, "arm": f.stem[5:], "pass": j.get("pass"),
        "stage_budget_pass": j.get("stage_budget_pass"), "selection_pass": j.get("selection_pass"),
        "earliest_divergence": j.get("earliest_divergence"),
        "worst_stage": worst.get("name"), "worst_rel": worst.get("rel"),
        "selected": [sel.get("native_count"), sel.get("reference_count")],
        "same_labels": sel.get("same_labels_in_order"), "sel_box_px": sel.get("box_max_abs_px"),
        "sel_score": sel.get("score_max_abs"),
        "post_boxes_cos": st.get("post.boxes", {}).get("cosine"), "post_scores_cos": st.get("post.scores", {}).get("cosine"),
        "timing_ms": j.get("timing_ms"),
    })
json.dump(rows, sys.stdout, indent=1)
print()
