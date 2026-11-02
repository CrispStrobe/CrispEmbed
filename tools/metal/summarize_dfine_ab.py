#!/usr/bin/env python3
"""Summarize tools/metal/oxr_dfine_metal_ab.sh output: one JSON row per page and arm."""
import json
import math
import sys
from pathlib import Path


def selected_cosines(sel):
    """Owner metric (docs/precision-acceptance.md): final boxes/scores cosine over
    the OXR-selected regions. Regions are paired order-insensitively (same label,
    greedy highest IoU); unmatched regions on either side contribute zero vectors
    (box 0,0,0,0 / score 0) so a missing or extra region lowers the cosine.
    Returns (boxes_cos, scores_cos, matched, native_count, reference_count)."""
    a, b = sel.get("regions") or [], sel.get("reference_regions") or []
    def iou(p, q):
        iw = max(0.0, min(p[2], q[2]) - max(p[0], q[0])); ih = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        inter = iw * ih
        u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
        return inter / u if u > 0 else 0.0
    pairs, used = [], set()
    for i, r in enumerate(b):
        best, bj = 0.0, None
        for j, n in enumerate(a):
            if j in used or n["label"] != r["label"]:
                continue
            v = iou(n["bbox"], r["bbox"])
            if v > best:
                best, bj = v, j
        if bj is not None:
            used.add(bj); pairs.append((a[bj], r))
    na, nb = [], []
    for n, r in pairs:
        na += n["bbox"] + [n["score"]]; nb += r["bbox"] + [r["score"]]
    for j, n in enumerate(a):
        if j not in used: na += n["bbox"] + [n["score"]]; nb += [0.0] * 5
    for r in b:
        if not any(r is q for _, q in pairs): na += [0.0] * 5; nb += r["bbox"] + [r["score"]]
    def cos(u, v):
        d = sum(x * y for x, y in zip(u, v)); n = math.sqrt(sum(x * x for x in u) * sum(y * y for y in v))
        return d / n if n else 1.0
    boxes = cos([x for i, x in enumerate(na) if i % 5 != 4], [x for i, x in enumerate(nb) if i % 5 != 4])
    scores = cos(na[4::5], nb[4::5])
    return boxes, scores, len(pairs), len(a), len(b)

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
    sel_box_cos, sel_score_cos, matched, _, _ = selected_cosines(sel) if sel else (None, None, 0, 0, 0)
    nonpost = [s for s in j.get("stages", []) if not s["name"].startswith("post.") and s.get("found")]
    worst_cos = min(nonpost, key=lambda s: s["cosine"], default={})
    rows.append({
        "page": f.parent.name, "arm": f.stem[5:], "pass": j.get("pass"),
        "stage_budget_pass": j.get("stage_budget_pass"), "selection_pass": j.get("selection_pass"),
        "earliest_divergence": j.get("earliest_divergence"),
        "worst_stage": worst.get("name"), "worst_rel": worst.get("rel"),
        "selected": [sel.get("native_count"), sel.get("reference_count")],
        "same_labels": sel.get("same_labels_in_order"), "sel_box_px": sel.get("box_max_abs_px"),
        "sel_score": sel.get("score_max_abs"),
        "selected_boxes_cos": sel_box_cos, "selected_scores_cos": sel_score_cos, "selected_matched": matched,
        "worst_stage_cos": [worst_cos.get("name"), worst_cos.get("cosine")],
        "query_alignment": j.get("query_alignment_applied"),
        "post_boxes_cos": st.get("post.boxes", {}).get("cosine"), "post_scores_cos": st.get("post.scores", {}).get("cosine"),
        "timing_ms": j.get("timing_ms"),
    })
json.dump(rows, sys.stdout, indent=1)
print()
