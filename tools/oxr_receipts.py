#!/usr/bin/env python3
"""Aggregate committed OXR-1.0 hosted lanes into the publication receipts.

    python tools/oxr_receipts.py

Reads the lane directories listed below (tools/ci-heavy/oxr_qualify.sh output,
committed under tests/results/oxr/) and writes, in the shared schema that
tools/ci-heavy/ocr_gguf_publish.py checks:

  tests/results/oxr/hosted-parity.json   F32 arms (default + reused decode graph)
  tests/results/oxr/precision.json       one arm per reduced precision

Nothing is relaxed: a case's arm is `ids_exact` only when the native token IDs
equal the original Python model's, and the parity receipt is `completed` only
when every protocol case is present from one source commit.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/results/oxr"
CASES = ["formula", "table", "japanese", "receipt", "german"]
F32_LANES = ["hosted-41251971237", "hosted-41252889926"]
PRECISION_LANES = {"f16": ["hosted-41255835479", "hosted-41255837490"]}
ARMS = {"f32": "native-default.json", "f32-reuse": "native-reuse.json"}


def source(lane):
    return (OUT / lane / "provenance.txt").read_text().split()[0].split("=", 1)[1]


def case_dirs(lanes):
    found = {}
    for lane in lanes:
        for d in sorted((OUT / lane).iterdir()):
            if (d / "python.json").is_file():
                found[d.name] = (lane, d)
    return found


def arm(d, name, ref):
    path = d / name
    r = json.loads(path.read_text())
    native = r.get("native", {})
    ids = native.get("token_ids", [])
    return {"exit": 0 if r.get("request_parity_pass") else 1, "ids_exact": ids == ref["token_ids"], "ids": ids,
            "text_exact": native.get("text") == ref["text"], "eos": native.get("eos"),
            "native_over_python": r.get("native_over_python"), "speed_pass": r.get("speed_pass"),
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    f32 = case_dirs(F32_LANES)
    commits = {source(lane) for lane in F32_LANES}
    if len(commits) != 1:
        raise SystemExit(f"F32 lanes come from different source commits: {sorted(commits)}")
    cases, ref0 = [], None
    for cid in CASES:
        if cid not in f32:
            continue
        lane, d = f32[cid]
        ref = json.loads((d / "python.json").read_text())
        ref0 = ref0 or ref
        cases.append({"id": cid, "lane": lane, "reference_ids": ref["token_ids"], "reference_text": ref["text"],
                      "reference_eos": ref["eos"], "reference_capped": ref["capped"],
                      "arms": {a: arm(d, f, ref) for a, f in ARMS.items()}})
    parity = {"schema": "crispembed.oxr.native-parity.v1", "commit": commits.pop(),
              "reference_revision": ref0["model_revision"], "reference_versions": ref0["packages"],
              "max_new_tokens": ref0["max_tokens"], "completed": [c["id"] for c in cases] == CASES, "cases": cases}
    (OUT / "hosted-parity.json").write_text(json.dumps(parity, indent=2, ensure_ascii=False) + "\n")

    arms = []
    for dtype, lanes in PRECISION_LANES.items():
        found, rows = case_dirs(lanes), []
        for cid in CASES:
            if cid not in found:
                continue
            lane, d = found[cid]
            ref = json.loads((d / "python.json").read_text())
            a = arm(d, "native-default.json", ref)
            rows.append({"id": cid, "lane": lane, "commit": source(lane),
                         **{k: v for k, v in a.items() if k != "ids"}})
        arms.append({"name": f"{dtype}-cpu", "dtype": dtype, "cases": rows})
    precision = {"schema": "crispembed.oxr.precision.v1", "reference_revision": parity["reference_revision"],
                 "arms": arms}
    (OUT / "precision.json").write_text(json.dumps(precision, indent=2, ensure_ascii=False) + "\n")
    for c in cases:
        print(c["id"], len(c["reference_ids"]), {a: v["ids_exact"] for a, v in c["arms"].items()})
    for a in arms:
        print(a["name"], {c["id"]: c["ids_exact"] for c in a["cases"]})
    print("completed:", parity["completed"])


if __name__ == "__main__":
    main()
