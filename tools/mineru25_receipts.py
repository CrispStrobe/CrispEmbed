#!/usr/bin/env python3
"""Aggregate hosted MinerU2.5-Pro case runs into the committed receipts.

    python tools/mineru25_receipts.py RUN_DIR [RUN_DIR ...]

Each RUN_DIR is one downloaded heavy-cpu.yml artifact from
tools/ci-heavy/mineru25_qualify.sh. Writes tests/results/mineru25/native-parity.json
(F32 ordinary/reuse arms and all F32 gates) and precision.json (every other
precision, one arm per dtype). Nothing is relaxed: a receipt is `completed` only
when every protocol case is present from one source commit, and the publish
gate separately requires every arm token-ID exact.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/mineru25-pro-publication-v1.json"
OUT = ROOT / "tests/results/mineru25"
# -xg: CRISPEMBED_MINERU25_PRO_EXACT_GELU=1 (F32 vision quick_gelu) A/B arms.
F32_ARMS = ("f32", "f32-reuse", "f32-xg", "f32-xg-reuse")


def case_run(run_dir, by_input):
    d = Path(run_dir)
    summary = json.loads((d / "summary.json").read_text())
    ref = json.loads((d / "python.json").read_text())
    prov = (d / "provenance.txt").read_text()
    commit = re.search(r"^source=([0-9a-f]{40})$", prov, re.M).group(1)
    gates = dict(line.split("=", 1) for line in (d / "results.txt").read_text().split() if "=" in line)
    case_id = by_input[(summary["case"], summary["task"])]
    arms = {}
    for f in sorted(d.glob("native-*-reuse*.json")):
        m = re.fullmatch(r"native-(.+)-reuse([01])", f.stem)
        r = json.loads(f.read_text())
        n = r.get("native", {})
        ids = n.get("token_ids", [])
        name = m.group(1) + ("-reuse" if m.group(2) == "1" else "")
        arms[name] = {"exit": 0 if r.get("request_parity_pass") else 1, "ids_exact": ids == ref["token_ids"],
                      "ids": ids, "text_exact": n.get("text") == ref["text"], "eos": n.get("eos"),
                      "request_parity_pass": r.get("request_parity_pass"), "speed_pass": r.get("speed_pass"),
                      "native_over_python": r.get("native_over_python"), "ground_truth": r.get("ground_truth"),
                      "gguf_sha256": r.get("gguf_sha256"), "report_sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    return commit, {"id": case_id, "image": summary["case"], "task": summary["task"],
                    "reference_ids": ref["token_ids"], "reference_text": ref["text"], "reference_eos": ref["eos"],
                    "reference_capped": ref["capped"], "max_tokens": ref["max_tokens"],
                    "gates": gates, "arms": arms}, ref


def main(argv):
    cfg = json.loads(PROTOCOL.read_text())
    by_input = {tuple(v): k for k, v in cfg["case_inputs"].items()}
    runs = [case_run(a, by_input) for a in argv[1:]]
    commits = {c for c, _, _ in runs}
    if len(commits) != 1:
        raise SystemExit(f"runs come from different source commits: {sorted(commits)}")
    order = {k: i for i, k in enumerate(cfg["parity_cases"])}
    cases = sorted((c for _, c, _ in runs), key=lambda c: order[c["id"]])
    ref = runs[0][2]
    common = {"commit": commits.pop(), "reference_revision": ref["model_revision"],
              "reference_versions": {**ref["packages"], "mineru_vl_utils": ref["blueprint_revision"]},
              "max_new_tokens": ref["max_tokens"],
              "completed": [c["id"] for c in cases] == cfg["parity_cases"]}
    f32_keys = {"reference", "preprocess", "stages-f32", "decode-f32", "decode-f32-reuse"}
    xg_keys = {"stages-f32-xg", "decode-f32-xg", "decode-f32-xg-reuse"}
    parity = {"schema": "crispembed.mineru25.native-parity.v1", **common,
              "f32_gates_pass": all(all(c["gates"].get(k) == "pass" for k in f32_keys) for c in cases),
              "f32_exact_gelu_gates_pass": all(all(c["gates"].get(k) == "pass" for k in xg_keys) for c in cases),
              "cases": [{**c, "arms": {k: v for k, v in c["arms"].items() if k in F32_ARMS},
                         "gates": {k: v for k, v in c["gates"].items() if k in f32_keys | xg_keys}} for c in cases]}
    dtypes = sorted({k for c in cases for k in c["arms"] if k not in F32_ARMS})
    precision = {"schema": "crispembed.mineru25.precision.v1", **common, "arms": [
        {"name": f"{dt}-cpu", "dtype": dt, "cases": [
            {"id": c["id"], **{k: v for k, v in c["arms"][dt].items() if k != "ids"},
             "stages": c["gates"].get(f"stages-{dt}")} for c in cases if dt in c["arms"]]}
        for dt in dtypes]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "native-parity.json").write_text(json.dumps(parity, indent=2, ensure_ascii=False) + "\n")
    (OUT / "precision.json").write_text(json.dumps(precision, indent=2, ensure_ascii=False) + "\n")
    for c in cases:
        row = " ".join(f"{k}:{'exact' if a['ids_exact'] else 'DIFF'}" for k, a in c["arms"].items())
        print(f"{c['id']:14} ref={len(c['reference_ids'])} eos={c['reference_eos']} {row} gates={c['gates']}")
    print("completed:", parity["completed"], "f32_gates_pass:", parity["f32_gates_pass"])


if __name__ == "__main__":
    main(sys.argv)
