#!/usr/bin/env python3
"""Aggregate hosted TeleOCR lanes (tools/ci-heavy/teleocr_qualify.sh) into the shared receipts.

    python tools/teleocr_receipts.py RUN_DIR [RUN_DIR ...]

Each RUN_DIR is one downloaded heavy-cpu.yml artifact. Writes
tests/results/teleocr/hosted-parity.json (F32 arm) and hosted-precision.json
(every other precision) in the schema tools/ci-heavy/ocr_gguf_publish.py
gates on. Every lane freezes its own original-model reference; they must agree
token for token, and all runs must come from one source commit. Numerical,
coverage and confidence verdicts are carried along unchanged; nothing is
relaxed.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/teleocr-publication-v1.json"
OUT = ROOT / "tests/results/teleocr"


def main(argv):
    cfg = json.loads(PROTOCOL.read_text())
    cases = cfg["parity_cases"]
    commits, references, arms, packages = set(), {}, {}, {}
    for run in map(Path, argv[1:]):
        commits.add(re.search(r"^source=([0-9a-f]{40})$", (run / "provenance.txt").read_text(), re.M).group(1))
        packages = {l.split("==")[0].lower(): l.split("==")[1] for l in (run / "packages.txt").read_text().split()
                    if "==" in l}
        for case in cases:
            ref = json.loads((run / "reference" / f"{case}.json").read_text())
            if case in references and references[case]["generated_ids"] != ref["generated_ids"]:
                raise SystemExit(f"lanes froze different references for {case}")
            references[case] = ref
        for native in sorted(run.glob("native-*")):
            dtype = native.name.removeprefix("native-")
            for case in cases:
                path = native / f"{case}.json"
                if not path.exists():
                    continue
                r = json.loads(path.read_text())
                ids = r.get("generated_ids", [])
                arms.setdefault(dtype, {})[case] = {
                    "exit": 0 if r.get("decoded_pass") else 1,
                    "ids_exact": ids == references[case]["generated_ids"], "ids": ids,
                    "decoded_pass": r.get("decoded_pass"), "numerical_pass": r.get("numerical_pass"),
                    "coverage_complete": r.get("coverage_complete"), "confidence_pass": r.get("confidence_pass"),
                    "reached_eos": r.get("reached_eos"), "tokens": r.get("tokens"), "seconds": r.get("seconds"),
                    "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if len(commits) != 1:
        raise SystemExit(f"runs come from different source commits: {sorted(commits)}")
    any_ref = references[cases[0]]
    common = {"commit": commits.pop(), "reference_revision": cfg["upstream"]["revision"],
              "reference_versions": {k: packages.get(k) for k in ("torch", "transformers", "pillow", "numpy")},
              "max_new_tokens": any_ref["environment"]["max_new_tokens"],
              "completed": sorted(references) == sorted(cases) and all(c in arms.get("f32", {}) for c in cases)}
    parity = {"schema": "crispembed.teleocr.hosted-parity.v1", **common,
              "f32_numerical_pass": all(arms.get("f32", {}).get(c, {}).get("numerical_pass") for c in cases),
              "cases": [{"id": c, "reference_ids": references[c]["generated_ids"],
                         "reference_text": references[c]["text"], "reference_eos": references[c]["reached_eos"],
                         "arms": {"f32": arms["f32"][c]} if c in arms.get("f32", {}) else {}} for c in cases]}
    precision = {"schema": "crispembed.teleocr.hosted-precision.v1", **common, "arms": [
        {"name": f"{dt}-cpu", "dtype": dt,
         "cases": [{"id": c, **{k: v for k, v in arms[dt][c].items() if k != "ids"}} for c in cases if c in arms[dt]]}
        for dt in sorted(arms) if dt != "f32"]}
    (OUT / "hosted-parity.json").write_text(json.dumps(parity, indent=2, ensure_ascii=False) + "\n")
    (OUT / "hosted-precision.json").write_text(json.dumps(precision, indent=2, ensure_ascii=False) + "\n")
    for dt in sorted(arms):
        row = " ".join(f"{c}:{'exact' if arms[dt][c]['ids_exact'] else 'DIFF'}" for c in cases if c in arms[dt])
        print(f"{dt:6} {row}")
    print("completed:", common["completed"], "f32_numerical_pass:", parity["f32_numerical_pass"])


if __name__ == "__main__":
    main(sys.argv)
