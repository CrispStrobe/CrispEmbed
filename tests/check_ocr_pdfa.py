#!/usr/bin/env python3
"""Validate generated archival PDFs with the independent veraPDF CLI.

The XML report, every failed rule, stdout/stderr and negative controls remain
in the CI artifact. A successful process exit alone is not a conformance pass.
"""

import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def validate(executable, path, destination):
    result = subprocess.run([executable, "--format", "xml", "--flavour", "2b", str(path)],
                            capture_output=True, timeout=90)
    destination.with_suffix(".xml").write_bytes(result.stdout)
    destination.with_suffix(".stderr.txt").write_bytes(result.stderr)
    report = ET.fromstring(result.stdout)
    reports = report.findall(".//validationReport")
    if len(reports) != 1:
        raise AssertionError(f"{path.name}: expected exactly one validation report, got {len(reports)}")
    validation = reports[0]
    if "pdf/a-2b" not in validation.attrib.get("profileName", "").lower():
        raise AssertionError(f"{path.name}: wrong validation profile {validation.attrib}")
    failures = [ET.tostring(rule, encoding="unicode")
                for rule in validation.findall(".//rule[@status='failed']")]
    return {"file": path.name, "exit_code": result.returncode,
            "compliant": validation.attrib.get("isCompliant") == "true",
            "failed_rules": failures}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verapdf", required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    options = parser.parse_args()
    root = options.artifacts
    reports = root / "verapdf"
    reports.mkdir(exist_ok=True)
    candidates = sorted(root.glob("archival-*.pdf")) + sorted(root.glob("bidi-*.pdf"))
    candidates = [path for path in candidates if not path.name.startswith("archival-control-")]
    if len(candidates) < 15:
        raise AssertionError(f"Expected all 15 archival image/text/bidi cases, got {len(candidates)}")
    receipt = {"validator": subprocess.check_output([options.verapdf, "--version"], text=True).strip(),
               "profile": "2b", "positive": [], "negative": []}
    try:
        for path in candidates:
            record = validate(options.verapdf, path, reports / path.stem)
            receipt["positive"].append(record)
        # Equal-length key replacement preserves all xref offsets and page
        # content: only the required resource being tested is removed.
        source = (root / "archival-rgb.pdf").read_bytes()
        for name, old, new in (("missing-output-intent", b"/OutputIntents", b"/InvalidIntent"),
                               ("missing-xmp", b"/Metadata", b"/NoXmpTag")):
            assert len(old) == len(new) and old in source
            path = reports / (name + ".pdf")
            path.write_bytes(source.replace(old, new, 1))
            receipt["negative"].append(validate(options.verapdf, path, reports / name))
    finally:
        (reports / "receipt.json").write_text(json.dumps(receipt, indent=2))
    failures = [r for r in receipt["positive"] if not r["compliant"] or r["exit_code"] != 0]
    false_passes = [r for r in receipt["negative"] if r["compliant"] or not r["failed_rules"]]
    if failures or false_passes:
        raise AssertionError(json.dumps({"positive_failures": failures, "false_passes": false_passes}, indent=2))
    print(f"veraPDF: {len(receipt['positive'])} PDF/A-2b exports passed; 2 invalid controls rejected")


if __name__ == "__main__":
    main()
