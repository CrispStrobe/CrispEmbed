#!/usr/bin/env python3
"""Schema and coverage guard for the public OCR engine matrix.

This is intentionally model-free: it prevents a supported engine from
silently disappearing from the matrix while allowing its artifact to be
uncached or its runtime load to be host-constrained.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "tests/regression/ocr_engine_matrix.json"

EXPECTED = {
    "PARSeq", "DBNet + TrOCR", "Tesseract-LSTM", "PP-FormulaNet-L",
    "MixTeX", "Texo-Distill", "PosFormer", "BTTR", "HMER", "SMT",
    "SMT++ full-page", "Polyphonic-TrOMR", "Flova/omr_transformer",
    "Transcoda-59M", "GOT-OCR2", "GLM-OCR", "InternVL2/2.5",
    "Qwen2.5-VL/Qwen3-VL", "DeepSeek-OCR-2", "Unlimited-OCR",
    "SmolDocling", "Qari-OCR", "PP-OCRv6",
}
STATUSES = {"runnable", "model-needed", "port/model-needed", "runtime-load-blocked"}


def main() -> int:
    doc = json.loads(MATRIX.read_text())
    rows = doc.get("engines")
    assert doc.get("version") == 1, "unsupported matrix version"
    assert isinstance(rows, list), "engines must be a list"

    names = [row.get("name") for row in rows]
    assert len(names) == len(set(names)), "duplicate engine names"
    assert set(names) == EXPECTED, f"coverage mismatch: missing={EXPECTED - set(names)}, extra={set(names) - EXPECTED}"

    for row in rows:
        assert row.get("lane"), f"{row.get('name')}: missing lane"
        assert row.get("runtime"), f"{row.get('name')}: missing runtime"
        assert row.get("fixture"), f"{row.get('name')}: missing fixture"
        assert row.get("status") in STATUSES, f"{row.get('name')}: invalid status"
        if row["status"] != "runnable":
            assert row.get("note") or row["status"] in {"model-needed", "port/model-needed"}, \
                f"{row.get('name')}: constrained status needs an explanation"

    print(f"engine matrix OK: {len(rows)} engines, {sum(r['status'] == 'runnable' for r in rows)} runnable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
