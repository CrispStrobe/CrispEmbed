#!/usr/bin/env python3
"""Pure unit checks for preprocessor benchmark outcome policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ocr_preprocessor_benchmark", ROOT / "tests/ocr_preprocessor_benchmark.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    raw = {"status": "ok", "cleanup": ["raw"]}
    assert MODULE.outcome(raw, raw) == "neutral"
    assert MODULE.outcome({"status": "error", "cleanup": ["--cleanup"]}, raw) == "error"
    assert MODULE.outcome({"status": "ok", "cleanup": ["--cleanup"], "text_delta_vs_raw": 0.0}, raw) == "neutral"
    assert MODULE.outcome({"status": "ok", "cleanup": ["--cleanup"], "text_delta_vs_raw": 0.02}, raw) == "neutral"
    assert MODULE.outcome({"status": "ok", "cleanup": ["--cleanup"], "text_delta_vs_raw": 0.3}, raw) == "unavailable"
    assert MODULE.outcome({"status": "ok", "cleanup": ["--cleanup"]}, raw) == "unavailable"
    print("preprocessor benchmark policy OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
