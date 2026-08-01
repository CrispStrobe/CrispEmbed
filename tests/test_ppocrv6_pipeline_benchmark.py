#!/usr/bin/env python3
"""Pure parser/policy checks for the PP-OCRv6 benchmark wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ppocrv6_pipeline_benchmark", ROOT / "tests/ppocrv6_pipeline_benchmark.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def main() -> int:
    match = MODULE.ROW.match("  INFO: fixture.png: 4 regions, 33 chars (conf=0.91, time_ms=123.4)")
    assert match is not None
    assert match.group("fixture") == "fixture.png"
    assert int(match.group("regions")) == 4
    assert int(match.group("chars")) == 33
    assert float(match.group("confidence")) == 0.91
    assert float(match.group("time_ms")) == 123.4
    rows = MODULE.parse_rows(match.string, "medium")
    assert rows[0]["variant"] == "medium"
    assert rows[0]["time_ms"] == 123.4
    print("PP-OCRv6 benchmark parser OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
