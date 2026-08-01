#!/usr/bin/env python3
"""Integrity checks for public-domain derived robustness fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/regression/images/derived/MANIFEST.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    data = json.loads(MANIFEST.read_text())
    records = data["records"]
    assert len(records) >= 30, len(records)
    seen = set()
    operations = set()
    for record in records:
        target = ROOT / record["file"]
        parent = ROOT / record["parent"]
        assert target.is_file(), target
        assert parent.is_file(), parent
        assert sha256(parent) == record["parent_sha256"], parent
        assert sha256(target) == record["sha256"], target
        assert record["file"] not in seen
        seen.add(record["file"])
        operations.add(record["recipe"]["op"])
    required = {"rotate", "border", "gradient", "blend", "speckle", "resize", "jpeg", "perspective",
                "rotate-alternating-horizontal-bands"}
    assert required <= operations, sorted(required - operations)
    print(f"derived fixtures OK: {len(records)} records, {len(operations)} transformations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
