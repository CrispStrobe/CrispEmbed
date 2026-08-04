#!/usr/bin/env python3
"""Guard: every ocr_orchestrator engine id is reachable by a CLI name.

The ppocrv6 and easyocr lanes were fully implemented but CLI-unreachable for
months because nothing asserted enum <-> CLI-name coverage; the same later
held for deepseek_ocr2 and tesseract_fraktur.  This test parses the C-ABI id
map in src/crispembed.cpp (``map_engine``) and the CLI name map in
examples/cli/main.cpp (``eng_id``) and fails when an id gains an engine but
no CLI spelling.  Dependency-free; runs in lint CI.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def abi_ids() -> dict[int, str]:
    src = (ROOT / "src" / "crispembed.cpp").read_text()
    m = re.search(r"static ocr_orchestrator::engine map_engine\(int e\).*?\n}\n", src, re.S)
    assert m, "map_engine not found in src/crispembed.cpp"
    ids = {}
    for case, engine in re.findall(r"case (\d+):\s*return E::(\w+);", m.group(0)):
        ids[int(case)] = engine
    assert len(ids) >= 18, f"map_engine parse suspiciously short: {ids}"
    return ids


def cli_ids() -> set[int]:
    src = (ROOT / "examples" / "cli" / "main.cpp").read_text()
    m = re.search(r"auto eng_id = \[\]\(const std::string & n\) -> int \{(.*?)\n\s*\};", src, re.S)
    assert m, "eng_id lambda not found in examples/cli/main.cpp"
    ids = {0}  # default: dbnet_trocr
    for num in re.findall(r"return (\d+);", m.group(1)):
        ids.add(int(num))
    return ids


def main() -> int:
    abi = abi_ids()
    cli = cli_ids()
    missing = {i: name for i, name in abi.items() if i not in cli}
    if missing:
        print(f"FAIL: engines with a C-ABI id but no CLI name: {missing}")
        return 1
    print(f"CLI engine-name coverage OK: {len(abi)} ids, all reachable by name")
    return 0


if __name__ == "__main__":
    sys.exit(main())
