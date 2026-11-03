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


def vlm_default(path: str, eid: int) -> str | None:
    """The registry alias the VLM default ternary picks for engine `eid`."""
    src = (ROOT / path).read_text()
    m = re.search(r'const char \* dflt = \(eid == 2\)(.*?);', src, re.S)
    assert m, f"VLM default ternary not found in {path}"
    hit = re.search(r'\(eid == ' + str(eid) + r'\)\s*\?\s*"([^"]+)"', "(eid == 2)" + m.group(1))
    return hit.group(1) if hit else None


def minicpm_default_consistent() -> list[str]:
    """CLI and server must resolve the same default for MiniCPM-OCR (id 23) and
    neither may demand an explicit --ocr-rec for it (gates are still required)."""
    errs = []
    for path in ("examples/cli/main.cpp", "examples/server/server.cpp"):
        if vlm_default(path, 23) != "minicpm-ocr":
            errs.append(f"{path}: engine 23 default is {vlm_default(path, 23)!r}, want 'minicpm-ocr'")
        src = (ROOT / path).read_text()
        gate = re.search(r'if \(eid == 23 && \((.*?)\)\) \{', src, re.S)
        if not gate:
            errs.append(f"{path}: MiniCPM gate check not found")
            continue
        g = gate.group(1)
        if "rec" in g and "empty()" in g:
            errs.append(f"{path}: MiniCPM still requires an explicit model path")
        for need in ("CRISPEMBED_MINICPM_OCR", "CRISPEMBED_QWEN35_OCR"):
            if f'core_env::on("{need}")' not in g:
                errs.append(f"{path}: MiniCPM gate {need} missing")
    return errs


def main() -> int:
    errs = minicpm_default_consistent()
    if errs:
        print("FAIL: " + "; ".join(errs))
        return 1
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
