#!/usr/bin/env python
"""Guard the registry's `languages` field (examples/cli/model_mgr.cpp).

The field records which scripts a recognizer's dictionary can emit, scanned
from the shipped GGUF by tools/scan_model_languages.py. This test does not
re-scan (that needs the model files); it guards the two ways the field can
silently rot:

  1. a typo'd or invented script label, which would render but mean nothing;
  2. drift on the facts that motivated the field in the first place —
     PP-OCRv6's tiny recognizer has NO kana and is not Japanese-capable
     (issue #44), while small/medium do, and tesseract-kor covers hangul but
     NOT the CJK ideographs that mixed hanja Korean needs.

Run: python tests/test_registry_languages.py
"""

import pathlib
import re
import sys

REGISTRY = pathlib.Path(__file__).resolve().parent.parent / "examples" / "cli" / "model_mgr.cpp"

# Must match the SCRIPTS labels in tools/scan_model_languages.py.
KNOWN = {
    "latin", "cjk", "kana", "hangul", "cyrillic",
    "greek", "arabic", "hebrew", "devanagari", "thai",
}

# name -> (scripts that MUST be present, scripts that MUST be absent).
# Each entry is a measured fact with a reason it matters.
INVARIANTS = {
    # issue #44: the tiny recognizer cannot read Japanese at all.
    "ppocrv6-tiny-rec": ({"latin", "cjk"}, {"kana"}),
    "ppocrv6-small-rec": ({"latin", "cjk", "kana"}, set()),
    "ppocrv6-medium-rec": ({"latin", "cjk", "kana"}, set()),
    # The CJK tesseract lane: jpn needs kana, chi_sim must not claim it.
    "tesseract-jpn": ({"cjk", "kana"}, set()),
    "tesseract-chi-sim": ({"cjk"}, {"kana", "hangul"}),
    # Korean hanja is NOT covered — hangul only.
    "tesseract-kor": ({"hangul"}, {"cjk"}),
    "tesseract-rus": ({"cyrillic"}, set()),
    "tesseract-ara": ({"arabic"}, set()),
    "tesseract-eng": ({"latin"}, {"cjk", "cyrillic", "arabic"}),
}

ENTRY = re.compile(r'\{\s*"([a-z0-9._-]+)",\s*"[^"]*\.gguf"(.*?)\},', re.S)


def parse():
    text = REGISTRY.read_text()
    found = {}
    for match in ENTRY.finditer(text):
        name, body = match.group(1), match.group(2)
        fields = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
        # The languages literal, when present, is the last string of the row
        # and is built only from known script labels joined by "+".
        if fields:
            last = fields[-1]
            if last and all(part in KNOWN for part in last.split("+")):
                found[name] = last
                continue
        found[name] = ""
    return found


def main():
    entries = parse()
    assert entries, "no registry entries parsed — did the row format change?"
    failures = []

    for name, scripts in entries.items():
        if not scripts:
            continue
        for part in scripts.split("+"):
            if part not in KNOWN:
                failures.append(f"{name}: unknown script label {part!r}")

    for name, (required, forbidden) in INVARIANTS.items():
        if name not in entries:
            failures.append(f"{name}: missing from the registry")
            continue
        scripts = set(entries[name].split("+")) if entries[name] else set()
        if not scripts:
            failures.append(f"{name}: languages field is empty but this model was scanned")
            continue
        for want in sorted(required - scripts):
            failures.append(f"{name}: expected {want!r} in dict coverage, got {entries[name]!r}")
        for deny in sorted(forbidden & scripts):
            failures.append(f"{name}: {deny!r} must NOT be claimed (measured absent), got {entries[name]!r}")

    if failures:
        for line in failures:
            print("FAIL:", line, file=sys.stderr)
        return 1

    scanned = sum(1 for v in entries.values() if v)
    print(f"registry languages OK: {scanned} scanned models, {len(INVARIANTS)} invariants held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
