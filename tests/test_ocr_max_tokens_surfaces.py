#!/usr/bin/env python3
"""Guard: the VLM generation cap must reach every whole-page VLM engine, on every surface.

`--ocr-max-tokens` is one user-visible flag sitting on top of FOUR
hand-maintained lists over the same set of engines:

  1. `is_vlm_engine()`            (src/ocr_orchestrator.cpp) — the canonical set
  2. `run_engine()`               (src/ocr_orchestrator.cpp) — must apply
                                   `st.params.vlm_max_tokens` for each of them
  3. `crispembed_ocr_model_set_max_tokens()` (src/crispembed.cpp) — the single-model
                                   `--ocr` lane
  4. `is_vlm` in the CLI          (examples/cli/main.cpp) — decides whether
                                   `model_a` resolves down the VLM or the DETECTOR branch

Until 2026-08-25 the flag was accepted, printed in `--help`, and silently did
NOTHING on `--ocr-pipeline` (both the CLI and the server stage builders
hardcoded `vlm_max_tokens = 0`), and did nothing on got / glm / deepseek_ocr2 /
unlimited_ocr on ANY surface, because those four engines had no setter at all.
Nothing failed; the model simply kept generating, which reads as "it rambles",
not as a bug. List (4) drifting from (1) is worse still — it hands a VLM engine
a detector model path, and the failure surfaces deep inside a vision graph.

This is the multi-surface dispatch trap the dev guide calls the #1 recurring
bug. Source-parsing rather than runtime, so it needs no models and runs in the
lint tier. Same shape as tests/test_cli_engine_names.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Engines whose generation length is not a knob: line/formula recognisers and
# detectors. Anything NOT here and reachable as a stage must honour the cap.
NON_GENERATIVE = {
    "dbnet_trocr", "surya", "tesseract", "tesseract_fraktur", "parseq",
    "ppocrv6", "easyocr", "unified",
}

failures: list[str] = []


def fail(msg: str) -> None:
    failures.append(msg)
    print(f"  FAIL {msg}")


def orchestrator_src() -> str:
    return (ROOT / "src" / "ocr_orchestrator.cpp").read_text()


def vlm_engines() -> set[str]:
    """The canonical set: `is_vlm_engine()`."""
    src = orchestrator_src()
    m = re.search(r"static bool is_vlm_engine\(engine e\) \{(.*?)\n\}", src, re.S)
    assert m, "is_vlm_engine not found in src/ocr_orchestrator.cpp"
    body = m.group(1)
    head = body.split("return true;")[0]
    names = set(re.findall(r"case engine::(\w+):", head))
    assert len(names) >= 10, f"is_vlm_engine parse suspiciously short: {names}"
    return names


def engine_ids() -> dict[str, int]:
    """engine name → C-ABI int, from `map_engine()` — the shipped contract.

    NOT the enum's ordinal position: the two are supposed to agree, but
    `map_engine` is what every C consumer and the CLI actually index by, so it
    is what the CLI's is_vlm ids must be checked against.
    """
    src = (ROOT / "src" / "crispembed.cpp").read_text()
    m = re.search(r"static ocr_orchestrator::engine map_engine\(int e\).*?\n\}\n", src, re.S)
    assert m, "map_engine not found in src/crispembed.cpp"
    ids = {eng: int(case) for case, eng in re.findall(r"case (\d+):\s*return E::(\w+);", m.group(0))}
    assert len(ids) >= 18, f"map_engine parse suspiciously short: {ids}"
    return ids


def run_engine_cases_applying_cap() -> set[str]:
    """Engines whose `run_engine()` block reads `vlm_max_tokens`.

    Handles C fallthrough groups: `case engine::olmocr:` immediately followed by
    `case engine::qwen2vl: {` share one body, so both are credited with what
    that body does.
    """
    body = orchestrator_src()
    marks = [(mm.start(), mm.end(), mm.group(1)) for mm in re.finditer(r"    case engine::(\w+):", body)]
    applying: set[str] = set()
    for i, (pos, end_of_label, name) in enumerate(marks):
        # Walk forward over any labels that fall through into the same body.
        group = [name]
        j = i
        while j + 1 < len(marks) and body[marks[j][1]:marks[j + 1][0]].strip() == "":
            j += 1
            group.append(marks[j][2])
        end = marks[j + 1][0] if j + 1 < len(marks) else len(body)
        if "vlm_max_tokens" in body[pos:end]:
            applying.update(group)
    return applying


# orchestrator engine -> the OCR_MODEL_* tag `--ocr` resolves it to
# (src/crispembed.cpp detect_arch). Several engines share one tag.
ENGINE_TO_OCR_MODEL = {
    "got": "GOT_OCR", "glm": "GLM_OCR", "qwen2vl": "QWEN2VL", "qwen3vl": "QWEN2VL",
    "olmocr": "QWEN2VL", "internvl2": "INTERNVL2", "deepseek_ocr2": "DEEPSEEK_OCR2",
    "granite_vision": "GRANITE_VISION", "lightonocr": "LIGHTONOCR",
    "unlimited_ocr": "UNLIMITED_OCR", "lfm2_vl": "LFM2_VL",
    # pix2struct has no --ocr route at all: it is reached by --pix2struct, whose
    # own --pix2struct-max-tokens is passed per call rather than via a setter.
    "pix2struct": None,
}


def c_abi_setter_engines() -> set[str]:
    """OCR_MODEL_* tags handled by crispembed_ocr_model_set_max_tokens()."""
    src = (ROOT / "src" / "crispembed.cpp").read_text()
    m = re.search(r"void crispembed_ocr_model_set_max_tokens\(void \* ctx, int max_tokens\) \{(.*?)\n\}", src, re.S)
    assert m, "crispembed_ocr_model_set_max_tokens not found"
    return set(re.findall(r"case OCR_MODEL_(\w+):", m.group(1)))


def cli_is_vlm_ids() -> set[int]:
    """The engine ids the CLI treats as whole-page VLMs."""
    src = (ROOT / "examples" / "cli" / "main.cpp").read_text()
    m = re.search(r"const bool is_vlm =\s*(.*?);", src, re.S)
    assert m, "is_vlm expression not found in examples/cli/main.cpp"
    expr = m.group(1)
    ids: set[int] = set()
    for lo, hi in re.findall(r"eid >= (\d+) && eid <= (\d+)", expr):
        ids.update(range(int(lo), int(hi) + 1))
    ids.update(int(n) for n in re.findall(r"eid == (\d+)", expr))
    return ids


def main() -> int:
    canonical = vlm_engines()
    ids_by_name = engine_ids()
    print(f"VLM engines per is_vlm_engine(): {len(canonical)}")

    # (1) vs (2): every VLM engine's run_engine block applies the cap.
    applying = run_engine_cases_applying_cap()
    for eng in sorted(canonical):
        if eng in NON_GENERATIVE:
            continue
        if eng not in applying:
            fail(f"engine::{eng} is a VLM engine but its run_engine() block never reads vlm_max_tokens "
                 f"— --ocr-pipeline --ocr-max-tokens silently no-ops on it")

    # (1) vs (4): the CLI's is_vlm list must agree, or model_a resolves down the
    # detector branch and the engine is handed a DBNet path.
    cli_ids = cli_is_vlm_ids()
    for eng in sorted(canonical):
        eid = ids_by_name.get(eng)
        if eid is None:
            fail(f"engine::{eng} is in is_vlm_engine() but has no case in map_engine()")
            continue
        if eid not in cli_ids:
            fail(f"engine::{eng} (id {eid}) is a VLM in the orchestrator but NOT in the CLI's is_vlm "
                 f"list — model_a would resolve down the detector branch")
    # The reverse direction, minus `unified`: the CLI deliberately routes the
    # metadata-dispatched lane through the single-model branch too, and it is
    # not an engine in its own right.
    for eid in sorted(cli_ids):
        name = next((n for n, o in ids_by_name.items() if o == eid), None)
        if name and name != "unified" and name not in canonical:
            fail(f"CLI treats engine id {eid} (engine::{name}) as a VLM but is_vlm_engine() does not")

    # (3): the single-model --ocr lane must reach the same generative engines.
    abi = c_abi_setter_engines()
    print(f"engines handled by crispembed_ocr_model_set_max_tokens(): {len(abi)}")
    for eng in sorted(canonical):
        if eng in NON_GENERATIVE:
            continue
        if eng not in ENGINE_TO_OCR_MODEL:
            fail(f"engine::{eng} is a VLM engine with no entry in this test's ENGINE_TO_OCR_MODEL table "
                 f"— add it (or mark it None if it has no --ocr route)")
            continue
        tag = ENGINE_TO_OCR_MODEL[eng]
        if tag is None:
            continue
        if tag not in abi:
            fail(f"engine::{eng} resolves to OCR_MODEL_{tag}, which has no case in "
                 f"crispembed_ocr_model_set_max_tokens() — --ocr --ocr-max-tokens silently no-ops on it")

    # The two stage builders must not throw the value away. This is the exact
    # line that was `= 0` on both surfaces.
    for rel, var in (("examples/cli/main.cpp", "ocr_max_tokens"),
                     ("examples/server/server.cpp", "ocr_max_tokens")):
        src = (ROOT / rel).read_text()
        if re.search(r"st\.vlm_max_tokens\s*=\s*0\s*;", src):
            fail(f"{rel} hardcodes st.vlm_max_tokens = 0 — the pipeline lane drops --ocr-max-tokens")
        if not re.search(r"st\.vlm_max_tokens\s*=\s*" + var + r"\s*;", src):
            fail(f"{rel} never assigns {var} to st.vlm_max_tokens")

    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
