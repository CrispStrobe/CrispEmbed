#!/usr/bin/env python3
"""Guard: a custom OCR prompt must reach every prompt-following VLM, on every surface.

Issue #56: `--ocr-prompt` did not exist, the CLI treated the unknown flag and
its value as input texts, and `--ocr` mode never reads texts — so the run
"worked" and produced prompt-independent output. The orchestrator already had
`vlm_prompt`, but both stage builders hardcoded it to nullptr.

The prompt now flows through four hand-maintained lists over the same engines:

  1. `crispembed_ocr_model_set_prompt()`   (src/crispembed.cpp)  — single-model `--ocr`
  2. `run_engine()`                        (src/ocr_orchestrator.cpp) — must forward
                                            `st.params.vlm_prompt`
  3. `takes_prompt` in the CLI             (examples/cli/main.cpp) — decides the
                                            "prompt ignored" warning
  4. the CLI and server stage builders     — must pass the flag, not nullptr

Source-parsing, so it needs no models and runs in the lint tier. Same shape as
tests/test_ocr_max_tokens_surfaces.py.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Orchestrator engines whose stage must forward vlm_prompt, and the setter
# (or recognize argument) that proves it.
ORCH_PROMPT = {
    "qwen2vl": "qwen2vl_ocr_set_prompt",
    "qwen3vl": "qwen2vl_ocr_set_prompt",
    "internvl2": "internvl2_ocr_set_prompt",
    "granite_vision": "vlm_prompt",
    "lfm2_vl": "lfm2_vl_ocr_set_prompt",
    "unified": "crispembed_ocr_model_set_prompt",
}

# OCR_MODEL_* tags crispembed_ocr_model_set_prompt() must accept.
ABI_PROMPT = {"QWEN2VL", "INTERNVL2", "LFM2_VL", "GRANITE_VISION"}

# CLI engine ids (eng_id in examples/cli/main.cpp) that accept a prompt.
CLI_PROMPT_NAMES = {"qwen2vl", "internvl2", "granite-vision", "qwen3vl", "unified", "olmocr", "lfm2-vl"}


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def case_body(src: str, eng: str) -> str:
    m = re.search(r"case engine::" + eng + r": \{(.*?)\n    case engine::", src, re.S)
    if not m:
        m = re.search(r"case engine::" + eng + r": \{(.*?)\n    default:", src, re.S)
    assert m, f"run_engine case for {eng} not found"
    return m.group(1)


def main() -> int:
    errors: list[str] = []

    orch = read("src/ocr_orchestrator.cpp")
    for eng, marker in ORCH_PROMPT.items():
        body = case_body(orch, eng)
        if "vlm_prompt" not in body or marker not in body:
            errors.append(f"run_engine({eng}) does not forward st.params.vlm_prompt via {marker}")

    abi = read("src/crispembed.cpp")
    m = re.search(r"int crispembed_ocr_model_set_prompt\(void \* ctx, const char \* prompt\) \{(.*?)\n\}", abi, re.S)
    if not m:
        errors.append("crispembed_ocr_model_set_prompt not found")
    else:
        handled = set(re.findall(r"case OCR_MODEL_(\w+):", m.group(1)))
        for tag in sorted(ABI_PROMPT - handled):
            errors.append(f"crispembed_ocr_model_set_prompt() does not handle OCR_MODEL_{tag}")

    cli = read("examples/cli/main.cpp")
    ids = {name: int(i) for name, i in re.findall(r'if \(n == "([\w-]+)"[^)]*\) return (\d+);', cli)}
    tm = re.search(r"const bool takes_prompt =(.*?);", cli, re.S)
    if not tm:
        errors.append("CLI takes_prompt list not found")
    else:
        listed = {int(x) for x in re.findall(r"eid == (\d+)", tm.group(1))}
        want = {ids[n] for n in CLI_PROMPT_NAMES if n in ids}
        if listed != want:
            errors.append(f"CLI takes_prompt ids {sorted(listed)} != expected {sorted(want)}")
    if '"--ocr-prompt"' not in cli:
        errors.append("CLI does not parse --ocr-prompt")
    if "crispembed_ocr_model_set_prompt(octx" not in cli:
        errors.append("CLI --ocr lane does not call crispembed_ocr_model_set_prompt")

    for rel in ("examples/cli/main.cpp", "examples/server/server.cpp"):
        src = read(rel)
        if re.search(r"st\.vlm_prompt = nullptr;", src):
            errors.append(f"{rel}: stage builder hardcodes st.vlm_prompt = nullptr")

    if errors:
        for e in errors:
            print("FAIL:", e)
        return 1
    print(f"OCR prompt surfaces OK: {len(ORCH_PROMPT)} orchestrator stages, {len(ABI_PROMPT)} ABI engines, "
          f"{len(CLI_PROMPT_NAMES)} CLI engine names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
