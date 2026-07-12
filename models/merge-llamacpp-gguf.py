#!/usr/bin/env python3
"""Unified entry point: import a stock llama.cpp vision-language model (LLM +
mmproj GGUF pair) into a single CrispEmbed-loadable GGUF.

Auto-detects the model family from the mmproj's `clip.projector_type` and
dispatches to the matching per-family merge script (all built on the shared
`gguf_merge_core.py`):

    projector_type      family        CrispEmbed engine   merge script
    ------------------  ------------  ------------------  --------------------------------
    qwen2vl_merger      Qwen2-VL      qwen2vl_ocr         merge-llamacpp-qwen2vl-gguf.py
    idefics3            SmolVLM       smoldocling         merge-llamacpp-smolvlm-gguf.py
    internvl            InternVL2.5/3 internvl2_ocr       merge-llamacpp-internvl-gguf.py

Each was validated end-to-end (correct OCR + diff-harness parity with the native
converter) on the small ggml-org GGUFs. Interop only — do NOT link libmtmd.

    python merge-llamacpp-gguf.py --llm MODEL.gguf --mmproj mmproj-MODEL.gguf --output out.gguf
    python merge-llamacpp-gguf.py --mmproj mmproj-MODEL.gguf --detect   # print family and exit
"""
import argparse
import os
import subprocess
import sys

import gguf_merge_core as core

HERE = os.path.dirname(os.path.abspath(__file__))

# clip.projector_type -> (family label, merge script filename)
DISPATCH = {
    "qwen2vl_merger": ("Qwen2-VL", "merge-llamacpp-qwen2vl-gguf.py"),
    "idefics3":       ("SmolVLM (Idefics3)", "merge-llamacpp-smolvlm-gguf.py"),
    "internvl":       ("InternVL2.5/3", "merge-llamacpp-internvl-gguf.py"),
}


def detect_family(mmproj_path):
    """Return (projector_type, family_label, script) for an mmproj GGUF."""
    md = core.read_gguf(mmproj_path).metadata
    proj = md.get("clip.projector_type")
    if proj is None:
        sys.exit(f"error: {mmproj_path} has no clip.projector_type "
                 f"(not a llama.cpp mmproj?)")
    if proj not in DISPATCH:
        supported = ", ".join(sorted(DISPATCH))
        sys.exit(f"error: unsupported projector_type '{proj}'. "
                 f"Supported: {supported}. "
                 f"(Add a per-family merge script + entry to import it.)")
    label, script = DISPATCH[proj]
    return proj, label, script


def main():
    ap = argparse.ArgumentParser(description="Import a llama.cpp VL model into CrispEmbed (auto-detect family)")
    ap.add_argument("--llm", help="llama.cpp LLM GGUF")
    ap.add_argument("--mmproj", required=True, help="llama.cpp mmproj GGUF")
    ap.add_argument("--output", help="output CrispEmbed GGUF")
    ap.add_argument("--detect", action="store_true", help="print the detected family and exit")
    a = ap.parse_args()

    proj, label, script = detect_family(a.mmproj)
    print(f"Detected: projector_type='{proj}' -> {label}  ({script})")
    if a.detect:
        return 0
    if not a.llm or not a.output:
        ap.error("need --llm and --output (or --detect)")

    script_path = os.path.join(HERE, script)
    cmd = [sys.executable, script_path, "--llm", a.llm, "--mmproj", a.mmproj, "--output", a.output]
    print(f"$ {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
