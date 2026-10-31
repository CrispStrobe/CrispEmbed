#!/usr/bin/env python3
"""Exercise real native TeleOCR document parsing and retain raw failed results.

This checks parser operation on a bounded image, not benchmark accuracy or
independent original-model parity. Every task result is retained before its
layout/table interpretation, including when a later check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--mode", choices=("detection", "segmentation"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if args.max_tokens < 1 or args.threads < 1:
        ap.error("max-tokens and threads must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("output directory is not empty; retain previous evidence and choose another directory")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.update(CRISPEMBED_TELEOCR="1", CRISPEMBED_TELEOCR_GPU="0", QWEN2VL_OCR_FORCE_CPU="1")

    spec = importlib.util.spec_from_file_location("teleocr_parse", ROOT / "tools/teleocr_parse.py")
    parser = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parser
    spec.loader.exec_module(parser)
    steps, checks = [], []
    report = {
        "kind": "teleocr-native-document-parser", "independent_model_parity": False,
        "model": args.model.name, "model_sha256": sha256(args.model),
        "library_sha256": sha256(args.lib), "image": args.image.name,
        "image_sha256": sha256(args.image), "mode": args.mode,
        "max_tokens": args.max_tokens, "threads": args.threads, "backend": "cpu",
        "test_sha256": sha256(__file__), "parser_sha256": sha256(ROOT / "tools/teleocr_parse.py"),
        "checks": checks, "passed": False,
    }

    def check(name, condition, **data):
        checks.append({"name": name, "passed": bool(condition), **data})
        if not condition:
            raise AssertionError(name)

    started = time.monotonic()
    try:
        with Image.open(args.image) as image:
            page_image = image.convert("RGB")
        with parser.TeleocrParser(args.model, threads=args.threads, lib_path=str(args.lib.resolve()),
                                  max_tokens=args.max_tokens) as engine:
            generate = engine._generate

            def traced_generate(image, prompt, kind):
                step = {"index": len(steps), "kind": kind, "prompt": prompt,
                        "image_size": list(image.size)}
                steps.append(step)
                write_json(args.out / "generations.json", steps)
                try:
                    text, tokens, confidence = generate(image, prompt, kind)
                    scores = np.asarray(engine.ocr.confidences()).tolist()
                    finite = all(math.isfinite(value) for value in scores)
                    step.update(text=text, tokens=tokens,
                                mean_confidence=confidence if math.isfinite(confidence) else None,
                                confidences=[value if math.isfinite(value) else None for value in scores],
                                confidence_finite=finite,
                                confidence_range=finite and all(0 <= value <= 1 for value in scores),
                                token_cap_reached=tokens >= args.max_tokens)
                    write_json(args.out / "generations.json", steps)
                    return text, tokens, confidence
                except Exception as exc:
                    step["error"] = f"{type(exc).__name__}: {exc}"
                    write_json(args.out / "generations.json", steps)
                    raise

            # Capture-only instrumentation; the parser receives exactly the
            # unchanged native text, token count and mean confidence.
            engine._generate = traced_generate
            page = engine.parse(page_image, mode=args.mode)
        page["page"] = 1
        write_json(args.out / "document.json", {"pages": [page]})
        (args.out / "document.md").write_text(page["markdown"] + "\n", encoding="utf-8")
        check("layout-and-extraction-ran", len(steps) >= 2, generations=len(steps))
        check("typed-blocks", bool(page["blocks"]) and
              all(block["type"] in parser.BLOCK_TYPES for block in page["blocks"]),
              count=len(page["blocks"]), types=[block["type"] for block in page["blocks"]])
        if args.mode == "segmentation":
            check("segmentation-polygons", any(block["polygon"] for block in page["blocks"]))
        check("confidence-finite-and-bounded", all(step["confidence_finite"] and step["confidence_range"]
                                                  and step["tokens"] > 0 for step in steps))
        check("no-generation-cap", not any(step["token_cap_reached"] for step in steps))
        check("markdown-nonempty", bool(page["markdown"].strip()))
        check("structured-json-roundtrip", json.loads((args.out / "document.json").read_text()) ==
              {"pages": [page]})
        check("markdown-file-exact", (args.out / "document.md").read_text(encoding="utf-8") ==
              page["markdown"] + "\n")
        report["warnings"] = page["warnings"]
        report["generated_tokens"] = page["stats"]["generated_tokens"]
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["generations_retained"] = len(steps)
        write_json(args.out / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
