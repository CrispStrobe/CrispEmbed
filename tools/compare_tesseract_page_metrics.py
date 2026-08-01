#!/usr/bin/env python3
"""Compare official Tesseract page metrics with CrispEmbed's Fraktur lane.

The native test binary owns the detector/recognizer setup; this tool only
normalizes the two command outputs into one JSON record. Large model paths are
passed through arguments/environment and are never written into the report.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


INFO_RE = re.compile(
    r"INFO: regions=(?P<regions>\d+) chars=(?P<chars>\d+) "
    r"confidence=(?P<confidence>[0-9.]+) stage_ms=(?P<stage_ms>[0-9.]+)"
)
NATIVE_TEXT_RE = re.compile(r"BEGIN native Fraktur full_text\n(?P<text>.*?)\n  END native Fraktur full_text", re.S)


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=900, check=False)


def official_metrics(image: Path, lang: str, psm: int) -> dict:
    proc = run(["tesseract", str(image), "stdout", "--psm", str(psm), "-l", lang, "tsv"])
    words = []
    lines = set()
    for line in proc.stdout.splitlines()[1:]:
        fields = line.split("\t", 11)
        if len(fields) < 12:
            continue
        if fields[0] == "4":
            lines.add(tuple(fields[1:5]))
            continue
        if fields[0] != "5":
            continue
        text = fields[11].strip()
        if not text:
            continue
        try:
            confidence = float(fields[10])
        except ValueError:
            continue
        words.append((text, confidence))
    return {
        "returncode": proc.returncode,
        "lines": len(lines),
        "words": len(words),
        "chars": sum(len(text) for text, _ in words),
        "mean_word_confidence": (sum(conf for _, conf in words) / len(words) / 100.0) if words else 0.0,
        "stderr": proc.stderr[-500:],
    }


def official_text(image: Path, lang: str, psm: int) -> str:
    proc = run(["tesseract", str(image), "stdout", "--psm", str(psm), "-l", lang])
    return " ".join(proc.stdout.split())


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def token_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, 1):
        current = [i]
        for j, right_token in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_token != right_token),
            ))
        previous = current
    return previous[-1]


def native_metrics(args: argparse.Namespace, image: Path) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "CRISPEMBED_FRAKTUR_DET_MODEL": str(args.det_model),
            "CRISPEMBED_FRAKTUR_MODEL": str(args.rec_model),
            "CRISPEMBED_FRAKTUR_IMAGE": str(image),
            "CRISPEMBED_TESSERACT_PAGESEG": "1",
            "CRISPEMBED_FRAKTUR_DUMP": "1",
        }
    )
    for key in (
        "CRISPEMBED_TESSERACT_PAGESEG_PROJECTION",
        "CRISPEMBED_TESSERACT_COMPONENT_PAGESEG",
        "CRISPEMBED_TESSERACT_COMPONENT_BASELINE",
    ):
        env.pop(key, None)
    if args.workers:
        env["CRISPEMBED_TESSERACT_WORKERS"] = str(args.workers)
    if args.beam:
        env["CRISPEMBED_TESSERACT_BEAM_WIDTH"] = str(args.beam)
    if args.projection:
        env["CRISPEMBED_TESSERACT_PAGESEG_PROJECTION"] = "1"
    elif args.component:
        env["CRISPEMBED_TESSERACT_COMPONENT_PAGESEG"] = "1"
    elif args.baseline:
        env["CRISPEMBED_TESSERACT_COMPONENT_BASELINE"] = "1"
    proc = run([str(args.native_test)], env)
    matches = INFO_RE.findall(proc.stdout + proc.stderr)
    if not matches:
        raise RuntimeError("native regression emitted no Fraktur INFO metrics")
    regions, chars, confidence, stage_ms = matches[-1]
    text_match = NATIVE_TEXT_RE.search(proc.stdout + proc.stderr)
    return {
        "returncode": proc.returncode,
        "regions": int(regions),
        "chars": int(chars),
        "mean_confidence": float(confidence),
        "stage_ms": float(stage_ms),
        "pageseg_policy": "projection" if args.projection else "component" if args.component else "baseline" if args.baseline else "legacy-fallback",
        "text": " ".join(text_match.group("text").split()) if text_match else "",
        "stderr": proc.stderr[-500:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--det-model", type=Path, required=True)
    parser.add_argument("--rec-model", type=Path, required=True)
    parser.add_argument("--native-test", type=Path, default=Path("build/test-ocr-orchestrator"))
    parser.add_argument("--lang", default="frk")
    parser.add_argument("--psm", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--beam", type=int, default=0)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--projection", action="store_true")
    policy.add_argument("--component", action="store_true", help="use the opt-in component prototype")
    policy.add_argument("--baseline", action="store_true", help="use the opt-in baseline-row matcher")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    official = official_metrics(args.image, args.lang, args.psm)
    reference_text = official_text(args.image, args.lang, args.psm)
    native = native_metrics(args, args.image)
    native_text = native.pop("text")
    char_denominator = max(1, len(reference_text))
    word_reference = reference_text.split()
    word_native = native_text.split()
    result = {
        "fixture": str(args.image),
        "official_tesseract": official,
        "native_crispembed": native,
        "comparison": {
            "region_delta_vs_official_lines": native["regions"] - official["lines"],
            "char_delta": native["chars"] - official["chars"],
            "confidence_delta": native["mean_confidence"] - official["mean_word_confidence"],
            "cer": edit_distance(reference_text, native_text) / char_denominator,
            "wer": token_distance(word_reference, word_native) / max(1, len(word_reference)),
        },
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0 if official["returncode"] == 0 and native["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
