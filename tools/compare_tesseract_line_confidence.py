#!/usr/bin/env python3
"""Compare official and native Tesseract confidence on one line fixture."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


NATIVE_RE = re.compile(r"text: '(?P<text>.*?)' \((?P<chars>\d+) chars\) char_conf=(?P<count>\d+) "
                       r"sequence_conf=(?P<confidence>[0-9.]+) word_conf=(?P<word_confidence>[0-9.]+)")


def run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, env=env, timeout=900, check=False)


def official(image: Path, lang: str, psm: int) -> dict:
    proc = run(["tesseract", str(image), "stdout", "--psm", str(psm), "-l", lang, "tsv"])
    words = []
    for raw in proc.stdout.splitlines()[1:]:
        fields = raw.split("\t", 11)
        if len(fields) < 12 or fields[0] != "5" or not fields[11].strip():
            continue
        try:
            confidence = float(fields[10]) / 100.0
        except ValueError:
            continue
        words.append((fields[11].strip(), confidence))
    return {
        "returncode": proc.returncode,
        "text": " ".join(text for text, _ in words),
        "words": len(words),
        "mean_word_confidence": sum(conf for _, conf in words) / len(words) if words else 0.0,
        "stderr": proc.stderr[-500:],
    }


def native(args: argparse.Namespace, beam: int) -> dict:
    env = os.environ.copy()
    if beam:
        env["CRISPEMBED_TESSERACT_BEAM_WIDTH"] = str(beam)
    else:
        env.pop("CRISPEMBED_TESSERACT_BEAM_WIDTH", None)
        env.pop("CRISPEMBED_TESSERACT_RECODE_BEAM_WIDTH", None)
    proc = run([str(args.native_test), "--tesseract-image", str(args.model), str(args.image)], env)
    matches = NATIVE_RE.findall(proc.stdout + proc.stderr)
    if not matches:
        raise RuntimeError("native confidence test emitted no result")
    text, chars, count, confidence, word_confidence = matches[-1]
    return {
        "returncode": proc.returncode,
        "text": text,
        "chars": int(chars),
        "char_confidences": int(count),
        "sequence_confidence": float(confidence),
        "word_confidence": float(word_confidence),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-test", type=Path, default=Path("build/test-confidence"))
    parser.add_argument("--lang", default="frk")
    parser.add_argument("--psm", type=int, default=7)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = official(args.image, args.lang, args.psm)
    greedy = native(args, 0)
    beam = native(args, args.beam) if args.beam > 1 else None
    result = {
        "fixture": str(args.image),
        "official_tesseract": reference,
        "native_greedy": greedy,
        "native_beam": beam,
        "comparison": {
            "greedy_text_matches": greedy["text"] == reference["text"],
            "greedy_confidence_delta": greedy["sequence_confidence"] - reference["mean_word_confidence"],
            "greedy_word_confidence_delta": greedy["word_confidence"] - reference["mean_word_confidence"],
            "beam_text_matches": beam["text"] == reference["text"] if beam else None,
            "beam_confidence_delta": beam["sequence_confidence"] - reference["mean_word_confidence"] if beam else None,
        },
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0 if reference["returncode"] == 0 and greedy["returncode"] == 0 and (not beam or beam["returncode"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
