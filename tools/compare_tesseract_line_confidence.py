#!/usr/bin/env python3
"""Compare official and native Tesseract confidence on one line fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


NATIVE_RE = re.compile(r"text: '(?P<text>.*?)' \((?P<chars>\d+) chars\) char_conf=(?P<count>\d+) "
                       r"char_min=(?P<char_min>[0-9.]+) char_mean=(?P<char_mean>[0-9.]+) "
                       r"sequence_conf=(?P<confidence>[0-9.]+) word_conf=(?P<word_confidence>[0-9.]+)")


def run(command: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    # Tesseract's diagnostics can contain locale- or image-derived bytes that
    # are not valid UTF-8. OCR text remains parsed from stdout; replacement
    # decoding keeps stderr useful without allowing the benchmark harness to
    # fail before it records metrics.
    return subprocess.run(command, text=True, errors="replace", capture_output=True, env=env, timeout=900, check=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official(image: Path, lang: str, psm: int, tessdata_dir: Path | None) -> dict:
    started = time.perf_counter()
    command = ["tesseract", str(image), "stdout", "--psm", str(psm), "-l", lang]
    if tessdata_dir is not None:
        command.extend(["--tessdata-dir", str(tessdata_dir)])
    command.append("tsv")
    env = os.environ.copy()
    # A stale TESSDATA_PREFIX can override Homebrew's valid tessdata path and
    # make Tesseract treat the PNG payload as a filename. Explicit CLI
    # tessdata selection must be independent of the caller's shell state.
    env.pop("TESSDATA_PREFIX", None)
    proc = run(command, env=env)
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
    confidences = sorted(confidence for _, confidence in words)
    return {
        "returncode": proc.returncode,
        "text": " ".join(text for text, _ in words),
        "words": len(words),
        "mean_word_confidence": sum(conf for _, conf in words) / len(words) if words else 0.0,
        "min_word_confidence": confidences[0] if confidences else 0.0,
        "median_word_confidence": confidences[len(confidences) // 2] if confidences else 0.0,
        "max_word_confidence": confidences[-1] if confidences else 0.0,
        "stderr": proc.stderr[-500:],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def native(args: argparse.Namespace, beam: int) -> dict:
    started = time.perf_counter()
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
    text, chars, count, char_min, char_mean, confidence, word_confidence = matches[-1]
    return {
        "returncode": proc.returncode,
        "text": text,
        "chars": int(chars),
        "char_confidences": int(count),
        "char_confidence_min": float(char_min),
        "char_confidence_mean": float(char_mean),
        "sequence_confidence": float(confidence),
        "word_confidence": float(word_confidence),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def confidence_acceptance_checks(args: argparse.Namespace, reference: dict, greedy: dict, beam: dict | None) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if getattr(args, "require_official_words", False):
        checks["official_words_present"] = reference.get("words", 0) > 0
    if args.max_greedy_word_confidence_delta is not None:
        checks["max_greedy_word_confidence_delta"] = (
            abs(greedy["word_confidence"] - reference["mean_word_confidence"])
            <= args.max_greedy_word_confidence_delta
        )
    if args.require_beam_sequence_only:
        checks["beam_sequence_only"] = beam is not None and beam["char_confidences"] == 0
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--native-test", type=Path, default=Path("build/test-confidence"))
    parser.add_argument("--lang", default="frk")
    parser.add_argument("--psm", type=int, default=7)
    parser.add_argument("--tessdata-dir", type=Path,
                        help="explicit Tesseract tessdata directory for the official subprocess")
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--max-greedy-word-confidence-delta", type=float,
                        help="fail if absolute greedy word-confidence delta exceeds this value")
    parser.add_argument("--require-beam-sequence-only", action="store_true",
                        help="fail unless beam output has no fabricated per-character confidences")
    parser.add_argument("--require-official-words", action="store_true",
                        help="fail if the official TSV reference contains no recognized words")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = official(args.image, args.lang, args.psm, args.tessdata_dir)
    greedy = native(args, 0)
    beam = native(args, args.beam) if args.beam > 1 else None
    checks = confidence_acceptance_checks(args, reference, greedy, beam)
    result = {
        "fixture": str(args.image),
        "provenance": {
            "recognizer_model_sha256": sha256_file(args.model),
            "confidence_reference": "official-tsv-level5-word-confidence-vs-native-recognizer-contract",
            "tessdata_dir": str(args.tessdata_dir) if args.tessdata_dir else None,
        },
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
        "acceptance": {"passed": all(checks.values()) if checks else None, "checks": checks},
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0 if reference["returncode"] == 0 and greedy["returncode"] == 0 and (not beam or beam["returncode"] == 0) and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
