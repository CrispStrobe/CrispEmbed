#!/usr/bin/env python3
"""Compare official Tesseract page metrics with CrispEmbed's Fraktur lane.

The native test binary owns the detector/recognizer setup; this tool only
normalizes the two command outputs into one JSON record. Large model paths are
passed through arguments/environment and are never written into the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


INFO_RE = re.compile(
    r"INFO: regions=(?P<regions>\d+) chars=(?P<chars>\d+) "
    r"confidence=(?P<confidence>[0-9.]+) stage_ms=(?P<stage_ms>[0-9.]+)"
)
BENCH_RE = re.compile(
    r"\[tesseract-stage-bench\] detect=(?P<detect>[0-9.]+) ms group=(?P<group>[0-9.]+) ms "
    r"crop=(?P<crop>[0-9.]+) ms recognize=(?P<recognize>[0-9.]+) ms total=(?P<total>[0-9.]+) ms "
    r"boxes=(?P<boxes>\d+) lines=(?P<lines>\d+)"
)
NATIVE_TEXT_RE = re.compile(r"BEGIN native Fraktur full_text\n(?P<text>.*?)\n  END native Fraktur full_text", re.S)


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=900, check=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_metrics(image: Path, lang: str, psm: int) -> dict:
    started = time.perf_counter()
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
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
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


def selected_pageseg_policy(args: argparse.Namespace) -> str:
    if args.projection:
        return "projection"
    if args.component:
        return "component"
    if args.baseline:
        return "baseline"
    return "legacy-fallback"


def native_metrics(args: argparse.Namespace, image: Path) -> dict:
    started = time.perf_counter()
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
    if args.benchmark:
        env["CRISPEMBED_OCR_ORCH_BENCH"] = "1"
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
    bench_matches = BENCH_RE.findall(proc.stdout + proc.stderr)
    benchmark = None
    if bench_matches:
        detect, group, crop, recognize, total, boxes, lines = bench_matches[-1]
        benchmark = {
            "detect_ms": float(detect),
            "group_ms": float(group),
            "crop_ms": float(crop),
            "recognize_ms": float(recognize),
            "total_ms": float(total),
            "boxes": int(boxes),
            "lines": int(lines),
        }
    return {
        "returncode": proc.returncode,
        "regions": int(regions),
        "chars": int(chars),
        "mean_confidence": float(confidence),
        "stage_ms": float(stage_ms),
        "pageseg_policy": selected_pageseg_policy(args),
        "text": " ".join(text_match.group("text").split()) if text_match else "",
        "stderr": proc.stderr[-500:],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "benchmark": benchmark,
    }


def acceptance_checks(args: argparse.Namespace, native: dict, comparison: dict) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if args.min_native_regions is not None:
        checks["min_native_regions"] = native["regions"] >= args.min_native_regions
    if args.max_cer is not None:
        checks["max_cer"] = comparison["cer"] <= args.max_cer
    if args.max_wer is not None:
        checks["max_wer"] = comparison["wer"] <= args.max_wer
    return checks


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
    parser.add_argument("--benchmark", action="store_true", help="include native detect/group/crop/recognize timings")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--projection", action="store_true")
    policy.add_argument("--component", action="store_true", help="use the opt-in component prototype")
    policy.add_argument("--baseline", action="store_true", help="use the opt-in baseline-row matcher")
    parser.add_argument("--min-native-regions", type=int, help="fail if native region count is below this value")
    parser.add_argument("--max-cer", type=float, help="fail if character error rate exceeds this value")
    parser.add_argument("--max-wer", type=float, help="fail if word error rate exceeds this value")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    official = official_metrics(args.image, args.lang, args.psm)
    reference_text = official_text(args.image, args.lang, args.psm)
    official["text"] = reference_text
    native = native_metrics(args, args.image)
    native_text = native["text"]
    char_denominator = max(1, len(reference_text))
    word_reference = reference_text.split()
    word_native = native_text.split()
    comparison = {
        "region_delta_vs_official_lines": native["regions"] - official["lines"],
        "char_delta": native["chars"] - official["chars"],
        "confidence_delta": native["mean_confidence"] - official["mean_word_confidence"],
        "cer": edit_distance(reference_text, native_text) / char_denominator,
        "wer": token_distance(word_reference, word_native) / max(1, len(word_reference)),
    }
    checks = acceptance_checks(args, native, comparison)
    result = {
        "fixture": str(args.image),
        "provenance": {
            "detector_model_sha256": sha256_file(args.det_model),
            "recognizer_model_sha256": sha256_file(args.rec_model),
            "ordering": "official-tsv-level4-vs-native-reading-order-index",
        },
        "official_tesseract": official,
        "native_crispembed": native,
        "comparison": comparison,
        "acceptance": {"passed": all(checks.values()) if checks else None, "checks": checks},
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
    print(serialized, end="")
    return 0 if official["returncode"] == 0 and native["returncode"] == 0 and all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
