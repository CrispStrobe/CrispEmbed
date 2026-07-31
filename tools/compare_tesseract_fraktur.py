#!/usr/bin/env python3
"""Compare the native Fraktur page/line path with system Tesseract.

The native side is the model-gated orchestrator test, which keeps the model
and detector outside Git. The comparison deliberately uses NFC and whitespace
normalization only: compatibility normalization would erase historical long-s
(`ſ`) and would make the result less useful for Fraktur regression work.
"""

import argparse
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path


def normalize(text):
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    return " ".join(text.split())


def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def native_text(args):
    env = os.environ.copy()
    env.update(
        CRISPEMBED_FRAKTUR_DET_MODEL=args.detector,
        CRISPEMBED_FRAKTUR_MODEL=args.model,
        CRISPEMBED_FRAKTUR_IMAGE=args.image,
        CRISPEMBED_FRAKTUR_DUMP="1",
    )
    proc = subprocess.run([args.native_test], env=env, text=True, capture_output=True, check=True)
    match = re.search(r"BEGIN native Fraktur full_text\n(.*?)\n  END native Fraktur full_text", proc.stdout, re.S)
    if not match:
        raise RuntimeError("native test did not emit the Fraktur full-text markers")
    return match.group(1)


def system_text(args):
    proc = subprocess.run(
        [args.tesseract, args.image, "stdout", "--oem", "1", "--psm", str(args.psm), "-l", "frk"],
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-test", required=True, help="built test-ocr-orchestrator executable")
    parser.add_argument("--tesseract", default="tesseract")
    parser.add_argument("--psm", type=int, default=3)
    args = parser.parse_args()

    native = normalize(native_text(args))
    system = normalize(system_text(args))
    char_distance = distance(native, system)
    native_words, system_words = native.split(), system.split()
    word_distance = distance(native_words, system_words)
    result = {
        "image": str(Path(args.image)),
        "native_chars": len(native),
        "system_chars": len(system),
        "native_words": len(native_words),
        "system_words": len(system_words),
        "char_edit_distance": char_distance,
        "cer_vs_system": char_distance / max(1, len(system)),
        "word_edit_distance": word_distance,
        "wer_vs_system": word_distance / max(1, len(system_words)),
        "native_text": native,
        "system_text": system,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
