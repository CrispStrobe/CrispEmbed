#!/usr/bin/env python3
"""Run the opt-in PP-OCRv6 full graph against regenerated gold archives.

The large GGUFs and gold archives live in the external model cache, so this
lane is intentionally opt-in for CI and local artifact-equipped machines.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


FIXTURES = {
    "tiny": {
        # The legacy 16-tensor Arabic reference is not a full graph gold
        # archive. Keep the tier selectable, but require a regenerated ref
        # before accepting tiny graph parity.
        "arabic": ("tests/regression/images/cc0/arabic_printed_line.png", "PP-OCRv6_tiny_rec-arabic-ref-regenerated.gguf"),
    },
    "small": {
        "arabic": ("tests/regression/images/cc0/arabic_printed_line.png", "PP-OCRv6_small_rec-arabic-ref-regenerated.gguf"),
        "receipt": ("tests/regression/images/cc0/receipt_example.png", "PP-OCRv6_small_rec-receipt-ref-regenerated.gguf"),
        "german": ("tests/regression/images/cc0/german_official_document.jpg", "PP-OCRv6_small_rec-german-ref-regenerated.gguf"),
    },
    "medium": {
        "fox": ("tests/regression/images/fox.png", "PP-OCRv6_medium_rec-fox-ref-regenerated.gguf"),
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", type=Path, default=Path("build"))
    ap.add_argument("--model-dir", type=Path, default=Path("/Volumes/backups/ai/crispembed-gguf"))
    ap.add_argument("--ref-dir", type=Path, default=Path("/Volumes/backups/ai/crispembed-gguf"))
    ap.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
    ap.add_argument("--variants", nargs="+", choices=tuple(FIXTURES), default=["small"],
                    help="recognizer tiers to validate (default: small)")
    ap.add_argument("--require", action="store_true", help="fail instead of skipping when artifacts are absent")
    args = ap.parse_args()

    exe = args.build_dir / ("test-ppocrv6-rec" if args.backend == "cpu" else "test-ppocrv6-rec")
    missing = [exe] if not exe.exists() else []
    selected = {}
    for variant in args.variants:
        model = args.model_dir / f"PP-OCRv6_{variant}_rec-f32.gguf"
        if not model.exists():
            missing.append(model)
        selected[variant] = []
        for name, (image, reference) in FIXTURES[variant].items():
            ref = args.ref_dir / reference
            if not ref.exists():
                missing.append(ref)
            if not Path(image).exists():
                missing.append(Path(image))
            selected[variant].append((name, image, ref, model))
    if missing:
        message = "PP-OCRv6 graph gold lane skipped; missing: " + ", ".join(str(p) for p in missing)
        if args.require:
            print(message, file=sys.stderr)
            return 2
        print(message)
        return 0

    threshold = 0.9999
    for variant in args.variants:
        for name, image, ref, model in selected[variant]:
            env = os.environ.copy()
            env.update(
                PPOCRV6_REF=str(ref),
                CRISPEMBED_PPOCRV6_GRAPH="1",
                CRISPEMBED_PPOCRV6_SVTR_GRAPH="1",
                CRISPEMBED_PPOCRV6_SVTR_DECODER_GRAPH="1",
            )
            if args.backend == "cpu":
                env["CRISPEMBED_PPOCRV6_FORCE_CPU"] = "1"
            result = subprocess.run(
                [str(exe), str(model), image], env=env, text=True, capture_output=True, check=False
            )
            output = result.stdout + result.stderr
            match = re.findall(r"logits cos=([0-9.]+)", output)
            texts = re.findall(r"^text=(.*)$", output, re.MULTILINE)
            if result.returncode != 0 or not match or not texts:
                print(f"{args.backend} {variant} {name}: FAILED to run/decode\n{output}", file=sys.stderr)
                return 1
            cosine = float(match[-1])
            print(f"{args.backend} {variant} {name}: logits_cos={cosine:.6f} text={texts[-1]}")
            if cosine < threshold:
                print(f"{args.backend} {variant} {name}: cosine {cosine:.6f} < {threshold:.4f}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
