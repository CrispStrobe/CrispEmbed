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
    "arabic": "tests/regression/images/cc0/arabic_printed_line.png",
    "receipt": "tests/regression/images/cc0/receipt_example.png",
    "german": "tests/regression/images/cc0/german_official_document.jpg",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", type=Path, default=Path("build"))
    ap.add_argument("--model-dir", type=Path, default=Path("/Volumes/backups/ai/crispembed-gguf"))
    ap.add_argument("--ref-dir", type=Path, default=Path("/Volumes/backups/ai/crispembed-gguf"))
    ap.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
    ap.add_argument("--require", action="store_true", help="fail instead of skipping when artifacts are absent")
    args = ap.parse_args()

    exe = args.build_dir / ("test-ppocrv6-rec" if args.backend == "cpu" else "test-ppocrv6-rec")
    model = args.model_dir / "PP-OCRv6_small_rec-f32.gguf"
    missing = [p for p in (exe, model) if not p.exists()]
    missing += [
        args.ref_dir / f"PP-OCRv6_small_rec-{name}-ref-regenerated.gguf"
        for name in FIXTURES
        if not (args.ref_dir / f"PP-OCRv6_small_rec-{name}-ref-regenerated.gguf").exists()
    ]
    missing += [Path(path) for path in FIXTURES.values() if not Path(path).exists()]
    if missing:
        message = "PP-OCRv6 graph gold lane skipped; missing: " + ", ".join(str(p) for p in missing)
        if args.require:
            print(message, file=sys.stderr)
            return 2
        print(message)
        return 0

    threshold = 0.9999
    for name, image in FIXTURES.items():
        env = os.environ.copy()
        env.update(
            PPOCRV6_REF=str(args.ref_dir / f"PP-OCRv6_small_rec-{name}-ref-regenerated.gguf"),
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
            print(f"{args.backend} {name}: FAILED to run/decode\n{output}", file=sys.stderr)
            return 1
        cosine = float(match[-1])
        print(f"{args.backend} {name}: logits_cos={cosine:.6f} text={texts[-1]}")
        if cosine < threshold:
            print(f"{args.backend} {name}: cosine {cosine:.6f} < {threshold:.4f}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
