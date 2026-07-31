#!/usr/bin/env python3
"""Compare NumPy/PyTorch reference logits with the native PP-LCNet harness."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--native", required=True, type=Path)
    ap.add_argument("--image", type=Path)
    args = ap.parse_args()
    ref_cmd = [sys.executable, str(Path(__file__).with_name("pplcnet_orientation_reference.py")), str(args.model_dir)]
    native_cmd = [str(args.native), str(args.gguf)]
    if args.image:
        ref_cmd.append(str(args.image)); native_cmd.append(str(args.image))
    ref = subprocess.check_output(ref_cmd, text=True)
    native = subprocess.check_output(native_cmd, text=True)
    rm = re.search(r"logits=([-+0-9.eE]+),([-+0-9.eE]+)", ref)
    nm = re.search(r"logit0=([-+0-9.eE]+) logit180=([-+0-9.eE]+)", native)
    if not rm or not nm:
        raise SystemExit("could not parse reference/native logits")
    r = [float(rm.group(1)), float(rm.group(2))]
    n = [float(nm.group(1)), float(nm.group(2))]
    dot = sum(a * b for a, b in zip(r, n))
    cosine = dot / ((sum(a * a for a in r) * sum(b * b for b in n)) ** 0.5)
    max_abs = max(abs(a - b) for a, b in zip(r, n))
    print(f"pplcnet-diff cosine={cosine:.9f} max_abs={max_abs:.9g} reference={r} native={n}")
    if cosine < 0.999 or max_abs > 0.1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
