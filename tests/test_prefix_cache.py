#!/usr/bin/env python3
"""C4 cross-call prefix KV cache parity test (drives the CLI binary).

The CLI encodes a `-f FILE` of texts (one per line) reusing ONE context, so
consecutive texts sharing a leading instruction prefix exercise the
cross-call prefix cache. We run the same file twice — cache ON (default) vs
OFF (CRISPEMBED_DECODER_PREFIX_CACHE=0) — and compare embeddings per line.

Covers same-prefix-twice (build+hit), prefix-change (invalidation), and
no-prefix (byte-identical full path).

Gate: min cosine >= 0.9999 (bit-equal on CPU).

Usage:
  CRISPEMBED_BIN=build/crispembed \
  CRISPEMBED_DECODER_MODEL=/tmp/jina-v5-nano-q8_0.gguf \
  python tests/test_prefix_cache.py [--gpu-backend cpu]
"""
import json
import os
import subprocess
import sys

import numpy as np

BIN = os.environ.get("CRISPEMBED_BIN", "build/crispembed")
MODEL = os.environ.get("CRISPEMBED_DECODER_MODEL")

PA = "Represent this sentence for searching relevant passages about science: "
PB = "Classify the sentiment of the following customer product review text: "

TEXTS = [
    PA + "the mitochondria is the powerhouse of the cell",           # 0 full
    PA + "photosynthesis converts sunlight into chemical energy",     # 1 build+hit
    PA + "black holes warp spacetime around their event horizon",     # 2 hit
    PB + "this blender broke after only two uses very disappointed",  # 3 prefix change -> full
    PB + "absolutely love this coffee maker best purchase ever",      # 4 build+hit (new prefix)
    "Berlin is the capital of Germany and a major cultural hub",      # 5 no prefix -> full
    "Quantum computing uses qubits instead of classical bits",        # 6 no prefix -> full
    PA + "gravity is the curvature of spacetime by mass and energy",  # 7 prefix A again -> build+hit
]


def run(cache_on, extra_args):
    fpath = "/tmp/c4_texts.txt"
    with open(fpath, "w") as f:
        f.write("\n".join(TEXTS) + "\n")
    env = dict(os.environ)
    if cache_on:
        env.pop("CRISPEMBED_DECODER_PREFIX_CACHE", None)
        env["CRISPEMBED_DECODER_PREFIX_DEBUG"] = "1"
    else:
        env["CRISPEMBED_DECODER_PREFIX_CACHE"] = "0"
    cmd = [BIN, "-m", MODEL, "-f", fpath, "--json"] + extra_args
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr[-2000:])
        raise SystemExit(f"CLI failed (rc={p.returncode})")
    data = json.loads(p.stdout)
    if isinstance(data, dict):
        data = [data]
    vecs = [np.array(o["embedding"], dtype=np.float64) for o in data]
    return np.array(vecs), p.stderr


def main():
    if not MODEL:
        print("SKIP: set CRISPEMBED_DECODER_MODEL")
        return 0
    extra = sys.argv[1:]  # e.g. --gpu-backend cpu (passed through to CLI)

    ref, _ = run(cache_on=False, extra_args=extra)
    cached, dbg = run(cache_on=True, extra_args=extra)

    n_hit = dbg.count("HIT")
    print(f"cache activity: {n_hit} hit/build lines")
    for l in dbg.splitlines():
        if "prefix_cache" in l:
            print("  " + l.strip())

    if ref.shape != cached.shape or ref.shape[0] != len(TEXTS):
        raise SystemExit(f"shape mismatch: ref={ref.shape} cached={cached.shape}")

    print(f"\n{'idx':>3} {'cos':>13} {'max_abs':>11}  text")
    min_cos, max_abs = 1.0, 0.0
    for i in range(len(TEXTS)):
        a, b = ref[i], cached[i]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        mab = float(np.max(np.abs(a - b)))
        min_cos, max_abs = min(min_cos, cos), max(max_abs, mab)
        print(f"{i:>3} {cos:>13.10f} {mab:>11.3e}  {TEXTS[i][:46]}")

    print(f"\nmin_cos = {min_cos:.10f}   max_abs = {max_abs:.3e}   hits = {n_hit}")
    # Expect 3 build/hit activations: PA group (build+hit on line 1, hit on line 2),
    # PB group (build+hit on line 4). Line 7 (PA again) falls back to full because
    # the live cache holds the PB prefix and the previous line shared no prefix.
    ok = min_cos >= 0.9999 and n_hit >= 3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
