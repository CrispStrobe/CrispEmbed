#!/usr/bin/env python3
"""Smoke test for Ettin CrossEncoder reranker GGUF — validates ranking + score range.

Requires a converted GGUF and built libcrispembed.so.  Runs with any quant.

Usage:
  python tests/test_ettin_reranker_smoke.py \
      --gguf /path/to/ettin-reranker-150m-v1-q8_0.gguf \
      --lib build/libcrispembed.so
"""

import argparse
import ctypes
import os
import sys


# Reference scores from Python CrossEncoder (f32, cross-encoder/ettin-reranker-150m-v1)
PAIRS = [
    ("What is the capital of France?", "Paris is the capital of France.",      +11.85),
    ("What is the capital of France?", "Cats are popular pets.",               -4.41),
    ("How do I cook pasta?",           "Boil water, add pasta, cook 8-10 minutes.", +7.79),
    ("How do I cook pasta?",           "The stock market hit a record high today.", -5.57),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, help="Path to Ettin reranker GGUF")
    parser.add_argument("--lib", default=None, help="Path to libcrispembed shared library")
    parser.add_argument("--tol", type=float, default=1.5,
                        help="Max absolute score diff tolerated vs reference (default: 1.5)")
    args = parser.parse_args()

    if args.lib is None:
        for candidate in [
            "build/libcrispembed.so", "build/libcrispembed.dylib",
            "build/Release/crispembed.dll", "build/crispembed.dll",
        ]:
            if os.path.exists(candidate):
                args.lib = candidate
                break
    if not args.lib or not os.path.exists(args.lib):
        print("ERROR: could not find libcrispembed - pass --lib or build first")
        return 1

    lib = ctypes.CDLL(os.path.abspath(args.lib))
    lib.crispembed_init.restype = ctypes.c_void_p
    lib.crispembed_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_is_reranker.restype = ctypes.c_int
    lib.crispembed_is_reranker.argtypes = [ctypes.c_void_p]
    lib.crispembed_rerank.restype = ctypes.c_float
    lib.crispembed_rerank.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.crispembed_free.argtypes = [ctypes.c_void_p]

    ctx = lib.crispembed_init(args.gguf.encode(), 4)
    if not ctx:
        print("FAIL: crispembed_init failed")
        return 1
    if not lib.crispembed_is_reranker(ctx):
        print("FAIL: loaded model is not a reranker")
        lib.crispembed_free(ctx)
        return 1

    scores = []
    for q, d, _ in PAIRS:
        scores.append(lib.crispembed_rerank(ctx, q.encode(), d.encode()))
    lib.crispembed_free(ctx)

    n_pass = 0
    n_fail = 0

    print(f"\n{'Pair':<55s} {'CE':>8s} {'Ref':>8s} {'Diff':>6s} {'Status':>6s}")
    print("-" * 90)
    for (q, d, ref), ce in zip(PAIRS, scores):
        diff = abs(ce - ref)
        status = "PASS" if diff < args.tol else "FAIL"
        label = f"{q[:25]} / {d[:25]}"
        print(f"{label:<55s} {ce:>+8.2f} {ref:>+8.2f} {diff:>6.2f} {status:>6s}")
        if status == "PASS":
            n_pass += 1
        else:
            n_fail += 1

    # Ranking check: relevant docs must score higher than irrelevant for each query
    rank_ok = True
    if scores[0] <= scores[1]:
        print("FAIL: France/relevant should score higher than France/irrelevant")
        rank_ok = False
    if scores[2] <= scores[3]:
        print("FAIL: Pasta/relevant should score higher than Pasta/irrelevant")
        rank_ok = False

    # Score range: relevant should be positive, irrelevant should be negative
    sign_ok = True
    if scores[0] < 0:
        print("FAIL: France/relevant score should be positive")
        sign_ok = False
    if scores[1] > 0:
        print("FAIL: France/irrelevant score should be negative")
        sign_ok = False

    if rank_ok:
        n_pass += 1
        print("\nranking:      PASS (relevant > irrelevant for both queries)")
    else:
        n_fail += 1
    if sign_ok:
        n_pass += 1
        print("sign:         PASS (relevant positive, irrelevant negative)")
    else:
        n_fail += 1

    print(f"\n=== {n_pass} passed, {n_fail} failed ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
