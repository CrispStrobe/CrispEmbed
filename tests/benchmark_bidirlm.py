#!/usr/bin/env python3
"""In-process BidirLM-Omni smoke benchmark and vector regression check.

This intentionally benchmarks the C API through ctypes with one model load.
It covers:
  * text embedding
  * audio embedding from f32le 16 kHz mono PCM
  * vision embedding from preprocessed float32 patch rows

Use --save-baseline before an optimization and --compare-baseline after it.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import numpy as np

from crispembed import CrispEmbed


TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming natural language processing.",
    "BidirLM-Omni embeds text, audio, and vision into one shared space.",
]


def bench(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times)), float(np.mean(times))


def encode_image_raw(model: CrispEmbed, patches: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out_dim = ctypes.c_int(0)
    ptr = model._lib.crispembed_encode_image(
        model._ctx,
        patches.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(int(patches.shape[0])),
        grid.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int(1),
        ctypes.byref(out_dim),
    )
    if not ptr or out_dim.value <= 0:
        raise RuntimeError("crispembed_encode_image failed")
    return np.ctypeslib.as_array(ptr, shape=(out_dim.value,)).copy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def env_default(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    return fallback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lib", default=env_default("CRISPEMBED_LIB"))
    ap.add_argument("--audio", default=env_default("CRISPEMBED_BIDIRLM_AUDIO"))
    ap.add_argument("--patches", default=env_default("CRISPEMBED_BIDIRLM_PATCHES"))
    ap.add_argument("--grid-thw", default=env_default("CRISPEMBED_BIDIRLM_GRID_THW", "1,2,2"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--save-baseline")
    ap.add_argument("--compare-baseline")
    ap.add_argument("--min-cos", type=float, default=0.99999)
    ap.add_argument("--max-diff", type=float, default=1e-3)
    args = ap.parse_args()

    os.environ.setdefault("CRISPEMBED_FORCE_CPU", "1")

    if not args.lib:
        raise RuntimeError("set --lib or CRISPEMBED_LIB")
    if not args.audio:
        raise RuntimeError("set --audio or CRISPEMBED_BIDIRLM_AUDIO")
    if not args.patches:
        raise RuntimeError("set --patches or CRISPEMBED_BIDIRLM_PATCHES")

    model = CrispEmbed(args.model, n_threads=args.threads, lib_path=args.lib)
    audio = np.fromfile(args.audio, dtype=np.float32)
    patches = np.fromfile(args.patches, dtype=np.float32)
    if patches.size % 1536 != 0:
        raise RuntimeError(f"{args.patches} is not a multiple of 1536 float32 values")
    patches = np.ascontiguousarray(patches.reshape(-1, 1536), dtype=np.float32)
    grid_vals = [int(x) for x in args.grid_thw.split(",")]
    if len(grid_vals) != 3:
        raise RuntimeError("--grid-thw must be T,H,W")
    grid = np.ascontiguousarray([grid_vals], dtype=np.int32)
    if int(np.prod(grid)) != int(patches.shape[0]):
        raise RuntimeError(f"grid {grid_vals} implies {int(np.prod(grid))} patches, got {patches.shape[0]}")

    vectors = {
        "text": np.ascontiguousarray(model.encode(TEXTS), dtype=np.float32),
        "audio": np.ascontiguousarray(model.encode_audio(audio, sr=16000), dtype=np.float32),
        "vision": np.ascontiguousarray(encode_image_raw(model, patches, grid), dtype=np.float32),
    }
    if vectors["audio"].size == 0:
        raise RuntimeError("audio vector is empty")
    if vectors["vision"].size == 0:
        raise RuntimeError("vision vector is empty")

    results = {}
    results["text_batch"] = bench(lambda: model.encode(TEXTS), warmup=args.warmup, iters=args.iters)
    results["audio"] = bench(lambda: model.encode_audio(audio, sr=16000), warmup=args.warmup, iters=args.iters)
    results["vision_raw"] = bench(lambda: encode_image_raw(model, patches, grid), warmup=args.warmup, iters=args.iters)

    summary = {
        "model": args.model,
        "threads": args.threads,
        "iters": args.iters,
        "grid_thw": grid_vals,
        "n_patches": int(patches.shape[0]),
        "dims": {k: list(v.shape) for k, v in vectors.items()},
        "bench_ms": {k: {"median": v[0], "mean": v[1]} for k, v in results.items()},
    }

    if args.save_baseline:
        np.savez(args.save_baseline, **vectors)
        summary["saved_baseline"] = args.save_baseline

    if args.compare_baseline:
        base = np.load(args.compare_baseline)
        checks = {}
        ok = True
        for key, cur in vectors.items():
            ref = base[key]
            if ref.shape != cur.shape:
                checks[key] = {"ok": False, "error": f"shape {cur.shape} != {ref.shape}"}
                ok = False
                continue
            if key == "text":
                cos_vals = [cosine(ref[i], cur[i]) for i in range(ref.shape[0])]
                max_abs = float(np.max(np.abs(ref - cur)))
                min_cos = float(np.min(cos_vals))
            else:
                min_cos = cosine(ref, cur)
                max_abs = float(np.max(np.abs(ref - cur)))
            passed = min_cos >= args.min_cos and max_abs <= args.max_diff
            checks[key] = {"ok": passed, "min_cos": min_cos, "max_abs_diff": max_abs}
            ok = ok and passed
        summary["regression"] = checks
        print(json.dumps(summary, indent=2))
        return 0 if ok else 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
