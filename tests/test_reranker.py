#!/usr/bin/env python3
"""Validate a CrispEmbed reranker GGUF against FlagEmbedding's FlagReranker.

Usage:
  # Convert first:
  python models/convert-bert-to-gguf.py --model BAAI/bge-reranker-v2-m3 \
      --output $CRISPEMBED_CACHE_DIR/bge-reranker-v2-m3.gguf --crisp

  # Run test:
  python tests/test_reranker.py \
      --gguf $CRISPEMBED_CACHE_DIR/bge-reranker-v2-m3.gguf \
      --lib build/libcrispembed.so

  pip install FlagEmbedding
"""

import argparse
import ctypes
import glob
import os
import sys
import numpy as np

# Patch transformers torch.load safety check for models without safetensors.
_noop = lambda: None
try:
    import importlib
    for _mn in ("transformers.modeling_utils", "transformers.utils.import_utils"):
        _m = importlib.import_module(_mn)
        if hasattr(_m, "check_torch_load_is_safe"):
            setattr(_m, "check_torch_load_is_safe", _noop)
except Exception:
    pass


TEST_PAIRS = [
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of France?", "Cats are popular pets."),
    ("How do I cook pasta?", "Boil water, add pasta, cook 8-10 minutes."),
    ("How do I cook pasta?", "The stock market hit a record high today."),
    ("What year did WWII end?", "World War II ended in 1945."),
    ("What year did WWII end?", "Pineapples grow on tropical plants."),
]


def load_lib(lib_path: str):
    lib_path = os.path.abspath(lib_path)
    if sys.platform == "win32":
        dll_dir = os.path.dirname(lib_path)
        extra_dirs = [dll_dir, os.path.join(dll_dir, "bin")]
        for _v in ("v13.0", "v12.6", "v12.4", "v12.1", "v12.0", "v11.8"):
            _b = f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{_v}"
            if os.path.isdir(_b):
                extra_dirs += [_b + "/bin", _b + "/bin/x64"]
                break
        for _ds in glob.glob("C:/Windows/System32/DriverStore/FileRepository/nvdm*.inf_amd64_*/"):
            if os.path.isdir(_ds):
                extra_dirs.append(_ds)
        os.environ["PATH"] = ";".join(extra_dirs) + ";" + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            for _d in extra_dirs:
                try:
                    os.add_dll_directory(_d)
                except OSError:
                    pass
    lib = ctypes.CDLL(lib_path)
    lib.crispembed_init.restype = ctypes.c_void_p
    lib.crispembed_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_is_reranker.restype = ctypes.c_int
    lib.crispembed_is_reranker.argtypes = [ctypes.c_void_p]
    lib.crispembed_rerank.restype = ctypes.c_float
    lib.crispembed_rerank.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.crispembed_free.argtypes = [ctypes.c_void_p]
    return lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf",   required=True, help="Path to reranker GGUF")
    parser.add_argument("--lib",    default=None,  help="Path to libcrispembed shared library")
    parser.add_argument("--hf",     default="BAAI/bge-reranker-v2-m3",
                        help="HF model ID for reference")
    parser.add_argument("--tol",    type=float, default=0.5,
                        help="Max absolute score diff tolerated (default: 0.5)")
    args = parser.parse_args()

    if args.lib is None:
        for candidate in [
            "build/libcrispembed.so", "build/libcrispembed.dylib",
            "build/Release/crispembed.dll", "build/crispembed.dll",
            "build-cuda/crispembed.dll", "build-vulkan/crispembed.dll",
        ]:
            if os.path.exists(candidate):
                args.lib = candidate
                break
    if not args.lib or not os.path.exists(args.lib):
        print("ERROR: could not find libcrispembed - pass --lib or build first")
        return 1

    lib = load_lib(args.lib)
    ctx = lib.crispembed_init(args.gguf.encode(), 4)
    if not ctx:
        print("ERROR: crispembed_init failed")
        return 1

    if not lib.crispembed_is_reranker(ctx):
        print("ERROR: loaded model is not a reranker")
        return 1

    ce_scores = [lib.crispembed_rerank(ctx, q.encode(), d.encode())
                 for q, d in TEST_PAIRS]

    from FlagEmbedding import FlagReranker
    rk = FlagReranker(args.hf, use_fp16=False, device="cpu")
    hf_scores = rk.compute_score([list(p) for p in TEST_PAIRS])

    print(f"\n{'Pair':<60s} {'CE':>10s} {'HF':>10s} {'Diff':>8s} {'Status':>8s}")
    print("-" * 100)
    all_pass = True
    for (q, d), ce, hf in zip(TEST_PAIRS, ce_scores, hf_scores):
        diff = abs(ce - hf)
        status = "PASS" if diff < args.tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        label = f"{q[:30]} / {d[:30]}"
        print(f"{label:<60s} {ce:>+10.4f} {hf:>+10.4f} {diff:>8.4f} {status:>8s}")

    ce_rank = np.argsort(-np.array(ce_scores)).tolist()
    hf_rank = np.argsort(-np.array(hf_scores)).tolist()
    print(f"\nRanking (CE): {ce_rank}")
    print(f"Ranking (HF): {hf_rank}")

    lib.crispembed_free(ctx)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
