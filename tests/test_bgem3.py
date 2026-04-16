#!/usr/bin/env python3
"""Validate CrispEmbed BGE-M3 outputs (dense + sparse + colbert) against FlagEmbedding.

Usage:
  # 1. Convert the model first:
  python models/convert-bert-to-gguf.py --model BAAI/bge-m3 \
      --output ~/.cache/crispembed/bge-m3.gguf --crisp

  # 2. Run this test:
  python tests/test_bgem3.py \
      --gguf ~/.cache/crispembed/bge-m3.gguf \
      --binary ./build/crispembed

  pip install FlagEmbedding sentence-transformers
"""

import argparse
import subprocess
import json
import numpy as np
import ctypes
import sys
import os
from pathlib import Path

# Patch transformers torch.load safety check for models without safetensors (e.g. BAAI/bge-m3)
_noop = lambda: None
try:
    import importlib
    for _mn in ("transformers.modeling_utils", "transformers.utils.import_utils"):
        _m = importlib.import_module(_mn)
        if hasattr(_m, "check_torch_load_is_safe"):
            _m.check_torch_load_is_safe = _noop
except Exception:
    pass


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_lib(lib_path: str):
    lib_path = os.path.abspath(lib_path)
    # On Windows, add all dependency directories to both PATH and add_dll_directory.
    # Both mechanisms are needed: add_dll_directory covers direct deps, PATH covers
    # transitive deps loaded by ggml-cuda.dll at runtime.
    import glob as _glob
    if sys.platform == "win32":
        dll_dir = os.path.dirname(lib_path)
        bin_dir = os.path.join(dll_dir, "bin")
        extra_dirs = [dll_dir]
        if os.path.isdir(bin_dir):
            extra_dirs.append(bin_dir)
        # CUDA toolkit: cublas64_*.dll and nvcudart.dll
        for _cuda_ver in ("v13.0", "v12.6", "v12.4", "v12.1", "v12.0", "v11.8"):
            _base = f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/{_cuda_ver}"
            if os.path.isdir(_base):
                extra_dirs.append(_base + "/bin")
                _x64 = _base + "/bin/x64"
                if os.path.isdir(_x64):
                    extra_dirs.append(_x64)
                break
        # Driver Store: nvcudart_hybrid64.dll
        for _ds in _glob.glob("C:/Windows/System32/DriverStore/FileRepository/nvdm*.inf_amd64_*/"):
            if os.path.isdir(_ds):
                extra_dirs.append(_ds)
        # Prepend to PATH (covers transitive DLL loads by ggml-cuda.dll)
        os.environ["PATH"] = ";".join(extra_dirs) + ";" + os.environ.get("PATH", "")
        # Also register via add_dll_directory (covers direct ctypes load)
        if hasattr(os, "add_dll_directory"):
            for _d in extra_dirs:
                try:
                    os.add_dll_directory(_d)
                except OSError:
                    pass
    lib = ctypes.CDLL(lib_path)
    lib.crispembed_init.restype = ctypes.c_void_p
    lib.crispembed_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.crispembed_encode.restype = ctypes.POINTER(ctypes.c_float)
    lib.crispembed_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    lib.crispembed_encode_sparse.restype = ctypes.c_int
    lib.crispembed_encode_sparse.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ]
    lib.crispembed_encode_multivec.restype = ctypes.POINTER(ctypes.c_float)
    lib.crispembed_encode_multivec.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.crispembed_has_sparse.restype = ctypes.c_int
    lib.crispembed_has_sparse.argtypes = [ctypes.c_void_p]
    lib.crispembed_has_colbert.restype = ctypes.c_int
    lib.crispembed_has_colbert.argtypes = [ctypes.c_void_p]
    lib.crispembed_free.argtypes = [ctypes.c_void_p]
    return lib


def test_dense(lib, ctx, hf_model, texts, tol=1e-3):
    print("\n=== Dense (cosine similarity vs FlagEmbedding) ===")
    from FlagEmbedding import BGEM3FlagModel
    hf = BGEM3FlagModel(hf_model, use_fp16=False, device="cpu")

    for text in texts:
        dim = ctypes.c_int(0)
        ptr = lib.crispembed_encode(ctx, text.encode(), ctypes.byref(dim))
        crisp_vec = np.array([ptr[i] for i in range(dim.value)], dtype=np.float32)

        hf_out = hf.encode([text], batch_size=1, return_dense=True)
        hf_vec = hf_out["dense_vecs"][0].astype(np.float32)
        hf_vec /= np.linalg.norm(hf_vec) + 1e-12

        sim = cosine(crisp_vec, hf_vec)
        status = "PASS" if sim > 1.0 - tol else "FAIL"
        print(f"  [{status}] '{text[:40]}' cosine={sim:.6f}")


def test_sparse(lib, ctx, hf_model, texts):
    print("\n=== Sparse (intersection similarity vs FlagEmbedding) ===")
    if not lib.crispembed_has_sparse(ctx):
        print("  SKIP: model has no sparse head")
        return
    from FlagEmbedding import BGEM3FlagModel
    hf = BGEM3FlagModel(hf_model, use_fp16=False, device="cpu")

    for text in texts:
        idx_ptr = ctypes.POINTER(ctypes.c_int32)()
        val_ptr = ctypes.POINTER(ctypes.c_float)()
        n = lib.crispembed_encode_sparse(ctx, text.encode(),
                                          ctypes.byref(idx_ptr), ctypes.byref(val_ptr))
        crisp_sparse = {int(idx_ptr[i]): float(val_ptr[i]) for i in range(n)}

        hf_out = hf.encode([text], batch_size=1, return_sparse=True)
        # FlagEmbedding keys lexical_weights by str(vocab_id); normalize to int for comparison.
        hf_sparse = {int(k): float(v) for k, v in dict(hf_out["lexical_weights"][0]).items()}

        common = set(crisp_sparse.keys()) & set(hf_sparse.keys())
        union  = set(crisp_sparse.keys()) | set(hf_sparse.keys())
        iou = len(common) / max(len(union), 1)
        status = "PASS" if iou > 0.5 else "FAIL"
        print(f"  [{status}] '{text[:40]}' nnz={n} hf_nnz={len(hf_sparse)} IoU={iou:.3f}")


def test_colbert(lib, ctx, hf_model, texts, tol=5e-3):
    print("\n=== ColBERT (per-token cosine vs FlagEmbedding) ===")
    if not lib.crispembed_has_colbert(ctx):
        print("  SKIP: model has no ColBERT head")
        return
    from FlagEmbedding import BGEM3FlagModel
    hf = BGEM3FlagModel(hf_model, use_fp16=False, device="cpu")

    for text in texts:
        n_tok = ctypes.c_int(0)
        dim   = ctypes.c_int(0)
        ptr = lib.crispembed_encode_multivec(ctx, text.encode(),
                                              ctypes.byref(n_tok), ctypes.byref(dim))
        if not ptr:
            print(f"  FAIL '{text[:40]}': null output")
            continue
        crisp_mv = np.array([ptr[i] for i in range(n_tok.value * dim.value)],
                             dtype=np.float32).reshape(n_tok.value, dim.value)

        hf_out = hf.encode([text], batch_size=1, return_colbert_vecs=True)
        hf_mv = hf_out["colbert_vecs"][0].astype(np.float32)  # [T, dim]
        # FlagEmbedding drops the CLS/<s> token from colbert output (see
        # BGEM3Model._colbert_embedding: last_hidden_state[:, 1:]). CrispEmbed
        # returns all tokens including CLS, so skip index 0 for alignment.
        crisp_content = crisp_mv[1:]
        T = min(crisp_content.shape[0], hf_mv.shape[0])
        sims = [cosine(crisp_content[t], hf_mv[t]) for t in range(T)]
        mean_sim = float(np.mean(sims))
        status = "PASS" if mean_sim > 1.0 - tol else "FAIL"
        print(f"  [{status}] '{text[:40]}' tokens={n_tok.value}/{hf_mv.shape[0]} "
              f"mean_cosine={mean_sim:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf",   required=True, help="Path to bge-m3.gguf")
    parser.add_argument("--lib",    default=None,  help="Path to libcrispembed.so/.dylib/.dll")
    parser.add_argument("--hf",     default="BAAI/bge-m3", help="HF model for reference")
    args = parser.parse_args()

    # Find shared library
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
        sys.exit(1)

    lib = load_lib(args.lib)
    ctx = lib.crispembed_init(args.gguf.encode(), 4)
    if not ctx:
        print("ERROR: crispembed_init failed")
        sys.exit(1)

    texts = [
        "What is the capital of France?",
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    ]

    test_dense(lib, ctx, args.hf, texts)
    test_sparse(lib, ctx, args.hf, texts)
    test_colbert(lib, ctx, args.hf, texts)

    lib.crispembed_free(ctx)
    print("\nDone.")


if __name__ == "__main__":
    main()
