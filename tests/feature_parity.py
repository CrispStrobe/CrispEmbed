#!/usr/bin/env python3
"""Exercise the Python wrapper across all CrispEmbed capabilities.

Usage:
  python tests/feature_parity.py --dense-model /path/to/model.gguf
  python tests/feature_parity.py --dense-model /path/to/model.gguf \
      --retrieval-model /path/to/bge-m3.gguf \
      --reranker-model /path/to/bge-reranker.gguf

Environment:
  CRISPEMBED_LIB=/path/to/libcrispembed.so|.dylib|.dll
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from crispembed import CrispEmbed  # noqa: E402


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def l2_norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def check_dense(model_path: str, lib_path: str | None, n_threads: int) -> None:
    print(f"[python] dense model: {model_path}")
    model = CrispEmbed(model_path, n_threads=n_threads, lib_path=lib_path)

    vec = model.encode("Hello world")
    assert_true(vec.ndim == 1 and vec.shape[0] > 0, "single encode returned invalid shape")
    assert_true(abs(l2_norm(vec) - 1.0) < 1e-3, "single encode is not normalized")

    texts = [
        "query: crisp embeddings are fast",
        "dense retrieval with ggml",
        "batch inference should preserve order",
    ]
    batch = model.encode(texts)
    assert_true(batch.shape == (len(texts), vec.shape[0]), "batch encode returned unexpected shape")
    assert_true(np.allclose(batch[0], model.encode(texts[0]), atol=1e-5), "batch encode disagrees with single encode")

    base_dim = vec.shape[0]
    trunc_dim = min(128, base_dim)
    model.set_dim(trunc_dim)
    vec_trunc = model.encode("Hello world")
    assert_true(vec_trunc.shape == (trunc_dim,), "set_dim did not truncate embedding")
    assert_true(abs(l2_norm(vec_trunc) - 1.0) < 1e-3, "truncated embedding is not normalized")
    model.set_dim(0)
    assert_true(model.encode("Hello world").shape == (base_dim,), "set_dim(0) did not restore native dim")

    original = model.prefix
    model.set_prefix("query: ")
    assert_true(model.prefix == "query: ", "prefix getter did not reflect configured prefix")
    prefixed = model.encode("hello")
    model.set_prefix("")
    cleared = model.encode("hello")
    assert_true(model.prefix == "", "prefix was not cleared")
    assert_true(prefixed.shape == cleared.shape, "prefix changed output dimension unexpectedly")
    assert_true(not np.allclose(prefixed, cleared, atol=1e-6), "prefix had no effect on embedding output")
    model.set_prefix(original)

    docs = [
        "Paris France capital city and Eiffel Tower.",
        "A bicycle uses two wheels and a chain.",
        "Berlin Germany capital city and Brandenburg Gate.",
    ]
    ranked = model.rerank_biencoder("paris france capital", docs, top_n=2)
    assert_true(len(ranked) == 2, "rerank_biencoder top_n was not applied")
    assert_true(ranked[0]["index"] == 0, "rerank_biencoder did not rank the relevant document first")
    assert_true(ranked[0]["score"] >= ranked[1]["score"], "rerank_biencoder results are not sorted")

    print("[python] dense, batch, matryoshka, prefix, and bi-encoder rerank: PASS")


def check_retrieval(model_path: str, lib_path: str | None, n_threads: int) -> None:
    print(f"[python] retrieval model: {model_path}")
    model = CrispEmbed(model_path, n_threads=n_threads, lib_path=lib_path)

    assert_true(model.has_sparse, "retrieval model does not report sparse support")
    sparse = model.encode_sparse("Paris is the capital of France.")
    assert_true(len(sparse) > 0, "encode_sparse returned no entries")
    assert_true(all(weight > 0.0 for weight in sparse.values()), "encode_sparse returned non-positive weights")

    assert_true(model.has_colbert, "retrieval model does not report colbert support")
    multi = model.encode_multivec("Paris is the capital of France.")
    assert_true(multi.ndim == 2 and multi.shape[0] > 0 and multi.shape[1] > 0, "encode_multivec returned invalid shape")
    token_norms = np.linalg.norm(multi, axis=1)
    assert_true(np.allclose(token_norms, 1.0, atol=5e-3), "encode_multivec token vectors are not normalized")

    print("[python] sparse and colbert retrieval: PASS")


def check_reranker(model_path: str, lib_path: str | None, n_threads: int) -> None:
    print(f"[python] reranker model: {model_path}")
    model = CrispEmbed(model_path, n_threads=n_threads, lib_path=lib_path)

    assert_true(model.is_reranker, "reranker model does not report reranker support")
    positive = model.rerank("capital of france", "Paris is the capital of France.")
    negative = model.rerank("capital of france", "Bicycles have handlebars and pedals.")
    assert_true(math.isfinite(positive) and math.isfinite(negative), "rerank returned non-finite score")
    assert_true(positive > negative, "reranker failed to score the relevant document higher")

    print("[python] cross-encoder rerank: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-model", required=True, help="GGUF path for a dense embedding model")
    parser.add_argument("--retrieval-model", help="GGUF path for a sparse+ColBERT model such as bge-m3")
    parser.add_argument("--reranker-model", help="GGUF path for a reranker model")
    parser.add_argument("--lib", default=os.environ.get("CRISPEMBED_LIB"), help="Path to libcrispembed shared library")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads to request")
    args = parser.parse_args()

    check_dense(args.dense_model, args.lib, args.threads)

    if args.retrieval_model:
        check_retrieval(args.retrieval_model, args.lib, args.threads)
    else:
        print("[python] sparse and colbert retrieval: SKIP (no --retrieval-model)")

    if args.reranker_model:
        check_reranker(args.reranker_model, args.lib, args.threads)
    else:
        print("[python] cross-encoder rerank: SKIP (no --reranker-model)")

    print("[python] feature parity script completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
