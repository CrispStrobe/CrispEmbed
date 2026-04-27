#!/usr/bin/env python3
"""Exercise the CrispEmbed CLI across all inference capabilities.

Usage:
  python tests/test_cli_parity.py --cli build-cuda/crispembed.exe \
      --dense-model "$CRISPEMBED_DENSE_MODEL" \
      --retrieval-model "$CRISPEMBED_RETRIEVAL_MODEL" \
      --reranker-model "$CRISPEMBED_RERANKER_MODEL"
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def run_cli(cli: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [cli, *args],
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )


def require_success(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise AssertionError(
            f"{label} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def parse_json_output(result: subprocess.CompletedProcess[str], label: str):
    require_success(result, label)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{label} did not return valid JSON:\n{result.stdout}") from exc


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def check_dense(cli: str, model_path: str) -> None:
    print(f"[cli] dense model: {model_path}")

    dense = parse_json_output(
        run_cli(cli, ["-m", model_path, "--json", "Hello world"]),
        "dense encode",
    )
    assert_true(isinstance(dense, list) and len(dense) == 1, "dense encode returned unexpected JSON shape")
    vec = dense[0]["embedding"]
    assert_true(len(vec) > 0, "dense encode returned empty embedding")
    assert_true(abs(l2_norm(vec) - 1.0) < 1e-3, "dense embedding is not normalized")

    batch_texts = [
        "query: crisp embeddings are fast",
        "dense retrieval with ggml",
        "batch inference should preserve order",
    ]
    batch = parse_json_output(
        run_cli(cli, ["-m", model_path, "--json", *batch_texts]),
        "batch encode",
    )
    assert_true(len(batch) == len(batch_texts), "batch encode returned wrong result count")
    assert_true(all(len(item["embedding"]) == len(vec) for item in batch), "batch encode changed embedding dimension")

    dim = min(128, len(vec))
    trunc = parse_json_output(
        run_cli(cli, ["-m", model_path, "-d", str(dim), "--json", "Hello world"]),
        "matryoshka encode",
    )
    trunc_vec = trunc[0]["embedding"]
    assert_true(len(trunc_vec) == dim, "matryoshka encode returned wrong dimension")
    assert_true(abs(l2_norm(trunc_vec) - 1.0) < 1e-3, "truncated embedding is not normalized")

    caps = parse_json_output(
        run_cli(cli, ["-m", model_path, "--prefix", "query: ", "--json", "--capabilities"]),
        "capabilities",
    )
    assert_true(caps["prefix"] == "query: ", "capabilities did not report prefix")
    assert_true(caps["dim"] == len(vec), "capabilities reported wrong dimension")

    prefixed = parse_json_output(
        run_cli(cli, ["-m", model_path, "--prefix", "query: ", "--json", "hello"]),
        "prefixed encode",
    )[0]["embedding"]
    cleared = parse_json_output(
        run_cli(cli, ["-m", model_path, "--json", "hello"]),
        "plain encode",
    )[0]["embedding"]
    assert_true(prefixed != cleared, "prefix had no effect on embedding output")

    docs = [
        "Paris France capital city and Eiffel Tower.",
        "A bicycle uses two wheels and a chain.",
        "Berlin Germany capital city and Brandenburg Gate.",
    ]
    biencoder = parse_json_output(
        run_cli(
            cli,
            ["-m", model_path, "--json", "--biencoder", "paris france capital", "--top-n", "2", *docs],
        ),
        "bi-encoder rerank",
    )
    results = biencoder["results"]
    assert_true(len(results) == 2, "bi-encoder rerank did not honor top-n")
    assert_true(results[0]["index"] == 0, "bi-encoder rerank did not rank the relevant document first")
    assert_true(results[0]["score"] >= results[1]["score"], "bi-encoder rerank results are not sorted")

    print("[cli] dense, capabilities, matryoshka, prefix, and bi-encoder rerank: PASS")


def check_retrieval(cli: str, model_path: str) -> None:
    print(f"[cli] retrieval model: {model_path}")

    caps = parse_json_output(
        run_cli(cli, ["-m", model_path, "--json", "--capabilities"]),
        "retrieval capabilities",
    )
    assert_true(caps["has_sparse"] is True, "retrieval model does not report sparse support")
    assert_true(caps["has_colbert"] is True, "retrieval model does not report colbert support")

    sparse = parse_json_output(
        run_cli(cli, ["-m", model_path, "--json", "--sparse", "Paris is the capital of France."]),
        "sparse encode",
    )
    sparse_entries = sparse[0]["sparse"]
    assert_true(len(sparse_entries) > 0, "sparse encode returned no entries")
    assert_true(all(item["weight"] > 0.0 for item in sparse_entries), "sparse encode returned non-positive weights")

    colbert = parse_json_output(
        run_cli(cli, ["-m", model_path, "--json", "--colbert", "Paris is the capital of France."]),
        "colbert encode",
    )
    entry = colbert[0]
    assert_true(entry["n_tokens"] > 0, "colbert encode returned no token vectors")
    assert_true(entry["dim"] > 0, "colbert encode returned zero-width vectors")
    assert_true(len(entry["vectors"]) == entry["n_tokens"], "colbert token count mismatch")
    assert_true(all(len(token) == entry["dim"] for token in entry["vectors"]), "colbert vector width mismatch")

    print("[cli] sparse and colbert retrieval: PASS")


def check_reranker(cli: str, model_path: str) -> None:
    print(f"[cli] reranker model: {model_path}")

    caps = parse_json_output(
        run_cli(cli, ["-m", model_path, "--json", "--capabilities"]),
        "reranker capabilities",
    )
    assert_true(caps["is_reranker"] is True, "reranker model does not report reranker support")

    docs = [
        "Paris is the capital of France.",
        "Bicycles have handlebars and pedals.",
    ]
    reranked = parse_json_output(
        run_cli(cli, ["-m", model_path, "--json", "--rerank", "capital of france", *docs]),
        "cross-encoder rerank",
    )
    results = reranked["results"]
    assert_true(len(results) == len(docs), "cross-encoder rerank returned wrong result count")
    assert_true(results[0]["index"] == 0, "cross-encoder rerank did not rank the relevant document first")
    assert_true(results[0]["score"] >= results[1]["score"], "cross-encoder rerank results are not sorted")

    print("[cli] cross-encoder rerank: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, help="Path to crispembed CLI binary")
    parser.add_argument("--dense-model", required=True, help="GGUF path for a dense embedding model")
    parser.add_argument("--retrieval-model", help="GGUF path for a sparse+ColBERT model such as bge-m3")
    parser.add_argument("--reranker-model", help="GGUF path for a reranker model")
    args = parser.parse_args()

    cli_path = Path(args.cli)
    if not cli_path.exists():
        raise FileNotFoundError(f"CLI binary not found: {cli_path}")

    check_dense(str(cli_path), args.dense_model)

    if args.retrieval_model:
        check_retrieval(str(cli_path), args.retrieval_model)
    else:
        print("[cli] sparse and colbert retrieval: SKIP (no --retrieval-model)")

    if args.reranker_model:
        check_reranker(str(cli_path), args.reranker_model)
    else:
        print("[cli] cross-encoder rerank: SKIP (no --reranker-model)")

    print("[cli] parity script completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
