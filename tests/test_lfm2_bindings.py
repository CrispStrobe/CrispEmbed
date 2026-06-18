#!/usr/bin/env python3
"""Smoke-test LFM2 embedding wiring through the public Python binding.

Uses the registry name by default. If the model is not already cached, set
CRISPEMBED_ACCEPT_LICENSE=lfm1.0 before running so the native resolver may
download the LiquidAI-licensed GGUF.
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lfm2-embed-q4k")
    parser.add_argument("--lib", default=os.environ.get("CRISPEMBED_LIB"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    models = CrispEmbed.list_models(lib_path=args.lib)
    lfm = {m["name"]: m for m in models if m["name"].startswith("lfm2-embed")}
    assert_true("lfm2-embed" in lfm, "lfm2-embed missing from registry")
    assert_true("lfm2-embed-q4k" in lfm, "lfm2-embed-q4k missing from registry")
    assert_true(lfm["lfm2-embed-q4k"]["license"] == "lfm1.0", "LFM2 license tag mismatch")
    assert_true("LiquidAI/LFM2.5-Embedding-350M" in lfm["lfm2-embed-q4k"]["model_card_url"],
                "LFM2 model card URL mismatch")

    query_prefix = CrispEmbed.query_prefix(args.model, lib_path=args.lib)
    passage_prefix = CrispEmbed.passage_prefix(args.model, lib_path=args.lib)
    assert_true(query_prefix == "query: ", "LFM2 query prefix missing")
    assert_true(passage_prefix == "document: ", "LFM2 passage prefix missing")

    model = CrispEmbed(args.model, n_threads=args.threads, lib_path=args.lib)
    texts = [
        "red planet",
        "Mars is called the red planet",
        "Jupiter is the largest planet",
    ]
    single = np.vstack([model.encode(text) for text in texts])
    batch = model.encode(texts)
    assert_true(batch.shape == (3, 1024), f"unexpected LFM2 batch shape: {batch.shape}")
    assert_true(np.max(np.abs(single - batch)) == 0.0, "batch encode differs from single encode")
    assert_true(abs(float(np.linalg.norm(batch[0])) - 1.0) < 1e-3, "embedding is not normalized")

    model.set_dim(256)
    trunc = model.encode("red planet")
    assert_true(trunc.shape == (256,), "matryoshka truncation failed")
    assert_true(abs(float(np.linalg.norm(trunc)) - 1.0) < 1e-3, "truncated embedding is not normalized")

    mars = float(np.dot(batch[0], batch[1]))
    jupiter = float(np.dot(batch[0], batch[2]))
    assert_true(mars > jupiter, "LFM2 semantic ordering failed")

    print("[python] LFM2 registry, prefixes, batch, matryoshka, and similarity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
