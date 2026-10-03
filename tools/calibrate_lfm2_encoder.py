#!/usr/bin/env python3
"""Collect a fresh LFM2 encoder importance matrix from a separate JSONL corpus.

Each corpus record must contain a "text" string. Individual sentences and
longer groups are both evaluated; no held-out evaluation text is added.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from crispembed import CrispEmbed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--extra-corpus", action="append", type=Path, default=[],
                    help="Additional disjoint JSONL text corpus; repeatable")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--lib", default="build/libcrispembed.so")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=8)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Use a fresh output path: the native collector merges existing statistics")
    if args.threads < 1 or args.group_size < 1:
        ap.error("threads and group-size must be positive")
    sources = [args.corpus, *args.extra_corpus]
    corpora = [p.read_bytes() for p in sources]
    texts = [json.loads(line)["text"] for corpus in corpora
             for line in corpus.decode("utf-8").splitlines() if line.strip()]
    if not texts or any(not isinstance(t, str) for t in texts):
        ap.error("Corpus must contain JSONL records with text strings")
    samples = list(texts)
    if args.group_size > 1:
        samples += ["\n".join(texts[i:i + args.group_size]) for i in range(0, len(texts), args.group_size)]
    os.environ["CRISPEMBED_LFM2_ENCODER"] = "1"
    os.environ["CRISPEMBED_IMATRIX_OUT"] = str(args.output.resolve())
    model = CrispEmbed(args.model, n_threads=args.threads, lib_path=args.lib)
    tokens = max_tokens = 0
    try:
        if not model.has_masked_lm:
            raise ValueError("Expected an enabled LFM2 masked encoder")
        for i, text in enumerate(samples):
            ids, _ = model.encode_tokens(text, normalize=False)
            tokens += len(ids)
            max_tokens = max(max_tokens, len(ids))
            if (i + 1) % 20 == 0:
                print(f"Calibrated {i + 1}/{len(samples)} samples", flush=True)
    finally:
        # crispembed_free flushes the collector; wrappers have copied the arrays.
        del model
    if not args.output.exists() or args.output.stat().st_size == 0:
        raise RuntimeError("The native collector did not write importance statistics")
    print(json.dumps({"corpus": args.corpus.name, "corpus_sha256": hashlib.sha256(corpora[0]).hexdigest(),
                      "corpora": [{"name": p.name, "sha256": hashlib.sha256(data).hexdigest()}
                                  for p, data in zip(sources, corpora)], "max_tokens": max_tokens,
                      "sentences": len(texts), "samples": len(samples), "tokens": tokens,
                      "group_size": args.group_size, "threads": args.threads}))


if __name__ == "__main__":
    main()
