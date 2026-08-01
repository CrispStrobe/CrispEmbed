#!/usr/bin/env python3
"""Audit shipped GGUF repos for MISSING task/projection heads.

Background: several models were shipped as a plain encoder with the task head
dropped (splade-pp lost its MLM head; bge-reranker-base lost its classifier head)
— they load fine but silently can't do their job. This gate catches that class by
reading each shipped GGUF's tensor-info section (tokenless HTTP range request, no
download of the weights) and checking that the required head tensor is present.

Why header-only: the tensor NAMES live in the gguf tensor-info section, near the
start — but AFTER the tokenizer-vocab KV block, which scales with vocab (~1 MB at
30 K tokens, ~15 MB at Gemma's 256 K). We read READ_MB (default 24) to clear it.

IMPORTANT: a passing grep is NECESSARY, not SUFFICIENT — it proves the tensor is
present, not that inference is correct. For anything that fails or looks off,
confirm by actually running the model (--rerank / --colbert / encode + dim check).
This is a release/regression gate, not a fast unit test (it hits huggingface.co).

Usage:  python tests/audit_gguf_heads.py            # audit all, exit 1 on any miss
        python tests/audit_gguf_heads.py bge-reranker-base   # one model
No HF token needed (public repos). Stdlib only.
"""
import json, sys, urllib.request

READ_MB = 24
HF = "https://huggingface.co"

# Each entry: a shipped model that MUST carry a head. `any_of` = tensor-name
# substrings; the head is present if ANY appears. Keep markers broad enough to
# cover every code path that produces that head (e.g. colbert has two loaders).
MANIFEST = [
    # -- rerankers: cross-encoder scoring head (2-layer bge/jina, mxbai DeBERTa, ms-marco 1-layer) --
    ("bge-reranker-base",   "cstr/bge-reranker-base-GGUF",   ["classifier.dense.weight", "classifier.out_proj.weight", "classifier.weight"]),
    ("bge-reranker-v2-m3",  "cstr/bge-reranker-v2-m3-GGUF",  ["classifier.dense.weight", "classifier.out_proj.weight", "classifier.weight"]),
    ("jina-reranker-v2",    "cstr/jina-reranker-v2-base-multilingual-GGUF", ["classifier.dense.weight", "classifier.out_proj.weight"]),
    ("mxbai-rerank-base",   "cstr/mxbai-rerank-base-v1-GGUF",   ["classifier"]),
    ("mxbai-rerank-xsmall", "cstr/mxbai-rerank-xsmall-v1-GGUF", ["classifier"]),
    ("ms-marco-L6",         "cstr/ms-marco-MiniLM-L-6-v2-GGUF",  ["classifier.weight", "classifier.dense.weight"]),
    ("ms-marco-L12",        "cstr/ms-marco-MiniLM-L-12-v2-GGUF", ["classifier.weight", "classifier.dense.weight"]),
    # -- fixed-label NER: BERT/XLM-R + Linear classifier (>1 labels) --
    ("bert-base-NER",       "cstr/bert-base-NER-GGUF", ["ner.classifier.weight", "classifier.weight"]),
    ("xlmr-ner-hrl",        "cstr/xlmr-ner-hrl-GGUF",  ["ner.classifier.weight", "classifier.weight"]),
    # -- SPLADE: MLM head (transform + decode to vocab) --
    ("splade-pp-en-v1",     "cstr/splade-pp-en-v1-GGUF", ["mlm_transform.weight", "cls.predictions", "lm_head"]),
    ("splade-v3",           "cstr/splade-v3-GGUF",       ["mlm_transform.weight", "cls.predictions", "lm_head"]),
    # -- GLiNER: span + prompt representation heads --
    ("gliner-deberta",      "cstr/gliner-deberta-GGUF",        ["prompt_rep", "span.out_project", "token_rep"]),
    ("gliner-lfm",          "cstr/sauerkraut-gliner-lfm-GGUF", ["prompt_rep", "span.out_project", "token_rep"]),
    # -- ColBERT: per-token projection (encoder path: colbert_linear; lfm2/decoder path: colbert.projection) --
    ("lfm2-colbert",        "cstr/lfm2-colbert-GGUF", ["colbert.projection.weight", "colbert_linear.weight"]),
    # -- special-head embedder: EmbeddingGemma Matryoshka Dense(x2) projection --
    ("embeddinggemma-300m", "cstr/embeddinggemma-300m-GGUF", ["dense.0.weight", "dense.1.weight"]),
]


def _get(url, range_bytes=None):
    req = urllib.request.Request(url, headers={"User-Agent": "crispembed-audit"})
    if range_bytes is not None:
        req.add_header("Range", f"bytes=0-{range_bytes}")
    return urllib.request.urlopen(req, timeout=60)


def pick_file(repo):
    """Pick a representative shipped .gguf: prefer q8_0, else f16/base, else first."""
    with _get(f"{HF}/api/models/{repo}") as r:
        sibs = [s["rfilename"] for s in json.load(r).get("siblings", []) if s["rfilename"].endswith(".gguf")]
    if not sibs:
        return None
    for pref in ("q8_0", "-f16.gguf"):
        m = next((f for f in sibs if pref in f), None)
        if m:
            return m
    # a bare base like "<name>.gguf" (no -quant suffix) is highest precision
    base = next((f for f in sibs if f.count("-") <= 2 and not any(q in f for q in ("q4", "q5", "q6", "q8", "iq4", "f16", "f32"))), None)
    return base or sibs[0]


def audit_one(label, repo, markers):
    try:
        f = pick_file(repo)
        if not f:
            return label, repo, "?", "NO_GGUF"
        url = f"{HF}/{repo}/resolve/main/{f}"
        with _get(url, READ_MB * 1_000_000) as r:
            data = r.read()
        hit = any(m.encode() in data for m in markers)
        return label, f, "OK" if hit else "MISSING_HEAD", f"read {len(data)//1_000_000}MB"
    except Exception as e:
        return label, repo, "ERROR", str(e)[:60]


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else None
    rows = [e for e in MANIFEST if not target or target in e[0]]
    if not rows:
        print(f"no manifest entry matching '{target}'"); return 2
    print(f"GGUF head audit — {len(rows)} models, reading {READ_MB}MB of each header\n")
    bad = []
    for label, repo, markers in rows:
        lbl, f, status, note = audit_one(label, repo, markers)
        mark = {"OK": "✅", "MISSING_HEAD": "❌ HEADLESS", "ERROR": "⚠️  ERROR", "NO_GGUF": "⚠️  NO GGUF"}[status]
        print(f"  {mark:14s} {lbl:22s} [{f}]  {note}")
        if status != "OK":
            bad.append(lbl)
    print()
    if bad:
        print(f"FAIL: {len(bad)} model(s) missing a head or unverifiable: {', '.join(bad)}")
        print("Reconvert with the head-aware converter, or confirm by running the model.")
        return 1
    print("PASS: every audited model carries its required head.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
