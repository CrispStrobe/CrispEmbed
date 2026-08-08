#!/usr/bin/env python
"""E2 — Japanese reranker eval for CrispEmbed cross-encoder rerankers.

Shape: Japanese query + relevant Japanese doc + irrelevant Japanese doc.
Assert the relevant doc outranks, report the score gap (not just ordering).
Include an English-only reranker as negative control.

The three checks:
  1. JA query + JA relevant doc scores HIGHER than JA irrelevant doc
  2. Score gap is meaningful (not noise-level)
  3. Negative control: English-only reranker may fail or produce
     degenerate rankings on Japanese

Usage:
    python tests/reranker_language_eval.py ./build/crispembed <models-dir> out.json
"""
import json
import subprocess
import sys
from pathlib import Path

BIN = sys.argv[1]
CACHE = Path(sys.argv[2]).expanduser()

# Japanese fixture: query + relevant doc + irrelevant doc
JA_CASES = [
    {
        "name": "cats-weather",
        "query": "猫がソファで寝ている",  # cat sleeping on sofa
        "relevant": "ソファーの上で猫が眠っています。家の中はとても静かです。",  # cat sleeping on sofa, house quiet
        "irrelevant": "明日の東京の天気は雨の予報です。傘を持っていきましょう。",  # Tokyo weather forecast
    },
    {
        "name": "cooking-sports",
        "query": "日本料理の作り方",  # how to make Japanese food
        "relevant": "味噌汁は大豆を発酵させた味噌と出汁で作る日本の伝統的なスープです。",  # miso soup recipe
        "irrelevant": "サッカーの試合は午後三時から始まります。チケットはまだ買えます。",  # soccer match info
    },
]

# English control case (should work on all rerankers)
EN_CASE = {
    "name": "en-control",
    "query": "cat sleeping on furniture",
    "relevant": "The cat is curled up on the couch, fast asleep.",
    "irrelevant": "The weather forecast calls for rain tomorrow.",
}

MODELS = [
    # Multilingual rerankers
    ("bge-reranker-v2-m3-q4_k.gguf", "bge-reranker-v2-m3 (q4_k)", True),
    ("jina-reranker-v2-base-multilingual-q4_k.gguf", "jina-reranker-v2-base-multilingual (q4_k)", True),
    # English-only negative control
    ("bge-reranker-base-q4_k.gguf", "bge-reranker-base (q4_k) [EN-only control]", False),
]


def rerank(model_path, query, docs):
    """Run reranker and return scores for each doc."""
    cmd = [BIN, "-m", str(model_path), "--rerank", query, "--json", "-t", "4"] + docs
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
    try:
        data = json.loads(p.stdout)
    except Exception as e:
        return None, f"JSON parse failed: {e}"
    # Output is {"query": ..., "results": [{index, score, document}, ...]}
    if isinstance(data, dict) and "results" in data:
        return data["results"], None
    if isinstance(data, list):
        return data, None
    return None, f"unexpected JSON shape: {type(data)}"


results = []
for fname, label, multi in MODELS:
    path = CACHE / fname
    if not path.exists():
        print(f"SKIP (not cached): {label}", flush=True)
        continue

    model_results = {"label": label, "multi": multi, "cases": []}

    # Run Japanese cases
    for case in JA_CASES:
        docs = [case["relevant"], case["irrelevant"]]
        rows, err = rerank(path, case["query"], docs)
        if rows is None:
            print(f"FAIL (run): {label} / {case['name']}: {err}", flush=True)
            model_results["cases"].append({"name": case["name"], "error": err})
            continue
        # rows should be [{score, index, text}, ...] sorted by score desc
        score_rel = None
        score_irr = None
        for r in rows:
            if r.get("index") == 0 or r.get("document_index") == 0:
                score_rel = r.get("score", r.get("relevance_score"))
            elif r.get("index") == 1 or r.get("document_index") == 1:
                score_irr = r.get("score", r.get("relevance_score"))

        if score_rel is None or score_irr is None:
            # Try positional: first row = highest, sorted by score
            if len(rows) >= 2:
                scores = [r.get("score", r.get("relevance_score", 0)) for r in rows]
                # We need to figure out which score belongs to which doc
                # The output is sorted by score desc, with index or document_index
                for r in rows:
                    idx = r.get("index", r.get("document_index", -1))
                    s = r.get("score", r.get("relevance_score"))
                    if idx == 0:
                        score_rel = s
                    elif idx == 1:
                        score_irr = s

        if score_rel is None or score_irr is None:
            print(f"FAIL (parse): {label} / {case['name']}: could not extract scores from {rows}", flush=True)
            model_results["cases"].append({"name": case["name"], "error": "score parse"})
            continue

        gap = score_rel - score_irr
        correct = score_rel > score_irr
        cr = {
            "name": case["name"],
            "score_relevant": score_rel,
            "score_irrelevant": score_irr,
            "gap": gap,
            "correct_order": correct,
        }
        model_results["cases"].append(cr)

    # Run English control
    docs = [EN_CASE["relevant"], EN_CASE["irrelevant"]]
    rows, err = rerank(path, EN_CASE["query"], docs)
    if rows is not None:
        score_rel = score_irr = None
        for r in rows:
            idx = r.get("index", r.get("document_index", -1))
            s = r.get("score", r.get("relevance_score"))
            if idx == 0:
                score_rel = s
            elif idx == 1:
                score_irr = s
        if score_rel is not None and score_irr is not None:
            model_results["en_control"] = {
                "score_relevant": score_rel,
                "score_irrelevant": score_irr,
                "gap": score_rel - score_irr,
                "correct_order": score_rel > score_irr,
            }

    results.append(model_results)

    # Print summary
    print(f"\n{label}", flush=True)
    for cr in model_results["cases"]:
        if "error" in cr:
            print(f"    {cr['name']:20s} ERROR: {cr['error']}", flush=True)
        else:
            print(
                f"    {cr['name']:20s} {'PASS' if cr['correct_order'] else 'FAIL'}  "
                f"rel={cr['score_relevant']:.4f} irr={cr['score_irrelevant']:.4f} "
                f"gap={cr['gap']:+.4f}",
                flush=True,
            )
    if "en_control" in model_results:
        ec = model_results["en_control"]
        print(
            f"    {'en-control':20s} {'PASS' if ec['correct_order'] else 'FAIL'}  "
            f"rel={ec['score_relevant']:.4f} irr={ec['score_irrelevant']:.4f} "
            f"gap={ec['gap']:+.4f}",
            flush=True,
        )

out = Path(sys.argv[3])
out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nwrote {out}")
