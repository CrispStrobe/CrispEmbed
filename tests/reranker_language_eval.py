#!/usr/bin/env python
"""E2+E3 — Multilingual reranker eval for CrispEmbed cross-encoder rerankers.

Shape per language: native query + relevant native doc + irrelevant native doc.
Assert the relevant doc outranks, report the score gap (not just ordering).
Include an English-only reranker as negative control.

Languages: Japanese (E2, 2026-08-08), Arabic + Korean (E3, 2026-08-17).

Usage:
    python tests/reranker_language_eval.py ./build/crispembed <models-dir> out.json
"""
import json
import subprocess
import sys
from pathlib import Path

BIN = sys.argv[1]
CACHE = Path(sys.argv[2]).expanduser()

# Per-language fixtures: query + relevant doc + irrelevant doc
LANG_CASES = {
    "ja": [
        {
            "name": "ja-cats-weather",
            "query": "猫がソファで寝ている",  # cat sleeping on sofa
            "relevant": "ソファーの上で猫が眠っています。家の中はとても静かです。",  # cat sleeping on sofa, house quiet
            "irrelevant": "明日の東京の天気は雨の予報です。傘を持っていきましょう。",  # Tokyo weather forecast
        },
        {
            "name": "ja-cooking-sports",
            "query": "日本料理の作り方",  # how to make Japanese food
            "relevant": "味噌汁は大豆を発酵させた味噌と出汁で作る日本の伝統的なスープです。",  # miso soup recipe
            "irrelevant": "サッカーの試合は午後三時から始まります。チケットはまだ買えます。",  # soccer match info
        },
    ],
    "ar": [
        {
            "name": "ar-cats-weather",
            "query": "قطة نائمة على الأريكة",  # cat sleeping on sofa
            "relevant": "القطة تنام فوق الكنبة في غرفة الجلوس. البيت هادئ جداً.",  # cat sleeping on sofa, house quiet
            "irrelevant": "توقعات الطقس تشير إلى أمطار غزيرة في طوكيو غداً.",  # Tokyo weather forecast
        },
        {
            "name": "ar-cooking-sports",
            "query": "طريقة طبخ الأرز العربي",  # how to cook Arabic rice
            "relevant": "الكبسة هي طبق أرز سعودي تقليدي يُطهى مع اللحم والبهارات والطماطم.",  # kabsa recipe
            "irrelevant": "مباراة كرة القدم تبدأ في الساعة الثالثة بعد الظهر. التذاكر متوفرة.",  # soccer match info
        },
    ],
    "ko": [
        {
            "name": "ko-cats-weather",
            "query": "고양이가 소파에서 자고 있다",  # cat sleeping on sofa
            "relevant": "고양이가 거실 소파 위에서 잠들어 있습니다. 집 안이 매우 조용합니다.",  # cat sleeping on sofa, house quiet
            "irrelevant": "내일 도쿄의 날씨는 비가 올 것으로 예상됩니다. 우산을 챙기세요.",  # Tokyo weather forecast
        },
        {
            "name": "ko-cooking-sports",
            "query": "한국 음식 만드는 방법",  # how to make Korean food
            "relevant": "김치찌개는 김치와 돼지고기를 넣고 끓이는 한국의 전통 찌개입니다.",  # kimchi jjigae recipe
            "irrelevant": "축구 경기는 오후 세 시에 시작합니다. 아직 티켓을 구할 수 있습니다.",  # soccer match info
        },
    ],
}

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

    # Run all language cases
    all_cases = [(lang, c) for lang, cases in LANG_CASES.items() for c in cases]
    for lang, case in all_cases:
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
            "lang": lang,
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
