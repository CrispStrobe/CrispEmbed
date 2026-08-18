#!/usr/bin/env python
"""Multilingual sanity eval for CrispEmbed embedding models.

Three checks per (model, language) pair, all relative (no ground-truth
vectors needed):

  1. Paraphrase       cos(X_cat_a, X_cat_b) > cos(X_cat_a, X_weather)
     -- monolingual semantics are represented at all.
  2. Cross-lingual    cos(X_cat_a, en_cat)  > cos(X_cat_a, en_weather)
     -- language<->EN alignment (what a "multilingual" claim is FOR).
  3. Non-degenerate   cos(X_cat_a, X_weather) < 0.95
     -- the UNK-collapse failure mode: a model whose tokenizer maps all
        non-English text to the same handful of tokens produces near-
        identical vectors for UNRELATED sentences, faking a pass on (1).

Margins are reported, not just pass/fail: a 0.001 margin is not a pass.
English-only models are included as a NEGATIVE CONTROL -- if they also
"pass", the test is not measuring what it claims to.

Languages: Japanese (E1, 2026-08-08), Arabic + Korean (E3, 2026-08-17).
"""
import json
import subprocess
import sys
import math
from pathlib import Path

BIN = sys.argv[1]
CACHE = Path(sys.argv[2]).expanduser()

TEXTS = {
    # English anchors (shared across all language checks)
    "en_cat": "A cat is sleeping on the sofa.",
    "en_weather": "It will rain in Tokyo tomorrow.",
    # Japanese (E1, 2026-08-08)
    "ja_cat_a": "猫がソファの上で眠っている。",
    "ja_cat_b": "ソファーで猫が寝ています。",
    "ja_weather": "明日の東京の天気は雨でしょう。",
    # Arabic (E3, 2026-08-17) — RTL script, different normalization
    "ar_cat_a": "القطة نائمة على الأريكة.",
    "ar_cat_b": "تنام القطة فوق الكنبة.",
    "ar_weather": "سيكون الطقس ممطراً في طوكيو غداً.",
    # Korean (E3, 2026-08-17) — Hangul, agglutinative morphology
    "ko_cat_a": "고양이가 소파 위에서 자고 있다.",
    "ko_cat_b": "소파에서 고양이가 잠을 자고 있습니다.",
    "ko_weather": "내일 도쿄의 날씨는 비가 올 것입니다.",
}
KEYS = list(TEXTS)

# Per-language check definitions: (lang_code, cat_a_key, cat_b_key, weather_key)
LANG_CHECKS = [
    ("ja", "ja_cat_a", "ja_cat_b", "ja_weather"),
    ("ar", "ar_cat_a", "ar_cat_b", "ar_weather"),
    ("ko", "ko_cat_a", "ko_cat_b", "ko_weather"),
]

MODELS = [
    # (file, label, claimed_multilingual)
    # --- previously verified JA (99f39f64) ---
    ("bge-m3-iq4_xs.gguf", "bge-m3 (XLM-R, iq4_xs)", True),
    ("granite-embedding-107m-multilingual-q4_k.gguf", "granite-embedding-107m-multilingual (q4_k)", True),
    ("jina-v5-nano-q4_k.gguf", "jina-v5-nano (q4_k)", True),
    ("jina-v5-small-q4_k.gguf", "jina-v5-small (q4_k)", True),
    ("Qwen3-Embedding-0.6B-Q8_0.gguf", "Qwen3-Embedding-0.6B (q8_0)", True),
    ("LFM2.5-Embedding-350M-Q8_0.gguf", "LFM2.5-Embedding-350M (q8_0)", True),
    ("nomic-embed-text-v2-moe.Q4_K_M.gguf", "nomic-embed-text-v2-moe (q4_k_m)", True),
    ("arctic-embed-m-v2-q4_k-imatrix.gguf", "arctic-embed-m-v2 (q4_k-imatrix)", True),
    # --- E1: untested multilingual aliases (smallest → largest) ---
    ("paraphrase-multilingual-MiniLM-L12-v2-q8_0.gguf", "paraphrase-multilingual-MiniLM-L12-v2 (q8_0)", True),
    ("multilingual-e5-small-q8_0.gguf", "multilingual-e5-small (q8_0, no prefix)", True),
    ("granite-embedding-278m-multilingual-q8_0.gguf", "granite-embedding-278m-multilingual (q8_0)", True),
    ("multilingual-e5-base-q8_0.gguf", "multilingual-e5-base (q8_0, no prefix)", True),
    ("multilingual-e5-large-q8_0.gguf", "multilingual-e5-large (q8_0, no prefix)", True),
    # E1: non-multilingual granite-278m as additional data point
    ("granite-embedding-278m-q8_0.gguf", "granite-embedding-278m (q8_0, non-multilingual)", True),
    # negative controls -- English-only training
    ("all-MiniLM-L6-v2-q4_k.gguf", "all-MiniLM-L6-v2 (q4_k) [EN-only control]", False),
    ("all-mpnet-base-v2-q8_0.gguf", "all-mpnet-base-v2 (q8_0) [EN-only control]", False),
]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb + 1e-12)


def embed(model_path):
    cmd = [BIN, "-m", str(model_path), "--json", "-t", "4"] + [TEXTS[k] for k in KEYS]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
    try:
        rows = json.loads(p.stdout)
    except Exception as e:
        return None, f"JSON parse failed: {e}"
    if len(rows) != len(KEYS):
        return None, f"expected {len(KEYS)} rows, got {len(rows)}"
    return {k: rows[i]["embedding"] for i, k in enumerate(KEYS)}, None


results = []
for fname, label, multi in MODELS:
    path = CACHE / fname
    if not path.exists():
        print(f"SKIP (not cached): {label}", flush=True)
        continue
    emb, err = embed(path)
    if emb is None:
        print(f"FAIL (run): {label}: {err}", flush=True)
        results.append({"label": label, "multi": multi, "error": err})
        continue

    r = {"label": label, "multi": multi, "langs": {}}
    print(f"\n{label}", flush=True)

    for lang, cat_a, cat_b, weather in LANG_CHECKS:
        para = cos(emb[cat_a], emb[cat_b])
        unrel = cos(emb[cat_a], emb[weather])
        xl_pos = cos(emb[cat_a], emb["en_cat"])
        xl_neg = cos(emb[cat_a], emb["en_weather"])
        lr = {
            f"{lang}_para": para, f"{lang}_unrel": unrel,
            f"{lang}_margin": para - unrel,
            f"{lang}_xl_pos": xl_pos, f"{lang}_xl_neg": xl_neg,
            f"{lang}_xl_margin": xl_pos - xl_neg,
            f"c1_{lang}_semantics": para > unrel,
            f"c2_{lang}_crosslingual": xl_pos > xl_neg,
            f"c3_{lang}_nondegenerate": unrel < 0.95,
        }
        r["langs"][lang] = lr
        # Also copy to top level for backward compat with JA-only consumers
        r.update(lr)
        LANG_UPPER = lang.upper()
        print(
            f"    C1 {LANG_UPPER} paraphrase   "
            f"{'PASS' if lr[f'c1_{lang}_semantics'] else 'FAIL'}  "
            f"para={para:.4f} unrel={unrel:.4f} margin={para - unrel:+.4f}\n"
            f"    C2 {LANG_UPPER} cross-lingual "
            f"{'PASS' if lr[f'c2_{lang}_crosslingual'] else 'FAIL'}  "
            f"{lang}~en_same={xl_pos:.4f} {lang}~en_other={xl_neg:.4f} "
            f"margin={xl_pos - xl_neg:+.4f}\n"
            f"    C3 {LANG_UPPER} non-degenerate "
            f"{'PASS' if lr[f'c3_{lang}_nondegenerate'] else 'FAIL'}  "
            f"unrelated-{LANG_UPPER} cos={unrel:.4f}",
            flush=True)

    results.append(r)

out = Path(sys.argv[3])
out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nwrote {out}")
