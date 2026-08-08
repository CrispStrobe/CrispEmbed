#!/usr/bin/env python
"""Japanese sanity eval for CrispEmbed embedding models.

Three checks per model, all relative (no ground-truth vectors needed):

  1. JA paraphrase   cos(ja_cat_a, ja_cat_b) > cos(ja_cat_a, ja_weather)
     -- monolingual Japanese semantics are represented at all.
  2. Cross-lingual   cos(ja_cat_a, en_cat)   > cos(ja_cat_a, en_weather)
     -- JA<->EN alignment (what a "multilingual" claim is FOR).
  3. Non-degenerate  cos(ja_cat_a, ja_weather) < 0.95
     -- the UNK-collapse failure mode: a model whose tokenizer maps all
        Japanese to the same handful of tokens produces near-identical
        vectors for UNRELATED Japanese, which would fake a pass on (1)
        if only the positive pair were measured.

Margins are reported, not just pass/fail: a 0.001 margin is not a pass.
English-only models are included as a NEGATIVE CONTROL -- if they also
"pass", the test is not measuring what it claims to.
"""
import json
import subprocess
import sys
import math
from pathlib import Path

BIN = sys.argv[1]
CACHE = Path(sys.argv[2]).expanduser()

TEXTS = {
    "ja_cat_a": "猫がソファの上で眠っている。",
    "ja_cat_b": "ソファーで猫が寝ています。",
    "ja_weather": "明日の東京の天気は雨でしょう。",
    "en_cat": "A cat is sleeping on the sofa.",
    "en_weather": "It will rain in Tokyo tomorrow.",
}
KEYS = list(TEXTS)

MODELS = [
    # (file, label, claimed_multilingual)
    ("bge-m3-iq4_xs.gguf", "bge-m3 (XLM-R, iq4_xs)", True),
    ("granite-embedding-107m-multilingual-Q4_K_M.gguf", "granite-embedding-107m-multilingual (q4_k_m)", True),
    ("jina-v5-nano-q4_k.gguf", "jina-v5-nano (q4_k)", True),
    ("jina-v5-small-q4_k.gguf", "jina-v5-small (q4_k)", True),
    ("Qwen3-Embedding-0.6B-Q8_0.gguf", "Qwen3-Embedding-0.6B (q8_0)", True),
    ("LFM2.5-Embedding-350M-Q8_0.gguf", "LFM2.5-Embedding-350M (q8_0)", True),
    ("nomic-embed-text-v2-moe.Q4_K_M.gguf", "nomic-embed-text-v2-moe (q4_k_m)", True),
    ("arctic-embed-m-v2-q4_k-imatrix.gguf", "arctic-embed-m-v2 (q4_k-imatrix)", True),
    # negative controls -- English-only training
    ("all-MiniLM-L6-v2-Q4_K_M.gguf", "all-MiniLM-L6-v2 (q4_k_m) [EN-only control]", False),
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
    para = cos(emb["ja_cat_a"], emb["ja_cat_b"])
    ja_unrel = cos(emb["ja_cat_a"], emb["ja_weather"])
    xl_pos = cos(emb["ja_cat_a"], emb["en_cat"])
    xl_neg = cos(emb["ja_cat_a"], emb["en_weather"])
    r = {
        "label": label, "multi": multi,
        "ja_para": para, "ja_unrel": ja_unrel, "ja_margin": para - ja_unrel,
        "xl_pos": xl_pos, "xl_neg": xl_neg, "xl_margin": xl_pos - xl_neg,
        "c1_ja_semantics": para > ja_unrel,
        "c2_crosslingual": xl_pos > xl_neg,
        "c3_nondegenerate": ja_unrel < 0.95,
    }
    results.append(r)
    print(
        f"{label}\n"
        f"    C1 JA paraphrase   {'PASS' if r['c1_ja_semantics'] else 'FAIL'}  "
        f"para={para:.4f} unrel={ja_unrel:.4f} margin={r['ja_margin']:+.4f}\n"
        f"    C2 cross-lingual   {'PASS' if r['c2_crosslingual'] else 'FAIL'}  "
        f"ja~en_same={xl_pos:.4f} ja~en_other={xl_neg:.4f} margin={r['xl_margin']:+.4f}\n"
        f"    C3 non-degenerate  {'PASS' if r['c3_nondegenerate'] else 'FAIL'}  "
        f"unrelated-JA cos={ja_unrel:.4f}",
        flush=True)

out = Path(sys.argv[3])
out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\nwrote {out}")
