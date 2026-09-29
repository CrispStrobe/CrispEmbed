#!/usr/bin/env python3
"""most-embed-de (CC BY-NC 4.0, Ministral3 1B German retrieval): CrispEmbed vs
sentence-transformers on current code, through the published GGUFs.

Ported 2026-08-20 (decoder_embed Ministral3 path); upstream weights unchanged
since 2026-08-11. Shared code (decoder_embed, BPE tokenizer, GGUF loader) has
moved since, and there is no CI regression entry - this re-proves it.

Reference: SentenceTransformer("malteos/most-embed-de"), fp32 CPU, prompts
"query: " / "passage: " (config_sentence_transformers.json), mean pooling incl.
prompt, L2 normalise. C++: crispembed with the same explicit --prefix.

Gates: per-text cosine >= 0.999 (q8_0) / >= 0.985 (q4_k + q8 attention), and the
retrieval check - every query ranks its own passage first, as upstream does.

    gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/most_embed_de.py \\
        -f pip="torch sentence-transformers>=5.5 transformers>=5.11"
"""
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("HEAVY_OUT", "out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", "scratch"))
OUT.mkdir(parents=True, exist_ok=True)
SCR.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[2]
MODEL = "malteos/most-embed-de"
# Gates from the 2026-09-29 run (36587452534): q8_0 0.9996-0.9998, q4_k-attn-q8
# 0.9893-0.9939 (quantisation; retrieval top-1 identical to upstream on both).
GGUFS = {"q8_0": ("most-embed-de-q8_0.gguf", 0.999), "q4_k-attn-q8": ("most-embed-de-q4_k-attn-q8.gguf", 0.985)}
QUERIES = ["Wie hoch ist die Zugspitze?",
           "Wann fiel die Berliner Mauer?",
           "Welche Zutaten braucht man für einen Apfelstrudel?",
           "Wie funktioniert eine Wärmepumpe?"]
PASSAGES = ["Die Zugspitze ist mit 2962 Metern der höchste Berg Deutschlands und liegt im Wettersteingebirge.",
            "Am 9. November 1989 öffnete die DDR ihre Grenzen, und die Berliner Mauer fiel.",
            "Für einen Apfelstrudel braucht man Strudelteig, säuerliche Äpfel, Zucker, Zimt, Rosinen und Butter.",
            "Eine Wärmepumpe entzieht der Umgebung Wärme und hebt sie mit einem Verdichter auf ein höheres "
            "Temperaturniveau."]
res = {"model": MODEL, "arms": {}, "errors": []}


def cos_rows(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def top1(q, p):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    return (q @ p.T).argmax(1).tolist()


try:
    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer

    b = SCR / "build"
    subprocess.check_call(["cmake", "-S", str(REPO), "-B", str(b), "-DCMAKE_BUILD_TYPE=Release"], stdout=subprocess.DEVNULL)
    subprocess.check_call(["cmake", "--build", str(b), "--target", "crispembed-cli", f"-j{os.cpu_count() or 4}"],
                          stdout=subprocess.DEVNULL)
    exe = next(p for p in b.rglob("crispembed") if p.is_file() and os.access(p, os.X_OK))

    st = SentenceTransformer(MODEL, device="cpu")
    ref_q = st.encode(QUERIES, prompt_name="query", normalize_embeddings=True)
    ref_p = st.encode(PASSAGES, prompt_name="document", normalize_embeddings=True)
    res["ref_top1"] = top1(ref_q, ref_p)
    del st

    def cpp(gguf, prefix, texts):
        r = subprocess.run([str(exe), "-m", gguf, "--prefix", prefix, "-t", str(os.cpu_count() or 4), *texts],
                           capture_output=True, text=True, timeout=1800)
        rows = [[float(v) for v in l.split()] for l in r.stdout.splitlines() if l.strip()]
        if len(rows) != len(texts):
            raise RuntimeError(f"expected {len(texts)} rows, got {len(rows)}: {r.stderr[-800:]}")
        return np.array(rows)

    for tag, (fname, gate) in GGUFS.items():
        g = hf_hub_download("cstr/most-embed-de-GGUF", fname)
        q = cpp(g, "query: ", QUERIES)
        p = cpp(g, "passage: ", PASSAGES)
        cq, cp = cos_rows(q, ref_q), cos_rows(p, ref_p)
        res["arms"][tag] = {"cos_query": [round(float(x), 6) for x in cq], "cos_passage": [round(float(x), 6) for x in cp],
                            "min_cos": round(float(min(cq.min(), cp.min())), 6), "gate": gate,
                            "top1": top1(q, p), "dim": int(q.shape[1])}
        print(tag, res["arms"][tag], flush=True)
except Exception:
    res["errors"].append(traceback.format_exc())
    print(res["errors"][-1], file=sys.stderr)
finally:
    (OUT / "result.json").write_text(json.dumps(res, indent=1))

ideal = list(range(len(QUERIES)))
lines = [f"### {MODEL}: CrispEmbed vs sentence-transformers (fp32)\n",
         f"reference top-1 passage per query: {res.get('ref_top1')} (ideal {ideal})\n",
         "| GGUF | min cos | gate | per-query cos | per-passage cos | C++ top-1 |", "|---|---|---|---|---|---|"]
for tag, a in res["arms"].items():
    lines.append(f"| {tag} | {a['min_cos']} | {a['gate']} | {a['cos_query']} | {a['cos_passage']} | {a['top1']} |")
if res["errors"]:
    lines.append("\n**errors:**\n```\n" + res["errors"][-1][-1500:] + "\n```")
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
ok = (not res["errors"] and len(res["arms"]) == len(GGUFS)
      and all(a["min_cos"] >= a["gate"] and a["top1"] == res["ref_top1"] for a in res["arms"].values()))
sys.exit(0 if ok else 1)
