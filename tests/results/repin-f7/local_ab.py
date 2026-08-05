#!/usr/bin/env python3
"""Local Metal+CPU cross-check of the reranker -f7 quants (G3 precedent).

Reuses RERANK_EVAL / kendall_tau / rerank-score parsing from the canonical
Kaggle pipeline (tools/kaggle/crispembed-imatrix-quant/imatrix_quant.py) by
importing that module — no metric reimplementation, no drift.

Gold = the f16 artifact's own scores on the SAME backend as the arm under
test (each backend gets its own gold, so the tau/dscore numbers are
backend-internal like the Kaggle CPU run; cross-backend f16 drift is
reported separately).
"""
import importlib.util
import json
import sys as _sys
_sys.path.insert(0, "/Users/christianstrobele/code/CrispEmbed/tools/kaggle/crispembed-imatrix-quant")
import subprocess
import sys
from pathlib import Path

CLI = Path(sys.argv[1])            # crispembed binary (absolute)
MODELS = Path(sys.argv[2])         # dir with the .gguf files
OUT = Path(sys.argv[3])            # results json

spec = importlib.util.spec_from_file_location(
    "imatrix_quant",
    "/Users/christianstrobele/code/CrispEmbed/tools/kaggle/crispembed-imatrix-quant/imatrix_quant.py")
iq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iq)
RERANK_EVAL = iq.RERANK_EVAL
kendall_tau = iq.kendall_tau
assert len(RERANK_EVAL) == 30, len(RERANK_EVAL)


def rerank_scores(model, backend, query, docs):
    r = subprocess.run([str(CLI), "-m", str(model), "--gpu-backend", backend,
                        "--json", "--rerank", query, *docs],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rerank rc={r.returncode} for {model} [{backend}]:\n{r.stderr[-1500:]}")
    if backend == "metal" and "MTL0" not in r.stderr:
        raise RuntimeError(f"metal arm without MTL0 proof in stderr for {model}")
    return {res["index"]: res["score"] for res in json.loads(r.stdout)["results"]}


FAMILIES = {
    "jina-reranker-v2-base-multilingual": [
        ("f16",        "jina-reranker-v2-base-multilingual.gguf"),
        ("q8_0",       "jina-reranker-v2-base-multilingual-q8_0.gguf"),
        ("q4_k-im-old", "jina-reranker-v2-base-multilingual-q4_k-imatrix.gguf"),
        ("q4_k-im-f7", "jina-reranker-v2-base-multilingual-q4_k-imatrix-f7.gguf"),
        ("iq4_xs-f7",  "jina-reranker-v2-base-multilingual-iq4_xs-f7.gguf"),
    ],
    "bge-reranker-v2-m3": [
        ("f16",        "bge-reranker-v2-m3.gguf"),
        ("q8_0",       "bge-reranker-v2-m3-q8_0.gguf"),
        ("q4_k-im-old", "bge-reranker-v2-m3-q4_k-imatrix.gguf"),
        ("q4_k-im-f7", "bge-reranker-v2-m3-q4_k-imatrix-f7.gguf"),
        ("iq4_xs-f7",  "bge-reranker-v2-m3-iq4_xs-f7.gguf"),
    ],
}

only = sys.argv[4] if len(sys.argv) > 4 else None
results = json.loads(OUT.read_text()) if OUT.exists() else {}
for fam, arms in FAMILIES.items():
    if only and only not in fam:
        continue
    for backend in ("cpu", "metal"):
        gold = None
        for arm, fn in arms:
            model = MODELS / fn
            per_pair = []
            for q, docs in RERANK_EVAL:
                per_pair.append(rerank_scores(model, backend, q, docs))
            key = f"{fam}/{backend}/{arm}"
            if arm == "f16":
                gold = per_pair
                results[key] = {"raw_pair0": per_pair[0]}
            else:
                taus, dabs = [], []
                for g, s in zip(gold, per_pair):
                    taus.append(kendall_tau(g, s))
                    shared = [i for i in g if i in s]
                    dabs.append(sum(abs(g[i] - s[i]) for i in shared) / len(shared))
                results[key] = {
                    "tau": sum(taus) / len(taus),
                    "dscore": sum(dabs) / len(dabs),
                    "raw_pair0": per_pair[0],
                }
            print(f"{key}: " + json.dumps({k: v for k, v in results[key].items() if k != 'raw_pair0'}),
                  flush=True)
            OUT.write_text(json.dumps(results, indent=1))
print("done", flush=True)
