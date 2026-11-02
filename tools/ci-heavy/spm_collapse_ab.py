#!/usr/bin/env python3
"""A/B of CRISPEMBED_SPM_COLLAPSE_SPACES (default on) against the original HF models,
for every registry model on the native SentencePiece encoder path (hosted CPU).

    gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/spm_collapse_ab.py \
        -f args="multilingual-e5-small bge-reranker-base" -f isolated_lane=spm-ab-1 \
        -f pip="torch transformers>=4.51,<5 sentence-transformers einops gguf huggingface_hub sentencepiece protobuf"

Same 78-text panel as tools/ci-heavy/below_bar_recipes.py (13 CUDA-audit texts plus the
65-text held-out imatrix eval corpus, every text without a prefix). For each model, with the
switch on (default) and =0:
  * token IDs: native `--tokens-raw` ids vs the model's own HF tokenizer, per text;
  * embedders: final embedding cosine vs sentence-transformers (fp32, batch 1);
  * rerankers: scores for a fixed query vs the HF sequence-classification logit
    (compared raw and through a sigmoid; the closer mapping is reported).
The model's tokenizer.json normalizer is recorded, so a regression can be tied to the spec.
Exit status is non-zero when the default arm matches HF on fewer texts than =0 for any model.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path(os.environ.get("HEAVY_OUT", "/tmp/spm-ab-out"))
S = Path(os.environ.get("HEAVY_SCRATCH", "/tmp/spm-ab-scratch"))
OUT.mkdir(parents=True, exist_ok=True)
S.mkdir(parents=True, exist_ok=True)
NT = str(os.cpu_count() or 4)
QUERY = "What did Einstein publish in 1905?"

# registry alias -> (HF original, kind). Every alias is a registry model whose GGUF resolves to
# tokenizer family 2 (SentencePiece) on the encoder path; found by scanning the registry headers.
MODELS = {
    "multilingual-e5-small": ("intfloat/multilingual-e5-small", "embed"),
    "multilingual-e5-base": ("intfloat/multilingual-e5-base", "embed"),
    "multilingual-e5-large": ("intfloat/multilingual-e5-large", "embed"),
    "pixie-rune-v1": ("telepix/PIXIE-Rune-v1.0", "embed"),
    "arctic-embed-l-v2": ("Snowflake/snowflake-arctic-embed-l-v2.0", "embed"),
    "arctic-embed-m-v2": ("Snowflake/snowflake-arctic-embed-m-v2.0", "embed"),
    "nomic-embed-text-v2-moe": ("nomic-ai/nomic-embed-text-v2-moe", "embed"),
    "paraphrase-multilingual-MiniLM-L12-v2": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "embed"),
    "bge-m3": ("BAAI/bge-m3", "embed"),
    "granite-embedding-278m": ("ibm-granite/granite-embedding-278m-multilingual", "embed"),
    "granite-embedding-107m": ("ibm-granite/granite-embedding-107m-multilingual", "embed"),
    "bge-reranker-v2-m3": ("BAAI/bge-reranker-v2-m3", "rerank"),
    "bge-reranker-base": ("BAAI/bge-reranker-base", "rerank"),
    "jina-reranker-v2-base-multilingual": ("jinaai/jina-reranker-v2-base-multilingual", "rerank"),
    "mxbai-rerank-xsmall-v1": ("mixedbread-ai/mxbai-rerank-xsmall-v1", "rerank"),
    "mxbai-rerank-base-v1": ("mixedbread-ai/mxbai-rerank-base-v1", "rerank"),
    "xlmr-ner-hrl": ("Davlan/xlm-roberta-base-ner-hrl", "tokens"),
}
ARMS = {"default (collapse on)": {}, "CRISPEMBED_SPM_COLLAPSE_SPACES=0": {"CRISPEMBED_SPM_COLLAPSE_SPACES": "0"}}
LICENCE_ENV = {"CRISPEMBED_ACCEPT_LICENSE": "all"}  # owner-approved, internal testing only


def log(*a):
    print(*a, flush=True)


def sh(cmd, env=None, timeout=3600):
    t = time.time()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout,
                       env=dict(os.environ, **LICENCE_ENV, **(env or {})))
    return r, time.time() - t


def build():
    b = S / "build-cpu"
    r, _ = sh(["cmake", "-S", ".", "-B", b, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
               "-DCRISPEMBED_NATIVE=OFF", "-DGGML_METAL=OFF", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF",
               "-DCRISPEMBED_BUILD_SHARED=OFF"])
    if r.returncode:
        sys.exit("configure failed:\n" + r.stderr[-3000:])
    r, _ = sh(["cmake", "--build", b, "--parallel", NT, "--target", "crispembed-cli"])
    if r.returncode:
        sys.exit("build failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])
    return next(p for p in b.rglob("crispembed") if p.is_file() and os.access(p, os.X_OK))


def panel():
    audit = [l for l in (ROOT / "tests/results/gpu-audit/metal-m1/embed_texts.txt").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in (ROOT / "tools/kaggle/crispembed-imatrix-quant/eval_corpus.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    docs = [r["text"] for r in rows if not r.get("role", "").startswith("query")]
    qs = [r["text"] for r in rows if r.get("role", "").startswith("query")]
    texts = audit + docs + qs
    assert not any(t.startswith("-") for t in texts)
    return texts


def registry():
    src = (ROOT / "examples/cli/model_mgr.cpp").read_text()
    st = src.index("static const ModelEntry k_registry[]")
    body = re.sub(r'"\s*"', "", src[st:src.index("\n};", st)])
    return {a: (f, u) for a, f, u in re.findall(r'\{\s*"([^"]+)",\s*"([^"]+)",\s*"(https://huggingface\.co/[^"]+)"', body)}


def pick(reg, alias):
    for a in (f"{alias}-f16", f"{alias}-f32", f"{alias}-q8", alias):
        if a in reg:
            return a, reg[a]
    raise KeyError(alias)


def download(url, name):
    p = S / "gguf" / name
    p.parent.mkdir(exist_ok=True)
    if not p.exists():
        r, _ = sh(["curl", "-sfL", "--retry", "3", "-o", p, url], timeout=3600)
        if r.returncode:
            raise RuntimeError(f"download {url} failed")
    return p


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def normalizer_spec(tok):
    try:
        spec = json.loads(tok.backend_tokenizer.to_str()).get("normalizer")
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}

    def strip(n):
        if isinstance(n, dict):
            return {k: ("<charsmap>" if k == "precompiled_charsmap" else strip(v)) for k, v in n.items()}
        if isinstance(n, list):
            return [strip(x) for x in n]
        return n
    return strip(spec)


def native_json(cli, gguf, args, texts, env):
    r, dt = sh([cli, "-m", gguf, "-t", NT, "--json", "--prefix", ""] + args + texts, env=env)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])
    try:
        return json.loads(r.stdout), dt
    except json.JSONDecodeError:
        return [json.loads(l.rstrip().rstrip(",")) for l in r.stdout.splitlines() if l.strip().startswith("{")], dt


def run_model(cli, reg, alias, hf, kind, texts):
    from transformers import AutoTokenizer
    rec = {"alias": alias, "hf": hf, "kind": kind}
    used, (fname, url) = pick(reg, alias)
    rec.update(registry_alias=used, file=fname, url=url)
    gguf = download(url, fname)
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    rec["hf_tokenizer"] = type(tok).__name__
    rec["normalizer"] = normalizer_spec(tok)
    ml = tok.model_max_length if 0 < (tok.model_max_length or 0) < 100000 else 8192
    rec["hf_max_length"] = ml
    # Inputs the original truncates are left out of the A/B: whitespace handling is the question here,
    # and several native GGUFs abort on over-length input in BOTH arms (recorded as long_input_probe).
    full = [tok(t)["input_ids"] for t in texts]
    keep = [i for i, f in enumerate(full) if len(f) <= ml]
    rec["excluded_over_length"] = [i for i in range(len(texts)) if i not in keep]
    texts = [texts[i] for i in keep]
    hf_ids = [full[i] for i in keep]
    # Two references: the fast tokenizer.json (tokenizers, as transformers 4.x uses it) and the model's
    # own SentencePiece model (slow tokenizer). They differ on a lone trailing "▁" for text ending in
    # whitespace; transformers 5 rebuilds the tokenizer and agrees with SentencePiece there.
    slow_ids = None
    try:
        slow = AutoTokenizer.from_pretrained(hf, use_fast=False, trust_remote_code=True)
        slow_ids = [slow(t)["input_ids"] for t in texts]
        rec["slow_tokenizer"] = type(slow).__name__
    except Exception as e:  # noqa: BLE001
        rec["slow_tokenizer_error"] = f"{type(e).__name__}: {e}"[:300]
    ref, m = None, None
    if rec["excluded_over_length"]:
        long_t = [t for i, t in enumerate(full) if len(t) > ml]
        r, _ = sh([cli, "-m", gguf, "-t", NT, "--json", "--prefix", "", "--tokens-raw",
                   [tt for i, tt in enumerate(panel()) if i in rec["excluded_over_length"]][0]])
        rec["long_input_probe"] = {"rc": r.returncode, "stderr_tail": r.stderr[-300:], "hf_len": len(long_t[0])}
    try:
        import torch
        torch.set_num_threads(int(NT))
        if kind == "embed":
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(hf, device="cpu", trust_remote_code=True,
                                    model_kwargs={"torch_dtype": torch.float32})
            ref = [v.tolist() for v in m.encode(texts, batch_size=1, normalize_embeddings=True)]
        elif kind == "rerank":
            from transformers import AutoModelForSequenceClassification
            m = AutoModelForSequenceClassification.from_pretrained(hf, trust_remote_code=True,
                                                                   torch_dtype=torch.float32).eval()
            ref = []
            with torch.no_grad():
                for t in texts:
                    e = tok(QUERY, t, return_tensors="pt", truncation=True, max_length=512)
                    ref.append(float(m(**e).logits.reshape(-1)[0]))
    except Exception as e:  # noqa: BLE001
        rec["reference_error"] = f"{type(e).__name__}: {e}"[:1500]
    for arm, env in ARMS.items():
        a = {}
        try:
            j, dt = native_json(cli, gguf, ["--tokens-raw"], texts, env)
            nat = [x.get("token_ids", []) for x in j]
            bad = [i for i, (h, n) in enumerate(zip(hf_ids, nat)) if h != n]
            a["ids_equal"] = len(texts) - len(bad)
            if slow_ids is not None:
                a["ids_equal_sentencepiece"] = sum(h == n for h, n in zip(slow_ids, nat))
            a["ids_mismatch_texts"] = bad
            a["ids_mismatch_examples"] = []
            for i in bad[:6]:
                h, n = hf_ids[i], nat[i]
                k = next((q for q, (x, y) in enumerate(zip(h, n)) if x != y), min(len(h), len(n)))
                a["ids_mismatch_examples"].append({"text": i, "hf_len": len(h), "native_len": len(n), "first_diff": k,
                                                   "hf": tok.convert_ids_to_tokens(h[max(0, k - 2):k + 4]),
                                                   "native": tok.convert_ids_to_tokens(n[max(0, k - 2):k + 4]),
                                                   "excerpt": texts[i][:80]})
        except Exception as e:  # noqa: BLE001
            a["ids_error"] = f"{type(e).__name__}: {e}"[:1500]
        try:
            if kind == "embed" and ref is not None:
                j, dt = native_json(cli, gguf, [], texts, env)
                cs = [cos(x["embedding"], y) for x, y in zip(j, ref)]
                a.update(cos_min=min(cs), cos_mean=float(np.mean(cs)), worst_text=int(np.argmin(cs)),
                         per_text=[round(c, 6) for c in cs], wall_s=round(dt, 2))
            elif kind == "rerank" and ref is not None:
                j, dt = native_json(cli, gguf, ["--rerank", QUERY], texts, env)
                res = j["results"] if isinstance(j, dict) else j
                key = "index" if res and "index" in res[0] else None
                sc = [r["score"] for r in (sorted(res, key=lambda r: r[key]) if key else res)]
                raw = np.array(ref)
                sig = 1 / (1 + np.exp(-raw))
                best = min((("raw", raw), ("sigmoid", sig)), key=lambda kv: np.abs(np.array(sc) - kv[1]).max())
                d = np.abs(np.array(sc) - best[1])
                a.update(score_mapping=best[0], score_max_abs=float(d.max()), score_mean_abs=float(d.mean()),
                         score_cos=cos(sc, best[1]), worst_text=int(np.argmax(d)),
                         order_equal=bool(list(np.argsort(sc)) == list(np.argsort(best[1]))), wall_s=round(dt, 2))
        except Exception as e:  # noqa: BLE001
            a["output_error"] = f"{type(e).__name__}: {e}"[:1500]
        rec[arm] = a
        log(f"  {alias:40s} {arm:34s} ids {a.get('ids_equal')}/{len(texts)} "
            f"cos_min {a.get('cos_min')} score_max_abs {a.get('score_max_abs')} {a.get('ids_error', '')}"
            f"{a.get('output_error', '')}")
    os.unlink(gguf)
    return rec


def main():
    names = sys.argv[1:] or list(MODELS)
    cli = build()
    reg = registry()
    texts = panel()
    out = {"source": os.environ.get("GITHUB_SHA"), "n_texts": len(texts), "models": []}
    for n in names:
        hf, kind = MODELS[n]
        try:
            out["models"].append(run_model(cli, reg, n, hf, kind, texts))
        except Exception as e:  # noqa: BLE001
            out["models"].append({"alias": n, "error": f"{type(e).__name__}: {e}"[:1500]})
        (OUT / "spm-collapse-ab.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    d0, d1 = "default (collapse on)", "CRISPEMBED_SPM_COLLAPSE_SPACES=0"
    lines = [f"### SPM collapse A/B ({(out['source'] or '')[:8]}, {len(texts)} texts)", "",
             "| model | file | texts | ids = HF fast on / off | ids = SentencePiece on / off | cos min on / off (embed) or max abs score diff on / off (rerank) |",
             "|---|---|---:|---|---|---|"]
    regress = False
    for m in out["models"]:
        if "error" in m:
            lines.append(f"| {m['alias']} | error | | {m['error'][:80]} |")
            regress = True
            continue
        a, b = m.get(d0, {}), m.get(d1, {})
        ia, ib = a.get("ids_equal"), b.get("ids_equal")
        if ia is None:
            regress = True
        if m["kind"] == "embed":
            q = f"{a.get('cos_min', float('nan')):.5f} / {b.get('cos_min', float('nan')):.5f}"
        elif m["kind"] == "rerank":
            q = f"{a.get('score_max_abs', float('nan')):.5f} / {b.get('score_max_abs', float('nan')):.5f}"
        else:
            q = "-"
        sa, sb = a.get("ids_equal_sentencepiece"), b.get("ids_equal_sentencepiece")
        if sa is not None and sb is not None and sa < sb:
            regress = True
        n = len(texts) - len(m.get("excluded_over_length", []))
        lines.append(f"| {m['alias']} | {m.get('file')} | {n} | {ia} / {ib} | {sa} / {sb} | {q} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    sys.exit(1 if regress else 0)


if __name__ == "__main__":
    main()
