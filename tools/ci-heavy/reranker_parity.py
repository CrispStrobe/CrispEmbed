#!/usr/bin/env python3
"""Cross-encoder reranker parity against the original HF models, plus the over-length
input check for the SentencePiece encoders (hosted CPU, portable build).

    gh workflow run heavy-cpu.yml --ref <branch> -f script=tools/ci-heavy/reranker_parity.py \
        -f isolated_lane=reranker-parity-1 \
        -f pip="torch transformers>=4.51,<5 sentence-transformers einops gguf huggingface_hub sentencepiece protobuf"

Arguments: optional model aliases (default: every reranker in RERANKERS, then every
encoder in LONG_INPUT).

Rerankers. Query "What did Einstein publish in 1905?" against the 77-text panel of
tools/ci-heavy/spm_collapse_ab.py. Reference: AutoModelForSequenceClassification fp32,
raw logit, truncation at the tokenizer's model_max_length (what sentence-transformers
CrossEncoder does). Arms: default, CRISPEMBED_PAIR_ROBERTA_SEP=0 (single separator, the
old layout). Reported: max/mean |native - HF|, Spearman-free order equality, and a stage
diff (CRISPEMBED_DUMP_LAYERS_GGUF vs HF hidden_states) on the worst text of each arm, with
the pair length so a token-layout difference is visible as a length mismatch.

Long input. For each encoder, the longest panel text repeated past its context: default
arm must exit 0 and keep exactly the original's truncated token count (special tokens
included); CRISPEMBED_POS_EMBD_TRUNCATE=0 is recorded (the old abort). Embedders also get
the final-embedding cosine against sentence-transformers on that truncated input.
Exit status is non-zero when a default arm crashes or is worse than the legacy arm.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spm_collapse_ab as base  # noqa: E402  (build, panel, registry, pick, download, cos, sh)

OUT, S, NT, QUERY = base.OUT, base.S, base.NT, base.QUERY

RERANKERS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-base": "BAAI/bge-reranker-base",
    "jina-reranker-v2-base-multilingual": "jinaai/jina-reranker-v2-base-multilingual",
    "mxbai-rerank-xsmall-v1": "mixedbread-ai/mxbai-rerank-xsmall-v1",
    "mxbai-rerank-base-v1": "mixedbread-ai/mxbai-rerank-base-v1",
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ettin-reranker-17m": "cross-encoder/ettin-reranker-17m-v1",
}
LONG_INPUT = {
    "multilingual-e5-small": ("intfloat/multilingual-e5-small", "embed"),
    "multilingual-e5-base": ("intfloat/multilingual-e5-base", "embed"),
    "multilingual-e5-large": ("intfloat/multilingual-e5-large", "embed"),
    "granite-embedding-107m": ("ibm-granite/granite-embedding-107m-multilingual", "embed"),
    "granite-embedding-278m": ("ibm-granite/granite-embedding-278m-multilingual", "embed"),
    "paraphrase-multilingual-MiniLM-L12-v2": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "embed"),
    "bge-reranker-base": ("BAAI/bge-reranker-base", "rerank"),
    "xlmr-ner-hrl": ("Davlan/xlm-roberta-base-ner-hrl", "tokens"),
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", "embed"),
}
ARMS = {"default": {}, "CRISPEMBED_PAIR_ROBERTA_SEP=0": {"CRISPEMBED_PAIR_ROBERTA_SEP": "0"}}


def log(*a):
    print(*a, flush=True)


def rerank_native(cli, gguf, docs, env):
    j, dt = base.native_json(cli, gguf, ["--rerank", QUERY], docs, env)
    res = j["results"] if isinstance(j, dict) else j
    if res and "index" in res[0]:
        res = sorted(res, key=lambda r: r["index"])
    return [float(r["score"]) for r in res], dt


def stage_diff(cli, gguf, doc, hf_states, env, tag):
    """Native layer dump for one (QUERY, doc) pair vs HF hidden_states."""
    import gguf as gg
    path = S / f"dump-{tag}.gguf"
    if path.exists():
        path.unlink()
    r, _ = base.sh([cli, "-m", gguf, "-t", NT, "--json", "--prefix", "", "--rerank", QUERY, doc],
                   env=dict(env, CRISPEMBED_DUMP_LAYERS="1", CRISPEMBED_DUMP_LAYERS_GGUF=str(path)))
    if r.returncode or not path.exists():
        return {"error": f"rc {r.returncode}: {r.stderr[-300:]}"}
    rd = gg.GGUFReader(str(path))
    nat = {t.name: np.asarray(t.data, np.float64).reshape(-1, hf_states[0].shape[-1]) for t in rd.tensors}
    n_layer = len(hf_states) - 1
    pairs = [("emb_ln_out", 0)] + [(f"layer_{i}", i + 1) for i in range(n_layer - 1)] + [("encoder_out", n_layer)]
    out = {"hf_len": int(hf_states[0].shape[0])}
    stages = []
    for name, k in pairs:
        if name not in nat:
            continue
        a, b = nat[name], hf_states[k].astype(np.float64)
        out.setdefault("native_len", int(a.shape[0]))
        n = min(len(a), len(b))
        rowcos = [base.cos(a[i], b[i]) for i in range(n)]
        stages.append({"stage": name, "cls_cos": rowcos[0], "min_row_cos": float(min(rowcos)),
                       "max_abs": float(np.abs(a[:n] - b[:n]).max())})
    out["stages"] = stages
    if stages:
        w = min(stages, key=lambda s: s["min_row_cos"])
        out["worst_stage"] = w["stage"]
        out["worst_min_row_cos"] = w["min_row_cos"]
        out["first_below_0p999"] = next((s["stage"] for s in stages if s["min_row_cos"] < 0.999), None)
    path.unlink()
    return out


def run_reranker(cli, reg, alias, hf, texts):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(int(NT))
    rec = {"alias": alias, "hf": hf}
    used, (fname, url) = base.pick(reg, alias)
    rec.update(registry_alias=used, file=fname, url=url)
    gguf = base.download(url, fname)
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    ml = tok.model_max_length if 0 < (tok.model_max_length or 0) < 100000 else 512
    rec["hf_max_length"] = ml
    pair = tok("a b", "c d")
    rec["hf_pair_layout"] = tok.convert_ids_to_tokens(pair["input_ids"])
    m = AutoModelForSequenceClassification.from_pretrained(hf, trust_remote_code=True,
                                                           torch_dtype=torch.float32).eval()
    ref, lens = [], []
    with torch.no_grad():
        for t in texts:
            e = tok(QUERY, t, return_tensors="pt", truncation=True, max_length=ml)
            lens.append(int(e["input_ids"].shape[1]))
            ref.append(float(m(**e).logits.reshape(-1)[0]))
    rec["hf_pair_len_max"] = max(lens)
    ref = np.array(ref)
    for arm, env in ARMS.items():
        a = {}
        try:
            sc, dt = rerank_native(cli, gguf, texts, env)
            d = np.abs(np.array(sc) - ref)
            a.update(score_max_abs=float(d.max()), score_mean_abs=float(d.mean()), worst_text=int(np.argmax(d)),
                     order_equal=bool(list(np.argsort(sc)) == list(np.argsort(ref))),
                     top1_equal=bool(int(np.argmax(sc)) == int(np.argmax(ref))), wall_s=round(dt, 2),
                     per_text=[round(float(x), 5) for x in sc])
            w = texts[a["worst_text"]]
            with torch.no_grad():
                e = tok(QUERY, w, return_tensors="pt", truncation=True, max_length=ml)
                hs = m(**e, output_hidden_states=True).hidden_states
            a["stage_diff_worst"] = stage_diff(cli, gguf, w, [h[0].numpy() for h in hs], env, f"{alias}-{len(rec)}")
        except Exception as e:  # noqa: BLE001
            a["error"] = f"{type(e).__name__}: {e}"[:1500]
        rec[arm] = a
        sd = a.get("stage_diff_worst", {})
        log(f"  {alias:36s} {arm:32s} max|d| {a.get('score_max_abs')} mean {a.get('score_mean_abs')} "
            f"order {a.get('order_equal')} len hf/native {sd.get('hf_len')}/{sd.get('native_len')} "
            f"worst stage {sd.get('worst_stage')} {sd.get('worst_min_row_cos')} {a.get('error', '')}")
    rec["hf_scores"] = [round(float(x), 5) for x in ref]
    os.unlink(gguf)
    return rec


def run_long(cli, reg, alias, hf, kind, texts):
    import torch
    from transformers import AutoTokenizer
    rec = {"alias": alias, "hf": hf, "kind": kind}
    used, (fname, url) = base.pick(reg, alias)
    rec.update(registry_alias=used, file=fname, url=url)
    gguf = base.download(url, fname)
    tok = AutoTokenizer.from_pretrained(hf, trust_remote_code=True)
    longest = max(texts, key=len)
    text = longest
    while len(tok(text)["input_ids"]) < 1200:
        text = text + " " + longest
    ml = tok.model_max_length if 0 < (tok.model_max_length or 0) < 100000 else 512
    hf_ids = tok(text, truncation=True, max_length=ml)["input_ids"]
    rec.update(hf_max_length=ml, hf_truncated_len=len(hf_ids), untruncated_len=len(tok(text)["input_ids"]))
    for arm, env in {"default": {}, "CRISPEMBED_POS_EMBD_TRUNCATE=0": {"CRISPEMBED_POS_EMBD_TRUNCATE": "0"}}.items():
        r, _ = base.sh([cli, "-m", gguf, "-t", NT, "--json", "--prefix", "", "--tokens-raw", text], env=env)
        a = {"rc": r.returncode}
        if r.returncode == 0:
            try:
                j = json.loads(r.stdout)
                ids = (j[0] if isinstance(j, list) else j).get("token_ids", [])
                a.update(native_len=len(ids), ids_equal_hf=ids == hf_ids,
                         keeps_special=bool(ids) and ids[0] == hf_ids[0] and ids[-1] == hf_ids[-1])
            except Exception as e:  # noqa: BLE001
                a["parse_error"] = str(e)[:300]
        else:
            a["stderr_tail"] = r.stderr[-300:]
        if kind == "embed" and r.returncode == 0:
            try:
                from sentence_transformers import SentenceTransformer
                st = SentenceTransformer(hf, device="cpu", trust_remote_code=True,
                                         model_kwargs={"torch_dtype": torch.float32})
                ref = st.encode([text], normalize_embeddings=True)[0]
                j, _ = base.native_json(cli, gguf, [], [text], env)
                a["st_max_seq_length"] = st.max_seq_length
                a["cos_vs_st"] = base.cos(j[0]["embedding"], ref)
            except Exception as e:  # noqa: BLE001
                a["embed_error"] = f"{type(e).__name__}: {e}"[:300]
        if kind == "rerank" and r.returncode == 0:
            try:
                sc, _ = rerank_native(cli, gguf, [text], env)
                a["rerank_rc0_score"] = sc[0]
            except Exception as e:  # noqa: BLE001
                a["rerank_error"] = f"{type(e).__name__}: {e}"[:300]
        rec[arm] = a
        log(f"  long {alias:36s} {arm:32s} {a}")
    os.unlink(gguf)
    return rec


def main():
    names = sys.argv[1:]
    cli = base.build()
    reg = base.registry()
    texts = base.panel()
    out = {"source": os.environ.get("GITHUB_SHA"), "build_mode": "portable (-DGGML_NATIVE=OFF)",
           "n_texts": len(texts), "rerankers": [], "long_input": []}
    for n, hf in RERANKERS.items():
        if names and n not in names:
            continue
        try:
            out["rerankers"].append(run_reranker(cli, reg, n, hf, texts))
        except Exception as e:  # noqa: BLE001
            out["rerankers"].append({"alias": n, "error": f"{type(e).__name__}: {e}"[:1500]})
        (OUT / "reranker-parity.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for n, (hf, kind) in LONG_INPUT.items():
        if names and n not in names:
            continue
        try:
            out["long_input"].append(run_long(cli, reg, n, hf, kind, texts))
        except Exception as e:  # noqa: BLE001
            out["long_input"].append({"alias": n, "error": f"{type(e).__name__}: {e}"[:1500]})
        (OUT / "reranker-parity.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    bad = False
    d0, d1 = "default", "CRISPEMBED_PAIR_ROBERTA_SEP=0"
    lines = [f"### Reranker parity ({(out['source'] or '')[:8]}, portable, {len(texts)} texts)", "",
             "| model | file | HF pair layout | max abs Δ default / sep=0 | mean abs Δ default / sep=0 | order equal default / sep=0 | worst stage (default) |",
             "|---|---|---|---|---|---|---|"]
    for m in out["rerankers"]:
        if "error" in m:
            lines.append(f"| {m['alias']} | error | {m['error'][:100]} | | | | |")
            bad = True
            continue
        a, b = m.get(d0, {}), m.get(d1, {})
        if "error" in a or a.get("score_max_abs", 0) > b.get("score_max_abs", 1e9) + 1e-4:
            bad = True
        sd = a.get("stage_diff_worst", {})
        lines.append(f"| {m['alias']} | {m['file']} | `{' '.join(m['hf_pair_layout'])}` | "
                     f"{a.get('score_max_abs', float('nan')):.4f} / {b.get('score_max_abs', float('nan')):.4f} | "
                     f"{a.get('score_mean_abs', float('nan')):.4f} / {b.get('score_mean_abs', float('nan')):.4f} | "
                     f"{a.get('order_equal')} / {b.get('order_equal')} | "
                     f"{sd.get('worst_stage')} {sd.get('worst_min_row_cos', float('nan')):.5f} (len {sd.get('hf_len')}/{sd.get('native_len')}) |")
    lines += ["", "| encoder | HF truncated len | default rc / len / ids = HF | POS_EMBD_TRUNCATE=0 rc | cos vs ST (default) |",
              "|---|---:|---|---|---|"]
    for m in out["long_input"]:
        if "error" in m:
            lines.append(f"| {m['alias']} | error {m['error'][:100]} | | | |")
            bad = True
            continue
        a, b = m.get("default", {}), m.get("CRISPEMBED_POS_EMBD_TRUNCATE=0", {})
        if a.get("rc") != 0:
            bad = True
        lines.append(f"| {m['alias']} | {m['hf_truncated_len']} | {a.get('rc')} / {a.get('native_len')} / "
                     f"{a.get('ids_equal_hf')} | {b.get('rc')} | {a.get('cos_vs_st')} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
