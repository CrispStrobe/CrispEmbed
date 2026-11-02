#!/usr/bin/env python3
"""Metal-vs-CPU A/B for crispembed-cli embedding/rerank/sparse/ColBERT modes.

Each arm is a separate process of the SAME binary. The Metal arm must show a
Metal device on its own stderr and the CPU arm must not; the runner raises
otherwise, so a silently-CPU "Metal" arm cannot pass. Output: one JSON receipt
per (model, mode) with max-abs, min cosine and wall time, plus stderr tails.

Usage:
  ab_embed.py BIN MODEL MODE TEXTS OUT.json [--query Q] [--repeat N] [--env K=V ...]
  MODE in {embed, sparse, colbert, rerank}
"""
import argparse, json, math, os, subprocess, time


def run(bin_, model, mode, texts, backend, query, env_extra, threads):
    cmd = [bin_, "-m", model, "-t", str(threads), "--gpu-backend", backend, "--json", "-f", texts]
    if mode == "sparse":
        cmd.append("--sparse")
    elif mode == "colbert":
        cmd.append("--colbert")
    elif mode == "rerank":
        cmd += ["--rerank", query]
    env = dict(os.environ)
    env.update(env_extra)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{backend} arm failed rc={p.returncode}: {p.stderr[-2000:]}")
    err = p.stderr
    has_mtl = "MTL0" in err or "ggml_metal" in err
    if backend == "metal" and not has_mtl:
        raise RuntimeError("metal arm shows no Metal device on stderr")
    if backend == "cpu" and has_mtl:
        raise RuntimeError("cpu arm initialised Metal")
    out = json.loads(p.stdout)
    if mode == "rerank":
        out = out["results"]
    return out, wall, err[-1500:]


def vec_stats(a, b):
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    cos = num / (na * nb) if na and nb else float("nan")
    mx = max(abs(x - y) for x, y in zip(a, b)) if a else 0.0
    return cos, mx


def compare(mode, m, c):
    rows = []
    assert len(m) == len(c), "row count differs"
    if mode == "embed":
        for i, (x, y) in enumerate(zip(m, c)):
            cos, mx = vec_stats(x["embedding"], y["embedding"])
            rows.append({"i": i, "cos": cos, "max_abs": mx, "dim": len(x["embedding"])})
    elif mode == "colbert":
        for i, (x, y) in enumerate(zip(m, c)):
            assert x["n_tokens"] == y["n_tokens"], "token count differs"
            coss, mxs = [], []
            for a, b in zip(x["vectors"], y["vectors"]):
                cs, mm = vec_stats(a, b)
                coss.append(cs)
                mxs.append(mm)
            rows.append({"i": i, "n_tokens": x["n_tokens"], "cos_min": min(coss), "max_abs": max(mxs)})
    elif mode == "sparse":
        for i, (x, y) in enumerate(zip(m, c)):
            dm = {e["token_id"]: e["weight"] for e in x["sparse"]}
            dc = {e["token_id"]: e["weight"] for e in y["sparse"]}
            keys = set(dm) | set(dc)
            mx = max((abs(dm.get(k, 0.0) - dc.get(k, 0.0)) for k in keys), default=0.0)
            rows.append({"i": i, "terms_metal": len(dm), "terms_cpu": len(dc),
                         "same_terms": set(dm) == set(dc), "max_abs": mx})
    elif mode == "rerank":
        key = "index" if "index" in m[0] else None
        sm = [r["score"] for r in (sorted(m, key=lambda r: r[key]) if key else m)]
        sc = [r["score"] for r in (sorted(c, key=lambda r: r[key]) if key else c)]
        om = [r.get("index") for r in m]
        oc = [r.get("index") for r in c]
        rows.append({"scores_metal": sm, "scores_cpu": sc,
                     "max_abs": max(abs(a - b) for a, b in zip(sm, sc)), "same_order": om == oc})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("model")
    ap.add_argument("mode")
    ap.add_argument("texts")
    ap.add_argument("out")
    ap.add_argument("--query", default="What did Einstein publish in 1905?")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--ref", default=None,
                    help="higher-precision GGUF of the same model, run once on CPU as the reference")
    a = ap.parse_args()
    env_extra = dict(kv.split("=", 1) for kv in a.env)
    walls = {"metal": [], "cpu": []}
    outs, errs = {}, {}
    for _ in range(a.repeat):  # interleaved arms
        for be in ("metal", "cpu"):
            o, w, e = run(a.bin, a.model, a.mode, a.texts, be, a.query, env_extra, a.threads)
            walls[be].append(round(w, 3))
            outs[be], errs[be] = o, e
    rows = compare(a.mode, outs["metal"], outs["cpu"])
    ref_summary = None
    if a.ref and a.mode == "embed":
        ro, _, _ = run(a.bin, a.ref, a.mode, a.texts, "cpu", a.query, env_extra, a.threads)
        for r, x, y, z in zip(rows, outs["metal"], outs["cpu"], ro):
            r["cos_metal_vs_ref"] = vec_stats(x["embedding"], z["embedding"])[0]
            r["cos_cpu_vs_ref"] = vec_stats(y["embedding"], z["embedding"])[0]
        ref_summary = {"ref": os.path.basename(a.ref),
                       "cos_min_metal_vs_ref": min(r["cos_metal_vs_ref"] for r in rows),
                       "cos_min_cpu_vs_ref": min(r["cos_cpu_vs_ref"] for r in rows)}
    elif a.ref and a.mode == "rerank":
        ro, _, _ = run(a.bin, a.ref, a.mode, a.texts, "cpu", a.query, env_extra, a.threads)
        key = "index" if "index" in ro[0] else None
        sr = [r["score"] for r in (sorted(ro, key=lambda r: r[key]) if key else ro)]
        ref_summary = {"ref": os.path.basename(a.ref),
                       "max_abs_metal_vs_ref": max(abs(p - q) for p, q in zip(rows[0]["scores_metal"], sr)),
                       "max_abs_cpu_vs_ref": max(abs(p - q) for p, q in zip(rows[0]["scores_cpu"], sr)),
                       "order_ref": [r.get("index") for r in ro]}
    rec = {"model": os.path.basename(a.model), "mode": a.mode, "texts": os.path.basename(a.texts),
           "threads": a.threads, "env": env_extra, "wall_s": walls, "rows": rows,
           "stderr_tail": errs, "date": time.strftime("%Y-%m-%d")}
    if a.mode == "embed":
        rec["cos_min"] = min(r["cos"] for r in rows)
    elif a.mode == "colbert":
        rec["cos_min"] = min(r["cos_min"] for r in rows)
    rec["max_abs"] = max(r["max_abs"] for r in rows)
    if ref_summary:
        rec["reference"] = ref_summary
    json.dump(rec, open(a.out, "w"), indent=1)
    summ = {k: rec[k] for k in ("model", "mode", "wall_s", "cos_min", "max_abs", "reference") if k in rec}
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
