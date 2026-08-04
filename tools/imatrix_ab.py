#!/usr/bin/env python3
"""imatrix_ab.py — local A/B harness for importance-matrix quantization.

Treats the f16/f32 source GGUF's embeddings as gold and compares quantized arms
against it with CONTINUOUS metrics only. A thresholded pass/fail (or a bare mean)
cannot see imatrix quality — and cosine is scale-blind (HARD RULE #2b), so the
norm ratio |quant|/|gold| is reported alongside it. Everything runs serially:
one model in memory at a time (16 GB Mac / 8 GB VPS).

Two modes:

  pipeline (default)  calibrate -> quantize baseline -> quantize +imatrix -> A/B
      python tools/imatrix_ab.py --cli build/crispembed \\
          --quant build/crispembed-quantize --src model-f16.gguf --qtype q4_k

  compare             A/B already-built GGUFs (e.g. downloaded from HF) against
                      the gold — same texts, same binary, only the GGUF varies
      python tools/imatrix_ab.py --cli build/crispembed --src model-f16.gguf \\
          --compare shipped-q4_k.gguf imatrix-q4_k.gguf

Corpora: the role-tagged JSONL under tools/kaggle/crispembed-imatrix-quant/
(German + English prose, the model's own query prompt, code, newline-heavy
text). `role: query*` rows are embedded through the CLI's model-derived query
prefix; every other row is embedded with `--prefix ""` (documents take no prefix
in every family we ship). A missing corpus is fatal — silently falling back to a
handful of English one-liners is the defect this harness exists to catch.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent / "kaggle" / "crispembed-imatrix-quant"

# German retrieval sanity: each case is (query, [docs]) with docs[0] the only
# relevant one. Top-1 must be docs[0] — this catches a quant that still scores a
# respectable cosine but has stopped RANKING correctly.
DE_RETRIEVAL = [
    ("Wie hoch ist der Leitzins der Europäischen Zentralbank?", [
        "Die Europäische Zentralbank hat den Leitzins erneut angehoben, um die Inflation zu dämpfen.",
        "Der Rhein ist die verkehrsreichste Wasserstraße Europas.",
        "Kartoffelsalat wird mit Essig, Öl und Brühe angemacht."]),
    ("Wie funktioniert eine Wärmepumpe?", [
        "Eine Wärmepumpe transportiert Wärme aus der Umgebungsluft ins Heizsystem, statt sie zu erzeugen.",
        "Der Deutsche Bundestag wird für vier Jahre gewählt.",
        "Die Zugspitze ist der höchste Berg Deutschlands."]),
    ("Welche Symptome hat eine Grippe?", [
        "Eine Influenza verursacht Fieber, Husten und Gliederschmerzen.",
        "Ein Sparplan auf einen Indexfonds gilt als kostengünstiger Einstieg.",
        "Die Hanse prägte den Ostseehandel über Jahrhunderte."]),
    ("Wie viel Mietkaution darf verlangt werden?", [
        "Eine Mietkaution darf höchstens drei Nettokaltmieten betragen.",
        "Antibiotika wirken nicht gegen Viren.",
        "Der Wolf ist wieder in mehreren Bundesländern heimisch."]),
    ("Wie backe ich ein Sauerteigbrot?", [
        "Mehl, Wasser, Salz und ein Sauerteigansatz werden verknetet und über Nacht gehen gelassen.",
        "Die Bundesnetzagentur überwacht den Wettbewerb im Strommarkt.",
        "Gravitationswellen wurden 2015 erstmals direkt nachgewiesen."]),
]


def load_corpus(stem):
    j = CORPUS_DIR / f"{stem}.jsonl"
    if not j.exists():
        sys.exit(f"missing corpus {j} — refusing to run on a fallback corpus")
    rows = [json.loads(l) for l in j.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [(r.get("role", "doc"), r["text"]) for r in rows if r.get("text")]


def split_roles(rows):
    return ([t for r, t in rows if r.startswith("query")],
            [t for r, t in rows if not r.startswith("query")])


def embed(cli, model, texts, as_query=False):
    """One process, model loaded once. as_query=False disables the auto prefix."""
    if not texts:
        return [], 0.0
    args = [cli, "-m", str(model), "--json"]
    if not as_query:
        args += ["--prefix", ""]
    t0 = time.time()
    r = subprocess.run(args + list(texts), capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        sys.exit(f"embed failed for {model}:\n{r.stderr[-2000:]}")
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        sys.exit(f"could not parse --json output for {model}:\n{r.stdout[:500]}")
    return [[float(x) for x in o["embedding"]] for o in data
            if isinstance(o, dict) and o.get("embedding")], dt


def embed_eval(cli, model, docs, queries):
    """Docs (prefix off) then queries (model's own prompt) — one fixed order so
    every arm's vectors line up index-for-index with the gold."""
    a, t1 = embed(cli, model, docs, as_query=False)
    b, t2 = embed(cli, model, queries, as_query=True)
    return a + b, t1 + t2


def norm(u):
    return math.sqrt(sum(x * x for x in u))


def cosine(a, b):
    na, nb = norm(a), norm(b)
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def stats(arm, gold):
    n = min(len(arm), len(gold))
    if not n:
        sys.exit("no vectors to compare — the CLI returned nothing")
    cs = [cosine(arm[i], gold[i]) for i in range(n)]
    nr = [norm(arm[i]) / norm(gold[i]) if norm(gold[i]) else float("nan")
          for i in range(n)]
    return {
        "n": n,
        "cos_min": min(cs), "cos_mean": sum(cs) / n, "cos_med": statistics.median(cs),
        "nr_mean": sum(nr) / n, "nr_min": min(nr), "nr_max": max(nr),
        "worst": min(range(n), key=lambda i: cs[i]),
    }


def retrieval_sanity(cli, model):
    """5 German cases; docs[0] is the only relevant one. Returns (hits, detail)."""
    hits, detail = 0, []
    for q, docs in DE_RETRIEVAL:
        qv, _ = embed(cli, model, [q], as_query=True)
        dv, _ = embed(cli, model, docs, as_query=False)
        if not qv or len(dv) != len(docs):
            detail.append("EMBED-FAIL")
            continue
        sims = [cosine(qv[0], d) for d in dv]
        top = max(range(len(sims)), key=lambda i: sims[i])
        hits += top == 0
        detail.append("/".join(f"{s:.3f}" for s in sims) + ("" if top == 0 else " <-MISS"))
    return hits, detail


def row(label, path, st, secs):
    mb = os.path.getsize(str(path)) / 1e6 if os.path.exists(str(path)) else 0.0
    return (f"  {label:<28s} {mb:7.1f} MB {secs:5.1f}s  "
            f"cos min={st['cos_min']:.6f} mean={st['cos_mean']:.6f} med={st['cos_med']:.6f}  "
            f"norm x{st['nr_mean']:.4f} [{st['nr_min']:.4f},{st['nr_max']:.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--quant", help="crispembed-quantize (pipeline mode only)")
    ap.add_argument("--src", required=True, help="f16/f32 gold GGUF")
    ap.add_argument("--qtype", default="q4_k")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="pre-built GGUFs to A/B against the gold (skips quantizing)")
    ap.add_argument("--workdir", default="/tmp/imatrix_ab")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-retrieval", action="store_true")
    args = ap.parse_args()

    calib = load_corpus("calib_corpus")
    eval_ = load_corpus("eval_corpus")
    cq, cd = split_roles(calib)
    eq, ed = split_roles(eval_)
    eval_order = [r for r in eval_ if not r[0].startswith("query")] + \
                 [r for r in eval_ if r[0].startswith("query")]
    print(f"corpus: calib {len(calib)} ({len(cd)} doc + {len(cq)} query-prompted, "
          f"{sum(chr(10) in t for _, t in calib)} newline-bearing) | "
          f"eval {len(eval_)} ({len(ed)} doc + {len(eq)} query-prompted)")

    if args.compare is not None:
        arms = [(Path(p).name, p) for p in args.compare]
    else:
        if not args.quant:
            sys.exit("--quant is required in pipeline mode")
        os.makedirs(args.workdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.src))[0]
        imat = os.path.join(args.workdir, base + ".imatrix")
        out_a = os.path.join(args.workdir, f"{base}-{args.qtype}.gguf")
        out_b = os.path.join(args.workdir, f"{base}-{args.qtype}-imatrix.gguf")
        if os.path.exists(imat):
            os.remove(imat)
        print(f"[1/3] calibration over {len(calib)} texts -> {imat}")
        env = dict(os.environ, CRISPEMBED_IMATRIX_OUT=imat)
        for texts, as_q in ((cd, False), (cq, True)):
            cmd = [args.cli, "-m", args.src, "--json"] + ([] if as_q else ["--prefix", ""])
            r = subprocess.run(cmd + texts, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"calibration (query={as_q}) failed:\n{r.stderr[-2000:]}")
        # Fail loudly: a missing/empty imatrix silently degrades the "+imatrix"
        # arm into a plain quant (clean_exit skips atexit; the collector flushes
        # explicitly from crispembed_free).
        if not os.path.exists(imat) or os.path.getsize(imat) == 0:
            sys.exit(f"calibration produced NO imatrix at {imat}")
        print(f"      {os.path.getsize(imat)//1024} KB")
        print(f"[2/3] quantize baseline / +imatrix ({args.qtype})")
        for out, extra in ((out_a, []), (out_b, ["--imatrix", imat])):
            r = subprocess.run([args.quant, args.src, out, args.qtype] + extra,
                               capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"quantize {out} failed:\n{r.stderr[-2000:]}")
            if extra:
                tail = [l for l in r.stdout.splitlines() if "with imatrix" in l]
                if tail:
                    print("      " + tail[-1].strip())
        arms = [(f"{args.qtype} baseline", out_a), (f"{args.qtype} +imatrix", out_b)]

    print(f"[3/3] eval over {len(eval_)} held-out texts (serial)")
    gold, t_gold = embed_eval(args.cli, args.src, ed, eq)
    results = []
    for label, path in arms:
        vecs, secs = embed_eval(args.cli, path, ed, eq)
        results.append((label, path, stats(vecs, gold), secs))

    print(f"\n===== A/B vs {os.path.basename(args.src)} (gold, {t_gold:.1f}s) =====")
    for label, path, st, secs in results:
        print(row(label, path, st, secs))
        w = st["worst"]
        if w < len(eval_order):
            print(f"      worst text [{eval_order[w][0]}]: {eval_order[w][1][:70]!r}")
    if len(results) == 2:
        a, b = results[0][2], results[1][2]
        print(f"  delta (arm2-arm1): cos_min {b['cos_min']-a['cos_min']:+.6f}  "
              f"cos_mean {b['cos_mean']-a['cos_mean']:+.6f}")

    if not args.no_retrieval:
        print("\n===== German retrieval sanity (top-1 must be doc[0]) =====")
        for label, path in [("gold", args.src)] + [(l, p) for l, p, _, _ in results]:
            hits, detail = retrieval_sanity(args.cli, path)
            print(f"  {label:<28s} {hits}/{len(DE_RETRIEVAL)}   " + " | ".join(detail))

    if args.compare is None and not args.keep:
        for _, p in arms:
            if os.path.exists(p):
                os.remove(p)
        print("\n  (removed intermediate GGUFs; --keep to retain)")


if __name__ == "__main__":
    main()
