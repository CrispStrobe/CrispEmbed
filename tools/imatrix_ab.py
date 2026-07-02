#!/usr/bin/env python3
"""imatrix_ab.py — A/B harness for C1 (importance-matrix quantization).

Measures BOTH quality and speed of imatrix vs plain quantization, treating the
f16/f32 source model's embeddings as the gold reference. Everything runs
serially (16 GB Mac / 8 GB VPS constraint: never load two heavy models at once).

Pipeline:
  1. calibration : run crispembed with CRISPEMBED_IMATRIX_OUT over a calibration
                   corpus (one process, model loaded once) -> <model>.imatrix
  2. quant A     : crispembed-quantize <src> <out_a> <qtype>              (baseline)
  3. quant B     : crispembed-quantize <src> <out_b> <qtype> --imatrix    (candidate)
  4. eval        : embed a held-out corpus with src / A / B, report
                   mean cosine(A, src) vs mean cosine(B, src), and wall-clock.

Accept criterion (C1): mean cos(B,src) >= mean cos(A,src) (imatrix must not
regress; target: close part of the gap to the f16 gold).

Usage:
  python tools/imatrix_ab.py --cli build/crispembed --quant build/crispembed-quantize \\
      --src model-f16.gguf --qtype q4_k [--workdir /tmp/ab] [--keep]
"""
import argparse, json, os, subprocess, sys, time, math

# Small but domain-diverse calibration corpus. For production runs, replace with
# text resembling the target embedding domain (see PLAN.md C1).
CALIB = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models transform raw text into dense vector representations.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "The mitochondria is the powerhouse of the cell.",
    "Quarterly revenue grew 12% year over year, beating analyst expectations.",
    "To reset your password, click the link in the confirmation email.",
    "The French Revolution began in 1789 and reshaped European politics.",
    "SELECT user_id, COUNT(*) FROM orders GROUP BY user_id HAVING COUNT(*) > 5;",
    "Preheat the oven to 200 degrees and bake the bread for 35 minutes.",
    "Gravitational waves were first directly detected by LIGO in 2015.",
    "The customer complained that the package arrived damaged and late.",
    "In distributed systems, consensus protocols like Raft ensure consistency.",
    "She walked along the beach at sunset, listening to the waves.",
    "The GDP deflator measures the price level of all domestically produced goods.",
    "Kubernetes orchestrates containerized workloads across a cluster of nodes.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The novel explores themes of memory, loss, and the passage of time.",
    "Interest rates were held steady by the central bank amid inflation concerns.",
    "A binary search tree keeps keys in sorted order for O(log n) lookup.",
    "The hikers reached the summit just before the storm rolled in.",
    "Antibiotics are ineffective against viral infections such as the common cold.",
    "The startup raised a Series B round led by a prominent venture firm.",
    "Regular expressions match patterns of characters within strings of text.",
    "The treaty established new trade routes between the two nations.",
    "Neural networks learn hierarchical features through backpropagation.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Climate models predict rising sea levels over the coming century.",
    "The API returns a paginated JSON response with a cursor token.",
    "Shakespeare wrote both comedies and tragedies during his career.",
    "Vaccines train the immune system to recognize specific pathogens.",
    "The bridge was engineered to withstand magnitude eight earthquakes.",
    "Cache invalidation is one of the hardest problems in computer science.",
]

# Held-out eval corpus (disjoint from CALIB).
EVAL = [
    "A large language model can summarize documents and answer questions.",
    "The stock market rallied after the earnings report was released.",
    "import numpy as np; a = np.zeros((3, 3)); a[1, 1] = 1.0",
    "The Amazon rainforest produces a significant share of the world's oxygen.",
    "Please find attached the invoice for the services rendered last month.",
    "Dijkstra's algorithm finds the shortest path in a weighted graph.",
    "The orchestra performed a symphony by Beethoven to a full house.",
    "Enzymes act as catalysts to accelerate biochemical reactions.",
    "The refugees were granted asylum after a lengthy legal process.",
    "A hash map provides average constant-time insertion and lookup.",
    "The telescope captured images of a distant spiral galaxy.",
    "Our return policy allows exchanges within thirty days of purchase.",
]


def run(cmd, env=None):
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def embed(cli, model, texts):
    """Return (list_of_vectors, wall_seconds). One process, model loaded once."""
    t0 = time.time()
    r = run([cli, "-m", model, "--json", *texts])
    dt = time.time() - t0
    if r.returncode != 0:
        sys.exit(f"embed failed for {model}:\n{r.stderr[-2000:]}")
    # crispembed --json prints a single JSON array of {"text","embedding"} objects.
    vecs = []
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        sys.exit(f"could not parse --json output for {model}:\n{r.stdout[:500]}")
    for obj in data:
        emb = obj.get("embedding") if isinstance(obj, dict) else None
        if isinstance(emb, list) and emb:
            vecs.append([float(x) for x in emb])
    return vecs, dt


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def mean_cos(vs_a, vs_b):
    n = min(len(vs_a), len(vs_b))
    if n == 0:
        return float("nan"), 0
    return sum(cosine(vs_a[i], vs_b[i]) for i in range(n)) / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--src", required=True, help="f16/f32 source GGUF")
    ap.add_argument("--qtype", default="q4_k")
    ap.add_argument("--workdir", default="/tmp/imatrix_ab")
    ap.add_argument("--keep", action="store_true", help="keep intermediate GGUFs")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.src))[0]
    imat = os.path.join(args.workdir, base + ".imatrix")
    out_a = os.path.join(args.workdir, f"{base}-{args.qtype}.gguf")
    out_b = os.path.join(args.workdir, f"{base}-{args.qtype}-imatrix.gguf")

    # 1. calibration
    if os.path.exists(imat):
        os.remove(imat)
    print(f"[1/4] calibration over {len(CALIB)} texts -> {imat}")
    env = dict(os.environ, CRISPEMBED_IMATRIX_OUT=imat)
    t0 = time.time()
    r = run([args.cli, "-m", args.src, "--json", *CALIB], env=env)
    if r.returncode != 0 or not os.path.exists(imat):
        sys.exit(f"calibration failed:\n{r.stderr[-2000:]}")
    print(f"      done in {time.time()-t0:.1f}s  ({os.path.getsize(imat)//1024} KB)")

    # 2 + 3. quantize baseline and imatrix
    print(f"[2/4] quantize baseline  -> {out_a}")
    r = run([args.quant, args.src, out_a, args.qtype])
    if r.returncode != 0:
        sys.exit(f"baseline quant failed:\n{r.stderr[-2000:]}")
    print(f"[3/4] quantize +imatrix  -> {out_b}")
    r = run([args.quant, args.src, out_b, args.qtype, "--imatrix", imat])
    if r.returncode != 0:
        sys.exit(f"imatrix quant failed:\n{r.stderr[-2000:]}")
    n_im = [l for l in r.stdout.splitlines() if "with imatrix" in l]
    if n_im:
        print("      " + n_im[-1].strip())

    # 4. eval (serial: one model in memory at a time)
    print(f"[4/4] eval over {len(EVAL)} held-out texts")
    vs_src, t_src = embed(args.cli, args.src, EVAL)
    vs_a,   t_a   = embed(args.cli, out_a, EVAL)
    vs_b,   t_b   = embed(args.cli, out_b, EVAL)

    cos_a, na = mean_cos(vs_a, vs_src)
    cos_b, nb = mean_cos(vs_b, vs_src)
    sz = lambda p: os.path.getsize(p) / 1e6

    print("\n===== A/B RESULT (" + args.qtype + ") =====")
    print(f"  source     {os.path.basename(args.src):40s} {sz(args.src):7.1f} MB  embed {t_src:5.2f}s")
    print(f"  A baseline {os.path.basename(out_a):40s} {sz(out_a):7.1f} MB  embed {t_a:5.2f}s  cos(A,src)={cos_a:.6f}")
    print(f"  B +imatrix {os.path.basename(out_b):40s} {sz(out_b):7.1f} MB  embed {t_b:5.2f}s  cos(B,src)={cos_b:.6f}")
    print(f"  quality delta (B-A): {cos_b - cos_a:+.6f}   over n={min(na,nb)} texts")
    verdict = "PASS (imatrix >= baseline)" if cos_b >= cos_a - 1e-6 else "REGRESSION"
    print(f"  VERDICT: {verdict}")

    if not args.keep:
        for p in (out_a, out_b):
            if os.path.exists(p):
                os.remove(p)
        print("  (removed intermediate GGUFs; --keep to retain)")


if __name__ == "__main__":
    main()
