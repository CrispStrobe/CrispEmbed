#!/usr/bin/env python3
"""CrispEmbed — batch imatrix quantization + A/B + upload (Kaggle kernel).

Follows the standard CrispEmbed Kaggle regime (kaggle_harness = kh):
  * kh.init_progress()      — line-buffered I/O + JSONL progress, pushed to HF
  * kh.resolve_hf_token()   — env → Kaggle Secret → mounted DATASET (hf_token.txt)
  * kh.install_build_toolchain() + ccache warmed from the crispasr-ccache dataset
  * kh.build_heartbeat(...)  — 30 s heartbeat around every long step
Attach BOTH datasets in kernel-metadata.json:
    "dataset_sources": ["chr1str/crispasr-hf-token", "chr1str/crispasr-ccache"]

Processes a LIST of models in ONE kernel run (build once, then per model:
download source → calibrate → quantize (+imatrix) → A/B → upload → rm → next) —
the "rm, next" loop. Select with the MODELS env var (comma list) or DEFAULT_BATCH
below. Per-model failures are isolated (logged, skipped).

For each model the source is the existing full-precision GGUF ALREADY in its
cstr/<name>-GGUF repo (auto-detected: largest non-quant .gguf) — no HF
re-conversion, so LoRA models (jina-v5) and odd namings just work. Imatrix
outputs use DISTINCT names and NEVER overwrite the canonical q8_0/q4_k baselines.
"""
import os, re, sys, json, math, time, shutil, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
if not WORK.exists():
    WORK = Path("/tmp/crisp-imatrix-work"); WORK.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
BRANCH   = os.environ.get("CRISP_BRANCH", "main")   # C1 is on main

# ── bootstrap kaggle_harness (kh): CrispASR clone, else bundled sibling copy ───
CRISPASR_DIR = WORK / "CrispASR"
if not CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
            "https://github.com/CrispStrobe/CrispASR.git", str(CRISPASR_DIR)])
        sys.path.insert(0, str(CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).resolve().parent))   # bundled fallback
import kaggle_harness as kh

# ── models to process (each maps to repo cstr/<name>-GGUF) ────────────────────
# The first three re-run to backfill their A/B summary (added after their initial
# run). The rest extend imatrix coverage across the embedding roster.
DEFAULT_BATCH = [
    # backfill summaries:
    "lfm2-embed", "jina-v5-nano", "bge-m3",
    # (e5-large, jina-v5-small already have summaries)
    # new coverage:
    "bge-large-en-v1.5", "bge-base-en-v1.5", "bge-small-en-v1.5",
    "mxbai-embed-large-v1", "multilingual-e5-base", "multilingual-e5-small",
    "nomic-embed-text-v1.5", "nomic-embed-text-v2-moe", "arctic-embed-l-v2",
    "gte-base-en-v1.5", "gte-large-en-v1.5", "octen-0.6b", "f2llm-v2-0.6b",
    "qwen3-embed-0.6b", "pixie-rune-v1",
    # group 2 (<=2.4GB, fit Kaggle CPU RAM):
    "all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2", "gte-small",
    "arctic-embed-xs", "snowflake-arctic-embed-m", "snowflake-arctic-embed-l",
    "paraphrase-multilingual-MiniLM-L12-v2", "harrier-270m", "harrier-0.6b",
    # embeddinggemma-300m re-enabled: the dense.* keep-F32 guard in
    # tools/quantize.cpp fixes the "tensor read out of bounds" load failure.
    "embeddinggemma-300m",
    # group 3 — large decoder embedders (f32 base 16-30GB). Handled by the big-base
    # path: calibrate/gold on the q8_0 (fits RAM), quantize from f32 (streaming),
    # stage in /tmp. 4B first to validate before the 30GB 8B downloads.
    "octen-4b", "qwen3-embed-4b", "octen-8b", "qwen3-embed-8b",
]
RUN = [m.strip() for m in os.environ.get("MODELS", "").split(",") if m.strip()] or DEFAULT_BATCH

# Optional per-model overrides: {name: {"hf_out":..., "quants":...}}.
# Default hf_out = cstr/<name>-GGUF; default quants = QSPECS.
OVERRIDES = {}

# QSPECS: (qtype, use_imatrix, upload_name_template | None). upload_name None =
# A/B reference only (never uploaded — don't clobber baselines). imatrix variants
# get DISTINCT names: q4_k+imatrix -> *-q4_k-imatrix.gguf, iq4_xs -> *-iq4_xs.gguf.
QSPECS = [
    ("q8_0",   False, None),                          # A/B reference (baseline exists)
    ("q4_k",   False, None),                          # A/B baseline (no imatrix) — shows the delta
    ("q4_k",   True,  "{prefix}-q4_k-imatrix.gguf"),
    ("iq4_xs", True,  "{prefix}-iq4_xs.gguf"),
]

_CALIB_FB = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models transform raw text into dense vector representations.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "Quarterly revenue grew 12% year over year, beating analyst expectations.",
    "Gravitational waves were first directly detected by LIGO in 2015.",
    "In distributed systems, consensus protocols like Raft ensure consistency.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "A binary search tree keeps keys in sorted order for O(log n) lookup.",
    "Neural networks learn hierarchical features through backpropagation.",
    "Cache invalidation is one of the hardest problems in computer science.",
]
_EVAL_FB = [
    "A large language model can summarize documents and answer questions.",
    "The stock market rallied after the earnings report was released.",
    "import numpy as np; a = np.zeros((3, 3)); a[1, 1] = 1.0",
    "A hash map provides average constant-time insertion and lookup.",
    "Our return policy allows exchanges within thirty days of purchase.",
]


def read_corpus(name, fallback):
    p = Path(__file__).resolve().parent / name
    if p.exists():
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return fallback


def embed(cli, model, texts):
    t0 = time.time()
    r = subprocess.run([str(cli), "-m", str(model), "--json", *texts],
                       capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"embed failed for {model}:\n{r.stderr[-1500:]}")
    data = json.loads(r.stdout)
    return [[float(x) for x in o["embedding"]] for o in data if o.get("embedding")], dt


def mean_cos(a, b):
    def cos(u, v):
        d = sum(x*y for x, y in zip(u, v))
        nu = math.sqrt(sum(x*x for x in u)); nv = math.sqrt(sum(y*y for y in v))
        return d/(nu*nv) if nu and nv else 0.0
    n = min(len(a), len(b))
    return (sum(cos(a[i], b[i]) for i in range(n)) / n if n else float("nan")), n


_QUANT_RE = re.compile(r'(^|[-.])(q\d|iq\d|q4_k|q5_k|q6_k|q8_0|q4_0|q5_0|q5_1|bf16|imatrix)', re.I)
# LoRA / task-adapter variants (jina-v5 ships these at the SAME size as the base
# retrieval model, so "largest non-quant" alone would wrongly pick one).
_TASK_RE  = re.compile(r'-(classification|clustering|text-matching|retrieval|separation|code|sts)$', re.I)

def pick_base_gguf(ggs, name):
    """Full-precision source, from a {filename: size} dict. Prefer the exact base
    name ({name}.gguf / -f16 / -f32); else the largest .gguf that is neither a
    quant nor a LoRA task-adapter variant. Returns (filename, prefix)."""
    if not ggs:
        raise RuntimeError(f"no .gguf for {name}")
    for cand in (f"{name}.gguf", f"{name}-f16.gguf", f"{name}-f32.gguf"):
        if cand in ggs:
            prefix = cand[:-5]
            for suf in ("-f16", "-f32"):
                if prefix.endswith(suf): prefix = prefix[:-len(suf)]
            return cand, prefix
    def ok(stem):
        return not _QUANT_RE.search(stem) and not _TASK_RE.search(stem)
    base = {f: sz for f, sz in ggs.items() if ok(f[:-5])}
    pool = base or {f: sz for f, sz in ggs.items() if not _QUANT_RE.search(f[:-5])} or ggs
    fn = max(pool, key=pool.get)
    prefix = fn[:-5]
    for suf in ("-f16", "-f32", ".f16", ".f32"):
        if prefix.endswith(suf):
            prefix = prefix[: -len(suf)]
    return fn, prefix


# Bases larger than this can't be loaded for inference on Kaggle's ~13GB RAM, so
# calibrate + A/B-gold run on the q8_0 (fits RAM; the imatrix is activation stats,
# ~identical on q8 vs f32) while quantization still reads the full-precision base
# (streaming, per-tensor). Big files stage in /tmp (~70GB) not /kaggle/working (~20GB).
BIG_BYTES = 10 * 1000**3

def process(name, cli, quant, api, calib, eval_):
    from huggingface_hub import hf_hub_download
    ov = OVERRIDES.get(name, {})
    hf_out = ov.get("hf_out", f"cstr/{name}-GGUF")
    quants = ov.get("quants", QSPECS)
    kh.step("model.start", model=name, repo=hf_out)

    ggs = {s.rfilename: (s.size or 0) for s in api.repo_info(hf_out, files_metadata=True).siblings
           if s.rfilename.endswith(".gguf")}
    base_fn, prefix = pick_base_gguf(ggs, name)
    base_sz = ggs.get(base_fn, 0)
    big = base_sz > BIG_BYTES
    stage = Path("/tmp/crisp-stage") if big else WORK
    srcdir = stage / "src" / name
    srcdir.mkdir(parents=True, exist_ok=True)

    # For big bases (>10GB) download + quantize + calibrate all from the q8_0
    # (4-8GB). crispembed-quantize dequantizes the q8 source before re-quantizing,
    # and q8_0 is ~lossless (cos ~0.9998) so q4-from-q8 ≈ q4-from-f32 — while
    # avoiding the 16-30GB f32 download that stalls on Kaggle nodes. Small models
    # still use the f32 base directly.
    q8_fn = f"{prefix}-q8_0.gguf"
    src_fn = q8_fn if (big and q8_fn in ggs) else base_fn
    with kh.build_heartbeat(f"{name}.download.src"):
        qsrc = Path(hf_hub_download(hf_out, src_fn, token=api.token, local_dir=str(srcdir)))
    csrc = qsrc
    goldlabel = "q8_0" if src_fn.endswith("-q8_0.gguf") else "full-precision"
    kh.step("model.source", model=name, src=src_fn, prefix=prefix,
            src_gb=round(ggs.get(src_fn, 0)/1e9, 2), big=big)

    imat = stage / f"{prefix}.imatrix"; imat.unlink(missing_ok=True)
    env = dict(os.environ, CRISPEMBED_IMATRIX_OUT=str(imat))
    with kh.build_heartbeat(f"{name}.calibrate"):
        cal = subprocess.run([str(cli), "-m", str(csrc), "--json", *calib],
                             env=env, capture_output=True, text=True)
    # Fail LOUDLY if calibration didn't produce the imatrix — otherwise the quantizer
    # silently falls back to NON-imatrix and uploads mislabeled "-imatrix" quants
    # (observed on qwen3-embed-8b). Surface the CLI stderr for diagnosis.
    if cal.returncode != 0:
        raise RuntimeError(f"calibration rc={cal.returncode} for {name}; stderr tail:\n{cal.stderr[-1200:]}")
    if not imat.exists() or imat.stat().st_size == 0:
        raise RuntimeError(f"calibration produced NO imatrix at {imat} for {name} "
                           f"(rc=0); stdout {len(cal.stdout)}B; stderr tail:\n{cal.stderr[-1200:]}")

    gold, _ = embed(cli, csrc, eval_)   # gold = calib source (q8_0 for big, ~lossless)

    report = []
    for qtype, use_im, up_tmpl in quants:
        if big and qtype == "q8_0" and not use_im:
            continue  # q8_0 IS the gold for big models — no A/B needed
        tag = f"{qtype}{'-im' if use_im else ''}"
        out = stage / f"{prefix}-{tag}.gguf"
        cmd = [str(quant), str(qsrc), str(out), qtype] + (["--imatrix", str(imat)] if use_im else [])
        with kh.build_heartbeat(f"{name}.quant.{tag}"):
            subprocess.check_call(cmd)
        vecs, dt = embed(cli, out, eval_)
        cos, n = mean_cos(vecs, gold)
        mb = out.stat().st_size / 1e6
        upname = up_tmpl.format(prefix=prefix) if up_tmpl else "(A/B only)"
        kh.step(f"{name}.ab.{tag}", imatrix=use_im, cos_vs_gold=round(cos, 6),
                gold=goldlabel, size_mb=round(mb, 1), upload=upname)
        report.append(f"{qtype:7s} imatrix={int(use_im)}  cos_vs_{goldlabel}={cos:.6f}  {mb:7.1f}MB  -> {upname}")
        if up_tmpl:
            with kh.build_heartbeat(f"{name}.upload.{tag}"):
                api.upload_file(path_or_fileobj=str(out), path_in_repo=upname,
                    repo_id=hf_out, repo_type="model",
                    commit_message=f"{qtype} +imatrix (cos_vs_{goldlabel}={cos:.4f})")
        out.unlink(missing_ok=True)

    summary = (f"imatrix A/B — {name} ({hf_out}), cos vs {goldlabel} gold, "
               f"n={len(eval_)}, calib={len(calib)}, quant_src={base_fn}\n" + "\n".join(report) + "\n")
    summ = stage / f"{prefix}-imatrix-ab.txt"; summ.write_text(summary)
    for p, msg in [(summ, "A/B summary (cos vs gold)"), (imat, "importance matrix (calibration)")]:
        with kh.build_heartbeat(f"{name}.upload.meta"):
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                repo_id=hf_out, repo_type="model", commit_message=msg)
    imat.unlink(missing_ok=True); summ.unlink(missing_ok=True)
    shutil.rmtree(srcdir, ignore_errors=True)   # free the multi-GB source(s)
    kh.step("model.done", model=name)
    return name, report


def main():
    kh.init_progress()
    token = kh.resolve_hf_token()
    kh.step("harness_ready", n_models=len(RUN), hf_token_ok=bool(token))
    calib = read_corpus("calib_corpus.txt", _CALIB_FB)
    eval_ = read_corpus("eval_corpus.txt", _EVAL_FB)

    # build crispembed-cli + crispembed-quantize (CPU; GPU attached only for internet)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "huggingface_hub", "hf_transfer", "gguf"])
    repo = WORK / "CrispEmbed"
    if not repo.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(repo)])
        subprocess.check_call(["git", "-C", str(repo), "submodule", "update", "--init", "--recursive"])
    kh.install_build_toolchain()
    build = repo / "build"; build.mkdir(exist_ok=True)
    GPU = os.environ.get("CRISP_GPU", "0") != "0"
    flags = (kh.cuda_build_flags(kh.detect_cuda_arch()) if GPU else ["-DGGML_CUDA=OFF"]) + kh.cache_and_link_flags()
    kh.sh_with_progress(f"cmake -S {repo} -B {build} -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(f"cmake --build {build} --target crispembed-cli crispembed-quantize "
                            f"-j{kh.safe_build_jobs(gpu=GPU)}")
    cli, quant = build / "crispembed", build / "crispembed-quantize"
    kh.step("built")

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    # Idempotent: skip models whose repo already has imatrix quants (unless FORCE=1),
    # so re-running the batch only processes newly-added models.
    force = os.environ.get("FORCE", "0") != "0"
    def is_done(hf_out):
        try:
            fs = api.list_repo_files(hf_out)
        except Exception:
            return False
        return (any(f.endswith("-iq4_xs.gguf") for f in fs)
                and any(f.endswith("-imatrix-ab.txt") for f in fs))

    import traceback
    results, failures, skipped = [], [], []
    for name in RUN:
        hf_out = OVERRIDES.get(name, {}).get("hf_out", f"cstr/{name}-GGUF")
        if not force and is_done(hf_out):
            print(f"[skip] {name} — already has imatrix quants", flush=True)
            kh.step("model.skip", model=name); skipped.append(name); continue
        try:
            results.append(process(name, cli, quant, api, calib, eval_))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {name}: {err}\n{traceback.format_exc()[-1500:]}", flush=True)
            kh.step("model.fail", model=name, error=err[:300])
            failures.append((name, err))

    # Downloadable batch summary (kernels stdout is not captured; kaggle_usage #15).
    lines = ["===== BATCH SUMMARY ====="]
    for name, rep in results:
        lines.append(f"\n## {name}")
        lines += ["  " + r for r in rep]
    if failures:
        lines.append("\nFAILED:")
        lines += [f"  {n}: {e}" for n, e in failures]
    text = "\n".join(lines) + "\n"
    (WORK / "batch_summary.txt").write_text(text)   # kept in working dir → downloadable
    print("\n" + text, flush=True)
    kh.step("all_done", ok=len(results), skipped=len(skipped), failed=len(failures),
            failures=",".join(n for n, _ in failures))
    print(f"[DONE] ok={len(results)} skipped={len(skipped)} failed={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
