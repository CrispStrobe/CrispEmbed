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
    "qwen3-embed-0.6b", "embeddinggemma-300m", "pixie-rune-v1",
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

def pick_base_gguf(api, repo, name):
    """Full-precision source. Prefer the exact base name ({name}.gguf /
    -f16 / -f32); else the largest .gguf that is neither a quant nor a LoRA
    task-adapter variant. Returns (filename, prefix)."""
    info = api.repo_info(repo, files_metadata=True)
    ggs = {s.rfilename: (s.size or 0) for s in info.siblings if s.rfilename.endswith(".gguf")}
    if not ggs:
        raise RuntimeError(f"no .gguf in {repo}")
    for cand in (f"{name}.gguf", f"{name}-f16.gguf", f"{name}-f32.gguf"):
        if cand in ggs:
            return cand, name
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


def process(name, cli, quant, api, calib, eval_):
    ov = OVERRIDES.get(name, {})
    hf_out = ov.get("hf_out", f"cstr/{name}-GGUF")
    quants = ov.get("quants", QSPECS)
    kh.step("model.start", model=name, repo=hf_out)

    with kh.build_heartbeat(f"{name}.download_src"):
        base_fn, prefix = pick_base_gguf(api, hf_out, name)
        from huggingface_hub import hf_hub_download
        f16 = Path(hf_hub_download(hf_out, base_fn, token=api.token,
                                   local_dir=str(WORK / "src" / name)))
    kh.step("model.source", model=name, src=f"{hf_out}/{base_fn}",
            prefix=prefix, size_mb=round(f16.stat().st_size / 1e6, 1))

    imat = WORK / f"{prefix}.imatrix"
    imat.unlink(missing_ok=True)
    env = dict(os.environ, CRISPEMBED_IMATRIX_OUT=str(imat))
    with kh.build_heartbeat(f"{name}.calibrate"):
        subprocess.run([str(cli), "-m", str(f16), "--json", *calib],
                       env=env, check=True, capture_output=True, text=True)

    gold, _ = embed(cli, f16, eval_)

    report = []
    for qtype, use_im, up_tmpl in quants:
        tag = f"{qtype}{'-im' if use_im else ''}"
        out = WORK / f"{prefix}-{tag}.gguf"
        cmd = [str(quant), str(f16), str(out), qtype] + (["--imatrix", str(imat)] if use_im else [])
        with kh.build_heartbeat(f"{name}.quant.{tag}"):
            subprocess.check_call(cmd)
        vecs, dt = embed(cli, out, eval_)
        cos, n = mean_cos(vecs, gold)
        mb = out.stat().st_size / 1e6
        upname = up_tmpl.format(prefix=prefix) if up_tmpl else "(A/B only)"
        kh.step(f"{name}.ab.{tag}", imatrix=use_im, cos_vs_f16=round(cos, 6),
                size_mb=round(mb, 1), upload=upname)
        report.append(f"{qtype:7s} imatrix={int(use_im)}  cos_vs_f16={cos:.6f}  {mb:7.1f}MB  -> {upname}")
        if up_tmpl:
            with kh.build_heartbeat(f"{name}.upload.{tag}"):
                api.upload_file(path_or_fileobj=str(out), path_in_repo=upname,
                    repo_id=hf_out, repo_type="model",
                    commit_message=f"{qtype} +imatrix (cos_vs_f16={cos:.4f})")
        out.unlink(missing_ok=True)

    summary = (f"imatrix A/B — {name} ({hf_out}), cos vs full-precision gold, "
               f"n={len(eval_)}, calib={len(calib)}\n" + "\n".join(report) + "\n")
    summ = WORK / f"{prefix}-imatrix-ab.txt"
    summ.write_text(summary)
    for p, msg in [(summ, "A/B summary (cos vs gold)"), (imat, "importance matrix (calibration)")]:
        with kh.build_heartbeat(f"{name}.upload.meta"):
            api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                repo_id=hf_out, repo_type="model", commit_message=msg)
    f16.unlink(missing_ok=True); imat.unlink(missing_ok=True); summ.unlink(missing_ok=True)
    shutil.rmtree(WORK / "src" / name, ignore_errors=True)  # free the multi-GB source
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

    import traceback
    results, failures = [], []
    for name in RUN:
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
    kh.step("all_done", ok=len(results), failed=len(failures),
            failures=",".join(n for n, _ in failures))
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
