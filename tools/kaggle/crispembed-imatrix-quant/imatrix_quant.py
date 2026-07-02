#!/usr/bin/env python3
"""CrispEmbed — per-model imatrix quantization + A/B + upload (Kaggle kernel).

Follows the standard CrispEmbed Kaggle regime (kaggle_harness = kh):
  * kh.init_progress()      — line-buffered I/O + JSONL progress, pushed to HF
  * kh.resolve_hf_token()   — env → Kaggle Secret → mounted DATASET (hf_token.txt)
  * kh.install_build_toolchain() + ccache warmed from the crispasr-ccache dataset
  * kh.build_heartbeat(...)  — 30 s heartbeat around every long step (build,
    download, convert, calibrate, quantize, upload) so the kernel never idles out
Attach BOTH datasets in kernel-metadata.json:
    "dataset_sources": ["chr1str/crispasr-hf-token", "chr1str/crispasr-ccache"]

Runs ONE model per invocation (the C1 rollout: run → upload → rm → next model).
Select the model with the MODEL constant below or the MODEL env var. Per run it:
  1. builds crispembed-cli + crispembed-quantize from origin/main (C1 merged),
  2. downloads the existing full-precision GGUF from the target repo (or, if
     none, converts from HF),
  3. calibration pass (CRISPEMBED_IMATRIX_OUT) over calib_corpus.txt,
  4. for each quant spec: quantize (+imatrix where flagged) → A/B cosine vs the
     f16 gold on eval_corpus.txt → upload → rm the quant,
  5. uploads imatrix variants under DISTINCT names (never overwriting the
     canonical q8_0/q4_k baselines) + the .imatrix artifact, then cleans up.
"""
import os, sys, json, math, time, subprocess
from pathlib import Path

# ── which model to process this run (edit, or set MODEL=… env) ────────────────
MODEL = os.environ.get("MODEL", "lfm2-embed")

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

# ── model registry (converters take --model <dir> --output <f16> --dtype f16) ──
# hf_src + hf_out (existing GGUF repos) + filename prefixes are the canonical
# values from examples/cli/model_mgr.cpp k_registry[]. Imatrix quants upload into
# the SAME repos users already download from.
#
# QSPECS: (qtype, use_imatrix, upload_name_template | None). upload_name None
# means DO NOT upload (a baseline of that type already exists in the repo — never
# overwrite it). imatrix variants get DISTINCT names so they never clobber the
# canonical q8_0/q4_k baselines: q4_k+imatrix → *-q4_k-imatrix.gguf, iq4_xs → *-iq4_xs.gguf.
QSPECS = [
    ("q8_0",   False, None),                          # A/B reference only; baseline exists
    ("q4_k",   False, None),                          # A/B baseline (no imatrix) — shows the delta; not uploaded
    ("q4_k",   True,  "{prefix}-q4_k-imatrix.gguf"),  # new file, does not touch *-q4_k.gguf
    ("iq4_xs", True,  "{prefix}-iq4_xs.gguf"),         # new file
]

MODELS = {
    "lfm2-embed": dict(
        hf_src="LiquidAI/LFM2.5-Embedding-350M",
        converter="models/convert-lfm2-embed-to-gguf.py",
        conv_args=["--dtype", "f16"],
        hf_out="cstr/lfm2-embed-GGUF", prefix="lfm2-embed", src_gguf="lfm2-embed-f16.gguf",
        quants=QSPECS,
    ),
    "jina-v5-nano": dict(
        hf_src="jinaai/jina-embeddings-v5-text-nano",
        converter="models/convert-decoder-embed-to-gguf.py",
        conv_args=["--dtype", "f16", "--crisp"],
        hf_out="cstr/jina-v5-nano-GGUF", prefix="jina-v5-nano", src_gguf="jina-v5-nano.gguf",
        quants=QSPECS,
    ),
    "jina-v5-small": dict(
        hf_src="jinaai/jina-embeddings-v5-text-small",
        converter="models/convert-decoder-embed-to-gguf.py",
        conv_args=["--dtype", "f16", "--crisp"],
        hf_out="cstr/jina-v5-small-GGUF", prefix="jina-v5-small", src_gguf="jina-v5-small.gguf",
        quants=QSPECS,
    ),
    "bge-m3": dict(
        hf_src="BAAI/bge-m3",
        converter="models/convert-bert-to-gguf.py",
        conv_args=["--dtype", "f16", "--crisp"],
        hf_out="cstr/bge-m3-GGUF", prefix="bge-m3", src_gguf="bge-m3.gguf",
        quants=QSPECS,
    ),
    "e5-large": dict(
        hf_src="intfloat/multilingual-e5-large",
        converter="models/convert-bert-to-gguf.py",
        conv_args=["--dtype", "f16", "--crisp"],
        hf_out="cstr/multilingual-e5-large-GGUF", prefix="multilingual-e5-large", src_gguf="multilingual-e5-large.gguf",
        quants=QSPECS,
    ),
    # BidirLM-Omni: multimodal, no single-file converter in models/ yet — TODO.
}


def read_corpus(name, fallback):
    p = Path(__file__).resolve().parent / name
    if p.exists():
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    return fallback

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


def embed(cli, model, texts):
    t0 = time.time()
    r = subprocess.run([str(cli), "-m", str(model), "--json", *texts],
                       capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"embed failed for {model}:\n{r.stderr[-2000:]}")
    data = json.loads(r.stdout)
    return [[float(x) for x in o["embedding"]] for o in data if o.get("embedding")], dt


def mean_cos(a, b):
    def cos(u, v):
        d = sum(x*y for x, y in zip(u, v))
        nu = math.sqrt(sum(x*x for x in u)); nv = math.sqrt(sum(y*y for y in v))
        return d/(nu*nv) if nu and nv else 0.0
    n = min(len(a), len(b))
    return (sum(cos(a[i], b[i]) for i in range(n)) / n if n else float("nan")), n


def main():
    kh.init_progress()
    token = kh.resolve_hf_token()
    if MODEL not in MODELS:
        sys.exit(f"unknown MODEL={MODEL!r}; choices: {sorted(MODELS)}")
    cfg = MODELS[MODEL]
    kh.step("harness_ready", model=MODEL, hf_token_ok=bool(token))

    calib = read_corpus("calib_corpus.txt", _CALIB_FB)
    eval_ = read_corpus("eval_corpus.txt", _EVAL_FB)

    # 1. deps + clone + build (crispembed-cli for calibration/A/B, quantize tool)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
        "safetensors", "gguf", "huggingface_hub", "hf_transfer", "transformers",
        "sentencepiece"])
    kh.step("deps_installed")

    repo = WORK / "CrispEmbed"
    if not repo.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH,
                               REPO_URL, str(repo)])
        subprocess.check_call(["git", "-C", str(repo), "submodule", "update",
                               "--init", "--recursive"])
    kh.step("cloned")

    kh.install_build_toolchain()
    build = repo / "build"; build.mkdir(exist_ok=True)
    # CPU build by DEFAULT. These embedders (<=600M) calibrate + quantize fine on
    # CPU, and a CUDA build compiles ggml-cuda's ~254 template-instance TUs
    # (~15 min of nvcc; the CrispASR ccache seed barely hits them and the arch
    # pin differs). We keep enable_gpu:true in kernel-metadata ONLY because
    # Kaggle CPU workers get no internet (kaggle_usage.md #3) — the GPU is used
    # for internet (clone/download/upload), NOT for the build. Set CRISP_GPU=1
    # for large models (e.g. BidirLM-Omni) where GPU calibration is worth it.
    GPU = os.environ.get("CRISP_GPU", "0") != "0"
    flags = (kh.cuda_build_flags(kh.detect_cuda_arch()) if GPU else ["-DGGML_CUDA=OFF"])
    flags += kh.cache_and_link_flags()
    cfg_cmd = (f"cmake -G Ninja -S {repo} -B {build} -DCMAKE_BUILD_TYPE=Release "
               + " ".join(flags))
    kh.sh_with_progress(cfg_cmd)
    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(f"cmake --build {build} "
                            f"--target crispembed-cli crispembed-quantize "
                            f"-j{kh.safe_build_jobs(gpu=GPU)}")
    cli   = build / "crispembed"
    quant = build / "crispembed-quantize"
    kh.step("built")

    # 2. acquire the full-precision source.
    #    PREFER the existing validated GGUF in the target repo (cfg["src_gguf"]) —
    #    it's the exact artifact users run, already-correct, and sidesteps HF
    #    re-conversion (critical for LoRA models like jina-v5, whose HF repo has
    #    task adapters). Fall back to snapshot_download + converter only if unset.
    from huggingface_hub import hf_hub_download, snapshot_download, HfApi
    if cfg.get("src_gguf"):
        with kh.build_heartbeat("download.src_gguf"):
            f16 = Path(hf_hub_download(cfg["hf_out"], cfg["src_gguf"], token=token,
                                       local_dir=str(WORK / "src")))
        src_desc = f"{cfg['hf_out']}/{cfg['src_gguf']}"
    else:
        with kh.build_heartbeat("download.model"):
            src = snapshot_download(repo_id=cfg["hf_src"], token=token,
                                    cache_dir=str(WORK / "hf-cache"))
        f16 = WORK / f"{cfg['prefix']}-f16.gguf"
        with kh.build_heartbeat("convert.f16"):
            subprocess.check_call([sys.executable, str(repo / cfg["converter"]),
                "--model", str(src), "--output", str(f16), *cfg["conv_args"]])
        src_desc = f"converted from {cfg['hf_src']}"
    kh.step("source_ready", src=src_desc, size_mb=round(f16.stat().st_size / 1e6, 1))

    # 3. calibration -> imatrix
    imat = WORK / f"{cfg['prefix']}.imatrix"
    imat.unlink(missing_ok=True)
    env = dict(os.environ, CRISPEMBED_IMATRIX_OUT=str(imat))
    with kh.build_heartbeat("calibrate"):
        subprocess.run([str(cli), "-m", str(f16), "--json", *calib],
                       env=env, check=True, capture_output=True, text=True)
    kh.step("calibrated", n_texts=len(calib), imatrix_kb=imat.stat().st_size // 1024)

    gold, _ = embed(cli, f16, eval_)   # f16 gold, once

    api = HfApi(token=token) if token else None
    if api:
        try:
            api.create_repo(cfg["hf_out"], repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"repo: {e}", flush=True)

    # 4. per-quant: quantize -> A/B -> upload -> rm
    report = []
    for qtype, use_im, up_tmpl in cfg["quants"]:
        tag = f"{qtype}{'-im' if use_im else ''}"
        out = WORK / f"{cfg['prefix']}-{tag}.gguf"         # distinct local temp name
        cmd = [str(quant), str(f16), str(out), qtype]
        if use_im:
            cmd += ["--imatrix", str(imat)]
        with kh.build_heartbeat(f"quantize.{tag}"):
            subprocess.check_call(cmd)
        vecs, dt = embed(cli, out, eval_)
        cos, n = mean_cos(vecs, gold)
        mb = out.stat().st_size / 1e6
        upname = up_tmpl.format(prefix=cfg["prefix"]) if up_tmpl else "(A/B only, not uploaded)"
        kh.step(f"ab.{tag}", imatrix=use_im, cos_vs_f16=round(cos, 6),
                size_mb=round(mb, 1), embed_s=round(dt, 2), n=n, upload=upname)
        report.append(f"{qtype:7s} imatrix={int(use_im)}  cos_vs_f16={cos:.6f}  {mb:7.1f}MB  -> {upname}")
        # Upload ONLY imatrix variants, under DISTINCT names — never overwrite the
        # canonical q8_0/q4_k baselines already in the repo (up_tmpl is None for those).
        if api and up_tmpl:
            with kh.build_heartbeat(f"upload.{qtype}"):
                api.upload_file(path_or_fileobj=str(out), path_in_repo=upname,
                    repo_id=cfg["hf_out"], repo_type="model",
                    commit_message=f"{qtype} +imatrix (cos_vs_f16={cos:.4f})")
            print(f"[upload] {upname}", flush=True)
        out.unlink(missing_ok=True)   # free space before next quant

    # 5. upload the imatrix artifact only (small; reproducibility). Do NOT upload
    #    f16 — it would risk clobbering and is large. Then rm everything local.
    if api:
        with kh.build_heartbeat("upload.imatrix"):
            api.upload_file(path_or_fileobj=str(imat), path_in_repo=imat.name,
                repo_id=cfg["hf_out"], repo_type="model",
                commit_message="importance matrix (calibration)")
        print(f"[upload] {imat.name}", flush=True)
    f16.unlink(missing_ok=True); imat.unlink(missing_ok=True)

    # Write a downloadable A/B summary to /kaggle/working (kernels_output captures
    # working-dir files but NOT stdout — kaggle_usage.md #15).
    summary = (f"imatrix A/B — {MODEL} ({cfg['hf_out']}), cos vs f16 gold, n={len(eval_)}\n"
               + "\n".join(report) + "\n")
    try:
        (WORK / f"{cfg['prefix']}-imatrix-ab.txt").write_text(summary)
    except Exception as e:
        print(f"summary write failed: {e}", flush=True)

    kh.step("all_done", **{f"q{i}": r for i, r in enumerate(report)})
    print("\n===== A/B SUMMARY (" + MODEL + ", cos vs f16 gold) =====")
    for r in report:
        print("  " + r, flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
