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
  2. downloads the HF source and converts it to an f16 GGUF,
  3. calibration pass (CRISPEMBED_IMATRIX_OUT) over calib_corpus.txt,
  4. for each quant spec: quantize (+imatrix where flagged) → A/B cosine vs the
     f16 gold on eval_corpus.txt → upload → rm the quant,
  5. uploads the f16 + .imatrix, then removes everything local.
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
# quants: (qtype, use_imatrix). q8_0 barely benefits; 4-bit types do. The tool
# auto-keeps vision/lm-head/embedding tensors at Q8_0.
MODELS = {
    "lfm2-embed": dict(
        hf_src="LiquidAI/LFM2.5-Embedding-350M",
        converter="models/convert-lfm2-embed-to-gguf.py",
        conv_args=["--dtype", "f16"],
        hf_out="cstr/lfm2.5-embedding-350m-crispembed-GGUF", prefix="lfm2-embed",
        quants=[("q8_0", False), ("q4_k", True), ("iq4_xs", True)],
    ),
    "jina-v5-nano": dict(
        hf_src="jinaai/jina-embeddings-v5-nano",   # verify exact repo id + LoRA
        converter="models/convert-decoder-embed-to-gguf.py",
        conv_args=["--dtype", "f16", "--crisp"],
        hf_out="cstr/jina-v5-nano-crispembed-GGUF", prefix="jina-v5-nano",
        quants=[("q8_0", False), ("q4_k", True), ("iq4_xs", True)],
    ),
    "bge-m3": dict(
        hf_src="BAAI/bge-m3",
        converter="models/convert-bert-to-gguf.py",
        conv_args=["--dtype", "f16", "--crisp"],
        hf_out="cstr/bge-m3-crispembed-GGUF", prefix="bge-m3",
        quants=[("q8_0", False), ("q4_k", True), ("iq4_xs", True)],
    ),
    "e5-large": dict(
        hf_src="intfloat/multilingual-e5-large",
        converter="models/convert-bert-to-gguf.py",
        conv_args=["--dtype", "f16", "--crisp"],
        hf_out="cstr/multilingual-e5-large-crispembed-GGUF", prefix="multilingual-e5-large",
        quants=[("q8_0", False), ("q4_k", True), ("iq4_xs", True)],
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
    cfg_cmd = (f"cmake -G Ninja -S {repo} -B {build} -DCMAKE_BUILD_TYPE=Release "
               f"-DGGML_CUDA=OFF " + " ".join(kh.cache_and_link_flags()))
    kh.sh_with_progress(cfg_cmd)
    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(f"cmake --build {build} "
                            f"--target crispembed-cli crispembed-quantize "
                            f"-j{kh.safe_build_jobs(gpu=False)}")
    cli   = build / "crispembed"
    quant = build / "crispembed-quantize"
    kh.step("built")

    # 2. download + convert to f16
    from huggingface_hub import snapshot_download, HfApi
    with kh.build_heartbeat("download.model"):
        src = snapshot_download(repo_id=cfg["hf_src"], token=token,
                                cache_dir=str(WORK / "hf-cache"))
    f16 = WORK / f"{cfg['prefix']}-f16.gguf"
    with kh.build_heartbeat("convert.f16"):
        subprocess.check_call([sys.executable, str(repo / cfg["converter"]),
            "--model", str(src), "--output", str(f16), *cfg["conv_args"]])
    kh.step("f16_done", size_mb=round(f16.stat().st_size / 1e6, 1))

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
    for qtype, use_im in cfg["quants"]:
        out = WORK / f"{cfg['prefix']}-{qtype}.gguf"
        cmd = [str(quant), str(f16), str(out), qtype]
        if use_im:
            cmd += ["--imatrix", str(imat)]
        with kh.build_heartbeat(f"quantize.{qtype}"):
            subprocess.check_call(cmd)
        vecs, dt = embed(cli, out, eval_)
        cos, n = mean_cos(vecs, gold)
        mb = out.stat().st_size / 1e6
        kh.step(f"ab.{qtype}", imatrix=use_im, cos_vs_f16=round(cos, 6),
                size_mb=round(mb, 1), embed_s=round(dt, 2), n=n)
        report.append(f"{qtype:7s} imatrix={int(use_im)}  cos_vs_f16={cos:.6f}  {mb:7.1f}MB")
        if api:
            with kh.build_heartbeat(f"upload.{qtype}"):
                api.upload_file(path_or_fileobj=str(out), path_in_repo=out.name,
                    repo_id=cfg["hf_out"], repo_type="model",
                    commit_message=f"{qtype}{' +imatrix' if use_im else ''} (cos_vs_f16={cos:.4f})")
        out.unlink(missing_ok=True)   # free space before next quant

    # 5. upload f16 + imatrix, then rm everything
    if api:
        for p, msg in [(f16, "f16 source"), (imat, "importance matrix (calibration)")]:
            with kh.build_heartbeat(f"upload.{p.name}"):
                api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                    repo_id=cfg["hf_out"], repo_type="model", commit_message=msg)
    f16.unlink(missing_ok=True); imat.unlink(missing_ok=True)

    kh.step("all_done", **{f"q{i}": r for i, r in enumerate(report)})
    print("\n===== A/B SUMMARY (" + MODEL + ", cos vs f16 gold) =====")
    for r in report:
        print("  " + r, flush=True)
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
