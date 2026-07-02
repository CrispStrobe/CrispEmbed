#!/usr/bin/env python3
"""CrispEmbed reference-generation batch (Kaggle, chr1s4).

For each engine that lacks a per-stage reference on HF, this kernel:
  1. downloads the upstream source model,
  2. runs tools/dump_<engine>_reference.py -> <engine>-ref.gguf,
  3. builds + runs the engine's test-<engine>-diff to VERIFY the ref
     against the shipped GGUF (cos on the final/output stage),
  4. on PASS, uploads <engine>-ref.gguf to the engine's HF GGUF repo,
so the regression manifest's diff step auto-enables (opt-in by ref presence).

Follows the Kaggle regime (kaggle_usage.md):
  - kaggle_harness heartbeat + progress (kh.init_progress / build_heartbeat)
  - BOTH per-account datasets: chr1s4/crispasr-hf-token (HF creds) +
    chr1s4/crispembed-ccache (warm the CUDA build of the diff harnesses)
  - HF token via kh.resolve_hf_token() (env -> Secret -> dataset fallback)
  - does NOT pip install torch (Kaggle pre-installs it)
  - per-engine try/except/continue; writes results to /kaggle/working so
    kernels_output can retrieve them (logs are NOT captured by that API).

Engines split by source acquisition:
  - HF-loadable (reliable here): gliner, lilt, lfm2, lfm2_colbert, layout
  - upstream .pth (set PTH_URLS below to enable): esrgan, safmn, nafnet
  - bert_ner: no dumper exists yet -> skipped (write tools/dump_bert_ner_reference.py first)
"""
import json, os, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
CRISPEMBED_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
CRISPASR_URL   = "https://github.com/CrispStrobe/CrispASR.git"
REPO  = WORK / "CrispEmbed"
BUILD = REPO / "build"
RESULTS = WORK / "ref_gen_results.json"
PROGRESS = WORK / "progress.txt"

# ── clone repos + import the harness (bundled fallback) ─────────────────────
for url, dst in ((CRISPASR_URL, WORK / "CrispASR"), (CRISPEMBED_URL, REPO)):
    if not dst.exists():
        try:
            subprocess.check_call(["git", "clone", "--depth", "1", url, str(dst)])
        except Exception as e:
            print(f"clone {url} failed: {e}", flush=True)
sys.path.insert(0, str(WORK / "CrispASR" / "tools" / "kaggle"))
if not (WORK / "CrispASR" / "tools" / "kaggle" / "kaggle_harness.py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # bundled fallback
import kaggle_harness as kh

kh.init_progress()
tok = kh.resolve_hf_token()
if tok:
    os.environ["HF_TOKEN"] = tok
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

def log(msg):
    print(msg, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(msg + "\n")

# ── per-engine recipe table ─────────────────────────────────────────────────
# .pth URLs left blank on purpose (docs give only the GitHub repo, not the exact
# release asset). Fill these to enable the SR/denoise engines; verified URLs only.
PTH_URLS = {
    "esrgan": "",   # xinntao/Real-ESRGAN realesr-general-x4v3.pth (release asset)
    "safmn":  "",   # sunny2109/SAFMN SAFMN_DF2K_x4.pth
    "nafnet": "",   # megvii-research/NAFNet NAFNet-SIDD-width32.pth
}

# name: dumper, source(--model), ref file, model repo/file, diff binary+target,
#       upload repo, extra pip, verify(model, ref) -> argv, diff env
ENGINES = [
    dict(name="gliner", dumper="dump_gliner_reference.py",
         source="VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER", ref="gliner-ref.gguf",
         model_repo="cstr/sauerkraut-gliner-lfm-GGUF", model_file="gliner-lfm-q8_0.gguf",
         diff="test-gliner-diff", pip=["gliner"],
         verify=lambda m, r: (["./build/test-gliner-diff", m], {"GLINER_DIFF_REF": r}),
         upload_repo="cstr/sauerkraut-gliner-lfm-GGUF"),
    dict(name="lilt", dumper="dump_lilt_reference.py",
         source="SCUT-DLVCLab/lilt-roberta-en-base", ref="lilt-ref.gguf",
         model_repo="cstr/lilt-base-GGUF", model_file="lilt-base-f32.gguf",
         diff="test-lilt-diff", pip=[],
         verify=lambda m, r: (["./build/test-lilt-diff", m, r], {}),
         upload_repo="cstr/lilt-base-GGUF"),
    dict(name="lfm2", dumper="dump_lfm2_reference.py",
         source="LiquidAI/LFM2.5-Embedding-350M", ref="lfm2-ref.gguf",
         model_repo="cstr/lfm2-embed-GGUF", model_file="lfm2-embed-q8_0.gguf",
         diff="test-lfm2-diff", pip=[],
         verify=lambda m, r: (["./build/test-lfm2-diff", m, r], {}),
         upload_repo="cstr/lfm2-embed-GGUF"),
    dict(name="lfm2_colbert", dumper="dump_lfm2_colbert_reference.py",
         source="LiquidAI/LFM2.5-ColBERT-350M", ref="lfm2-colbert-ref.gguf",
         model_repo="cstr/lfm2-colbert-GGUF", model_file="lfm2-colbert-q8_0.gguf",
         diff="test-lfm2-colbert-diff", pip=[],
         verify=lambda m, r: (["./build/test-lfm2-colbert-diff", m, r], {}),
         upload_repo="cstr/lfm2-colbert-GGUF"),
    dict(name="layout", dumper="dump_layout_reference.py",
         source="cmarkea/dit-base-layout-detection", ref="layout-ref.gguf",
         model_repo="cstr/layout-heron-gguf", model_file="layout-heron-f32.gguf",
         diff="test-layout-diff", pip=[],
         verify=lambda m, r: (["./build/test-layout-diff", m, r], {}),
         upload_repo="cstr/layout-heron-gguf", source_optional=True),
]

def hf_get(repo, fname, dst_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=fname, local_dir=str(dst_dir),
                           token=os.environ.get("HF_TOKEN"))

def hf_put(path, repo, name):
    from huggingface_hub import HfApi
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj=str(path), path_in_repo=name, repo_id=repo, repo_type="model")

# ── build the diff-harness binaries (needs the ccache dataset to be quick) ───
def build_targets(targets):
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
    BUILD.mkdir(exist_ok=True)
    kh.sh_with_progress(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release "
                        + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(f"cmake --build {BUILD} --target {' '.join(targets)} "
                            f"-j{kh.safe_build_jobs(gpu=True)}")

# ── main ────────────────────────────────────────────────────────────────────
def main():
    results = {}
    tools = REPO / "tools"
    # 1. build every diff harness we intend to verify with
    targets = [e["diff"] for e in ENGINES]
    try:
        with kh.build_heartbeat("build"):
            build_targets(targets)
    except Exception as e:
        log(f"BUILD FAILED: {e}")
        RESULTS.write_text(json.dumps({"build_error": str(e)}, indent=2)); return

    # 2. per engine: source -> dump -> verify -> upload
    for e in ENGINES:
        name = e["name"]
        try:
            with kh.build_heartbeat(f"engine.{name}"):
                for pkg in e.get("pip", []):
                    kh.sh(f"pip install -q {pkg}")
                ref_path = WORK / e["ref"]
                dumper = tools / e["dumper"]
                src = e["source"]
                cmd = f"python {dumper} --model {src} --output {ref_path}"
                log(f"[{name}] dump: {cmd}")
                rc = kh.sh(cmd, check=False)
                if rc != 0 or not ref_path.exists():
                    if e.get("source_optional"):
                        log(f"[{name}] SKIP: dumper failed (source {src}) — needs recipe fix")
                        results[name] = "dump_failed"; continue
                    results[name] = "dump_failed"; continue
                model = hf_get(e["model_repo"], e["model_file"], WORK / name)
                argv, env = e["verify"](model, str(ref_path))
                log(f"[{name}] verify: {' '.join(argv)}  env={env}")
                r = subprocess.run(argv, cwd=str(REPO), env={**os.environ, **env},
                                   capture_output=True, text=True)
                out = (r.stdout + r.stderr)
                log(f"[{name}] diff tail: {out.strip().splitlines()[-3:] if out.strip() else 'no output'}")
                passed = ("0 failed" in out) or ("PASS" in out and "FAIL" not in out)
                if not passed:
                    results[name] = "verify_failed"; continue
                hf_put(ref_path, e["upload_repo"], e["ref"])
                log(f"[{name}] UPLOADED {e['ref']} -> {e['upload_repo']}")
                results[name] = "ok"
        except Exception as ex:
            log(f"[{name}] ERROR: {ex}")
            results[name] = f"error: {ex}"

    # 3. SR/denoise engines: only if a .pth URL is supplied (else skip clearly)
    for name in ("esrgan", "safmn", "nafnet"):
        if not PTH_URLS.get(name):
            log(f"[{name}] SKIP: set PTH_URLS['{name}'] to the verified upstream .pth URL")
            results[name] = "skipped_no_pth_url"

    RESULTS.write_text(json.dumps(results, indent=2))
    log(f"DONE: {json.dumps(results)}")

if __name__ == "__main__":
    main()
