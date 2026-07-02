#!/usr/bin/env python3
"""CrispEmbed reference-generation batch (Kaggle, chr1s4).

For each engine that lacks a per-stage reference on HF, this kernel:
  1. acquires the upstream source (HF model id, an HF-hosted .pth, or a release URL),
  2. runs tools/dump_<engine>_reference.py -> <engine>-ref.gguf,
  3. builds + runs the engine's test-<engine>-diff to VERIFY the ref against the
     shipped GGUF (cos on the final/output stage) — uploads ONLY on PASS,
  4. uploads <engine>-ref.gguf to the engine's HF GGUF repo so the regression
     manifest's diff step auto-enables (opt-in by ref presence).

Follows the Kaggle regime (kaggle_usage.md): kaggle_harness heartbeat/progress;
BOTH per-account datasets (chr1s4/crispasr-hf-token + chr1s4/crispembed-ccache);
resolve_hf_token; NO torch reinstall; per-engine try/except/continue; results +
progress written to /kaggle/working (kernels_output does not capture logs).

Source acquisition per engine (all confirmed 2026-07):
  - HF model id (dumper loads it):  gliner, lilt, lfm2, lfm2_colbert, layout
  - HF-hosted .pth (hf_hub_download): safmn (Meloo/SAFMN), nafnet (mikestealth/nafnet-models)
  - release URL (wget):              esrgan (xinntao/Real-ESRGAN v0.2.5.0)
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

for url, dst in ((CRISPASR_URL, WORK / "CrispASR"), (CRISPEMBED_URL, REPO)):
    if not dst.exists():
        # CrispEmbed needs the ggml submodule (CMakeLists.txt:70 aborts without it).
        extra = ["--recursive", "--shallow-submodules"] if dst == REPO else []
        try:
            subprocess.check_call(["git", "clone", "--depth", "1", *extra, url, str(dst)])
        except Exception as e:
            print(f"clone {url} failed: {e}", flush=True)
sys.path.insert(0, str(WORK / "CrispASR" / "tools" / "kaggle"))
if not (WORK / "CrispASR" / "tools" / "kaggle" / "kaggle_harness.py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
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
        f.write(str(msg) + "\n")

# name | dumper | source spec | ref file | model repo/file | diff binary
#      | upload repo | verify(model, ref)->(argv, env) | extra pip
# source spec is one of: ("hf", "<model id>") | ("pth_hf", "<repo>", "<file>")
#                        | ("url", "<download url>")
def dv(bin):  # default verify: <binary> <model.gguf> <ref.gguf>
    return lambda m, r: ([f"./build/{bin}", m, r], {})

ENGINES = [
    dict(name="gliner", dumper="dump_gliner_reference.py", source=("hf", "VAGOsolutions/SauerkrautLM-LFM2.5-GLiNER"),
         ref="gliner-ref.gguf", model_repo="cstr/sauerkraut-gliner-lfm-GGUF", model_file="gliner-lfm-q8_0.gguf",
         diff="test-gliner-diff", upload_repo="cstr/sauerkraut-gliner-lfm-GGUF", pip=["gliner"],
         verify=lambda m, r: (["./build/test-gliner-diff", m], {"GLINER_DIFF_REF": r})),
    dict(name="lilt", dumper="dump_lilt_reference.py", source=("hf", "SCUT-DLVCLab/lilt-roberta-en-base"),
         ref="lilt-ref.gguf", model_repo="cstr/lilt-base-GGUF", model_file="lilt-base-f32.gguf",
         diff="test-lilt-diff", upload_repo="cstr/lilt-base-GGUF", pip=[], verify=dv("test-lilt-diff")),
    dict(name="lfm2", dumper="dump_lfm2_reference.py", source=("hf", "LiquidAI/LFM2.5-Embedding-350M"),
         ref="lfm2-ref.gguf", model_repo="cstr/lfm2-embed-GGUF", model_file="lfm2-embed-q8_0.gguf",
         diff="test-lfm2-diff", upload_repo="cstr/lfm2-embed-GGUF", pip=[], verify=dv("test-lfm2-diff")),
    dict(name="lfm2_colbert", dumper="dump_lfm2_colbert_reference.py", source=("hf", "LiquidAI/LFM2.5-ColBERT-350M"),
         ref="lfm2-colbert-ref.gguf", model_repo="cstr/lfm2-colbert-GGUF", model_file="lfm2-colbert-q8_0.gguf",
         diff="test-lfm2-colbert-diff", upload_repo="cstr/lfm2-colbert-GGUF", pip=[], verify=dv("test-lfm2-colbert-diff")),
    dict(name="layout", dumper="dump_layout_reference.py", source=("hf", "cmarkea/dit-base-layout-detection"),
         ref="layout-ref.gguf", model_repo="cstr/layout-heron-gguf", model_file="layout-heron-f32.gguf",
         diff="test-layout-diff", upload_repo="cstr/layout-heron-gguf", pip=[], verify=dv("test-layout-diff"),
         source_optional=True),
    # SR / restoration — upstream .pth (URLs confirmed 2026-07)
    dict(name="esrgan", dumper="dump_esrgan_reference.py",
         source=("url", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"),
         ref="esrgan-ref.gguf", model_repo="cstr/esrgan-sr-GGUF", model_file="esrgan-x4-f32.gguf",
         diff="test-esrgan-diff", upload_repo="cstr/esrgan-sr-GGUF", pip=[], verify=dv("test-esrgan-diff")),
    dict(name="safmn", dumper="dump_safmn_reference.py", source=("pth_hf", "Meloo/SAFMN", "SAFMN_DF2K_x4.pth"),
         ref="safmn-ref.gguf", model_repo="cstr/safmn-sr-GGUF", model_file="safmn-x4-f32.gguf",
         diff="test-safmn-diff", upload_repo="cstr/safmn-sr-GGUF", pip=[], verify=dv("test-safmn-diff")),
    dict(name="nafnet", dumper="dump_nafnet_reference.py", source=("pth_hf", "mikestealth/nafnet-models", "NAFNet-SIDD-width32.pth"),
         ref="nafnet-ref.gguf", model_repo="cstr/nafnet-sidd-GGUF", model_file="nafnet-sidd-w32-f16.gguf",
         diff="test-nafnet-diff", upload_repo="cstr/nafnet-sidd-GGUF", pip=[], verify=dv("test-nafnet-diff")),
]

def hf_get(repo, fname, dst_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=fname, local_dir=str(dst_dir),
                           token=os.environ.get("HF_TOKEN"))

def hf_put(path, repo, name):
    from huggingface_hub import HfApi
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj=str(path), path_in_repo=name, repo_id=repo, repo_type="model")

def acquire_source(e):
    """Return the --model value the dumper expects."""
    kind = e["source"][0]
    if kind == "hf":
        return e["source"][1]                       # dumper loads the HF id directly
    if kind == "pth_hf":
        return hf_get(e["source"][1], e["source"][2], WORK / (e["name"] + "_src"))
    if kind == "url":
        dst = WORK / (e["name"] + "_src") / os.path.basename(e["source"][1])
        dst.parent.mkdir(parents=True, exist_ok=True)
        kh.sh(f"wget -q -O {dst} {e['source'][1]}")
        return str(dst)
    raise ValueError(f"unknown source kind {kind}")

def build_targets(targets):
    kh.install_build_toolchain()
    flags = kh.cuda_build_flags(kh.detect_cuda_arch()) + kh.cache_and_link_flags()
    BUILD.mkdir(exist_ok=True)
    kh.sh_with_progress(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(f"cmake --build {BUILD} --target {' '.join(targets)} -j{kh.safe_build_jobs(gpu=True)}")

def main():
    results = {}
    tools = REPO / "tools"
    try:
        build_targets(sorted({e["diff"] for e in ENGINES}))
    except Exception as e:
        log(f"BUILD FAILED: {e}"); RESULTS.write_text(json.dumps({"build_error": str(e)}, indent=2)); return

    for e in ENGINES:
        name = e["name"]
        try:
            with kh.build_heartbeat(f"engine.{name}"):
                for pkg in e.get("pip", []):
                    kh.sh(f"pip install -q {pkg}")
                src = acquire_source(e)
                ref_path = WORK / e["ref"]
                cmd = f"python {tools / e['dumper']} --model {src} --output {ref_path}"
                log(f"[{name}] dump: {cmd}")
                if kh.sh(cmd, check=False) != 0 or not ref_path.exists():
                    log(f"[{name}] dump_failed"); results[name] = "dump_failed"; continue
                model = hf_get(e["model_repo"], e["model_file"], WORK / name)
                argv, env = e["verify"](model, str(ref_path))
                log(f"[{name}] verify: {' '.join(argv)} env={env}")
                r = subprocess.run(argv, cwd=str(REPO), env={**os.environ, **env}, capture_output=True, text=True)
                out = r.stdout + r.stderr
                tail = out.strip().splitlines()[-3:] if out.strip() else ["<no output>"]
                log(f"[{name}] diff tail: {tail}")
                if not (("0 failed" in out) or ("PASS" in out and "FAIL" not in out)):
                    log(f"[{name}] verify_failed"); results[name] = "verify_failed"; continue
                hf_put(ref_path, e["upload_repo"], e["ref"])
                log(f"[{name}] UPLOADED {e['ref']} -> {e['upload_repo']}"); results[name] = "ok"
        except Exception as ex:
            log(f"[{name}] ERROR: {ex}"); results[name] = f"error: {ex}"

    RESULTS.write_text(json.dumps(results, indent=2))
    log(f"DONE: {json.dumps(results)}")

if __name__ == "__main__":
    main()
