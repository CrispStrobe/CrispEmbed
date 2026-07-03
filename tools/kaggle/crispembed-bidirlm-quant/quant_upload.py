#!/usr/bin/env python3
"""CrispEmbed — BidirLM-Omni-2.5B reconvert + quantize + upload (all variants, both repos).

The shipped bidirlm-omni GGUFs had a converter bug that cratered the TEXT tower to
cos 0.047 (vision fine). A fresh re-export with the current convert-decoder-embed-to-gguf.py
fixes it (text 0.999 / vision 0.997). The q8_0 defaults were re-uploaded from a local run;
this kernel backfills EVERY variant of BOTH repos on Kaggle's fast HF link (a slow home
connection can't push ~30 GB reliably).

  full-omni  -> cstr/bidirlm-omni-2.5b-GGUF           (text + audio + vision in one file)
  text-only  -> cstr/bidirlm-omni-2.5b-textonly-GGUF  (--text-only: skips audio/vision towers)

Runs under chr1s4. Follows the proven crispembed-quant-upload pattern; stages under /tmp
(the ~70 GB writable layer, NOT /kaggle/working which is ~20 GB) and uploads-then-deletes.
"""

import os, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
_CRISPASR_DIR = WORK / "CrispASR"

if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", CRISPASR_URL, str(_CRISPASR_DIR)])
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass
if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.step("harness_ready", hf_token_ok=bool(hf_token))

# --- deps. Pin transformers==4.57.6: the Kaggle image's build crashes BidirLM's Qwen2
#     tokenizer (_patch_mistral_regex). Don't reinstall torch. ---
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "safetensors", "gguf", "huggingface_hub", "hf_transfer", "transformers==4.57.6"])
kh.step("deps_installed")

# --- clone CrispEmbed main (has --text-only converter + crispembed-quantize) ---
REPO = WORK / "CrispEmbed"
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", "main",
                           "https://github.com/CrispStrobe/CrispEmbed.git", str(REPO)])
    subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive"])
kh.step("cloned")

# --- download the model once (trust_remote_code files come with the snapshot) ---
from huggingface_hub import snapshot_download, HfApi
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
scratch = Path("/tmp") / "bidirlm-work"
scratch.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download.model"):
    src = snapshot_download(repo_id="BidirLM/BidirLM-Omni-2.5B-Embedding",
                            cache_dir=str(scratch / "hf-cache"), token=hf_token)
print(f"[model] {src}", flush=True)
kh.step("model_downloaded")

# --- build crispembed-quantize (CPU) ---
kh.install_build_toolchain()
BUILD = REPO / "build"
BUILD.mkdir(exist_ok=True)
kh.sh_with_progress(f"cmake -G Ninja -S {REPO} -B {BUILD} -DCMAKE_BUILD_TYPE=Release "
                    f"-DGGML_CUDA=OFF " + " ".join(kh.cache_and_link_flags()))
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispembed-quantize -j{kh.safe_build_jobs(gpu=False)}")
QUANTIZE = str(BUILD / "crispembed-quantize")
CONVERTER = str(REPO / "models" / "convert-decoder-embed-to-gguf.py")
kh.step("quantize_built")

api = HfApi(token=hf_token) if hf_token else None
QUANTS = ["q8_0", "q6_k", "q5_k", "q4_k"]  # biggest→smallest; upload+delete each

# (converter flags, HF repo, file prefix)
VARIANTS = [
    ([], "cstr/bidirlm-omni-2.5b-GGUF", "bidirlm-omni-2.5b"),
    (["--text-only"], "cstr/bidirlm-omni-2.5b-textonly-GGUF", "bidirlm-omni-2.5b-textonly"),
]

def upload(path, repo, name, msg):
    if not api:
        print(f"[skip upload] no token: {name}", flush=True); return
    sz = Path(path).stat().st_size / (1024**3)
    print(f"[upload] {name} ({sz:.1f} GiB) -> {repo}", flush=True)
    with kh.build_heartbeat(f"upload.{name}"):
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                        repo_id=repo, repo_type="model", commit_message=msg)
    print(f"[upload] done {name}", flush=True)

for flags, repo, prefix in VARIANTS:
    if api:
        try: api.create_repo(repo, repo_type="model", exist_ok=True)
        except Exception as e: print(f"[repo] {e}", flush=True)
    f16 = scratch / f"{prefix}-f16.gguf"
    with kh.build_heartbeat(f"convert.{prefix}.f16"):
        subprocess.check_call([sys.executable, CONVERTER, "--model", str(src),
                               "--output", str(f16), "--dtype", "f16"] + flags)
    print(f"[convert] {prefix}-f16.gguf = {f16.stat().st_size/(1024**3):.2f} GiB", flush=True)
    kh.step(f"{prefix}_f16_done")

    for qt in QUANTS:
        q = scratch / f"{prefix}-{qt}.gguf"
        with kh.build_heartbeat(f"quantize.{prefix}.{qt}"):
            subprocess.check_call([QUANTIZE, str(f16), str(q), qt])
        upload(q, repo, f"{prefix}-{qt}.gguf", f"{qt} (re-export 2026-07 — fixed text tower)")
        q.unlink(missing_ok=True)  # free disk before the next quant
        kh.step(f"{prefix}_{qt}_done")

    upload(f16, repo, f"{prefix}-f16.gguf", "F16 (re-export 2026-07 — fixed text tower)")
    f16.unlink(missing_ok=True)
    kh.step(f"{prefix}_all_uploaded")

kh.step("all_done")
print("\n[DONE] bidirlm-omni reconvert + quantize + upload complete (both repos)", flush=True)
