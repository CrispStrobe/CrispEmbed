#!/usr/bin/env python3
"""CrispEmbed — DeepSeek-OCR-2 quantize + upload.

Download F16 GGUF from HF, quantize to Q8_0 + Q4_K, upload back.
Follows the proven qwen2vl_convert.py pattern exactly.
"""

import os, subprocess, sys, shutil, time
from pathlib import Path

WORK = Path("/kaggle/working")
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
_CRISPASR_DIR = WORK / "CrispASR"

# Clone CrispASR for kaggle_harness; fall back to bundled copy
if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
            CRISPASR_URL, str(_CRISPASR_DIR)])
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass
if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.step("harness_ready", hf_token_ok=bool(hf_token))

# --- Step 1: Install deps ---
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "safetensors", "huggingface_hub", "hf_transfer",
])
kh.step("deps_installed")

# --- Step 2: Download F16 GGUF from HF ---
print("[2] downloading DeepSeek-OCR-2 F16 GGUF", flush=True)
from huggingface_hub import hf_hub_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "deepseek-cache"
        break
scratch.mkdir(parents=True, exist_ok=True)

with kh.build_heartbeat("download.f16"):
    f16_path = hf_hub_download(
        "cstr/deepseek-ocr2-crispembed-GGUF",
        "deepseek-ocr2-f16.gguf",
        cache_dir=str(scratch),
        token=hf_token,
    )
f16_size = os.path.getsize(f16_path) / (1024**3)
print(f"[2] F16: {f16_size:.2f} GiB at {f16_path}", flush=True)
kh.step("f16_downloaded", size_gb=round(f16_size, 2))

# --- Step 3: Clone CrispEmbed and build quantizer ---
print("[3] building crispembed-quantize", flush=True)

REPO = WORK / "CrispEmbed"
if not REPO.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--branch", "main",
        "https://github.com/CrispStrobe/CrispEmbed.git", str(REPO),
    ])
    subprocess.check_call(["git", "-C", str(REPO), "submodule", "update",
                           "--init", "--recursive"])
kh.step("cloned")

kh.install_build_toolchain()
BUILD = WORK / "build"
BUILD.mkdir(exist_ok=True)

cmake_cfg = (
    f"cmake -G Ninja -S {REPO} -B {BUILD} "
    f"-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF "
    + " ".join(kh.cache_and_link_flags())
)
kh.sh_with_progress(cmake_cfg)

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} --target crispembed-quantize "
        f"-j{kh.safe_build_jobs(gpu=False)}"
    )
kh.step("quantize_built")
QUANTIZE = BUILD / "crispembed-quantize"

# --- Step 4: Quantize ---
OUT_Q8 = WORK / "deepseek-ocr2-q8_0.gguf"
print("[4] quantizing F16 -> Q8_0", flush=True)
with kh.build_heartbeat("quantize.q8"):
    subprocess.check_call([str(QUANTIZE), f16_path, str(OUT_Q8), "q8_0"])
q8_gb = OUT_Q8.stat().st_size / (1024**3)
print(f"[4] Q8_0: {q8_gb:.2f} GiB", flush=True)
kh.step("q8_done", size_gb=round(q8_gb, 2))

OUT_Q4 = WORK / "deepseek-ocr2-q4_k.gguf"
print("[5] quantizing F16 -> Q4_K", flush=True)
with kh.build_heartbeat("quantize.q4k"):
    subprocess.check_call([str(QUANTIZE), f16_path, str(OUT_Q4), "q4_k"])
q4_gb = OUT_Q4.stat().st_size / (1024**3)
print(f"[5] Q4_K: {q4_gb:.2f} GiB", flush=True)
kh.step("q4k_done", size_gb=round(q4_gb, 2))

# --- Step 5: Upload to HF ---
HF_REPO = "cstr/deepseek-ocr2-crispembed-GGUF"
if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    try:
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[6] repo: {e}", flush=True)

    for path, name, msg in [
        (OUT_Q8, "deepseek-ocr2-q8_0.gguf", "Q8_0 quantized"),
        (OUT_Q4, "deepseek-ocr2-q4_k.gguf", "Q4_K quantized"),
    ]:
        if path.exists():
            sz = path.stat().st_size / (1024**3)
            print(f"[6] uploading {name} ({sz:.1f} GiB)", flush=True)
            with kh.build_heartbeat(f"upload.{name}"):
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=name,
                    repo_id=HF_REPO, repo_type="model",
                    commit_message=msg,
                )
            print(f"[6] uploaded {name}", flush=True)
    kh.step("uploaded")
else:
    print("[6] SKIP upload — no HF token", flush=True)

kh.step("all_done")
print("\n[DONE] DeepSeek-OCR-2 quantization complete", flush=True)
