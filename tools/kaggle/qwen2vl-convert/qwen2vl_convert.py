# %% [markdown]
# # CrispEmbed — Qwen2.5-VL-3B GGUF conversion (v2: with tokenizer)
#
# Convert `Qwen/Qwen2.5-VL-3B-Instruct` to F16 GGUF with BPE tokenizer,
# quantize to Q8_0 + Q4_K, upload all to HF.

# %% [code]
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

# %% [code]
REPO = WORK / "CrispEmbed"
BRANCH = "main"

print("[1] cloning CrispEmbed", flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", BRANCH,
    "https://github.com/CrispStrobe/CrispEmbed.git", str(REPO),
])
subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive"])
kh.step("cloned", branch=BRANCH)

# %% [code]
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "safetensors", "gguf", "huggingface_hub", "transformers", "hf_transfer",
])
kh.step("deps_installed")

# %% [code]
from huggingface_hub import snapshot_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        scratch = Path(candidate) / "qwen25vl-cache"
        break
scratch.mkdir(parents=True, exist_ok=True)
free_gb = shutil.disk_usage(scratch).free / (1024**3)
print(f"[3] scratch: {scratch} (free: {free_gb:.1f} GiB)", flush=True)

HF_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
print(f"[3] downloading {HF_MODEL}", flush=True)
with kh.build_heartbeat("model.download"):
    src = snapshot_download(repo_id=HF_MODEL, cache_dir=str(scratch))
kh.step("model_downloaded", src=src)

# %% [code]
OUT_F16 = WORK / "qwen2.5-vl-3b-f16.gguf"
converter = REPO / "models" / "convert-qwen2vl-to-gguf.py"

print("[4] converting to F16 GGUF (with tokenizer)", flush=True)
with kh.build_heartbeat("convert.f16"):
    subprocess.check_call([
        sys.executable, str(converter),
        "--model", src,
        "--output", str(OUT_F16),
        "--dtype", "f16",
        "--load-dtype", "bfloat16",
    ])
size_gb = OUT_F16.stat().st_size / (1024**3)
print(f"[4] F16: {size_gb:.2f} GiB", flush=True)
kh.step("f16_done", size_gb=round(size_gb, 2))

# %% [code]
# Build quantizer
print("[5] building crispembed-quantize", flush=True)
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

# %% [code]
# Quantize Q8_0
OUT_Q8 = WORK / "qwen2.5-vl-3b-q8_0.gguf"
print("[6] quantizing F16 -> Q8_0", flush=True)
with kh.build_heartbeat("quantize.q8"):
    subprocess.check_call([str(QUANTIZE), str(OUT_F16), str(OUT_Q8), "q8_0"])
q8_gb = OUT_Q8.stat().st_size / (1024**3)
print(f"[6] Q8_0: {q8_gb:.2f} GiB", flush=True)
kh.step("q8_done", size_gb=round(q8_gb, 2))

# Quantize Q4_K
OUT_Q4 = WORK / "qwen2.5-vl-3b-q4_k.gguf"
print("[7] quantizing F16 -> Q4_K", flush=True)
with kh.build_heartbeat("quantize.q4k"):
    subprocess.check_call([str(QUANTIZE), str(OUT_F16), str(OUT_Q4), "q4_k"])
q4_gb = OUT_Q4.stat().st_size / (1024**3)
print(f"[7] Q4_K: {q4_gb:.2f} GiB", flush=True)
kh.step("q4k_done", size_gb=round(q4_gb, 2))

# Delete F16 to free space before upload
OUT_F16.unlink(missing_ok=True)
kh.step("f16_deleted_for_space")

# %% [code]
HF_REPO = "cstr/qwen2.5-vl-3b-crispembed-GGUF"
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    try:
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[8] repo: {e}", flush=True)

    for path, name, msg in [
        (OUT_Q8, "qwen2.5-vl-3b-q8_0.gguf", "Q8_0 v2 (with tokenizer)"),
        (OUT_Q4, "qwen2.5-vl-3b-q4_k.gguf", "Q4_K v2 (with tokenizer, vision Q8_0 floor)"),
    ]:
        if path.exists():
            sz = path.stat().st_size / (1024**3)
            print(f"[8] uploading {name} ({sz:.1f} GiB)", flush=True)
            with kh.build_heartbeat(f"upload.{name}"):
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=name,
                    repo_id=HF_REPO, repo_type="model",
                    commit_message=msg,
                )
            print(f"[8] uploaded {name}", flush=True)
    kh.step("uploaded")
else:
    print("[8] no HF_TOKEN — skipping upload", flush=True)
    kh.step("upload_skipped")

kh.step("done")
