# %% [markdown]
# # CrispEmbed — Qwen2.5-VL-3B GGUF conversion
#
# Convert `Qwen/Qwen2.5-VL-3B-Instruct` to F16 GGUF, upload to HF.
#
# Kaggle setup:
#   - Accelerator: None (CPU only)
#   - Internet: ON
#   - Attach dataset: chr1s4/crispasr-hf-token

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
        pass  # fall through to bundled copy

if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.step("harness_ready", hf_token_ok=bool(hf_token))

# %% [code]
REPO = WORK / "CrispEmbed"
BRANCH = "feat/keyven-german-ocr"

print("[1] cloning CrispEmbed", flush=True)
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", BRANCH,
    "https://github.com/CrispStrobe/CrispEmbed.git", str(REPO),
])
kh.step("cloned", branch=BRANCH)

# %% [code]
# torch is pre-installed on Kaggle — only install small deps
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

print("[4] converting to F16 GGUF", flush=True)
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
HF_REPO = "cstr/qwen2.5-vl-3b-crispembed-GGUF"
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    try:
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[5] repo: {e}", flush=True)

    print(f"[5] uploading F16 ({size_gb:.1f} GiB)", flush=True)
    with kh.build_heartbeat("upload.f16"):
        api.upload_file(
            path_or_fileobj=str(OUT_F16),
            path_in_repo="qwen2.5-vl-3b-f16.gguf",
            repo_id=HF_REPO, repo_type="model",
            commit_message="Add F16 GGUF (vision + 36-layer LLM)",
        )
    print("[5] uploaded F16", flush=True)
    kh.step("uploaded")
else:
    print("[5] no HF_TOKEN — skipping upload", flush=True)
    kh.step("upload_skipped")

kh.step("done")
