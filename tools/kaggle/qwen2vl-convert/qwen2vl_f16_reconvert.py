# %% [markdown]
# # CrispEmbed — Qwen2.5-VL F16 re-conversion (add tokenizer)
#
# Re-convert F16 GGUF with BPE tokenizer data, upload to HF.
# Quick run: ~3 min (model cached from previous kernel).

# %% [code]
import os, subprocess, sys, shutil, time
from pathlib import Path

WORK = Path("/kaggle/working")
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
_CRISPASR_DIR = WORK / "CrispASR"

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
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", "main",
    "https://github.com/CrispStrobe/CrispEmbed.git", str(REPO),
])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "safetensors", "gguf", "huggingface_hub", "transformers", "hf_transfer",
])
kh.step("cloned_and_deps")

# %% [code]
from huggingface_hub import snapshot_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

scratch = Path("/kaggle/temp/qwen25vl-cache")
scratch.mkdir(parents=True, exist_ok=True)

HF_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
with kh.build_heartbeat("model.download"):
    src = snapshot_download(repo_id=HF_MODEL, cache_dir=str(scratch))
kh.step("model_downloaded")

# %% [code]
OUT_F16 = WORK / "qwen2.5-vl-3b-f16.gguf"
with kh.build_heartbeat("convert.f16"):
    subprocess.check_call([
        sys.executable, str(REPO / "models" / "convert-qwen2vl-to-gguf.py"),
        "--model", src, "--output", str(OUT_F16),
        "--dtype", "f16", "--load-dtype", "bfloat16",
    ])
size_gb = OUT_F16.stat().st_size / (1024**3)
kh.step("f16_done", size_gb=round(size_gb, 2))

# %% [code]
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    with kh.build_heartbeat("upload.f16"):
        api.upload_file(
            path_or_fileobj=str(OUT_F16),
            path_in_repo="qwen2.5-vl-3b-f16.gguf",
            repo_id="cstr/qwen2.5-vl-3b-crispembed-GGUF",
            repo_type="model",
            commit_message="F16 v2 (with BPE tokenizer)",
        )
    kh.step("uploaded")
else:
    kh.step("upload_skipped")

kh.step("done")
