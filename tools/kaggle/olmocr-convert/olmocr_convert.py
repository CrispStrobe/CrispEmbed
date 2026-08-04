# %% [markdown]
# # CrispEmbed — olmOCR-2-7B GGUF conversion
#
# Convert `allenai/olmOCR-2-7B-1025` (Apache-2.0 Qwen2.5-VL-7B document
# fine-tune) to F16 GGUF with BPE tokenizer, quantize to Q8_0 + Q4_K,
# upload to HF. 7B sizing: F16 ~16 GiB + Q8 ~8.7 + Q4_K ~4.7 exceeds the
# 20 GB /kaggle/working cap, so everything big stages under /tmp
# (~70 GB ephemeral layer) and uploads straight to HF.

# %% [code]
import os, subprocess, sys, shutil
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
if not hf_token:
    raise SystemExit("no HF token — upload would be impossible, aborting early")

# %% [code]
# Big files live on the ephemeral layer, NOT /kaggle/working (20 GB cap).
for candidate in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(candidate):
        SCRATCH = Path(candidate) / "olmocr-work"
        break
SCRATCH.mkdir(parents=True, exist_ok=True)
free_gb = shutil.disk_usage(SCRATCH).free / (1024**3)
print(f"[0] scratch: {SCRATCH} (probe free: {free_gb:.1f} GiB — budget ~70)", flush=True)

REPO = SCRATCH / "CrispEmbed"
BRANCH = "feat/olmocr-lane"

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

HF_MODEL = "allenai/olmOCR-2-7B-1025"
print(f"[3] downloading {HF_MODEL}", flush=True)
with kh.build_heartbeat("model.download"):
    src = snapshot_download(repo_id=HF_MODEL, cache_dir=str(SCRATCH / "hf-cache"))
kh.step("model_downloaded", src=src)

# The converter writes general.name from the path's last component, which for
# a snapshot_download is a revision hash. The runtime detects the olmOCR
# contract from that name, so hand the converter a properly named alias.
named_src = SCRATCH / "olmOCR-2-7B-1025"
if not named_src.exists():
    named_src.symlink_to(src)
src = str(named_src)

# %% [code]
OUT_F16 = SCRATCH / "olmocr-2-7b-f16.gguf"
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

# Free the HF snapshot (~16 GiB) before quantizing — /tmp is ~70 GB total.
shutil.rmtree(SCRATCH / "hf-cache", ignore_errors=True)
kh.step("hf_cache_freed")

# %% [code]
# Build quantizer
print("[5] building crispembed-quantize", flush=True)
kh.install_build_toolchain()
BUILD = SCRATCH / "build"
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
from huggingface_hub import HfApi
api = HfApi(token=hf_token)
HF_REPO = "cstr/olmOCR-2-7B-1025-GGUF"
try:
    api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
except Exception as e:
    print(f"[6] repo: {e}", flush=True)

def upload(path: Path, msg: str):
    sz = path.stat().st_size / (1024**3)
    print(f"[up] uploading {path.name} ({sz:.1f} GiB)", flush=True)
    with kh.build_heartbeat(f"upload.{path.name}"):
        api.upload_file(
            path_or_fileobj=str(path), path_in_repo=path.name,
            repo_id=HF_REPO, repo_type="model", commit_message=msg,
        )
    print(f"[up] uploaded {path.name}", flush=True)

# Quantize + upload one at a time, deleting as we go (disk budget).
OUT_Q8 = SCRATCH / "olmocr-2-7b-q8_0.gguf"
print("[6] quantizing F16 -> Q8_0", flush=True)
with kh.build_heartbeat("quantize.q8"):
    subprocess.check_call([str(QUANTIZE), str(OUT_F16), str(OUT_Q8), "q8_0"])
print(f"[6] Q8_0: {OUT_Q8.stat().st_size / (1024**3):.2f} GiB", flush=True)
kh.step("q8_done")
upload(OUT_Q8, "Q8_0 from allenai/olmOCR-2-7B-1025 (CrispEmbed converter)")
OUT_Q8.unlink()

OUT_Q4 = SCRATCH / "olmocr-2-7b-q4_k.gguf"
print("[7] quantizing F16 -> Q4_K", flush=True)
with kh.build_heartbeat("quantize.q4k"):
    subprocess.check_call([str(QUANTIZE), str(OUT_F16), str(OUT_Q4), "q4_k"])
print(f"[7] Q4_K: {OUT_Q4.stat().st_size / (1024**3):.2f} GiB", flush=True)
kh.step("q4k_done")
upload(OUT_Q4, "Q4_K from allenai/olmOCR-2-7B-1025 (CrispEmbed converter, vision Q8_0 floor)")
OUT_Q4.unlink()

OUT_F16.unlink(missing_ok=True)
kh.step("done")
print("[8] all done", flush=True)
