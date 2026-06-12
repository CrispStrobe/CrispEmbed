# %% [markdown]
# # CrispEmbed — Keyven german-ocr-3.1 end-to-end test
#
# Load Keyven's llama.cpp split GGUFs, run smoke test + OCR.

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
kh.resolve_hf_token()
kh.step("harness_ready")

# %% [code]
REPO = WORK / "CrispEmbed"
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.check_call([
    "git", "clone", "--depth", "1", "--branch", "main",
    "https://github.com/CrispStrobe/CrispEmbed.git", str(REPO),
])
subprocess.check_call(["git", "-C", str(REPO), "submodule", "update",
                        "--init", "--recursive"])
kh.step("cloned")

# %% [code]
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
        f"cmake --build {BUILD} --target test-qwen2vl "
        f"-j{kh.safe_build_jobs(gpu=False)}"
    )
kh.step("built")

# %% [code]
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "huggingface_hub", "hf_transfer", "Pillow",
])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

scratch = Path("/kaggle/temp/keyven-cache")
scratch.mkdir(parents=True, exist_ok=True)

print("[3] downloading Keyven GGUFs", flush=True)
with kh.build_heartbeat("download.keyven"):
    llm_path = hf_hub_download(
        "Keyven/german-ocr-3.1", "german-ocr-3.1-Q4_K_M.gguf",
        cache_dir=str(scratch))
    mmproj_path = hf_hub_download(
        "Keyven/german-ocr-3.1", "mmproj-german-ocr-3.1-F16.gguf",
        cache_dir=str(scratch))
kh.step("downloaded", llm=llm_path, mmproj=mmproj_path)

# Create test invoice
img = Image.new("RGB", (640, 480), "white")
draw = ImageDraw.Draw(img)
draw.text((60, 60), "Rechnung Nr. 2024-0042", fill="black")
draw.text((60, 130), "Firma Mustermann GmbH", fill="black")
draw.text((60, 260), "1  Beratungsleistung  10h  150.00", fill="black")
draw.text((60, 325), "Gesamt: 2500.00 EUR", fill="black")
test_img = str(WORK / "test_invoice.png")
img.save(test_img)

# %% [code]
# Run smoke test with split model
print("[4] running test-qwen2vl with --mmproj", flush=True)
TEST_BIN = str(BUILD / "test-qwen2vl")
with kh.build_heartbeat("test.keyven"):
    result = subprocess.run(
        [TEST_BIN, llm_path, test_img, "--mmproj", mmproj_path],
        capture_output=True, text=True, timeout=600)

print("=== STDOUT ===", flush=True)
print(result.stdout, flush=True)
print("=== STDERR (last 1000) ===", flush=True)
print(result.stderr[-1000:] if result.stderr else "(empty)", flush=True)
print(f"=== exit code: {result.returncode} ===", flush=True)

kh.step("test_done", returncode=result.returncode,
        stdout_tail=result.stdout[-300:] if result.stdout else "")

kh.step("done")
