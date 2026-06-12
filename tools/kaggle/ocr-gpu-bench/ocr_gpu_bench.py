#!/usr/bin/env python3
"""CrispEmbed OCR GPU benchmark — builds CrispEmbed with CUDA on Kaggle,
runs all OCR models and reports timing + correctness.

Runs on P100 (sm_60) or T4 (sm_75).
"""
import os, sys, subprocess, time, json
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
BRANCH = "feat/posformer-port"

# --- Auth via kaggle_harness ---
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
try:
    import kaggle_harness as kh
    kh.init_progress()
except ImportError:
    print("WARNING: kaggle_harness not available")

def run(cmd, **kwargs):
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True, **kwargs)

def timed(label, cmd):
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    dt = time.time() - t0
    print(f"[{label}] {dt:.1f}s | exit={result.returncode}")
    if result.stdout.strip():
        for line in result.stdout.strip().split("\n")[-5:]:
            print(f"  {line}")
    if result.returncode != 0 and result.stderr.strip():
        for line in result.stderr.strip().split("\n")[-5:]:
            print(f"  ERR: {line}")
    return dt, result

# --- Step 1: Clone and build ---
print("=" * 60)
print("Step 1: Clone CrispEmbed and build with CUDA")
print("=" * 60)

EMBED_DIR = WORK / "CrispEmbed"
if not EMBED_DIR.exists():
    run(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")

BUILD_DIR = EMBED_DIR / "build"
BUILD_DIR.mkdir(exist_ok=True)

# Detect GPU
gpu_info = subprocess.run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
                          shell=True, capture_output=True, text=True)
gpu_name = gpu_info.stdout.strip() if gpu_info.returncode == 0 else "unknown"
has_gpu = gpu_info.returncode == 0 and "GPU" not in gpu_name.lower() or len(gpu_name) > 3
print(f"GPU: {gpu_name}")

# Install deps
run("pip install -q gguf safetensors")

# Build — with CUDA if GPU available, otherwise CPU-only
os.chdir(str(BUILD_DIR))
cuda_flag = "-DGGML_CUDA=ON" if has_gpu else ""
# P100 needs cuda_driver stub — Kaggle's CUDA 12.8 sometimes misses it
if has_gpu:
    # Find libcuda.so stub for cmake
    import glob
    stubs = glob.glob("/usr/local/cuda/targets/*/lib/stubs/libcuda.so")
    if stubs:
        stub_dir = os.path.dirname(stubs[0])
        cuda_flag += f" -DCMAKE_LIBRARY_PATH={stub_dir}"

cmake_cmd = (
    f"cmake {EMBED_DIR} -G 'Unix Makefiles' "
    f"-DCMAKE_BUILD_TYPE=Release {cuda_flag}"
)
try:
    run(cmake_cmd)
except subprocess.CalledProcessError:
    # CUDA cmake failed — fall back to CPU-only
    print("CUDA cmake failed, falling back to CPU-only build")
    run(f"cmake {EMBED_DIR} -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release")

run(f"make -j$(nproc)")

# --- Step 2: Download models ---
print("\n" + "=" * 60)
print("Step 2: Download GGUF models from HuggingFace")
print("=" * 60)

MODELS_DIR = WORK / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Get HF token
hf_token = None
for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
          "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
    if Path(p).exists():
        hf_token = Path(p).read_text().strip()
        break

from huggingface_hub import hf_hub_download

models = {
    "posformer": ("cstr/posformer-crohme-GGUF", "posformer-crohme-q8_0.gguf"),
    "surya-det": ("cstr/surya-det-GGUF", "surya-det-q8_0.gguf"),
    "bttr": ("cstr/bttr-hw-GGUF", "bttr-hw-q8_0.gguf"),
}

model_paths = {}
for name, (repo, filename) in models.items():
    try:
        path = hf_hub_download(repo_id=repo, filename=filename,
                               local_dir=str(MODELS_DIR), token=hf_token)
        model_paths[name] = path
        print(f"  {name}: {path} ({Path(path).stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

# --- Step 3: Create test images ---
print("\n" + "=" * 60)
print("Step 3: Create test images")
print("=" * 60)

from PIL import Image, ImageDraw
import numpy as np

# Document image for surya-det
doc = Image.new('RGB', (800, 600), 'white')
d = ImageDraw.Draw(doc)
for i, line in enumerate([
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning transforms language processing.",
    "CrispEmbed: lightweight embedding via ggml.",
]):
    d.text((40, 40 + i * 60), line, fill='black')
doc.save(str(WORK / "test_doc.png"))

# Math formula for handwriting OCR
hw = Image.new('L', (200, 60), 255)
d2 = ImageDraw.Draw(hw)
d2.text((10, 15), "x + y = z", fill=0)
hw.save(str(WORK / "test_formula.bmp"))

print("Test images created")

# --- Step 4: Run benchmarks ---
print("\n" + "=" * 60)
print("Step 4: OCR Benchmarks (GPU)")
print("=" * 60)

results = {}
os.chdir(str(BUILD_DIR))

# Surya-det
if "surya-det" in model_paths:
    dt, r = timed("surya-det",
        f"./test-surya-det {model_paths['surya-det']} {WORK}/test_doc.png")
    results["surya-det"] = {"time_s": dt, "exit": r.returncode,
                            "output": r.stderr[-500:] if r.stderr else r.stdout[-500:]}

# PosFormer
if "posformer" in model_paths:
    dt, r = timed("posformer",
        f"./test-posformer {model_paths['posformer']} {WORK}/test_formula.bmp")
    results["posformer"] = {"time_s": dt, "exit": r.returncode,
                            "output": r.stderr[-200:] if r.stderr else ""}

# BTTR
if "bttr" in model_paths:
    dt, r = timed("bttr",
        f"./test-bttr {model_paths['bttr']} {WORK}/test_formula.bmp")
    results["bttr"] = {"time_s": dt, "exit": r.returncode,
                       "output": r.stderr[-200:] if r.stderr else ""}

# --- Step 5: Summary ---
print("\n" + "=" * 60)
print("Step 5: Summary")
print("=" * 60)

print(f"\nGPU: {gpu_name}")
print(f"{'Model':<15} {'Time':>8} {'Status':>8}")
print("-" * 35)
for name, r in results.items():
    status = "PASS" if r["exit"] == 0 else "FAIL"
    print(f"{name:<15} {r['time_s']:>7.1f}s {status:>8}")

# Save results
with open(str(WORK / "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nDone!")
