#!/usr/bin/env python3
"""Kaggle kernel: Test CrispEmbed surya-det on GPU (P100/T4).

Builds CrispEmbed with CUDA, downloads the surya-det GGUF, and runs
text detection on a synthetic + real test image. Compares CPU vs GPU.

Usage (Kaggle):
  1. Upload this kernel with GPU enabled
  2. Add chr1s4/crispasr-hf-token dataset for HF auth
  3. Add chr1s4/crispasr-ccache dataset for build cache
  4. Run — it will build, download, and test automatically
"""

import os
import subprocess
import sys
import time

# --- Kaggle harness setup ---
if not os.path.exists("CrispASR"):
    subprocess.run(["git", "clone", "--depth=1",
                    "https://github.com/CrispStrobe/CrispASR.git"], check=True)
sys.path.insert(0, "CrispASR")
try:
    import kaggle_harness as kh
    kh.setup_hf_auth()
    print("Kaggle harness loaded + HF auth set up")
except Exception as e:
    print(f"Kaggle harness setup: {e} (continuing without)")

# --- Clone and build CrispEmbed with CUDA ---
if not os.path.exists("CrispEmbed"):
    subprocess.run(["git", "clone", "--recursive",
                    "https://github.com/CrispStrobe/CrispEmbed.git"], check=True)

os.chdir("CrispEmbed")

# Check GPU
gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                           "--format=csv,noheader"], capture_output=True, text=True)
print(f"GPU: {gpu_info.stdout.strip()}")

# Warm ccache if available
ccache_dir = "/kaggle/working/CrispEmbed/.ccache"
os.makedirs(ccache_dir, exist_ok=True)
os.environ["CCACHE_DIR"] = ccache_dir
for mount in ["/kaggle/input/crispasr-ccache", "/kaggle/input/datasets/chr1s4/crispasr-ccache"]:
    tar_path = os.path.join(mount, "ccache.tar")
    if os.path.exists(tar_path):
        print(f"Warming ccache from {tar_path}")
        subprocess.run(f"tar xf {tar_path} -C /kaggle/working/CrispEmbed/", shell=True)
        break

# Build with CUDA
print("\n=== Building CrispEmbed with CUDA ===")
os.makedirs("build", exist_ok=True)
t0 = time.time()
subprocess.run([
    "cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
    "-G", "Ninja",
], check=True, capture_output=True)

# Build test binary + CLI
subprocess.run(["ninja", "-C", "build", "-j4",
                "test-surya-det", "crispembed-cli"], check=True)
build_time = time.time() - t0
print(f"Build time: {build_time:.0f}s")

# --- Download surya-det GGUF ---
print("\n=== Downloading surya-det models ===")
from huggingface_hub import hf_hub_download

models = {}
for variant in ["surya-det-q8_0.gguf", "surya-det-f16.gguf"]:
    path = hf_hub_download("cstr/surya-det-GGUF", variant)
    models[variant] = path
    sz = os.path.getsize(path) / 1024 / 1024
    print(f"  {variant}: {sz:.0f} MB")

# --- Create test image ---
print("\n=== Creating test image ===")
from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (1200, 1200), (255, 255, 255))
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
except Exception:
    font = ImageFont.load_default()

# Draw text lines
texts = [
    "This is a test document for text detection.",
    "CrispEmbed surya-det should find these lines.",
    "Testing on Kaggle GPU (P100 or T4).",
    "Line four with some numbers: 12345.",
    "Final line of the test document.",
]
for i, text in enumerate(texts):
    draw.text((100, 100 + i * 60), text, fill=(0, 0, 0), font=font)

# Draw a table
for r in range(4):
    for c in range(3):
        x, y = 100 + c * 300, 500 + r * 40
        draw.rectangle([x, y, x + 290, y + 35], outline=(0, 0, 0))
        draw.text((x + 10, y + 8), f"Cell {r},{c}", fill=(0, 0, 0), font=font)

test_img = "/kaggle/working/surya_test.png"
img.save(test_img)
print(f"  Test image: 1200x1200, {len(texts)} text lines + table")

# --- Run detection on GPU ---
print("\n=== Running surya-det on GPU ===")

results = {}
for variant, label in [("surya-det-q8_0.gguf", "Q8_0"), ("surya-det-f16.gguf", "F16")]:
    model_path = models[variant]

    # Run 3 times: first is warmup, average last 2
    times = []
    n_detections = 0
    for run in range(3):
        t0 = time.time()
        result = subprocess.run(
            ["build/test-surya-det", model_path, test_img],
            capture_output=True, text=True, timeout=300
        )
        elapsed = time.time() - t0
        times.append(elapsed)

        if run == 0:
            # Print full output on first run
            print(f"\n--- {label} (run {run+1}: {elapsed:.1f}s) ---")
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")
            for line in result.stderr.strip().split("\n")[-10:]:
                if line.strip():
                    print(f"  [stderr] {line}")
            # Count detections
            for line in result.stdout.split("\n"):
                if "detection" in line.lower() or "bbox" in line.lower() or "region" in line.lower():
                    n_detections += 1
        else:
            print(f"  {label} run {run+1}: {elapsed:.1f}s")

    avg = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    results[label] = {"avg": avg, "times": times, "detections": n_detections}

# --- Run detection on CPU for comparison ---
print("\n=== Running surya-det on CPU ===")
os.environ["CRISPEMBED_FORCE_CPU"] = "1"

for variant, label in [("surya-det-q8_0.gguf", "Q8_0-CPU")]:
    model_path = models[variant.replace("-CPU", "")]
    t0 = time.time()
    result = subprocess.run(
        ["build/test-surya-det", model_path, test_img],
        capture_output=True, text=True, timeout=600
    )
    elapsed = time.time() - t0
    results[label] = {"avg": elapsed, "times": [elapsed]}
    print(f"  {label}: {elapsed:.1f}s")

del os.environ["CRISPEMBED_FORCE_CPU"]

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"GPU: {gpu_info.stdout.strip()}")
print(f"Build time: {build_time:.0f}s")
print()
print(f"{'Variant':<15} {'Avg (s)':>10} {'Speedup':>10}")
print("-" * 40)

cpu_time = results.get("Q8_0-CPU", {}).get("avg", 1.0)
for label in ["Q8_0", "F16", "Q8_0-CPU"]:
    if label in results:
        avg = results[label]["avg"]
        speedup = cpu_time / avg if avg > 0 else 0
        suffix = "" if "CPU" in label else f" ({speedup:.1f}x)"
        print(f"{label:<15} {avg:>10.1f} {suffix}")

print("\n=== DONE ===")
