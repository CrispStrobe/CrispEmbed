#!/usr/bin/env python3
"""Kaggle kernel: Test CrispEmbed surya-det on GPU (P100/T4).

Builds CrispEmbed with CUDA, downloads the surya-det GGUF, and runs
text detection on a test image. Compares CPU vs GPU timings.

Usage (Kaggle):
  1. Upload this kernel with GPU enabled
  2. Add chr1s4/crispasr-hf-token dataset for HF auth
  3. Run — it will build, download, and test automatically
"""

import os
import subprocess
import sys
import time

# --- Kaggle harness setup ---
# Clone CrispASR for kaggle_harness.py (3-tier auth, heartbeat, etc.)
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
os.makedirs("build", exist_ok=True)
os.chdir("build")

# Check GPU
gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                           "--format=csv,noheader"], capture_output=True, text=True)
print(f"GPU: {gpu_info.stdout.strip()}")

# Build with CUDA
print("\n=== Building CrispEmbed with CUDA ===")
t0 = time.time()
subprocess.run([
    "cmake", "..", "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_CUDA=ON",
    "-G", "Ninja",
], check=True, capture_output=True)
subprocess.run(["ninja", "-j4", "crispembed"], check=True)
build_time = time.time() - t0
print(f"Build time: {build_time:.0f}s")

os.chdir("..")

# --- Download surya-det GGUF ---
print("\n=== Downloading surya-det model ===")
from huggingface_hub import hf_hub_download

for variant in ["surya-det-f16.gguf", "surya-det-q8_0.gguf"]:
    path = hf_hub_download("cstr/surya-det-GGUF", variant)
    print(f"  {variant}: {path}")

# --- Create test image ---
print("\n=== Creating test image ===")
from PIL import Image, ImageDraw, ImageFont

img = Image.new("RGB", (1200, 1200), (255, 255, 255))
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
except:
    font = ImageFont.load_default()

# Draw text lines
for i, text in enumerate([
    "This is a test document for text detection.",
    "CrispEmbed surya-det should find these lines.",
    "Testing on Kaggle GPU (P100 or T4).",
    "Line four with some numbers: 12345.",
    "Final line of the test document.",
]):
    draw.text((100, 100 + i * 60), text, fill=(0, 0, 0), font=font)

# Draw a table
for r in range(4):
    for c in range(3):
        x, y = 100 + c * 300, 500 + r * 40
        draw.rectangle([x, y, x + 290, y + 35], outline=(0, 0, 0))
        draw.text((x + 10, y + 8), f"Cell {r},{c}", fill=(0, 0, 0), font=font)

img.save("/tmp/surya_test.png")
print("  Test image: 1200x1200, 5 text lines + table")

# --- Run detection ---
print("\n=== Running surya-det (GPU) ===")

for variant, label in [("surya-det-q8_0.gguf", "Q8_0"), ("surya-det-f16.gguf", "F16")]:
    model_path = hf_hub_download("cstr/surya-det-GGUF", variant)

    t0 = time.time()
    result = subprocess.run(
        ["build/crispembed", "-m", model_path, "--text-det", "/tmp/surya_test.png"],
        capture_output=True, text=True, timeout=120
    )
    elapsed = time.time() - t0

    print(f"\n--- {label} ({elapsed:.1f}s) ---")
    print(f"  stdout: {result.stdout[:500]}")
    if result.stderr:
        # Count detections from stderr
        lines = result.stderr.strip().split("\n")
        for line in lines[-10:]:
            print(f"  {line}")

# --- Compare CPU vs GPU timing ---
print("\n=== CPU vs GPU comparison ===")
print("(GPU results above; CPU baseline from VPS: ~60s F32, ~17s graph)")

print("\n=== DONE ===")
