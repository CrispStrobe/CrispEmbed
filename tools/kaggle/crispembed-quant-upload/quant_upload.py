#!/usr/bin/env python3
"""Kaggle kernel: quantize DeepSeek-OCR-2 F16 → Q8_0 + Q4_K and upload to HF.

Uses CrispASR kaggle_harness.py for proper HF token resolution, heartbeat, etc.
"""

import subprocess, os, sys, time, shutil

# ── Step 1: Clone CrispASR for the kaggle harness ──
os.chdir("/kaggle/working")
subprocess.run("git clone --depth 1 https://github.com/CrispStrobe/CrispASR.git", shell=True)
sys.path.insert(0, "/kaggle/working/CrispASR/tools/kaggle")
import kaggle_harness as kh

kh.resolve_tokens()
hf_token = os.environ.get("HF_TOKEN", "")
print(f"HF token: {hf_token[:15]}..." if hf_token else "ERROR: No HF token")
if not hf_token:
    sys.exit(1)

# ── Step 2: Clone CrispEmbed and build quantizer ──
os.chdir("/kaggle/working")
subprocess.run("git clone --depth 1 https://github.com/CrispStrobe/CrispEmbed.git", shell=True)
os.chdir("/kaggle/working/CrispEmbed")
subprocess.run("git submodule update --init --recursive", shell=True)

# Install build tools
subprocess.run("apt-get update -qq && apt-get install -y -qq ninja-build cmake", shell=True)

# Build quantizer only
subprocess.run("cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja", shell=True, check=True)
subprocess.run("ninja -C build -j$(nproc) crispembed-quantize", shell=True, check=True)

QUANTIZE = "/kaggle/working/CrispEmbed/build/crispembed-quantize"

# ── Step 3: Download DeepSeek-OCR-2 F16 from HF ──
print("\n" + "="*60)
print("Downloading DeepSeek-OCR-2 F16...")
print("="*60)

os.makedirs("/kaggle/working/models", exist_ok=True)
subprocess.run(
    f"huggingface-cli download cstr/deepseek-ocr2-crispembed-GGUF deepseek-ocr2-f16.gguf "
    f"--local-dir /kaggle/working/models --token {hf_token}",
    shell=True, check=True
)

f16 = "/kaggle/working/models/deepseek-ocr2-f16.gguf"
assert os.path.exists(f16), f"F16 not found at {f16}"
print(f"F16 size: {os.path.getsize(f16)/1e9:.1f} GB")

# ── Step 4: Quantize ──
print("\n" + "="*60)
print("Quantizing to Q8_0...")
print("="*60)

q8 = "/kaggle/working/models/deepseek-ocr2-q8_0.gguf"
q4 = "/kaggle/working/models/deepseek-ocr2-q4_k.gguf"

with kh.build_heartbeat():
    subprocess.run(f"{QUANTIZE} {f16} {q8} q8_0", shell=True, check=True)
    print(f"Q8_0 size: {os.path.getsize(q8)/1e9:.1f} GB")

    subprocess.run(f"{QUANTIZE} {f16} {q4} q4_k", shell=True, check=True)
    print(f"Q4_K size: {os.path.getsize(q4)/1e9:.1f} GB")

# ── Step 5: Upload to HF ──
print("\n" + "="*60)
print("Uploading quantized models to HF...")
print("="*60)

from huggingface_hub import HfApi
api = HfApi(token=hf_token)
repo = "cstr/deepseek-ocr2-crispembed-GGUF"

for path, name in [(q8, "deepseek-ocr2-q8_0.gguf"), (q4, "deepseek-ocr2-q4_k.gguf")]:
    if os.path.exists(path):
        sz = os.path.getsize(path) / 1e9
        print(f"Uploading {name} ({sz:.1f} GB)...")
        api.upload_file(path_or_fileobj=path, path_in_repo=name,
                        repo_id=repo, repo_type="model")
        print(f"  Done: {name}")

print("\n" + "="*60)
print("ALL DONE")
print("="*60)
