#!/usr/bin/env python3
"""Kaggle kernel: Convert Nanonets-OCR-s to CrispEmbed GGUF + quantize + upload.

Runs on Kaggle's 16GB RAM CPU instance. Uses CrispASR kaggle_harness for
build toolchain + HF token handling.

Attach datasets: chr1s4/crispasr-hf-token
"""
import subprocess, sys, os, gc

# Clone CrispEmbed
os.chdir("/kaggle/working")
subprocess.run("git clone --recursive https://github.com/CrispStrobe/CrispEmbed.git", shell=True, check=True)
os.chdir("CrispEmbed")

# Install gguf
subprocess.run([sys.executable, "-m", "pip", "install", "gguf", "safetensors", "transformers", "-q"], check=True)

# Get HF token for upload
hf_token = None
for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
          "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
    if os.path.exists(p):
        hf_token = open(p).read().strip()
        break
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print(f"HF token loaded ({len(hf_token)} chars)")

# Convert
print("Converting Nanonets-OCR-s...")
subprocess.run([
    sys.executable, "models/convert-qwen2vl-to-gguf.py",
    "--model", "nanonets/Nanonets-OCR-s",
    "--dtype", "f16",
    "--output", "/kaggle/working/nanonets-ocr-s-f16.gguf"
], check=True)
print(f"F16 GGUF: {os.path.getsize('/kaggle/working/nanonets-ocr-s-f16.gguf') / 1024**2:.0f} MB")

# Build quantizer
subprocess.run("cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release", shell=True, check=True)
subprocess.run("ninja -C build crispembed-quantize", shell=True, check=True)

# Quantize
for qt in ["q8_0", "q4_k"]:
    out = f"/kaggle/working/nanonets-ocr-s-{qt}.gguf"
    subprocess.run(["build/crispembed-quantize",
                    "/kaggle/working/nanonets-ocr-s-f16.gguf", out, qt], check=True)
    print(f"{qt}: {os.path.getsize(out) / 1024**2:.0f} MB")

# Upload to HuggingFace
if hf_token:
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=hf_token)
    repo_id = "cstr/nanonets-ocr-s-crispembed-GGUF"
    create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)
    for f in ["nanonets-ocr-s-f16.gguf", "nanonets-ocr-s-q8_0.gguf", "nanonets-ocr-s-q4_k.gguf"]:
        path = f"/kaggle/working/{f}"
        if os.path.exists(path):
            print(f"Uploading {f}...")
            api.upload_file(path_or_fileobj=path, path_in_repo=f, repo_id=repo_id, token=hf_token)
    print("Upload complete!")

print("DONE")
