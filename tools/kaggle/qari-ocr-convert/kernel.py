#!/usr/bin/env python3
"""Kaggle kernel: merge Qari-OCR LoRA into Qwen2-VL-2B and convert to GGUF.

Run on Kaggle with 16 GB RAM (GPU not needed).
Requires: CrispASR kaggle_harness.py for HF auth.

Steps:
1. Download Qwen2-VL-2B-Instruct base model
2. Download Qari-OCR LoRA adapter
3. Merge LoRA into base weights (tensor by tensor)
4. Convert to GGUF using CrispEmbed converter
5. Quantize to Q8_0 and Q4_K
6. Upload to HuggingFace
"""

import gc, json, os, shutil, subprocess, sys

# --- Install deps first ---
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gguf', 'safetensors', 'huggingface_hub'])

# --- Kaggle harness setup (for HF auth token) ---
# Clone CrispASR to get kaggle_harness.py
if not os.path.exists('/kaggle/working/CrispASR'):
    subprocess.run(['git', 'clone', '--depth=1',
                    'https://github.com/CrispStrobe/CrispASR.git',
                    '/kaggle/working/CrispASR'], check=False)

sys.path.insert(0, '/kaggle/working/CrispASR')
try:
    import kaggle_harness as kh
    kh.setup()
    print("Kaggle harness loaded, HF auth configured")
except Exception as e:
    print(f"Warning: kaggle_harness failed ({e}), trying manual HF auth")
    # Try reading token from Kaggle dataset
    for token_path in [
        '/kaggle/input/crispasr-hf-token/hf_token.txt',
        '/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt',
        '/kaggle/input/datasets/chr1str/crispasr-hf-token/hf_token.txt',
    ]:
        if os.path.exists(token_path):
            token = open(token_path).read().strip()
            os.environ['HF_TOKEN'] = token
            print(f"HF token loaded from {token_path}")
            break
    else:
        print("No HF token found — public models only")

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download, snapshot_download

# --- Config ---
BASE_MODEL = 'Qwen/Qwen2-VL-2B-Instruct'
ADAPTER_MODEL = 'NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct'
MERGED_DIR = '/kaggle/working/qari-ocr-merged'
GGUF_OUTPUT = '/kaggle/working/qari-ocr-2b-f16.gguf'

# --- Download ---
print("Downloading base model...")
base_dir = snapshot_download(BASE_MODEL, allow_patterns=['*.safetensors', '*.json'])
print(f"Base: {base_dir}")

print("Downloading adapter...")
adapter_dir = snapshot_download(ADAPTER_MODEL)
print(f"Adapter: {adapter_dir}")

# --- Load adapter config ---
with open(os.path.join(adapter_dir, 'adapter_config.json')) as f:
    acfg = json.load(f)
lora_scale = acfg['lora_alpha'] / acfg['r']
print(f"LoRA: r={acfg['r']}, alpha={acfg['lora_alpha']}, scale={lora_scale}")

# --- Build LoRA lookup ---
adapter = safe_open(os.path.join(adapter_dir, 'adapter_model.safetensors'), framework='pt')
adapter_keys = list(adapter.keys())

PREFIX = 'base_model.model.'
lora_lookup = {}
for k in adapter_keys:
    if '.lora_A.' in k:
        base = k[len(PREFIX):].replace('.lora_A.weight', '') + '.weight'
        b_key = k.replace('.lora_A.', '.lora_B.')
        if b_key in adapter_keys:
            lora_lookup[base] = (k, b_key)

print(f"LoRA pairs: {len(lora_lookup)}")

# --- Merge ---
os.makedirs(MERGED_DIR, exist_ok=True)

# Copy config + tokenizer files
for fname in os.listdir(base_dir):
    if fname.endswith('.json'):
        shutil.copy2(os.path.join(base_dir, fname), os.path.join(MERGED_DIR, fname))

# Process each shard
import glob
shards = sorted(glob.glob(os.path.join(base_dir, 'model-*.safetensors')))
merged_count = 0

for shard_path in shards:
    shard_name = os.path.basename(shard_path)
    print(f"\nProcessing {shard_name}...")

    tensors = {}
    with safe_open(shard_path, framework='pt') as f:
        for key in f.keys():
            t = f.get_tensor(key)

            if key in lora_lookup:
                a_key, b_key = lora_lookup[key]
                A = adapter.get_tensor(a_key).float()
                B = adapter.get_tensor(b_key).float()
                delta = (B @ A) * lora_scale
                t = (t.float() + delta).half()
                merged_count += 1
                print(f"  Merged: {key} (delta norm={delta.norm():.2f})")
                del A, B, delta

            tensors[key] = t

    out_path = os.path.join(MERGED_DIR, shard_name)
    save_file(tensors, out_path)
    print(f"  Saved {out_path} ({len(tensors)} tensors)")
    del tensors
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\nMerged {merged_count}/{len(lora_lookup)} LoRA tensors")

# --- Convert to GGUF ---
print("\n--- Converting to GGUF ---")

# Clone CrispEmbed for the converter (already cloned above for harness, but ensure it's there)
crispembed_dir = '/kaggle/working/CrispEmbed'
if not os.path.exists(crispembed_dir):
    subprocess.check_call(['git', 'clone', '--depth=1', '--recursive',
                           'https://github.com/CrispStrobe/CrispEmbed.git',
                           crispembed_dir])

# Run converter
cmd = [
    sys.executable, os.path.join(crispembed_dir, 'models/convert-qwen2vl-to-gguf.py'),
    '--model', MERGED_DIR,
    '--output', GGUF_OUTPUT,
    '--dtype', 'f16',
]
print(f"Running: {' '.join(cmd)}")
subprocess.check_call(cmd)

print(f"\nGGUF: {GGUF_OUTPUT} ({os.path.getsize(GGUF_OUTPUT) / 1e9:.2f} GB)")

# --- Quantize ---
# Build crispembed-quantize (needs ggml submodule)
print("\n--- Building quantizer ---")
subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'],
               cwd=crispembed_dir, check=False)
os.makedirs('/kaggle/working/build', exist_ok=True)
subprocess.check_call(['cmake', '-S', crispembed_dir,
                        '-B', '/kaggle/working/build',
                        '-DCMAKE_BUILD_TYPE=Release'])
subprocess.check_call(['cmake', '--build', '/kaggle/working/build',
                        '--target', 'crispembed-quantize', '-j4'])

quantizer = '/kaggle/working/build/crispembed-quantize'
for qtype in ['q8_0', 'q4_k']:
    out = GGUF_OUTPUT.replace('-f16.gguf', f'-{qtype}.gguf')
    print(f"\nQuantizing to {qtype}...")
    subprocess.check_call([quantizer, GGUF_OUTPUT, out, qtype])
    print(f"  {out} ({os.path.getsize(out) / 1e6:.0f} MB)")

# --- Upload to HuggingFace ---
print("\n--- Uploading to HuggingFace ---")
from huggingface_hub import HfApi, create_repo

repo_id = 'cstr/qari-ocr-crispembed-GGUF'
try:
    create_repo(repo_id, repo_type='model', exist_ok=True)
except: pass

api = HfApi()
for f in glob.glob('/kaggle/working/qari-ocr-2b-*.gguf'):
    fname = os.path.basename(f)
    print(f"Uploading {fname}...")
    api.upload_file(path_or_fileobj=f, path_in_repo=fname,
                    repo_id=repo_id, commit_message=f'Add {fname}')

print("\nDone!")
