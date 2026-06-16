#!/usr/bin/env python3
"""Kaggle kernel: merge Qari-OCR LoRA into Qwen2-VL-2B and convert to GGUF.

Run on Kaggle with 16 GB RAM (no GPU needed).
"""
import gc, json, os, shutil, subprocess, sys, glob

# --- Install deps ---
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'gguf', 'safetensors', 'huggingface_hub'])

# --- HF auth (for upload) ---
for token_path in [
    '/kaggle/input/crispasr-hf-token/hf_token.txt',
    '/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt',
    '/kaggle/input/datasets/chr1str/crispasr-hf-token/hf_token.txt',
]:
    if os.path.exists(token_path):
        token = open(token_path).read().strip()
        os.environ['HF_TOKEN'] = token
        print(f"HF token from {token_path}")
        break

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download, snapshot_download, HfApi, create_repo

BASE_MODEL = 'Qwen/Qwen2-VL-2B-Instruct'
ADAPTER_MODEL = 'NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct'
MERGED_DIR = '/kaggle/working/qari-ocr-merged'
GGUF_F16 = '/kaggle/working/qari-ocr-2b-f16.gguf'

# --- Download ---
print("Downloading base model...")
base_dir = snapshot_download(BASE_MODEL, allow_patterns=['*.safetensors', '*.json'])
print(f"Base: {base_dir}")

print("Downloading adapter...")
adapter_dir = snapshot_download(ADAPTER_MODEL)
print(f"Adapter: {adapter_dir}")

# --- Adapter config ---
with open(os.path.join(adapter_dir, 'adapter_config.json')) as f:
    acfg = json.load(f)
lora_scale = acfg['lora_alpha'] / acfg['r']
print(f"LoRA: r={acfg['r']}, alpha={acfg['lora_alpha']}, scale={lora_scale}")

# --- Build LoRA lookup ---
adapter = safe_open(os.path.join(adapter_dir, 'adapter_model.safetensors'), framework='pt')
PREFIX = 'base_model.model.'
lora_lookup = {}
for k in adapter.keys():
    if '.lora_A.' in k:
        base = k[len(PREFIX):].replace('.lora_A.weight', '') + '.weight'
        b_key = k.replace('.lora_A.', '.lora_B.')
        if b_key in list(adapter.keys()):
            lora_lookup[base] = (k, b_key)
print(f"LoRA pairs: {len(lora_lookup)}")

# --- Merge ---
os.makedirs(MERGED_DIR, exist_ok=True)
for fname in os.listdir(base_dir):
    if fname.endswith('.json'):
        shutil.copy2(os.path.join(base_dir, fname), os.path.join(MERGED_DIR, fname))

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
                if merged_count <= 5 or merged_count % 50 == 0:
                    print(f"  [{merged_count}] Merged: {key} (delta={delta.norm():.2f})")
                del A, B, delta
            tensors[key] = t
    save_file(tensors, os.path.join(MERGED_DIR, shard_name))
    print(f"  Saved ({len(tensors)} tensors)")
    del tensors; gc.collect()

print(f"\nMerged {merged_count}/{len(lora_lookup)} LoRA tensors")

# --- Convert to GGUF ---
print("\n=== GGUF Conversion ===")
crispembed_dir = '/kaggle/working/CrispEmbed'
if not os.path.exists(crispembed_dir):
    subprocess.check_call(['git', 'clone', '--depth=1', '--recursive',
                           'https://github.com/CrispStrobe/CrispEmbed.git',
                           crispembed_dir])

cmd = [sys.executable,
       os.path.join(crispembed_dir, 'models/convert-qwen2vl-to-gguf.py'),
       '--model', MERGED_DIR,
       '--output', GGUF_F16,
       '--dtype', 'f16']
print(f"Running: {' '.join(cmd)}")
subprocess.check_call(cmd)
print(f"F16 GGUF: {os.path.getsize(GGUF_F16) / 1e9:.2f} GB")

# --- Build quantizer ---
print("\n=== Building quantizer ===")
subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'],
               cwd=crispembed_dir, check=False)

# Strip test targets from CMakeLists (may reference files not in shallow clone)
import re
cml_path = os.path.join(crispembed_dir, 'CMakeLists.txt')
with open(cml_path) as f:
    cml = f.read()
cml = re.sub(r'(?:add_executable|target_link_libraries|target_include_directories)\(test-[^\)]*\)\n?', '', cml)
with open(cml_path, 'w') as f:
    f.write(cml)

build_dir = '/kaggle/working/build'
os.makedirs(build_dir, exist_ok=True)
subprocess.check_call(['cmake', '-S', crispembed_dir, '-B', build_dir,
                        '-DCMAKE_BUILD_TYPE=Release'])
subprocess.check_call(['cmake', '--build', build_dir,
                        '--target', 'crispembed-quantize', '-j4'])

# --- Quantize ---
quantizer = os.path.join(build_dir, 'crispembed-quantize')
for qtype in ['q8_0', 'q4_k']:
    out = GGUF_F16.replace('-f16.gguf', f'-{qtype}.gguf')
    print(f"\nQuantizing to {qtype}...")
    subprocess.check_call([quantizer, GGUF_F16, out, qtype])
    print(f"  {out} ({os.path.getsize(out) / 1e6:.0f} MB)")

# --- Upload ---
print("\n=== Uploading to HuggingFace ===")
repo_id = 'cstr/qari-ocr-crispembed-GGUF'
try:
    create_repo(repo_id, repo_type='model', exist_ok=True)
except Exception as e:
    print(f"Repo: {e}")

api = HfApi()
for f in sorted(glob.glob('/kaggle/working/qari-ocr-2b-*.gguf')):
    fname = os.path.basename(f)
    size_mb = os.path.getsize(f) / 1e6
    print(f"Uploading {fname} ({size_mb:.0f} MB)...")
    api.upload_file(path_or_fileobj=f, path_in_repo=fname,
                    repo_id=repo_id, commit_message=f'Add {fname}')

print("\n=== DONE ===")
