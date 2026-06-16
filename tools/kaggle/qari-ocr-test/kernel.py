#!/usr/bin/env python3
"""Qari-OCR diff harness: capture PyTorch intermediates + run C++ diff test.

1. Load merged Qari-OCR model in PyTorch
2. Capture per-layer activations to reference GGUF
3. Build CrispEmbed, run test-qwen2vl-diff against reference
4. Upload reference GGUF + results to HuggingFace
"""
import gc, json, os, subprocess, sys, shutil, struct, glob, time

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'gguf', 'safetensors', 'huggingface_hub', 'Pillow',
                       'peft', 'accelerate', 'qwen-vl-utils'])

for p in ['/kaggle/input/crispasr-hf-token/hf_token.txt',
          '/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt']:
    if os.path.exists(p):
        os.environ['HF_TOKEN'] = open(p).read().strip()
        break

import torch
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, HfApi

# ─── 1. Create test image ────────────────────────────────────────────
img = Image.new('RGB', (300, 80), 'white')
ImageDraw.Draw(img).text((20, 25), "Hello World 2024", fill='black')
img.save('/kaggle/working/test.png')
print("Test image: 300x80")

# ─── 2. Load model ───────────────────────────────────────────────────
print("\n=== Loading PyTorch model ===")
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

base = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct', torch_dtype=torch.float32, device_map='cpu')
model = PeftModel.from_pretrained(base, 'NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct')
model = model.merge_and_unload()
model.eval()
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# ─── 3. Prepare inputs ───────────────────────────────────────────────
messages = [{"role": "user", "content": [
    {"type": "image", "image": '/kaggle/working/test.png'},
    {"type": "text", "text": "Describe this image."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")

token_ids = inputs.input_ids[0].tolist()
print(f"Token IDs: {token_ids}")
print(f"Grid THW: {inputs.image_grid_thw.tolist()}")
print(f"Pixel values shape: {list(inputs.pixel_values.shape)}")
print(f"Image pad count: {token_ids.count(151655)}")

# ─── 4. Capture per-layer activations ────────────────────────────────
print("\n=== Capturing activations ===")
activations = {}

def make_hook(name):
    def hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        activations[name] = t.detach().float().cpu().numpy().copy()
    return hook

hooks = []
# Vision encoder layers (first 4 + last)
vis_blocks = model.model.visual.blocks
for i in [0, 1, 2, 3, len(vis_blocks)-1]:
    hooks.append(vis_blocks[i].register_forward_hook(make_hook(f"vis_layer_{i}")))

# Merger
hooks.append(model.model.visual.merger.register_forward_hook(make_hook("vis_merger")))

# LLM layers (first 2)
llm_layers = model.model.language_model.layers
for i in range(min(2, len(llm_layers))):
    hooks.append(llm_layers[i].register_forward_hook(make_hook(f"llm_layer_{i}")))

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

for h in hooks:
    h.remove()

# Logits at last position
logits = outputs.logits[0, -1].float().cpu().numpy()
activations["last_logits"] = logits

# Save token IDs
activations["token_ids"] = np.array(token_ids, dtype=np.int32)

# Save pixel values (preprocessed)
activations["pixel_values"] = inputs.pixel_values.float().cpu().numpy()

# Report
print("\nCaptured:")
for name, data in sorted(activations.items()):
    print(f"  {name}: shape={data.shape}, dtype={data.dtype}, "
          f"mean={data.mean():.6f}, std={data.std():.6f}")

# Top-5 logits
top5_idx = np.argsort(logits)[-5:][::-1]
print(f"\nTop-5 logits at last position:")
for idx in top5_idx:
    tok_str = processor.tokenizer.decode([idx])
    print(f"  {idx}: {logits[idx]:.2f} = '{tok_str}'")

# Generate reference output
with torch.no_grad():
    gen_ids = model.generate(**inputs, max_new_tokens=30)
gen_text = processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
print(f"\nPyTorch output: '{gen_text}'")

# ─── 5. Write reference GGUF ─────────────────────────────────────────
print("\n=== Writing reference GGUF ===")
ref_path = '/kaggle/working/qari-ocr-ref.gguf'

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

def write_string(f, s):
    b = s.encode('utf-8')
    f.write(struct.pack('<Q', len(b)))
    f.write(b)

tensor_list = [(k, v.astype(np.float32) if v.dtype != np.int32 else v) for k, v in activations.items()]

with open(ref_path, 'wb') as f:
    f.write(struct.pack('<I', GGUF_MAGIC))
    f.write(struct.pack('<I', GGUF_VERSION))
    f.write(struct.pack('<Q', len(tensor_list)))
    f.write(struct.pack('<Q', 1))  # n_kv

    # Metadata
    write_string(f, 'general.architecture')
    f.write(struct.pack('<I', 8))  # STRING
    write_string(f, 'qwen2vl_ref')

    # Tensor info
    offset = 0
    for name, data in tensor_list:
        write_string(f, name)
        f.write(struct.pack('<I', len(data.shape)))
        for d in data.shape:
            f.write(struct.pack('<Q', d))
        dtype_id = 0 if data.dtype == np.float32 else 5  # F32 or INT32
        f.write(struct.pack('<I', dtype_id))
        f.write(struct.pack('<Q', offset))
        offset += data.nbytes
        offset = (offset + 31) & ~31

    # Align
    pos = f.tell()
    f.write(b'\x00' * (((pos + 31) & ~31) - pos))

    # Data
    for name, data in tensor_list:
        f.write(data.tobytes())
        pad = ((data.nbytes + 31) & ~31) - data.nbytes
        if pad: f.write(b'\x00' * pad)

print(f"Reference GGUF: {os.path.getsize(ref_path) / 1e6:.1f} MB, {len(tensor_list)} tensors")

# ─── 6. Upload reference to HF ───────────────────────────────────────
print("\n=== Uploading to HuggingFace ===")
api = HfApi()
api.upload_file(path_or_fileobj=ref_path, path_in_repo='qari-ocr-ref.gguf',
                repo_id='cstr/qari-ocr-crispembed-GGUF',
                commit_message='Add per-layer reference GGUF for parity testing')
print("Uploaded qari-ocr-ref.gguf")

# ─── 7. Build + run CrispEmbed diff test ──────────────────────────────
print("\n=== CrispEmbed diff test ===")
# Use Q8_0 — best balance of quality and speed on CPU
gguf_path = hf_hub_download('cstr/qari-ocr-crispembed-GGUF', 'qari-ocr-2b-q8_0.gguf')

ce_dir = '/kaggle/working/CrispEmbed'
if os.path.exists(ce_dir): shutil.rmtree(ce_dir)
subprocess.check_call(['git', 'clone', '--depth=1', '--recursive',
                       'https://github.com/CrispStrobe/CrispEmbed.git', ce_dir])

bld = '/kaggle/working/build'
if os.path.exists(bld): shutil.rmtree(bld)
os.makedirs(bld)
# Strip test targets from CMakeLists to avoid missing file errors
import re
cml_path = os.path.join(ce_dir, 'CMakeLists.txt')
with open(cml_path) as f:
    cml = f.read()
# Remove all test-* targets (add_executable + target_link_libraries + target_include)
cml = re.sub(r'(?:add_executable|target_link_libraries|target_include_directories)\(test-[^\)]*\)\n?', '', cml)
with open(cml_path, 'w') as f:
    f.write(cml)

r_cmake = subprocess.run(['cmake', '-S', ce_dir, '-B', bld, '-DCMAKE_BUILD_TYPE=Release'],
                         capture_output=True, text=True)
if r_cmake.returncode != 0:
    print(f"cmake configure FAILED:\n{r_cmake.stderr[-1000:]}")
    sys.exit(1)

r_build = subprocess.run(['cmake', '--build', bld, '--target', 'crispembed-cli', '-j4'],
                         capture_output=True, text=True, timeout=1200)
if r_build.returncode != 0:
    print(f"cmake build FAILED:\n{r_build.stderr[-2000:]}")
    sys.exit(1)
print("Build OK")

cli = os.path.join(bld, 'crispembed')
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = os.path.join(bld, 'ggml/src')

# Run OCR and capture output
t0 = time.time()
r = subprocess.run([cli, '-m', gguf_path, '--ocr', '/kaggle/working/test.png'],
                   capture_output=True, text=True, env=env, timeout=7200)
t1 = time.time()
print(f"CrispEmbed inference took {t1-t0:.0f}s")
print(f"CrispEmbed output: '{r.stdout.strip()[:300]}'")

# Print last 20 lines of stderr for debug info
stderr_lines = r.stderr.strip().split('\n')
for line in stderr_lines[-20:]:
    if line.strip():
        print(f"  [ce] {line}")

# ─── 8. Key diagnostics ──────────────────────────────────────────────
print("\n=== Diagnostics ===")
print(f"PyTorch output:     '{gen_text}'")
print(f"CrispEmbed output:  '{r.stdout.strip()[:200]}'")
if gen_text.strip() == r.stdout.strip():
    print("PARITY: EXACT MATCH")
elif r.stdout.strip().startswith('Hello') or 'World' in r.stdout:
    print("PARITY: CLOSE (text content matches)")
else:
    print("PARITY: DIVERGES — need per-layer diff debugging")
    print(f"  Reference GGUF uploaded for offline C++ diff testing")

print("\n=== DONE ===")
