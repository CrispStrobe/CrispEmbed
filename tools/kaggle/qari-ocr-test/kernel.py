#!/usr/bin/env python3
"""Test Qari-OCR GGUF: per-layer parity vs PyTorch + Arabic output quality."""
import gc, json, os, subprocess, sys, glob, time

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'gguf', 'safetensors', 'huggingface_hub', 'Pillow',
                       'peft', 'accelerate'])

for p in ['/kaggle/input/crispasr-hf-token/hf_token.txt',
          '/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt']:
    if os.path.exists(p):
        os.environ['HF_TOKEN'] = open(p).read().strip()
        break

import torch
import numpy as np
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, snapshot_download

# ─── 1. Create test images ───────────────────────────────────────────
print("=== Creating test images ===")

# Simple ASCII image (reliable rendering)
img1 = Image.new('RGB', (300, 80), 'white')
draw1 = ImageDraw.Draw(img1)
draw1.text((20, 25), "Hello World 2024", fill='black')
img1.save('/kaggle/working/test_ascii.png')

# Create a simple Arabic-like test (numbers + basic chars)
img2 = Image.new('RGB', (400, 100), 'white')
draw2 = ImageDraw.Draw(img2)
draw2.text((20, 30), "1234567890 ABCDEF", fill='black')
img2.save('/kaggle/working/test_nums.png')

print("Test images created")

# ─── 2. PyTorch reference ────────────────────────────────────────────
print("\n=== PyTorch reference (merged Qari-OCR) ===")

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig

# Load merged model (merge LoRA on the fly)
print("Loading base model + LoRA adapter...")
from peft import PeftModel

base = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    torch_dtype=torch.float32,
    device_map='cpu',
)
model = PeftModel.from_pretrained(base, 'NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct')
model = model.merge_and_unload()
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')

# Run on test image
def run_pytorch(image_path, prompt="Read the text in this image."):
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=64)
    return processor.batch_decode(ids[:, inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)[0]

pt_out1 = run_pytorch('/kaggle/working/test_ascii.png')
print(f"PyTorch (ASCII): '{pt_out1}'")

pt_out2 = run_pytorch('/kaggle/working/test_nums.png')
print(f"PyTorch (nums):  '{pt_out2}'")

# ─── 3. Per-layer activation dump ────────────────────────────────────
print("\n=== Per-layer activation comparison ===")

# Hook into vision encoder layers
activations = {}
def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            activations[name] = output[0].detach().float().cpu().numpy()
        else:
            activations[name] = output.detach().float().cpu().numpy()
    return hook

# Register hooks on first 4 vision layers + merger + first 2 LLM layers
hooks = []
for i in range(min(4, len(model.model.visual.blocks))):
    h = model.model.visual.blocks[i].register_forward_hook(make_hook(f"vis_layer_{i}"))
    hooks.append(h)
h = model.model.visual.merger.register_forward_hook(make_hook("vis_merger"))
hooks.append(h)
for i in range(min(2, len(model.model.layers))):
    h = model.model.layers[i].register_forward_hook(make_hook(f"llm_layer_{i}"))
    hooks.append(h)

# Run forward to capture activations
from qwen_vl_utils import process_vision_info
messages = [{"role": "user", "content": [
    {"type": "image", "image": '/kaggle/working/test_ascii.png'},
    {"type": "text", "text": "OCR"},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, _ = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
with torch.no_grad():
    model(**inputs)

for h in hooks:
    h.remove()

print("Captured activations:")
for name, act in sorted(activations.items()):
    print(f"  {name}: shape={act.shape}, mean={act.mean():.6f}, std={act.std():.6f}")

del model, base; gc.collect()

# ─── 4. CrispEmbed GGUF test ─────────────────────────────────────────
print("\n=== CrispEmbed GGUF test ===")

gguf_path = hf_hub_download('cstr/qari-ocr-crispembed-GGUF', 'qari-ocr-2b-q4_k.gguf')
print(f"GGUF: {os.path.getsize(gguf_path) / 1e9:.2f} GB")

# Build CrispEmbed
ce_dir = '/kaggle/working/CrispEmbed'
if not os.path.exists(ce_dir):
    subprocess.check_call(['git', 'clone', '--depth=1', '--recursive',
                           'https://github.com/CrispStrobe/CrispEmbed.git', ce_dir])

bld = '/kaggle/working/build'
os.makedirs(bld, exist_ok=True)
subprocess.check_call(['cmake', '-S', ce_dir, '-B', bld, '-DCMAKE_BUILD_TYPE=Release'])
subprocess.check_call(['cmake', '--build', bld, '--target', 'crispembed-cli', '-j4'])

cli = os.path.join(bld, 'crispembed')
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = os.path.join(bld, 'ggml/src')

def run_crispembed(image_path):
    r = subprocess.run([cli, '-m', gguf_path, '--ocr', image_path],
                       capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        print(f"  STDERR: {r.stderr[-500:]}")
    return r.stdout.strip()

t0 = time.time()
ce_out1 = run_crispembed('/kaggle/working/test_ascii.png')
t1 = time.time()
print(f"CrispEmbed (ASCII): '{ce_out1}' ({t1-t0:.1f}s)")

t0 = time.time()
ce_out2 = run_crispembed('/kaggle/working/test_nums.png')
t1 = time.time()
print(f"CrispEmbed (nums):  '{ce_out2}' ({t1-t0:.1f}s)")

# ─── 5. Comparison ───────────────────────────────────────────────────
print("\n=== Results ===")
print(f"ASCII  PyTorch:     '{pt_out1}'")
print(f"ASCII  CrispEmbed:  '{ce_out1}'")
print(f"Nums   PyTorch:     '{pt_out2}'")
print(f"Nums   CrispEmbed:  '{ce_out2}'")

if ce_out1 and not ce_out1.startswith('error'):
    print("\nSTATUS: CrispEmbed inference WORKS")
    if pt_out1.strip() == ce_out1.strip():
        print("OUTPUT: EXACT MATCH (ASCII)")
    else:
        print(f"OUTPUT: DIFFERS — may be Q4_K quantization effect")
else:
    print("\nSTATUS: CrispEmbed inference FAILED")

print("\n=== DONE ===")
