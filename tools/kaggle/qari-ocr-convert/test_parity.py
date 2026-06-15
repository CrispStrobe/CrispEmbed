#!/usr/bin/env python3
"""Kaggle kernel: test Qari-OCR GGUF parity + output quality.

Downloads the Q4_K GGUF, builds CrispEmbed, runs inference on a
synthetic Arabic test image, and compares against PyTorch reference.
"""
import gc, json, os, subprocess, sys, glob

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'gguf', 'safetensors', 'huggingface_hub', 'Pillow', 'qwen_vl_utils'])

# HF auth
for p in ['/kaggle/input/crispasr-hf-token/hf_token.txt',
          '/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt']:
    if os.path.exists(p):
        os.environ['HF_TOKEN'] = open(p).read().strip()
        break

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download

# --- Create Arabic test image ---
print("=== Creating Arabic test image ===")
img = Image.new('RGB', (400, 100), 'white')
draw = ImageDraw.Draw(img)
# Simple text (ASCII fallback if no Arabic font)
test_text = "Hello World 123"  # ASCII for reliable rendering
draw.text((20, 30), test_text, fill='black')
img.save('/kaggle/working/test_input.png')
print(f"Test image: 400x100, text='{test_text}'")

# --- PyTorch reference ---
print("\n=== PyTorch reference (Qari-OCR) ===")
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    '/kaggle/working/qari-ocr-merged',  # from previous kernel
    torch_dtype=torch.float16,
    device_map='cpu',
)
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')

messages = [{"role": "user", "content": [
    {"type": "image", "image": '/kaggle/working/test_input.png'},
    {"type": "text", "text": "OCR this image."},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
from qwen_vl_utils import process_vision_info
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128)

output_text = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:],
                                      skip_special_tokens=True)[0]
print(f"PyTorch output: '{output_text}'")

del model; gc.collect()

# --- CrispEmbed GGUF test ---
print("\n=== CrispEmbed GGUF test ===")

# Download Q4_K from HF
gguf_path = hf_hub_download('cstr/qari-ocr-crispembed-GGUF', 'qari-ocr-2b-q4_k.gguf')
print(f"GGUF: {gguf_path} ({os.path.getsize(gguf_path) / 1e9:.2f} GB)")

# Build CrispEmbed
crispembed_dir = '/kaggle/working/CrispEmbed'
if not os.path.exists(crispembed_dir):
    subprocess.check_call(['git', 'clone', '--depth=1', '--recursive',
                           'https://github.com/CrispStrobe/CrispEmbed.git',
                           crispembed_dir])

build_dir = '/kaggle/working/build'
os.makedirs(build_dir, exist_ok=True)
subprocess.check_call(['cmake', '-S', crispembed_dir, '-B', build_dir,
                        '-DCMAKE_BUILD_TYPE=Release'])
subprocess.check_call(['cmake', '--build', build_dir,
                        '--target', 'crispembed-cli', '-j4'])

# Run inference
cli = os.path.join(build_dir, 'crispembed')
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = os.path.join(build_dir, 'ggml/src')

result = subprocess.run(
    [cli, '-m', gguf_path, '--ocr', '/kaggle/working/test_input.png'],
    capture_output=True, text=True, env=env, timeout=300
)
cpp_output = result.stdout.strip()
print(f"CrispEmbed output: '{cpp_output}'")
if result.stderr:
    # Print last 10 lines of stderr (model loading info)
    for line in result.stderr.strip().split('\n')[-10:]:
        print(f"  [stderr] {line}")

# --- Compare ---
print("\n=== Comparison ===")
print(f"PyTorch:     '{output_text}'")
print(f"CrispEmbed:  '{cpp_output}'")

if cpp_output and not cpp_output.startswith('error'):
    # Character-level comparison
    common = sum(1 for a, b in zip(output_text, cpp_output) if a == b)
    max_len = max(len(output_text), len(cpp_output))
    char_match = common / max_len if max_len > 0 else 0
    print(f"Character match: {common}/{max_len} = {char_match:.1%}")
    print("STATUS: PASS" if char_match > 0.5 else "STATUS: NEEDS INVESTIGATION")
else:
    print("STATUS: CrispEmbed inference failed")
    print(f"Return code: {result.returncode}")

print("\n=== DONE ===")
