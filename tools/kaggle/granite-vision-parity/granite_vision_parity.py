#!/usr/bin/env python3
"""Granite Vision 3.3-2B — reference dump + GGUF conversion for CrispEmbed parity.

This Kaggle kernel:
1. Downloads the model from HuggingFace
2. Runs a forward pass to capture per-stage reference activations
3. Writes a reference GGUF for crispembed-diff parity testing
4. Optionally converts to GGUF if not already done

Runs on Kaggle P100/T4 with 13-16 GB RAM (model is ~5GB F16).
"""

import gc
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(WORK)

# ── Bootstrap harness ───────────────────────────────────────────────────
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
    class kh:
        @staticmethod
        def log(msg): print(msg, flush=True)
        @staticmethod
        def build_heartbeat():
            import contextlib
            return contextlib.nullcontext()

import numpy as np
import torch
import torch.nn.functional as F

kh.log("=== Granite Vision 3.3-2B Parity Test ===")

# ── Download model ──────────────────────────────────────────────────────
kh.log("Downloading model...")
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    "ibm-granite/granite-vision-3.3-2b",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.tiktoken"],
    cache_dir=str(WORK / "hf_cache")
)
kh.log(f"Model at: {model_dir}")

# ── Load config ─────────────────────────────────────────────────────────
with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)
vc = cfg["vision_config"]
tc = cfg["text_config"]
kh.log(f"Vision: dim={vc['hidden_size']}, layers={vc['num_hidden_layers']}")
kh.log(f"LLM: dim={tc['hidden_size']}, layers={tc['num_hidden_layers']}")

# ── Load model with transformers ────────────────────────────────────────
kh.log("Loading model...")
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

# Use float16 to save memory
model = LlavaNextForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="cpu"
)
model.eval()
kh.log(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

processor = LlavaNextProcessor.from_pretrained(model_dir)

# ── Generate reference activations ──────────────────────────────────────
kh.log("Generating reference activations...")

# Create a deterministic test image
torch.manual_seed(42)
test_image = torch.randint(0, 256, (384, 384, 3), dtype=torch.uint8)

# Save test image for reproducibility
from PIL import Image
pil_image = Image.fromarray(test_image.numpy())
pil_image.save(str(WORK / "test_image.png"))

# Process with the model's processor
conversation = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What text is in this image?"}
    ]}
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=pil_image, text=prompt, return_tensors="pt")

# Move to CPU float16
for k in inputs:
    if isinstance(inputs[k], torch.Tensor):
        inputs[k] = inputs[k].to(dtype=torch.float16 if inputs[k].is_floating_point() else inputs[k].dtype)

# Capture intermediates via hooks
intermediates = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        intermediates[name] = output.detach().float().cpu().numpy()
    return hook

hooks = []
# Vision encoder layers
for i, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
    hooks.append(layer.register_forward_hook(make_hook(f"vis_layer_{i}")))

# Vision post-layernorm
hooks.append(model.vision_tower.vision_model.post_layernorm.register_forward_hook(
    make_hook("vis_post_ln")))

# Projector
hooks.append(model.multi_modal_projector.register_forward_hook(
    make_hook("projector")))

# LLM layers (sample a few)
for i in [0, 1, 10, 20, 39]:
    if i < len(model.language_model.model.layers):
        hooks.append(model.language_model.model.layers[i].register_forward_hook(
            make_hook(f"llm_layer_{i}")))

# LLM final norm
hooks.append(model.language_model.model.norm.register_forward_hook(
    make_hook("llm_norm")))

kh.log("Running forward pass...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

# Remove hooks
for h in hooks:
    h.remove()

# Decode output
generated_text = processor.decode(outputs[0], skip_special_tokens=True)
kh.log(f"Generated text: {generated_text}")

# ── Write reference GGUF ───────────────────────────────────────────────
kh.log("Writing reference GGUF...")

ref_tensors = {}
for name, data in intermediates.items():
    if data.ndim > 2:
        # Flatten batch dim
        data = data[0] if data.shape[0] == 1 else data
    ref_tensors[name] = data.astype(np.float32)
    kh.log(f"  {name}: shape={list(data.shape)}, mean={data.mean():.6f}")

# Also save the generated token IDs
gen_ids = outputs[0].cpu().numpy().astype(np.float32)
ref_tensors["generated_ids"] = gen_ids

# Write GGUF
def write_ref_gguf(path, tensors):
    MAGIC = 0x46554747; VERSION = 3; TYPE_STRING = 8; TYPE_F32 = 0
    def ws(f, s):
        b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)
    tensor_list = list(tensors.items())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC)); f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(tensor_list))); f.write(struct.pack("<Q", 1))
        ws(f, "general.architecture"); f.write(struct.pack("<I", TYPE_STRING)); ws(f, "granite_vision_ref")
        offset = 0
        for name, data in tensor_list:
            ws(f, name); f.write(struct.pack("<I", len(data.shape)))
            for d in data.shape: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", TYPE_F32)); f.write(struct.pack("<Q", offset))
            offset += data.nbytes; offset = (offset + 31) & ~31
        pos = f.tell(); aligned = (pos + 31) & ~31; f.write(b"\x00" * (aligned - pos))
        for name, data in tensor_list:
            f.write(data.astype(np.float32).tobytes())
            pad = ((data.nbytes + 31) & ~31) - data.nbytes
            if pad > 0: f.write(b"\x00" * pad)
    kh.log(f"Written {path}: {len(tensor_list)} tensors, {os.path.getsize(path)/1024/1024:.1f} MB")

write_ref_gguf(str(WORK / "granite-vision-ref.gguf"), ref_tensors)

# ── Summary ─────────────────────────────────────────────────────────────
kh.log(f"\n=== Summary ===")
kh.log(f"Reference GGUF: granite-vision-ref.gguf ({len(ref_tensors)} tensors)")
kh.log(f"Generated text: {generated_text[:200]}")
kh.log(f"Feature layers captured: {[k for k in ref_tensors if k.startswith('vis_')]}")
kh.log(f"LLM layers captured: {[k for k in ref_tensors if k.startswith('llm_')]}")
kh.log(f"\nDownload granite-vision-ref.gguf from kernel output for crispembed-diff.")

# Write progress file for monitoring
with open(WORK / "progress.txt", "w") as f:
    f.write(f"Status: DONE\n")
    f.write(f"Generated: {generated_text[:200]}\n")
    f.write(f"Tensors: {len(ref_tensors)}\n")
    f.write(f"Feature layers: {list(ref_tensors.keys())}\n")
