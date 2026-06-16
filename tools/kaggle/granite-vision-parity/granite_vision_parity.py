#!/usr/bin/env python3
"""Granite Vision 3.3-2B — minimal reference dump for CrispEmbed parity.

Strategy: load weights via safetensors (NOT full transformers model) to
stay within 13GB RAM on Kaggle P100. Run vision encoder + projector +
first few LLM layers in pure PyTorch, capture intermediates.
"""

import gc, json, math, os, struct, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(WORK)

# Bootstrap harness
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

class _kh_stub:
    @staticmethod
    def log(msg): print(msg, flush=True)
    @staticmethod
    def resolve_hf_token(): return None

kh = _kh_stub()
try:
    import kaggle_harness as _kh_real
    _kh_real.init_progress()
    kh = _kh_real
except Exception as e:
    print(f"Harness init failed ({e}), using stub", flush=True)

# Ensure safetensors is available
try:
    from safetensors import safe_open
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "--quiet"])
    from safetensors import safe_open

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

kh.log("=== Granite Vision 3.3-2B Parity (minimal) ===")

# Download model (using HF token from harness dataset for gated models)
kh.log("Downloading model...")
hf_token = None
try:
    hf_token = kh.resolve_hf_token()
except Exception:
    # Manual fallback: read from dataset file
    for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
              "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
        if os.path.exists(p):
            hf_token = open(p).read().strip()
            break
kh.log(f"HF token: {'available' if hf_token else 'not found'}")

from huggingface_hub import snapshot_download
model_dir = snapshot_download(
    "ibm-granite/granite-vision-3.3-2b",
    allow_patterns=["*.safetensors", "config.json", "model.safetensors.index.json"],
    cache_dir=str(WORK / "hf_cache"),
    token=hf_token)
kh.log(f"Model at: {model_dir}")

with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)
vc, tc = cfg["vision_config"], cfg["text_config"]
kh.log(f"Vision: dim={vc['hidden_size']}, layers={vc['num_hidden_layers']}")

# Find safetensors files
st_files = sorted(os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors"))
kh.log(f"Safetensors: {len(st_files)} files")

# Helper to load a tensor by name across shards
def load_tensor(name):
    for sp in st_files:
        with safe_open(sp, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name).float()
    raise KeyError(f"Tensor not found: {name}")

# ── Run SigLIP Vision Encoder ───────────────────────────────────────
kh.log("Running vision encoder...")
dim = vc["hidden_size"]  # 1152
n_layers = vc["num_hidden_layers"]  # 27
n_heads = vc["num_attention_heads"]  # 16
img_size = vc["image_size"]  # 384
ps = vc["patch_size"]  # 14
n_patches = (img_size // ps) ** 2  # 729
feat_layers = cfg["vision_feature_layer"]  # [-24, -20, -12, -1]
abs_feat_layers = [l + n_layers if l < 0 else l for l in feat_layers]

# Deterministic input
torch.manual_seed(42)
image = torch.rand(1, 3, img_size, img_size)

# Patch embed
pe_w = load_tensor("vision_tower.vision_model.embeddings.patch_embedding.weight")
pe_b = load_tensor("vision_tower.vision_model.embeddings.patch_embedding.bias")
x = F.conv2d(image, pe_w, pe_b, stride=ps)  # [1, dim, ph, pw]
x = x.flatten(2).transpose(1, 2)  # [1, n_patches, dim]

# Position embedding
pos_embed = load_tensor("vision_tower.vision_model.embeddings.position_embedding.weight")
x = x + pos_embed[:n_patches].unsqueeze(0)

intermediates = {"input": image[0].numpy().copy()}
intermediates["vis_patch_embed"] = x[0].detach().numpy().copy()

# Transformer layers
layer_outputs = {}
for li in range(n_layers):
    prefix = f"vision_tower.vision_model.encoder.layers.{li}"
    kh.log(f"  Vision layer {li}/{n_layers}")

    # LN1
    ln1_w = load_tensor(f"{prefix}.layer_norm1.weight")
    ln1_b = load_tensor(f"{prefix}.layer_norm1.bias")
    normed = F.layer_norm(x, (dim,), ln1_w, ln1_b)

    # MHSA
    d_head = dim // n_heads
    q_w = load_tensor(f"{prefix}.self_attn.q_proj.weight")
    q_b = load_tensor(f"{prefix}.self_attn.q_proj.bias")
    k_w = load_tensor(f"{prefix}.self_attn.k_proj.weight")
    k_b = load_tensor(f"{prefix}.self_attn.k_proj.bias")
    v_w = load_tensor(f"{prefix}.self_attn.v_proj.weight")
    v_b = load_tensor(f"{prefix}.self_attn.v_proj.bias")
    o_w = load_tensor(f"{prefix}.self_attn.out_proj.weight")
    o_b = load_tensor(f"{prefix}.self_attn.out_proj.bias")

    Q = F.linear(normed, q_w, q_b).reshape(1, -1, n_heads, d_head).transpose(1, 2)
    K = F.linear(normed, k_w, k_b).reshape(1, -1, n_heads, d_head).transpose(1, 2)
    V = F.linear(normed, v_w, v_b).reshape(1, -1, n_heads, d_head).transpose(1, 2)

    attn = F.scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(1, 2).reshape(1, -1, dim)
    attn = F.linear(attn, o_w, o_b)
    x = x + attn

    # LN2 + FFN
    ln2_w = load_tensor(f"{prefix}.layer_norm2.weight")
    ln2_b = load_tensor(f"{prefix}.layer_norm2.bias")
    normed2 = F.layer_norm(x, (dim,), ln2_w, ln2_b)

    fc1_w = load_tensor(f"{prefix}.mlp.fc1.weight")
    fc1_b = load_tensor(f"{prefix}.mlp.fc1.bias")
    fc2_w = load_tensor(f"{prefix}.mlp.fc2.weight")
    fc2_b = load_tensor(f"{prefix}.mlp.fc2.bias")

    h = F.gelu(F.linear(normed2, fc1_w, fc1_b), approximate="tanh")
    h = F.linear(h, fc2_w, fc2_b)
    x = x + h

    # Capture feature layer
    if li in abs_feat_layers:
        layer_outputs[li] = x[0].detach().numpy().copy()
        intermediates[f"vis_layer_{li}"] = layer_outputs[li]

    # Free weights to save memory
    del q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, fc1_w, fc1_b, fc2_w, fc2_b
    del ln1_w, ln1_b, ln2_w, ln2_b
    gc.collect()

kh.log("Vision encoder done")

# Concatenate multi-layer features
feat_concat = np.concatenate([layer_outputs[li] for li in sorted(abs_feat_layers)], axis=-1)
intermediates["vis_features_concat"] = feat_concat
kh.log(f"Concat features: {feat_concat.shape}")

# ── Projector ───────────────────────────────────────────────────────
kh.log("Running projector...")
proj_1_w = load_tensor("multi_modal_projector.linear_1.weight")
proj_1_b = load_tensor("multi_modal_projector.linear_1.bias")
proj_2_w = load_tensor("multi_modal_projector.linear_2.weight")
proj_2_b = load_tensor("multi_modal_projector.linear_2.bias")

feat_t = torch.from_numpy(feat_concat).unsqueeze(0)
proj = F.gelu(F.linear(feat_t, proj_1_w, proj_1_b))
proj = F.linear(proj, proj_2_w, proj_2_b)
intermediates["projector"] = proj[0].detach().numpy().copy()
kh.log(f"Projector output: {proj.shape}")

del proj_1_w, proj_1_b, proj_2_w, proj_2_b, feat_t
gc.collect()

# ── Write reference GGUF ───────────────────────────────────────────
kh.log("Writing reference GGUF...")

ref_tensors = {}
for name, data in intermediates.items():
    ref_tensors[name] = data.astype(np.float32)
    kh.log(f"  {name}: shape={list(data.shape)}, mean={data.mean():.6f}")

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

write_ref_gguf(str(WORK / "granite-vision-ref.gguf"), ref_tensors)
kh.log(f"Reference GGUF: {os.path.getsize(WORK / 'granite-vision-ref.gguf') / 1024 / 1024:.1f} MB")

# ── Summary ─────────────────────────────────────────────────────────
with open(WORK / "progress.txt", "w") as f:
    f.write(f"Status: DONE\n")
    f.write(f"Tensors: {len(ref_tensors)}\n")
    for name, data in ref_tensors.items():
        f.write(f"  {name}: {list(data.shape)}\n")

# Upload reference GGUF to HuggingFace
if hf_token:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(WORK / "granite-vision-ref.gguf"),
            path_in_repo="granite-vision-ref.gguf",
            repo_id="cstr/granite-vision-crispembed-GGUF")
        kh.log("Reference GGUF uploaded to HuggingFace")
    except Exception as e:
        kh.log(f"HF upload failed: {e}")

kh.log("=== DONE ===")
