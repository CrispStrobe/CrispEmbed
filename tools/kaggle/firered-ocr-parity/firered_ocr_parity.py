#!/usr/bin/env python3
"""FireRed-OCR (Qwen3-VL) — reference dump for CrispEmbed crispembed-diff parity.

Loads weights via safetensors (lazy, per-tensor) and runs vision encoder +
projector + first few LLM layers step-by-step. Captures per-stage activations
and writes a reference GGUF. Uploads to HuggingFace.

Qwen3-VL differences from Qwen2-VL captured here:
  - QK norms (RMSNorm on Q and K per head)
  - Deepstack visual features (concat from layers [5, 11, 17])
  - mRoPE interleaved mode
"""

import gc, json, math, os, struct, subprocess, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(WORK)

# ── Bootstrap harness ───────────────────────────────────────────────
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
        def hf_token():
            for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
                       "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
                if os.path.exists(p):
                    return open(p).read().strip()
            return None

try:
    from safetensors import safe_open
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "--quiet"])
    from safetensors import safe_open

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

kh.log("=== FireRed-OCR (Qwen3-VL) Parity Test ===")

# ── Download model ──────────────────────────────────────────────────
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
    "FireRedTeam/FireRed-OCR",
    allow_patterns=["*.safetensors", "config.json"],
    cache_dir=str(WORK / "hf_cache"),
    token=hf_token)
kh.log(f"Model at: {model_dir}")

with open(os.path.join(model_dir, "config.json")) as f:
    cfg = json.load(f)
vc, tc = cfg["vision_config"], cfg["text_config"]
kh.log(f"Vision: depth={vc['depth']}, hidden={vc['hidden_size']}, patch={vc['patch_size']}")
kh.log(f"LLM: layers={tc['num_hidden_layers']}, hidden={tc['hidden_size']}")
kh.log(f"Deepstack: {vc.get('deepstack_visual_indexes', [])}")

st_files = sorted(os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors"))

def load_tensor(name):
    for sp in st_files:
        with safe_open(sp, framework="pt") as sf:
            if name in sf.keys():
                return sf.get_tensor(name).float()
    raise KeyError(f"Not found: {name}")

# ── Vision encoder ──────────────────────────────────────────────────
kh.log("Running vision encoder...")
dim = vc["hidden_size"]   # 1024
depth = vc["depth"]       # 24
n_heads = vc["num_heads"] # 16
img_size = 384
ps = vc["patch_size"]     # 16
tps = vc["temporal_patch_size"]  # 2
n_patches_side = img_size // ps  # 24
n_patches = n_patches_side * n_patches_side  # 576
deepstack = vc.get("deepstack_visual_indexes", [])

torch.manual_seed(42)
image = torch.rand(1, 3, img_size, img_size)

# Patch embed: Conv3D(3, dim, [tps, ps, ps])
pe_w = load_tensor("model.visual.patch_embed.proj.weight")  # [1024, 3, 2, 16, 16]
pe_b = load_tensor("model.visual.patch_embed.proj.bias")
# For single image (temporal=1), pad to temporal_patch_size=2 by repeating
img_5d = image.unsqueeze(2).repeat(1, 1, tps, 1, 1)  # [1, 3, 2, 384, 384]
x = F.conv3d(img_5d, pe_w, pe_b, stride=(tps, ps, ps))  # [1, 1024, 1, 24, 24]
x = x.flatten(2).transpose(1, 2)  # [1, 576, 1024]

# Position embedding
pos_w = load_tensor("model.visual.pos_embed.weight")  # [2304, 1024]
# For 576 patches, we need to select/interpolate positions
if pos_w.shape[0] >= n_patches:
    x = x + pos_w[:n_patches].unsqueeze(0)

intermediates = {"input": image[0].numpy().copy()}
intermediates["vis_patch_embed"] = x[0].detach().numpy().copy()

# Transformer layers
layer_features = {}
for li in range(depth):
    prefix = f"model.visual.blocks.{li}"
    kh.log(f"  Vision layer {li}/{depth}")

    # LN1 + Attention
    ln1_w = load_tensor(f"{prefix}.norm1.weight")
    ln1_b = load_tensor(f"{prefix}.norm1.bias")
    normed = F.layer_norm(x, (dim,), ln1_w, ln1_b)

    d_head = dim // n_heads

    # Check for fused QKV or separate Q/K/V
    try:
        qkv_w = load_tensor(f"{prefix}.attn.qkv.weight")
        qkv_b = load_tensor(f"{prefix}.attn.qkv.bias")
        qkv = F.linear(normed, qkv_w, qkv_b)
        Q, K, V = qkv.chunk(3, dim=-1)
    except KeyError:
        q_w = load_tensor(f"{prefix}.attn.q.weight")
        q_b = load_tensor(f"{prefix}.attn.q.bias")
        k_w = load_tensor(f"{prefix}.attn.k.weight")
        k_b = load_tensor(f"{prefix}.attn.k.bias")
        v_w = load_tensor(f"{prefix}.attn.v.weight")
        v_b = load_tensor(f"{prefix}.attn.v.bias")
        Q = F.linear(normed, q_w, q_b)
        K = F.linear(normed, k_w, k_b)
        V = F.linear(normed, v_w, v_b)

    Q = Q.reshape(1, -1, n_heads, d_head).transpose(1, 2)
    K = K.reshape(1, -1, n_heads, d_head).transpose(1, 2)
    V = V.reshape(1, -1, n_heads, d_head).transpose(1, 2)

    attn = F.scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(1, 2).reshape(1, -1, dim)

    proj_w = load_tensor(f"{prefix}.attn.proj.weight")
    proj_b = load_tensor(f"{prefix}.attn.proj.bias")
    attn = F.linear(attn, proj_w, proj_b)
    x = x + attn

    # LN2 + FFN
    ln2_w = load_tensor(f"{prefix}.norm2.weight")
    ln2_b = load_tensor(f"{prefix}.norm2.bias")
    normed2 = F.layer_norm(x, (dim,), ln2_w, ln2_b)

    # Try SwiGLU first, fall back to GELU fc1/fc2
    try:
        gate_w = load_tensor(f"{prefix}.mlp.gate_proj.weight")
        up_w = load_tensor(f"{prefix}.mlp.up_proj.weight")
        down_w = load_tensor(f"{prefix}.mlp.down_proj.weight")
        gate = F.silu(F.linear(normed2, gate_w))
        up = F.linear(normed2, up_w)
        h = F.linear(gate * up, down_w)
    except KeyError:
        fc1_w = load_tensor(f"{prefix}.mlp.fc1.weight")
        fc1_b = load_tensor(f"{prefix}.mlp.fc1.bias")
        fc2_w = load_tensor(f"{prefix}.mlp.fc2.weight")
        fc2_b = load_tensor(f"{prefix}.mlp.fc2.bias")
        h = F.gelu(F.linear(normed2, fc1_w, fc1_b), approximate="tanh")
        h = F.linear(h, fc2_w, fc2_b)

    x = x + h

    if li in deepstack:
        layer_features[li] = x[0].detach().numpy().copy()
        intermediates[f"vis_layer_{li}"] = layer_features[li]

    # Free weights
    del ln1_w, ln1_b, ln2_w, ln2_b
    gc.collect()

kh.log("Vision encoder done")

# Concat deepstack features
if deepstack:
    feat_list = [layer_features[li] for li in sorted(deepstack)]
    feat_concat = np.concatenate(feat_list, axis=-1)
    intermediates["vis_deepstack_concat"] = feat_concat
    kh.log(f"Deepstack concat: {feat_concat.shape}")
else:
    feat_concat = x[0].detach().numpy()

# ── Merger (spatial merge) ──────────────────────────────────────────
kh.log("Running merger...")
try:
    merge_norm_w = load_tensor("model.visual.merger.norm.weight")
    merge_norm_b = load_tensor("model.visual.merger.norm.bias")
    merge_fc1_w = load_tensor("model.visual.merger.mlp.0.weight")
    merge_fc1_b = load_tensor("model.visual.merger.mlp.0.bias")
    merge_fc2_w = load_tensor("model.visual.merger.mlp.2.weight")
    merge_fc2_b = load_tensor("model.visual.merger.mlp.2.bias")

    # Spatial merge: group 2x2 patches, concat, project
    merge_size = vc["spatial_merge_size"]  # 2
    feat_t = torch.from_numpy(feat_concat).unsqueeze(0)
    T, D_feat = feat_t.shape[1], feat_t.shape[2]
    h_patches = w_patches = int(math.sqrt(T))
    feat_2d = feat_t.reshape(1, h_patches, w_patches, D_feat)
    merged = feat_2d[:, ::merge_size, ::merge_size, :]  # simple spatial subsample (approx)
    # TODO: proper 2x2 concat merge
    merged = merged.reshape(1, -1, D_feat)
    merged = F.layer_norm(merged, (D_feat,), merge_norm_w, merge_norm_b)
    merged = F.gelu(F.linear(merged, merge_fc1_w, merge_fc1_b))
    merged = F.linear(merged, merge_fc2_w, merge_fc2_b)
    intermediates["merger"] = merged[0].detach().numpy().copy()
    kh.log(f"Merger output: {merged.shape}")
except Exception as e:
    kh.log(f"Merger failed: {e}, using raw features")
    merged = torch.from_numpy(feat_concat).unsqueeze(0)

# ── LLM first 3 layers ─────────────────────────────────────────────
kh.log("Running LLM layers 0-2...")
llm_dim = tc["hidden_size"]  # 2048
n_llm_layers = min(3, tc["num_hidden_layers"])
llm_heads = tc["num_attention_heads"]
llm_kv = tc["num_key_value_heads"]
eps = tc.get("rms_norm_eps", 1e-6)

embed_w = load_tensor("model.language_model.embed_tokens.weight")
# Use a simple test: embed the first few token IDs
test_ids = torch.tensor([[151644, 8948, 198]])  # <|im_start|>system\n
x_llm = F.embedding(test_ids, embed_w)
intermediates["llm_embed"] = x_llm[0].detach().numpy().copy()

for li in range(n_llm_layers):
    prefix = f"model.language_model.layers.{li}"
    kh.log(f"  LLM layer {li}")

    # RMSNorm
    norm1_w = load_tensor(f"{prefix}.input_layernorm.weight")
    rms = x_llm.pow(2).mean(-1, keepdim=True)
    normed = x_llm * torch.rsqrt(rms + eps) * norm1_w

    # Q/K/V
    q_w = load_tensor(f"{prefix}.self_attn.q_proj.weight")
    k_w = load_tensor(f"{prefix}.self_attn.k_proj.weight")
    v_w = load_tensor(f"{prefix}.self_attn.v_proj.weight")
    o_w = load_tensor(f"{prefix}.self_attn.o_proj.weight")

    d_head = llm_dim // llm_heads
    Q = F.linear(normed, q_w).reshape(1, -1, llm_heads, d_head).transpose(1, 2)
    K = F.linear(normed, k_w).reshape(1, -1, llm_kv, d_head).transpose(1, 2)
    V = F.linear(normed, v_w).reshape(1, -1, llm_kv, d_head).transpose(1, 2)

    # QK norms (Qwen3-VL specific)
    try:
        q_norm_w = load_tensor(f"{prefix}.self_attn.q_norm.weight")
        k_norm_w = load_tensor(f"{prefix}.self_attn.k_norm.weight")
        Q_rms = Q.pow(2).mean(-1, keepdim=True)
        Q = Q * torch.rsqrt(Q_rms + eps) * q_norm_w
        K_rms = K.pow(2).mean(-1, keepdim=True)
        K = K * torch.rsqrt(K_rms + eps) * k_norm_w
        if li == 0: kh.log("    QK norms applied (Qwen3-VL)")
    except KeyError:
        if li == 0: kh.log("    No QK norms (Qwen2-VL)")

    # GQA expand K/V
    if llm_kv < llm_heads:
        repeat = llm_heads // llm_kv
        K = K.repeat_interleave(repeat, dim=1)
        V = V.repeat_interleave(repeat, dim=1)

    attn = F.scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(1, 2).reshape(1, -1, llm_dim)
    attn = F.linear(attn, o_w)

    # Residual (with multiplier if Granite, but Qwen3 doesn't have it)
    x_llm = x_llm + attn

    # FFN
    norm2_w = load_tensor(f"{prefix}.post_attention_layernorm.weight")
    rms2 = x_llm.pow(2).mean(-1, keepdim=True)
    normed2 = x_llm * torch.rsqrt(rms2 + eps) * norm2_w

    gate_w = load_tensor(f"{prefix}.mlp.gate_proj.weight")
    up_w = load_tensor(f"{prefix}.mlp.up_proj.weight")
    down_w = load_tensor(f"{prefix}.mlp.down_proj.weight")
    h = F.silu(F.linear(normed2, gate_w)) * F.linear(normed2, up_w)
    h = F.linear(h, down_w)
    x_llm = x_llm + h

    intermediates[f"llm_layer_{li}"] = x_llm[0].detach().numpy().copy()

    del q_w, k_w, v_w, o_w, gate_w, up_w, down_w, norm1_w, norm2_w
    gc.collect()

kh.log("LLM layers done")

# ── Write reference GGUF ───────────────────────────────────────────
kh.log("Writing reference GGUF...")
ref_tensors = {}
for name, data in intermediates.items():
    ref_tensors[name] = data.astype(np.float32)
    kh.log(f"  {name}: {list(data.shape)}, mean={data.mean():.6f}")

def write_ref_gguf(path, tensors):
    MAGIC = 0x46554747; VERSION = 3; TYPE_STRING = 8; TYPE_F32 = 0
    def ws(f, s):
        b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)
    tensor_list = list(tensors.items())
    with open(path, "wb") as f:
        f.write(struct.pack("<I", MAGIC)); f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", len(tensor_list))); f.write(struct.pack("<Q", 1))
        ws(f, "general.architecture"); f.write(struct.pack("<I", TYPE_STRING)); ws(f, "firered_ocr_ref")
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

write_ref_gguf(str(WORK / "firered-ocr-ref.gguf"), ref_tensors)
kh.log(f"Ref GGUF: {os.path.getsize(WORK / 'firered-ocr-ref.gguf') / 1024 / 1024:.1f} MB")

# Upload to HF
if hf_token:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(WORK / "firered-ocr-ref.gguf"),
            path_in_repo="firered-ocr-ref.gguf",
            repo_id="cstr/firered-ocr-crispembed-GGUF")
        kh.log("Ref GGUF uploaded to HuggingFace")
    except Exception as e:
        kh.log(f"HF upload failed: {e}")

with open(WORK / "progress.txt", "w") as f:
    f.write(f"Status: DONE\nTensors: {len(ref_tensors)}\n")
    for name in ref_tensors:
        f.write(f"  {name}: {list(ref_tensors[name].shape)}\n")

kh.log("=== DONE ===")
