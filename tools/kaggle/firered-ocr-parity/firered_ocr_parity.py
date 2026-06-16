#!/usr/bin/env python3
"""FireRed-OCR (Qwen3-VL) — reference dump for crispembed-diff parity."""

import gc, json, math, os, struct, subprocess, sys, time, traceback
from pathlib import Path

WORK = Path("/kaggle/working")
try:
    os.makedirs(WORK, exist_ok=True)
    os.chdir(WORK)
except Exception:
    WORK = Path(".")

# Write progress immediately — use str path for maximum compat
PROGRESS = str(WORK / "progress.txt")
with open(PROGRESS, "w") as _f:
    _f.write("STARTING\n")

def log(msg):
    print(msg, flush=True)
    try:
        with open(PROGRESS, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass

log("=== FireRed-OCR Parity v4 ===")
log(f"Python {sys.version}")

# Write progress immediately so we can see output even on crash
try:
    # HF token from dataset
    hf_token = None
    for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
              "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
        if os.path.exists(p):
            hf_token = open(p).read().strip()
            log(f"HF token from {p}")
            break
    if not hf_token:
        log("No HF token found (model is public, continuing)")

    # Install safetensors if needed
    try:
        from safetensors import safe_open
    except ImportError:
        log("Installing safetensors...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "--quiet"])
        from safetensors import safe_open

    import numpy as np
    import torch
    import torch.nn.functional as F

    # Download only what we need (config + single safetensors)
    log("Downloading model files...")
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download("FireRedTeam/FireRed-OCR", "config.json",
                                   cache_dir="/tmp/hf_cache", token=hf_token)
    log("Config downloaded")

    model_path = hf_hub_download("FireRedTeam/FireRed-OCR", "model.safetensors",
                                  cache_dir="/tmp/hf_cache", token=hf_token)
    log(f"Model downloaded: {os.path.getsize(model_path) / 1e9:.1f} GB")

    with open(config_path) as f:
        cfg = json.load(f)
    vc, tc = cfg["vision_config"], cfg["text_config"]
    log(f"Vision: depth={vc['depth']}, hidden={vc['hidden_size']}")
    log(f"LLM: layers={tc['num_hidden_layers']}, hidden={tc['hidden_size']}")

    def load_tensor(name):
        with safe_open(model_path, framework="pt") as sf:
            t = sf.get_tensor(name)
            return t.float()  # F32 for computation accuracy

    # List available tensor names for debugging
    with safe_open(model_path, framework="pt") as sf:
        all_keys = sorted(sf.keys())
        vis_keys = [k for k in all_keys if k.startswith("model.visual.blocks.0.")]
        log(f"Total tensors: {len(all_keys)}")
        log(f"Vision block 0 keys: {vis_keys}")

    # ── Vision encoder ──────────────────────────────────────────────
    log("Running vision encoder...")
    dim = vc["hidden_size"]
    depth = vc["depth"]
    n_heads = vc["num_heads"]
    ps = vc["patch_size"]
    tps = vc["temporal_patch_size"]
    img_size = 384
    deepstack = vc.get("deepstack_visual_indexes", [])

    torch.manual_seed(42)
    image = torch.rand(1, 3, img_size, img_size)

    # Patch embed
    pe_w = load_tensor("model.visual.patch_embed.proj.weight")
    pe_b = load_tensor("model.visual.patch_embed.proj.bias")
    img_5d = image.unsqueeze(2).repeat(1, 1, tps, 1, 1)
    x = F.conv3d(img_5d, pe_w, pe_b, stride=(tps, ps, ps))
    x = x.flatten(2).transpose(1, 2)
    del pe_w, pe_b, img_5d; gc.collect()

    n_patches = x.shape[1]
    log(f"Patches: {n_patches}, dim: {dim}")

    # Position embedding
    pos_w = load_tensor("model.visual.pos_embed.weight")
    if pos_w.shape[0] >= n_patches:
        x = x + pos_w[:n_patches].unsqueeze(0)
    del pos_w; gc.collect()

    intermediates = {"input": image[0].numpy().copy()}
    intermediates["vis_patch_embed"] = x[0].detach().numpy().copy()

    layer_features = {}
    for li in range(depth):
        prefix = f"model.visual.blocks.{li}"
        if li % 4 == 0:
            import psutil
            mem = psutil.virtual_memory()
            log(f"  Vision layer {li}/{depth} (RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB)")

        try:
            ln1_w = load_tensor(f"{prefix}.norm1.weight")
            ln1_b = load_tensor(f"{prefix}.norm1.bias")
        except KeyError as e:
            log(f"  Layer {li}: missing norm1 ({e}), trying layer_norm1...")
            ln1_w = load_tensor(f"{prefix}.layer_norm1.weight")
            ln1_b = load_tensor(f"{prefix}.layer_norm1.bias")
        normed = F.layer_norm(x, (dim,), ln1_w, ln1_b)
        del ln1_w, ln1_b

        # Attention (try fused QKV first)
        try:
            qkv_w = load_tensor(f"{prefix}.attn.qkv.weight")
            qkv_b = load_tensor(f"{prefix}.attn.qkv.bias")
            qkv = F.linear(normed, qkv_w, qkv_b)
            Q, K, V = qkv.chunk(3, dim=-1)
            del qkv_w, qkv_b, qkv
        except KeyError:
            Q = F.linear(normed, load_tensor(f"{prefix}.attn.q.weight"),
                         load_tensor(f"{prefix}.attn.q.bias"))
            K = F.linear(normed, load_tensor(f"{prefix}.attn.k.weight"),
                         load_tensor(f"{prefix}.attn.k.bias"))
            V = F.linear(normed, load_tensor(f"{prefix}.attn.v.weight"),
                         load_tensor(f"{prefix}.attn.v.bias"))
        del normed

        d_head = dim // n_heads
        Q = Q.reshape(1, -1, n_heads, d_head).transpose(1, 2)
        K = K.reshape(1, -1, n_heads, d_head).transpose(1, 2)
        V = V.reshape(1, -1, n_heads, d_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V)
        attn = attn.transpose(1, 2).reshape(1, -1, dim)
        del Q, K, V

        proj_w = load_tensor(f"{prefix}.attn.proj.weight")
        proj_b = load_tensor(f"{prefix}.attn.proj.bias")
        x = x + F.linear(attn, proj_w, proj_b)
        del attn, proj_w, proj_b

        # FFN
        try:
            ln2_w = load_tensor(f"{prefix}.norm2.weight")
            ln2_b = load_tensor(f"{prefix}.norm2.bias")
        except KeyError:
            ln2_w = load_tensor(f"{prefix}.layer_norm2.weight")
            ln2_b = load_tensor(f"{prefix}.layer_norm2.bias")
        normed2 = F.layer_norm(x, (dim,), ln2_w, ln2_b)
        del ln2_w, ln2_b

        try:
            # Qwen3-VL uses mlp.linear_fc1/linear_fc2, older uses mlp.fc1/fc2
            try:
                fc1_w = load_tensor(f"{prefix}.mlp.fc1.weight")
                fc1_b = load_tensor(f"{prefix}.mlp.fc1.bias")
            except KeyError:
                fc1_w = load_tensor(f"{prefix}.mlp.linear_fc1.weight")
                fc1_b = load_tensor(f"{prefix}.mlp.linear_fc1.bias")
            try:
                fc2_w = load_tensor(f"{prefix}.mlp.fc2.weight")
                fc2_b = load_tensor(f"{prefix}.mlp.fc2.bias")
            except KeyError:
                fc2_w = load_tensor(f"{prefix}.mlp.linear_fc2.weight")
                fc2_b = load_tensor(f"{prefix}.mlp.linear_fc2.bias")
            h = F.gelu(F.linear(normed2, fc1_w, fc1_b), approximate="tanh")
            h = F.linear(h, fc2_w, fc2_b)
            del fc1_w, fc1_b, fc2_w, fc2_b
        except KeyError:
            gate_w = load_tensor(f"{prefix}.mlp.gate_proj.weight")
            up_w = load_tensor(f"{prefix}.mlp.up_proj.weight")
            down_w = load_tensor(f"{prefix}.mlp.down_proj.weight")
            h = F.silu(F.linear(normed2, gate_w)) * F.linear(normed2, up_w)
            h = F.linear(h, down_w)
            del gate_w, up_w, down_w
        del normed2

        x = x + h
        del h; gc.collect()

        if li in deepstack:
            layer_features[li] = x[0].detach().numpy().copy()
            intermediates[f"vis_layer_{li}"] = layer_features[li]

    log("Vision encoder done")

    # Deepstack concat
    if deepstack and layer_features:
        feat_concat = np.concatenate([layer_features[li] for li in sorted(deepstack)], axis=-1)
        intermediates["vis_deepstack"] = feat_concat
        log(f"Deepstack: {feat_concat.shape}")

    # ── Write reference GGUF ───────────────────────────────────────
    log("Writing reference GGUF...")
    ref_tensors = {}
    for name, data in intermediates.items():
        ref_tensors[name] = data.astype(np.float32)
        log(f"  {name}: {list(data.shape)}, mean={data.mean():.6f}")

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
    log(f"Ref GGUF: {os.path.getsize(WORK / 'firered-ocr-ref.gguf') / 1024 / 1024:.1f} MB")

    # Upload
    if hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(WORK / "firered-ocr-ref.gguf"),
                path_in_repo="firered-ocr-ref.gguf",
                repo_id="cstr/firered-ocr-crispembed-GGUF")
            log("Uploaded to HF")
        except Exception as e:
            log(f"HF upload failed: {e}")

    log("=== DONE ===")

except Exception as e:
    log(f"FATAL ERROR: {e}")
    log(traceback.format_exc())
