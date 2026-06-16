#!/usr/bin/env python3
"""Granite Vision 3.3-2B — reference dump for crispembed-diff parity."""

import gc, json, math, os, struct, subprocess, sys, time, traceback
from pathlib import Path

WORK = Path("/kaggle/working")
os.chdir(WORK)

def log(msg):
    print(msg, flush=True)
    with open(WORK / "progress.txt", "a") as f:
        f.write(msg + "\n")

log("=== Granite Vision 3.3-2B Parity v5 ===")
log(f"Python {sys.version}")

try:
    # HF token
    hf_token = None
    for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
              "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
        if os.path.exists(p):
            hf_token = open(p).read().strip()
            log(f"HF token from {p}")
            break

    try:
        from safetensors import safe_open
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "safetensors", "--quiet"])
        from safetensors import safe_open

    import numpy as np
    import torch
    import torch.nn.functional as F

    log("Downloading model files...")
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download("ibm-granite/granite-vision-3.3-2b", "config.json",
                                   cache_dir=str(WORK / "hf_cache"), token=hf_token)
    log("Config downloaded")

    # Download shards one at a time
    shard1 = hf_hub_download("ibm-granite/granite-vision-3.3-2b",
                              "model-00001-of-00002.safetensors",
                              cache_dir=str(WORK / "hf_cache"), token=hf_token)
    log(f"Shard 1: {os.path.getsize(shard1)/1e9:.1f} GB")

    shard2 = hf_hub_download("ibm-granite/granite-vision-3.3-2b",
                              "model-00002-of-00002.safetensors",
                              cache_dir=str(WORK / "hf_cache"), token=hf_token)
    log(f"Shard 2: {os.path.getsize(shard2)/1e9:.1f} GB")

    with open(config_path) as f:
        cfg = json.load(f)
    vc, tc = cfg["vision_config"], cfg["text_config"]
    log(f"Vision: dim={vc['hidden_size']}, layers={vc['num_hidden_layers']}")
    log(f"LLM: dim={tc['hidden_size']}, layers={tc['num_hidden_layers']}")
    log(f"Feature layers: {cfg['vision_feature_layer']}")

    st_files = [shard1, shard2]

    def load_tensor(name):
        for sp in st_files:
            with safe_open(sp, framework="pt") as sf:
                if name in sf.keys():
                    return sf.get_tensor(name).float()
        raise KeyError(f"Not found: {name}")

    # ── SigLIP Vision Encoder ───────────────────────────────────────
    log("Running SigLIP vision encoder...")
    dim = vc["hidden_size"]   # 1152
    n_layers = vc["num_hidden_layers"]  # 27
    n_heads = vc["num_attention_heads"]  # 16
    img_size = vc["image_size"]  # 384
    ps = vc["patch_size"]  # 14
    feat_layers = [l + n_layers if l < 0 else l for l in cfg["vision_feature_layer"]]

    torch.manual_seed(42)
    image = torch.rand(1, 3, img_size, img_size)

    # Patch embed
    pe_w = load_tensor("vision_tower.vision_model.embeddings.patch_embedding.weight")
    pe_b = load_tensor("vision_tower.vision_model.embeddings.patch_embedding.bias")
    x = F.conv2d(image, pe_w, pe_b, stride=ps)
    x = x.flatten(2).transpose(1, 2)
    del pe_w, pe_b; gc.collect()

    n_patches = x.shape[1]
    log(f"Patches: {n_patches}")

    pos_w = load_tensor("vision_tower.vision_model.embeddings.position_embedding.weight")
    x = x + pos_w[:n_patches].unsqueeze(0)
    del pos_w; gc.collect()

    intermediates = {"input": image[0].numpy().copy()}
    intermediates["vis_patch_embed"] = x[0].detach().numpy().copy()

    layer_outputs = {}
    for li in range(n_layers):
        prefix = f"vision_tower.vision_model.encoder.layers.{li}"
        if li % 5 == 0: log(f"  Vision layer {li}/{n_layers}")

        ln1_w = load_tensor(f"{prefix}.layer_norm1.weight")
        ln1_b = load_tensor(f"{prefix}.layer_norm1.bias")
        normed = F.layer_norm(x, (dim,), ln1_w, ln1_b)
        del ln1_w, ln1_b

        d_head = dim // n_heads
        Q = F.linear(normed, load_tensor(f"{prefix}.self_attn.q_proj.weight"),
                     load_tensor(f"{prefix}.self_attn.q_proj.bias"))
        K = F.linear(normed, load_tensor(f"{prefix}.self_attn.k_proj.weight"),
                     load_tensor(f"{prefix}.self_attn.k_proj.bias"))
        V = F.linear(normed, load_tensor(f"{prefix}.self_attn.v_proj.weight"),
                     load_tensor(f"{prefix}.self_attn.v_proj.bias"))
        del normed

        Q = Q.reshape(1, -1, n_heads, d_head).transpose(1, 2)
        K = K.reshape(1, -1, n_heads, d_head).transpose(1, 2)
        V = V.reshape(1, -1, n_heads, d_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(Q, K, V)
        attn = attn.transpose(1, 2).reshape(1, -1, dim)
        del Q, K, V

        o_w = load_tensor(f"{prefix}.self_attn.out_proj.weight")
        o_b = load_tensor(f"{prefix}.self_attn.out_proj.bias")
        x = x + F.linear(attn, o_w, o_b)
        del attn, o_w, o_b

        ln2_w = load_tensor(f"{prefix}.layer_norm2.weight")
        ln2_b = load_tensor(f"{prefix}.layer_norm2.bias")
        normed2 = F.layer_norm(x, (dim,), ln2_w, ln2_b)
        del ln2_w, ln2_b

        fc1_w = load_tensor(f"{prefix}.mlp.fc1.weight")
        fc1_b = load_tensor(f"{prefix}.mlp.fc1.bias")
        fc2_w = load_tensor(f"{prefix}.mlp.fc2.weight")
        fc2_b = load_tensor(f"{prefix}.mlp.fc2.bias")
        h = F.gelu(F.linear(normed2, fc1_w, fc1_b), approximate="tanh")
        h = F.linear(h, fc2_w, fc2_b)
        del fc1_w, fc1_b, fc2_w, fc2_b, normed2

        x = x + h
        del h; gc.collect()

        if li in feat_layers:
            layer_outputs[li] = x[0].detach().numpy().copy()
            intermediates[f"vis_layer_{li}"] = layer_outputs[li]

    log("Vision encoder done")

    # Concat multi-layer features
    feat_concat = np.concatenate([layer_outputs[li] for li in sorted(feat_layers)], axis=-1)
    intermediates["vis_features_concat"] = feat_concat
    log(f"Feature concat: {feat_concat.shape}")

    # ── Projector ───────────────────────────────────────────────────
    log("Running projector...")
    proj_1_w = load_tensor("multi_modal_projector.linear_1.weight")
    proj_1_b = load_tensor("multi_modal_projector.linear_1.bias")
    proj_2_w = load_tensor("multi_modal_projector.linear_2.weight")
    proj_2_b = load_tensor("multi_modal_projector.linear_2.bias")
    feat_t = torch.from_numpy(feat_concat).unsqueeze(0)
    proj = F.gelu(F.linear(feat_t, proj_1_w, proj_1_b))
    proj = F.linear(proj, proj_2_w, proj_2_b)
    intermediates["projector"] = proj[0].detach().numpy().copy()
    log(f"Projector: {proj.shape}")
    del proj_1_w, proj_1_b, proj_2_w, proj_2_b, feat_t, proj; gc.collect()

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
    log(f"Ref GGUF: {os.path.getsize(WORK / 'granite-vision-ref.gguf') / 1024 / 1024:.1f} MB")

    if hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=str(WORK / "granite-vision-ref.gguf"),
                path_in_repo="granite-vision-ref.gguf",
                repo_id="cstr/granite-vision-crispembed-GGUF")
            log("Uploaded to HF")
        except Exception as e:
            log(f"HF upload failed: {e}")

    log("=== DONE ===")

except Exception as e:
    log(f"FATAL ERROR: {e}")
    log(traceback.format_exc())
