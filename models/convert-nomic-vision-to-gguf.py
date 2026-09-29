#!/usr/bin/env python3
"""Convert nomic-ai/nomic-embed-vision-v1.5 (Apache-2.0) to a vit_embed GGUF.

Architecture (modeling_hf_nomic_bert.py, NomicVisionModel):
  patch Linear(3*16*16 -> 768) over (c, p1, p2)  == conv2d kernel [768, 3, 16, 16]
  + CLS token, + learned pos_embed [197, 768]
  12 pre-LN blocks: fused Wqkv (q | k | v rows), 2-D rotary on patch tokens
    (not CLS; interleaved pairs; dims 0-31 row index, 32-63 column index,
    freq_i = 10000^(-i/16), ref grid 14x14), SDPA, out_proj;
    SwiGLU MLP: fc11(x) * silu(fc12(x)) -> LayerNorm(2048, eps 1e-5) -> fc2
  no final LayerNorm
  selector (NomicMultiHeadAttentionPooling): latent query attends over all
    197 tokens (Wq, Wkv = k | v, out_proj, no rope); then
    hidden[:, 0] + GatedMLP(LN(attn_out))   (no inner LN in this MLP)
  embedding = L2-normalised token 0 of that.

Tensor names follow convert-siglip-to-gguf.py (vit.* keys, enc.N.*); the
Nomic-only pieces are enc.N.attn.qkv (pre-fused), enc.N.ffn.fc11/fc12/fc2/norm
and nomic_pool.*.

    python models/convert-nomic-vision-to-gguf.py --model-dir <snapshot> --output nomic-embed-vision-v1.5-f32.gguf
"""
import argparse
import json
from pathlib import Path

import gguf
import numpy as np
from safetensors.numpy import load_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fp16", action="store_true", help="store 2-D weights as F16 (norms, biases, pos stay F32)")
    args = ap.parse_args()

    md = Path(args.model_dir)
    cfg = json.loads((md / "config.json").read_text())
    pp = json.loads((md / "preprocessor_config.json").read_text())
    sd = load_file(str(md / "model.safetensors"))

    D, L, NH = cfg["n_embd"], cfg["n_layer"], cfg["n_head"]
    inter = int(cfg["n_inner"])
    inter = (inter + 255) // 256 * 256  # NomciBertGatedMLP rounds hidden_features to a multiple of 256
    img, ps = cfg["img_size"], cfg["patch_size"]
    grid = img // ps
    ref = cfg.get("ref_feat_shape") or [grid, grid]
    assert cfg["activation_function"] == "swiglu" and cfg["prenorm"] and cfg.get("no_last_ln", False)
    assert cfg.get("use_rotary_pos_emb") and cfg.get("rotary_emb_fraction", 0) == 0
    assert sd["layers.0.mlp.fc11.weight"].shape == (inter, D)

    w = gguf.GGUFWriter(args.output, arch="vit")
    w.add_string("general.name", "nomic-embed-vision-v1.5")
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.source", "https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5")
    w.add_uint32("vit.hidden_size", D)
    w.add_uint32("vit.num_hidden_layers", L)
    w.add_uint32("vit.num_attention_heads", NH)
    w.add_uint32("vit.intermediate_size", inter)
    w.add_uint32("vit.image_size", img)
    w.add_uint32("vit.patch_size", ps)
    w.add_uint32("vit.num_patches", grid * grid)
    w.add_uint32("vit.num_channels", cfg.get("num_channels", 3))
    w.add_string("vit.model_type", "nomic")
    w.add_float32("vit.layer_norm_eps", cfg["layer_norm_epsilon"])
    w.add_float32("vit.mlp_ln_eps", 1e-5)  # nn.LayerNorm(hidden_features) default
    w.add_string("vit.hidden_act", "swiglu")
    w.add_bool("vit.has_cls_token", True)
    w.add_bool("vit.rope_2d", True)
    w.add_float32("vit.rope_theta", float(cfg.get("rotary_emb_base", 10000)))
    w.add_uint32("vit.rope_ref_h", int(ref[0]))
    w.add_uint32("vit.rope_ref_w", int(ref[1]))
    w.add_bool("vit.has_nomic_pool", True)
    w.add_array("vit.image_mean", [float(x) for x in pp["image_mean"]])
    w.add_array("vit.image_std", [float(x) for x in pp["image_std"]])

    def put(name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if args.fp16 and arr.ndim >= 2 and not name.startswith(("position_embd", "cls_token", "nomic_pool.latent", "patch_embed")):
            arr = arr.astype(np.float16)
        w.add_tensor(name, arr)

    # patch Linear [D, 3*ps*ps] over (c, p1, p2) == torch conv weight [D, 3, ps, ps]
    put("patch_embed.weight", sd["embeddings.proj.weight"].reshape(D, 3, ps, ps))
    put("patch_embed.bias", sd["embeddings.proj.bias"])
    put("cls_token", sd["embeddings.cls_token"].reshape(1, D))
    put("position_embd.weight", sd["embeddings.pos_embed"].reshape(-1, D))

    for i in range(L):
        p = f"layers.{i}."
        qkv_w, qkv_b = sd[p + "attn.Wqkv.weight"], sd[p + "attn.Wqkv.bias"]  # "(three h d)": q | k | v rows
        put(f"enc.{i}.ln1.weight", sd[p + "norm1.weight"])
        put(f"enc.{i}.ln1.bias", sd[p + "norm1.bias"])
        # already fused (q | k | v rows) - vit_embed uses enc.N.attn.qkv.* as-is
        put(f"enc.{i}.attn.qkv.weight", qkv_w)
        put(f"enc.{i}.attn.qkv.bias", qkv_b)
        put(f"enc.{i}.attn.o.weight", sd[p + "attn.out_proj.weight"])
        put(f"enc.{i}.attn.o.bias", sd[p + "attn.out_proj.bias"])
        put(f"enc.{i}.ln2.weight", sd[p + "norm2.weight"])
        put(f"enc.{i}.ln2.bias", sd[p + "norm2.bias"])
        for n in ("fc11", "fc12", "fc2"):
            put(f"enc.{i}.ffn.{n}.weight", sd[p + f"mlp.{n}.weight"])
            put(f"enc.{i}.ffn.{n}.bias", sd[p + f"mlp.{n}.bias"])
        put(f"enc.{i}.ffn.norm.weight", sd[p + "mlp.norm.weight"])
        put(f"enc.{i}.ffn.norm.bias", sd[p + "mlp.norm.bias"])

    s = "selector."
    put("nomic_pool.latent", sd[s + "attn.latent"].reshape(1, D))
    put("nomic_pool.q.weight", sd[s + "attn.Wq.weight"])
    put("nomic_pool.q.bias", sd[s + "attn.Wq.bias"])
    put("nomic_pool.kv.weight", sd[s + "attn.Wkv.weight"])  # k | v rows
    put("nomic_pool.kv.bias", sd[s + "attn.Wkv.bias"])
    put("nomic_pool.o.weight", sd[s + "attn.out_proj.weight"])
    put("nomic_pool.o.bias", sd[s + "attn.out_proj.bias"])
    put("nomic_pool.ln.weight", sd[s + "norm1.weight"])
    put("nomic_pool.ln.bias", sd[s + "norm1.bias"])
    for n in ("fc11", "fc12", "fc2"):
        put(f"nomic_pool.mlp.{n}.weight", sd[s + f"mlp.{n}.weight"])
        put(f"nomic_pool.mlp.{n}.bias", sd[s + f"mlp.{n}.bias"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
