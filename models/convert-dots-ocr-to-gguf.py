#!/usr/bin/env python3
"""Convert dots.ocr (rednote-hilab) to GGUF for CrispEmbed.

dots.ocr = Qwen2 LLM (standard RoPE, attention_bias) + custom vision (2D RoPE, PatchMerger).
NOT Qwen2-VL (no mRoPE). MIT license.

Usage:
    python convert-dots-ocr-to-gguf.py --model rednote-hilab/dots.ocr --output dots-ocr-f16.gguf --fp16
"""

import argparse, gc, json, os, struct, sys
import numpy as np
from safetensors import safe_open

GGUF_MAGIC = 0x46554747; GGUF_VERSION = 3
GGML_TYPE_F32 = 0; GGML_TYPE_F16 = 1
GGUF_TYPE_UINT32 = 4; GGUF_TYPE_FLOAT32 = 6; GGUF_TYPE_STRING = 8

def ws(f,s): b=s.encode("utf-8"); f.write(struct.pack("<Q",len(b))); f.write(b)
def wks(f,k,v): ws(f,k); f.write(struct.pack("<I",GGUF_TYPE_STRING)); ws(f,v)
def wku(f,k,v): ws(f,k); f.write(struct.pack("<I",GGUF_TYPE_UINT32)); f.write(struct.pack("<I",v))
def wkf(f,k,v): ws(f,k); f.write(struct.pack("<I",GGUF_TYPE_FLOAT32)); f.write(struct.pack("<f",v))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    model_dir = args.model
    if not os.path.isdir(model_dir):
        from huggingface_hub import hf_hub_download
        # Download config + safetensors index
        for f in ["config.json", "model.safetensors.index.json"]:
            hf_hub_download(args.model, f, local_dir=f"/tmp/{args.model.split('/')[-1]}")
        with open(f"/tmp/{args.model.split('/')[-1]}/model.safetensors.index.json") as f:
            idx = json.load(f)
        shards = sorted(set(idx["weight_map"].values()))
        for s in shards:
            hf_hub_download(args.model, s, local_dir=f"/tmp/{args.model.split('/')[-1]}")
        model_dir = f"/tmp/{args.model.split('/')[-1]}"

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    vc = cfg.get("vision_config", {})

    print(f"Vision: depth={vc.get('num_hidden_layers',42)}, hidden={vc.get('hidden_size',1536)}")
    print(f"LLM: layers={cfg['num_hidden_layers']}, hidden={cfg['hidden_size']}, heads={cfg['num_attention_heads']}/{cfg['num_key_value_heads']}")

    st_files = sorted(os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors"))

    # Name mapping
    def map_name(key):
        # Vision
        if key.startswith("vision_tower."):
            s = key[len("vision_tower."):]
            if s.startswith("patch_embed.patchifier.proj."): return "v.patch_embed." + s.split(".")[-1]
            if s.startswith("patch_embed.patchifier.norm."): return "v.patch_norm." + s.split(".")[-1]
            if s == "post_trunk_norm.weight": return "v.post_norm.weight"
            if s.startswith("blocks."):
                p = s.split("."); li = int(p[1]); r = ".".join(p[2:])
                r = r.replace("attn.qkv.", "attn.qkv.")
                r = r.replace("attn.proj.", "attn.proj.")
                r = r.replace("ffn.w1.", "ffn.gate.")
                r = r.replace("ffn.w2.", "ffn.down.")
                r = r.replace("ffn.w3.", "ffn.up.")
                return f"v.blk.{li}.{r}"
            if s.startswith("merger."): return f"v.merger.{s[len('merger.'):]}"
            return f"v.{s}"
        # LLM
        if key == "model.embed_tokens.weight": return "l.embed.weight"
        if key == "model.norm.weight": return "l.norm.weight"
        if key == "lm_head.weight": return "l.lm_head.weight"
        if key.startswith("model.layers."):
            p = key.split("."); li = int(p[2]); r = ".".join(p[3:])
            r = r.replace("self_attn.q_proj", "attn.q")
            r = r.replace("self_attn.k_proj", "attn.k")
            r = r.replace("self_attn.v_proj", "attn.v")
            r = r.replace("self_attn.o_proj", "attn.o")
            r = r.replace("mlp.gate_proj", "ffn.gate")
            r = r.replace("mlp.up_proj", "ffn.up")
            r = r.replace("mlp.down_proj", "ffn.down")
            r = r.replace("input_layernorm", "norm1")
            r = r.replace("post_attention_layernorm", "norm2")
            return f"l.blk.{li}.{r}"
        return None

    tmap = {}
    for sp in st_files:
        with safe_open(sp, framework="pt") as sf:
            for k in sf.keys():
                n = map_name(k)
                if n: tmap[n] = (k, sp)
    print(f"Mapped: {len(tmap)} tensors")

    # Get shapes
    tinfo = {}
    for gn, (sk, sp) in tmap.items():
        with safe_open(sp, framework="pt") as sf:
            t = sf.get_tensor(sk)
            shape = list(t.shape)
            dt = GGML_TYPE_F16 if (args.fp16 and len(shape) >= 2) else GGML_TYPE_F32
            nb = int(np.prod(shape)) * (2 if dt == GGML_TYPE_F16 else 4)
            tinfo[gn] = (shape, nb, dt)
            del t

    # Write GGUF
    n_kv = 16
    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC)); f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tinfo))); f.write(struct.pack("<Q", n_kv))

        wks(f, "general.architecture", "dots_ocr")
        wks(f, "general.name", "dots.ocr-3B")
        # Vision
        wku(f, "dots_ocr.vision.depth", vc.get("num_hidden_layers", 42))
        wku(f, "dots_ocr.vision.hidden_size", vc.get("hidden_size", 1536))
        wku(f, "dots_ocr.vision.num_heads", vc.get("num_attention_heads", 12))
        wku(f, "dots_ocr.vision.patch_size", vc.get("patch_size", 14))
        wku(f, "dots_ocr.vision.spatial_merge_size", vc.get("spatial_merge_size", 2))
        wku(f, "dots_ocr.vision.temporal_patch_size", vc.get("temporal_patch_size", 1))
        # LLM
        wku(f, "dots_ocr.hidden_size", cfg["hidden_size"])
        wku(f, "dots_ocr.num_hidden_layers", cfg["num_hidden_layers"])
        wku(f, "dots_ocr.num_attention_heads", cfg["num_attention_heads"])
        wku(f, "dots_ocr.num_key_value_heads", cfg["num_key_value_heads"])
        wku(f, "dots_ocr.intermediate_size", cfg["intermediate_size"])
        wku(f, "dots_ocr.vocab_size", cfg["vocab_size"])
        wkf(f, "dots_ocr.rope_theta", cfg.get("rope_theta", 1000000))
        wku(f, "dots_ocr.image_token_id", cfg.get("image_token_id", 151665))

        offset = 0
        order = list(tinfo.keys())
        for n in order:
            shape, nb, dt = tinfo[n]
            ws(f, n); f.write(struct.pack("<I", len(shape)))
            for d in shape: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dt)); f.write(struct.pack("<Q", offset))
            offset += nb; offset = (offset + 31) & ~31

        pos = f.tell(); al = (pos + 31) & ~31; f.write(b"\x00" * (al - pos))

        for i, n in enumerate(order):
            shape, nb, dt = tinfo[n]
            sk, sp = tmap[n]
            with safe_open(sp, framework="pt") as sf:
                t = sf.get_tensor(sk)
            if dt == GGML_TYPE_F16:
                d = t.half().numpy().tobytes()
            else:
                d = t.float().numpy().tobytes()
            del t; gc.collect()
            f.write(d)
            pad = ((len(d) + 31) & ~31) - len(d)
            if pad > 0: f.write(b"\x00" * pad)
            if (i + 1) % 100 == 0: print(f"  {i+1}/{len(order)}")

    print(f"Written: {args.output} ({os.path.getsize(args.output)/1024/1024:.0f} MB)")


if __name__ == "__main__":
    main()
