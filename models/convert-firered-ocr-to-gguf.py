#!/usr/bin/env python3
"""Convert FireRed-OCR (Qwen3-VL) to GGUF for CrispEmbed.

Usage:
    python convert-firered-ocr-to-gguf.py \
        --model FireRedTeam/FireRed-OCR \
        --output firered-ocr-f16.gguf --fp16

FireRed-OCR is a Qwen3-VL fine-tune. Key differences from Qwen2-VL:
  - QK norms (RMSNorm on Q and K per head)
  - Deepstack visual features (multi-layer concat from [5, 11, 17])
  - Vision: depth=24, hidden=1024, patch=16
  - mRoPE sections [24, 20, 20], interleaved=true
  - LLM: 28 layers, GQA 16/8, rope_theta=5M

Uses lazy safetensors loading for 8GB VPS compatibility.
"""

import argparse, gc, json, os, struct, sys
import numpy as np
from safetensors import safe_open

GGUF_MAGIC = 0x46554747; GGUF_VERSION = 3
GGML_TYPE_F32 = 0; GGML_TYPE_F16 = 1
GGUF_TYPE_UINT32 = 4; GGUF_TYPE_INT32 = 5; GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8; GGUF_TYPE_ARRAY = 9

def ws(f,s): b=s.encode("utf-8"); f.write(struct.pack("<Q",len(b))); f.write(b)
def wks(f,k,v): ws(f,k); f.write(struct.pack("<I",GGUF_TYPE_STRING)); ws(f,v)
def wku(f,k,v): ws(f,k); f.write(struct.pack("<I",GGUF_TYPE_UINT32)); f.write(struct.pack("<I",v))
def wkf(f,k,v): ws(f,k); f.write(struct.pack("<I",GGUF_TYPE_FLOAT32)); f.write(struct.pack("<f",v))
def wka(f,k,a): ws(f,k); f.write(struct.pack("<I",GGUF_TYPE_ARRAY)); f.write(struct.pack("<I",GGUF_TYPE_INT32)); f.write(struct.pack("<Q",len(a)));[f.write(struct.pack("<i",v)) for v in a]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    model_dir = args.model
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(args.model,
                                      allow_patterns=["*.safetensors", "config.json"],
                                      local_dir=f"/tmp/{args.model.split('/')[-1]}")

    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    vc, tc = cfg["vision_config"], cfg["text_config"]

    print(f"Vision: depth={vc['depth']}, hidden={vc['hidden_size']}, patch={vc['patch_size']}")
    print(f"LLM: layers={tc['num_hidden_layers']}, hidden={tc['hidden_size']}, heads={tc['num_attention_heads']}/{tc['num_key_value_heads']}")
    print(f"Deepstack: {vc.get('deepstack_visual_indexes', [])}")
    print(f"mRoPE: sections={tc['rope_scaling']['mrope_section']}, interleaved={tc['rope_scaling'].get('mrope_interleaved', False)}")

    st_files = sorted(os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors"))

    # Name mapping: Qwen3-VL HF → GGUF
    def map_name(key):
        if key.startswith("model.visual."):
            s = key[len("model.visual."):]
            # Patch embed
            if s == "patch_embed.proj.weight": return "v.patch_embed.weight"
            if s == "patch_embed.proj.bias": return "v.patch_embed.bias"
            if s == "pos_embed.weight": return "v.pos_embed"
            # Encoder layers
            if s.startswith("blocks."):
                p = s.split("."); li = int(p[1]); r = ".".join(p[2:])
                r = r.replace("attn.qkv.", "attn.qkv.")
                r = r.replace("attn.proj.", "attn.proj.")
                r = r.replace("mlp.linear_fc1.", "ffn.up.")
                r = r.replace("mlp.linear_fc2.", "ffn.down.")
                r = r.replace("mlp.fc1.", "ffn.up.")
                r = r.replace("mlp.fc2.", "ffn.down.")
                return f"v.blk.{li}.{r}"
            # Merger (spatial merge)
            if s.startswith("merger."): return f"v.{s}"
            return f"v.{s}"
        elif key.startswith("model.language_model."):
            s = key[len("model.language_model."):]
            if s == "embed_tokens.weight": return "llm.embed.weight"
            if s == "norm.weight": return "llm.norm.weight"
            if s.startswith("layers."):
                p = s.split("."); li = int(p[1]); r = ".".join(p[2:])
                r = r.replace("self_attn.q_proj", "attn.q")
                r = r.replace("self_attn.k_proj", "attn.k")
                r = r.replace("self_attn.v_proj", "attn.v")
                r = r.replace("self_attn.o_proj", "attn.o")
                r = r.replace("self_attn.q_norm", "attn.q_norm")
                r = r.replace("self_attn.k_norm", "attn.k_norm")
                r = r.replace("mlp.gate_proj", "ffn.gate")
                r = r.replace("mlp.up_proj", "ffn.up")
                r = r.replace("mlp.down_proj", "ffn.down")
                r = r.replace("input_layernorm", "norm1")
                r = r.replace("post_attention_layernorm", "norm2")
                return f"llm.blk.{li}.{r}"
            if s.startswith("lm_head."): return f"llm.lm_head.{s[len('lm_head.'):]}"
        return None

    # Collect metadata
    tmap = {}
    for sp in st_files:
        with safe_open(sp, framework="pt") as sf:
            for k in sf.keys():
                n = map_name(k)
                if n: tmap[n] = (k, sp)
    print(f"Mapped: {len(tmap)} tensors")

    # Get shapes (lazy). Flatten 5D tensors to 4D (GGUF max is 4D).
    tinfo = {}
    for gn, (sk, sp) in tmap.items():
        with safe_open(sp, framework="pt") as sf:
            t = sf.get_tensor(sk)
            shape = list(t.shape)
            if len(shape) == 5:
                shape = [shape[0], shape[1] * shape[2], shape[3], shape[4]]
            dt = GGML_TYPE_F16 if (args.fp16 and len(shape) >= 2) else GGML_TYPE_F32
            nb = int(np.prod(shape)) * (2 if dt == GGML_TYPE_F16 else 4)
            tinfo[gn] = (shape, nb, dt)
            del t

    # mRoPE config
    rope_sections = tc["rope_scaling"]["mrope_section"]
    mrope_interleaved = tc["rope_scaling"].get("mrope_interleaved", False)
    deepstack = vc.get("deepstack_visual_indexes", [])

    # Write GGUF
    n_kv = 22
    with open(args.output, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC)); f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tinfo))); f.write(struct.pack("<Q", n_kv))

        wks(f, "general.architecture", "qwen3vl")
        wks(f, "general.name", "FireRed-OCR-Qwen3VL-2B")
        # Vision
        wku(f, "qwen3vl.vision.depth", vc["depth"])
        wku(f, "qwen3vl.vision.hidden_size", vc["hidden_size"])
        wku(f, "qwen3vl.vision.num_heads", vc["num_heads"])
        wku(f, "qwen3vl.vision.patch_size", vc["patch_size"])
        wku(f, "qwen3vl.vision.spatial_merge_size", vc["spatial_merge_size"])
        wku(f, "qwen3vl.vision.temporal_patch_size", vc["temporal_patch_size"])
        wku(f, "qwen3vl.vision.out_hidden_size", vc["out_hidden_size"])
        wka(f, "qwen3vl.vision.deepstack_indexes", deepstack)
        # LLM
        wku(f, "qwen3vl.hidden_size", tc["hidden_size"])
        wku(f, "qwen3vl.num_hidden_layers", tc["num_hidden_layers"])
        wku(f, "qwen3vl.num_attention_heads", tc["num_attention_heads"])
        wku(f, "qwen3vl.num_key_value_heads", tc["num_key_value_heads"])
        wku(f, "qwen3vl.intermediate_size", tc["intermediate_size"])
        wku(f, "qwen3vl.vocab_size", tc["vocab_size"])
        wkf(f, "qwen3vl.rope_theta", tc.get("rope_theta", 5000000))
        wka(f, "qwen3vl.rope_sections", rope_sections + [0])  # pad to 4
        wku(f, "qwen3vl.mrope_interleaved", 1 if mrope_interleaved else 0)
        wku(f, "qwen3vl.has_qk_norm", 1)
        wku(f, "qwen3vl.tie_word_embeddings", 1 if cfg.get("tie_word_embeddings", True) else 0)
        wku(f, "qwen3vl.image_token_id", cfg.get("image_token_id", 151655))

        # Tensor headers
        offset = 0
        order = list(tinfo.keys())
        for n in order:
            shape, nb, dt = tinfo[n]
            ws(f, n); f.write(struct.pack("<I", len(shape)))
            for d in shape: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", dt)); f.write(struct.pack("<Q", offset))
            offset += nb; offset = (offset + 31) & ~31

        pos = f.tell(); al = (pos + 31) & ~31; f.write(b"\x00" * (al - pos))

        # Tensor data
        for i, n in enumerate(order):
            shape, nb, dt = tinfo[n]
            sk, sp = tmap[n]
            with safe_open(sp, framework="pt") as sf:
                t = sf.get_tensor(sk)
            # Flatten 5D→4D (Conv3D patch embed)
            if t.ndim == 5:
                t = t.reshape(t.shape[0], t.shape[1]*t.shape[2], t.shape[3], t.shape[4])
            # Convert BF16→FP16 directly (no F32 intermediate to save RAM)
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
