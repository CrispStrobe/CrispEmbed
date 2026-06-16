#!/usr/bin/env python3
"""Minimal Granite Vision GGUF converter — lazy safetensors loading."""
import json, os, struct, sys, numpy as np
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

model_dir = sys.argv[1]
output = sys.argv[2]
use_fp16 = "--fp16" in sys.argv

with open(os.path.join(model_dir, "config.json")) as f: cfg = json.load(f)
vc, tc = cfg["vision_config"], cfg["text_config"]
feat_layers = cfg["vision_feature_layer"]
tie = cfg.get("tie_word_embeddings", False)

st_files = sorted(os.path.join(model_dir,f) for f in os.listdir(model_dir) if f.endswith(".safetensors"))
print(f"Vision: dim={vc['hidden_size']}, layers={vc['num_hidden_layers']}")
print(f"LLM: dim={tc['hidden_size']}, layers={tc['num_hidden_layers']}")

# Map tensor names
def map_name(key):
    if key.startswith("vision_tower.vision_model."):
        s = key[len("vision_tower.vision_model."):]
        if s == "embeddings.patch_embedding.weight": return "vis.patch_embed.weight"
        if s == "embeddings.patch_embedding.bias": return "vis.patch_embed.bias"
        if s == "embeddings.position_embedding.weight": return "vis.pos_embed.weight"
        if s.startswith("encoder.layers."):
            p = s.split("."); li = int(p[2]); r = ".".join(p[3:])
            r = r.replace("self_attn.q_proj","attn.q").replace("self_attn.k_proj","attn.k")
            r = r.replace("self_attn.v_proj","attn.v").replace("self_attn.out_proj","attn.out")
            r = r.replace("mlp.fc1","ffn.up").replace("mlp.fc2","ffn.down")
            return f"vis.layer.{li}.{r}"
        if s == "post_layernorm.weight": return "vis.post_ln.weight"
        if s == "post_layernorm.bias": return "vis.post_ln.bias"
    elif key.startswith("multi_modal_projector."): return f"proj.{key[len('multi_modal_projector.'):]}"
    elif key == "image_newline": return "image_newline"
    elif key.startswith("language_model.model."):
        s = key[len("language_model.model."):]
        if s == "embed_tokens.weight": return "llm.embed.weight"
        if s == "norm.weight": return "llm.norm.weight"
        if s.startswith("layers."):
            p = s.split("."); li = int(p[1]); r = ".".join(p[2:])
            r = r.replace("self_attn.q_proj","attn.q").replace("self_attn.k_proj","attn.k")
            r = r.replace("self_attn.v_proj","attn.v").replace("self_attn.o_proj","attn.o")
            r = r.replace("mlp.gate_proj","ffn.gate").replace("mlp.up_proj","ffn.up").replace("mlp.down_proj","ffn.down")
            r = r.replace("input_layernorm","norm1").replace("post_attention_layernorm","norm2")
            return f"llm.layer.{li}.{r}"
    elif key.startswith("language_model.lm_head."): return f"llm.lm_head.{key[len('language_model.lm_head.'):]}"
    return None

# Collect metadata
tmap = {}  # gguf_name → (src_key, st_path)
for sp in st_files:
    with safe_open(sp, framework="pt") as sf:
        for k in sf.keys():
            n = map_name(k)
            if n: tmap[n] = (k, sp)
print(f"Mapped: {len(tmap)} tensors")

# Get shapes (lazy)
tinfo = {}
for gn, (sk, sp) in tmap.items():
    with safe_open(sp, framework="pt") as sf:
        t = sf.get_tensor(sk)
        shape = list(t.shape)
        dt = GGML_TYPE_F16 if (use_fp16 and len(shape)>=2) else GGML_TYPE_F32
        nb = int(np.prod(shape)) * (2 if dt==GGML_TYPE_F16 else 4)
        tinfo[gn] = (shape, nb, dt)
        del t

n_kv = 20
with open(output, "wb") as f:
    f.write(struct.pack("<I",GGUF_MAGIC)); f.write(struct.pack("<I",GGUF_VERSION))
    f.write(struct.pack("<Q",len(tinfo))); f.write(struct.pack("<Q",n_kv))
    wks(f,"general.architecture","granite_vision")
    wks(f,"general.name","Granite-Vision-3.3-2B")
    wku(f,"granite_vision.vis_dim",vc["hidden_size"])
    wku(f,"granite_vision.vis_layers",vc["num_hidden_layers"])
    wku(f,"granite_vision.vis_heads",vc["num_attention_heads"])
    wku(f,"granite_vision.vis_image_size",vc["image_size"])
    wku(f,"granite_vision.vis_patch_size",vc["patch_size"])
    wka(f,"granite_vision.feature_layers",feat_layers)
    wku(f,"granite_vision.llm_dim",tc["hidden_size"])
    wku(f,"granite_vision.llm_layers",tc["num_hidden_layers"])
    wku(f,"granite_vision.llm_heads",tc["num_attention_heads"])
    wku(f,"granite_vision.llm_kv_heads",tc["num_key_value_heads"])
    wku(f,"granite_vision.llm_ffn_dim",tc["intermediate_size"])
    wku(f,"granite_vision.vocab_size",tc["vocab_size"])
    wkf(f,"granite_vision.embedding_multiplier",tc.get("embedding_multiplier",1.0))
    wkf(f,"granite_vision.residual_multiplier",tc.get("residual_multiplier",1.0))
    wkf(f,"granite_vision.logits_scaling",tc.get("logits_scaling",1.0))
    wkf(f,"granite_vision.rope_theta",tc.get("rope_theta",10000.0))
    wku(f,"granite_vision.image_token_index",cfg.get("image_token_index",49155))
    wku(f,"granite_vision.tie_word_embeddings",1 if tie else 0)

    off = 0
    order = list(tinfo.keys())
    for n in order:
        shape,nb,dt = tinfo[n]
        ws(f,n); f.write(struct.pack("<I",len(shape)))
        for d in shape: f.write(struct.pack("<Q",d))
        f.write(struct.pack("<I",dt)); f.write(struct.pack("<Q",off))
        off += nb; off = (off+31)&~31

    pos = f.tell(); al = (pos+31)&~31; f.write(b"\x00"*(al-pos))

    for i,n in enumerate(order):
        shape,nb,dt = tinfo[n]
        sk,sp = tmap[n]
        with safe_open(sp, framework="pt") as sf:
            t = sf.get_tensor(sk); a = t.float().numpy(); del t
        d = a.astype(np.float16).tobytes() if dt==GGML_TYPE_F16 else a.astype(np.float32).tobytes()
        del a; f.write(d)
        pad = ((len(d)+31)&~31)-len(d)
        if pad>0: f.write(b"\x00"*pad)
        if (i+1)%100==0: print(f"  {i+1}/{len(order)}")

print(f"Written: {output} ({os.path.getsize(output)/1024/1024:.0f} MB)")
