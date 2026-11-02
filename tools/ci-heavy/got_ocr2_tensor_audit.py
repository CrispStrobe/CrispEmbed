#!/usr/bin/env python3
"""GOT-OCR2: compare every tensor of published GGUFs with the original checkpoint.

Dequantizes each GGUF tensor (F32/F16/BF16/Q8_0/Q4_K via gguf.quants) and compares
it with the matching stepfun-ai/GOT-OCR2_0 safetensors tensor (bf16 -> f32):
cosine, max |delta|, relative error and reference magnitude. Writes
$HEAVY_OUT/tensor-audit.json and prints the worst tensors per file.

    python tools/ci-heavy/got_ocr2_tensor_audit.py [file.gguf ...]
"""

import json
import os
import re
import sys
import urllib.request

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from huggingface_hub import hf_hub_download
from safetensors import safe_open

OUT = os.environ.get("HEAVY_OUT", ".")
SCR = os.environ.get("HEAVY_SCRATCH", ".")
REV = os.environ.get("GOT_REV", "5a0b0a1adc")


def gguf_to_hf(name):
    if name.startswith("l."):
        n = name[2:]
        if n == "embed_tokens.weight":
            return "model.embed_tokens.weight"
        if n == "output_norm.weight":
            return "model.norm.weight"
        m = re.match(r"blk\.(\d+)\.(.+)", n)
        i, rest = m.group(1), m.group(2)
        mp = {"input_layernorm.weight": "input_layernorm.weight",
              "post_attention_layernorm.weight": "post_attention_layernorm.weight",
              "attn_q.weight": "self_attn.q_proj.weight", "attn_q.bias": "self_attn.q_proj.bias",
              "attn_k.weight": "self_attn.k_proj.weight", "attn_k.bias": "self_attn.k_proj.bias",
              "attn_v.weight": "self_attn.v_proj.weight", "attn_v.bias": "self_attn.v_proj.bias",
              "attn_o.weight": "self_attn.o_proj.weight", "ffn_gate.weight": "mlp.gate_proj.weight",
              "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight"}
        return f"model.layers.{i}.{mp[rest]}"
    return None  # vision tensors: covered by the stage diff (cos 1.0 at F16)


def main():
    files = sys.argv[1:] or ["got-ocr2-f16.gguf", "got-ocr2-q4_k.gguf"]
    st = hf_hub_download("stepfun-ai/GOT-OCR2_0", "model.safetensors")
    f = safe_open(st, framework="pt", device="cpu")
    keys = list(f.keys())
    head = {k: list(f.get_slice(k).get_shape()) for k in keys if "lm_head" in k or "embed_tokens" in k}
    rec = {"original_keys": len(keys), "original_head_tensors": head, "files": {}}
    if "lm_head.weight" in keys:
        a = f.get_tensor("lm_head.weight").float().numpy()
        b = f.get_tensor("model.embed_tokens.weight").float().numpy()
        rec["lm_head_equals_embed"] = bool(np.array_equal(a, b))
        rec["lm_head_vs_embed_maxabs"] = float(np.abs(a - b).max())
        del a, b
    for fn in files:
        path = os.path.join(SCR, "models", fn)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(
                f"https://huggingface.co/cstr/got-ocr2-crispembed-GGUF/resolve/{REV}/{fn}", path)
        r = GGUFReader(path)
        rows = []
        for t in r.tensors:
            hk = gguf_to_hf(t.name)
            if hk is None:
                continue
            if hk not in keys:
                rows.append({"name": t.name, "missing_in_original": hk})
                continue
            ref = f.get_tensor(hk).float().numpy().astype(np.float64)
            got = dequantize(t.data, t.tensor_type).astype(np.float64).reshape(ref.shape)
            d = got - ref
            cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30))
            rows.append({"name": t.name, "type": t.tensor_type.name, "cos": cos,
                         "max_abs_delta": float(np.abs(d).max()), "ref_max_abs": float(np.abs(ref).max()),
                         "rel_l2": float(np.linalg.norm(d) / (np.linalg.norm(ref) + 1e-30))})
        rows.sort(key=lambda x: x.get("cos", -9))
        rec["files"][fn] = rows
        print(f"== {fn}: {len(rows)} LLM tensors; worst 12 by cosine")
        for x in rows[:12]:
            print("  ", json.dumps(x))
    with open(os.path.join(OUT, "tensor-audit.json"), "w") as fo:
        json.dump(rec, fo, indent=1)
    print("original head tensors:", head, "lm_head==embed:", rec.get("lm_head_equals_embed"))


if __name__ == "__main__":
    main()
