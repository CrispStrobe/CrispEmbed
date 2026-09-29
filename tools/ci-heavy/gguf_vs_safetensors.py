#!/usr/bin/env python3
"""Compare every LLM tensor of a Qwen2/3-VL-family CrispEmbed GGUF against the
upstream safetensors, to separate a conversion error from a runtime one.

Uni-MuMER-Qwen3-VL-2B (F16 and q8_0 alike) matches transformers exactly
through decoder layer 14 and diverges at layer 15 on every row but BOS - the
same on two images and two precisions, so deterministic. The base Qwen3-VL-2B
GGUF matches at every layer. A mis-mapped or stale tensor in the fine-tune's
GGUF would do exactly this.

    gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/gguf_vs_safetensors.py \\
        -f args="--hf phxember/Uni-MuMER-Qwen3-VL-2B --gguf cstr/uni-mumer-qwen3-vl-2b-GGUF/uni-mumer-qwen3-vl-2b-f16.gguf" \\
        -f pip="torch gguf safetensors numpy huggingface_hub"
Exit 1 if any tensor differs beyond F16 rounding.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

OUT = Path(os.environ.get("HEAVY_OUT", "out"))
OUT.mkdir(parents=True, exist_ok=True)
arg = lambda k: sys.argv[sys.argv.index(k) + 1]
hf_repo = arg("--hf")
g_repo, g_file = arg("--gguf").rsplit("/", 1)[0], arg("--gguf").rsplit("/", 1)[1]

SUFFIX = {
    "attn_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight", "attn_o.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

snap = Path(snapshot_download(hf_repo, allow_patterns=["*.safetensors", "*.json"]))
where = {}
for f in snap.glob("*.safetensors"):
    with safe_open(str(f), "pt") as h:
        for k in h.keys():
            where[k] = f
reader = GGUFReader(hf_hub_download(g_repo, g_file))

import torch  # noqa: E402  (bf16 safetensors need torch to read)

rows, bad = [], []
for t in reader.tensors:
    m = re.match(r"l\.blk\.(\d+)\.(.+)$", t.name)
    if not m or m.group(2) not in SUFFIX:
        continue
    il, suf = int(m.group(1)), m.group(2)
    cands = [f"model.language_model.layers.{il}.{SUFFIX[suf]}", f"model.layers.{il}.{SUFFIX[suf]}"]
    key = next((c for c in cands if c in where), None)
    if key is None:
        bad.append({"tensor": t.name, "error": "no upstream tensor"})
        continue
    with safe_open(str(where[key]), "pt") as h:
        ref = h.get_tensor(key).float().numpy()
    g = np.array(t.data, dtype=np.float32).reshape(ref.shape) if t.data.size == ref.size else None
    if g is None:
        bad.append({"tensor": t.name, "error": f"size {t.data.size} vs {ref.size} (quantized file? use an F16/F32 GGUF)"})
        continue
    d = np.abs(g - ref)
    rel = float(d.max() / (np.abs(ref).max() + 1e-12))
    cos = float((g * ref).sum() / (np.linalg.norm(g) * np.linalg.norm(ref) + 1e-30))
    e = {"tensor": t.name, "upstream": key, "max_abs": float(d.max()), "rel": rel, "cos": cos}
    rows.append(e)
    if cos < 0.99999 or rel > 1e-2:
        bad.append(e)

(OUT / "result.json").write_text(json.dumps({"bad": bad, "all": rows}, indent=1))
lines = [f"### {g_file} vs {hf_repo}: {len(rows)} LLM tensors compared, {len(bad)} off\n"]
lines += [f"- `{b['tensor']}`: {b}" for b in bad[:40]]
worst = sorted(rows, key=lambda e: e["cos"])[:5]
lines += ["\nlowest cos:"] + [f"- `{e['tensor']}` cos {e['cos']:.6f} rel {e['rel']:.2e}" for e in worst]
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(1 if bad else 0)
