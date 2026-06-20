#!/usr/bin/env python3
"""Patch a Granite-Vision GGUF: embed the BPE tokenizer + late-added scalars.

The original converter (convert-granite-vision-to-gguf.py, pre-tokenizer) wrote
no tokenizer and omitted attention_multiplier / rms_norm_eps, so the engine
could not detokenize and decoded attention with the wrong scale. This rewrites
an existing GGUF (preserving the quantized tensor data) with:

  tokenizer.tokens   (array<str>, id-ordered, len = vocab_size)
  tokenizer.merges   (array<str>, "left right")
  granite_vision.bos_token_id / eos_token_id
  granite_vision.attention_multiplier  (float32)
  granite_vision.rms_eps               (float32)

For freshly-converted models use the updated converter instead; this exists to
upgrade GGUFs that are already published/quantized.

Usage:
  python models/patch-granite-gguf-tokenizer.py <in.gguf> <hf_src_dir> <out.gguf>

<hf_src_dir> must contain tokenizer.json and config.json from
ibm-granite/granite-vision-3.3-2b.
"""
import json, sys, os
import numpy as np
import gguf


def load_tokenizer(src_dir):
    with open(os.path.join(src_dir, "tokenizer.json")) as f:
        tj = json.load(f)
    model = tj["model"]
    vocab = model["vocab"]                         # token -> id (base)
    inv = {i: tok for tok, i in vocab.items()}
    for a in tj.get("added_tokens", []):           # specials + <image>
        inv[a["id"]] = a["content"]
    n = max(inv) + 1
    tokens = [inv.get(i, f"<unused{i}>") for i in range(n)]

    merges = []
    for m in model.get("merges", []):
        merges.append(m if isinstance(m, str) else f"{m[0]} {m[1]}")
    return tokens, merges


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    in_path, src_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(os.path.join(src_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg["text_config"]
    attn_mul = float(tc.get("attention_multiplier", 0.015625))
    rms_eps = float(tc.get("rms_norm_eps", 1e-5))
    eos_id = int(tc.get("eos_token_id", 0))
    bos_id = int(tc.get("bos_token_id", 0))

    tokens, merges = load_tokenizer(src_dir)
    print(f"tokens={len(tokens)} merges={len(merges)} "
          f"attn_mul={attn_mul} rms_eps={rms_eps} eos={eos_id}")

    r = gguf.GGUFReader(in_path)
    arch = "granite_vision"
    w = gguf.GGUFWriter(out_path, arch)

    GT = gguf.GGUFValueType
    skip = {"general.architecture", "general.name"}
    # Copy every existing KV (except arch/name, which the writer sets) verbatim.
    for name, fld in r.fields.items():
        if name.startswith("GGUF.") or name in skip:
            continue
        t = fld.types[0]
        if t == GT.ARRAY:
            elem_t = fld.types[1]
            vals = [fld.parts[i] for i in fld.data]
            if elem_t == GT.STRING:
                vals = [bytes(v).decode("utf-8") for v in vals]
            else:
                vals = [int(v[0]) if np.issubdtype(v.dtype, np.integer)
                        else float(v[0]) for v in vals]
            w.add_array(name, vals)
        elif t == GT.STRING:
            w.add_string(name, bytes(fld.parts[fld.data[0]]).decode("utf-8"))
        else:
            val = fld.parts[fld.data[0]][0]
            if t in (GT.FLOAT32, GT.FLOAT64):
                w.add_float32(name, float(val))
            elif t == GT.BOOL:
                w.add_bool(name, bool(val))
            else:
                w.add_uint32(name, int(val))

    # New tokenizer + scalar KVs.
    w.add_array("tokenizer.tokens", tokens)
    w.add_array("tokenizer.merges", merges)
    w.add_float32("granite_vision.attention_multiplier", attn_mul)
    w.add_float32("granite_vision.rms_eps", rms_eps)
    w.add_uint32("granite_vision.bos_token_id", bos_id)
    w.add_uint32("granite_vision.eos_token_id", eos_id)

    # Copy all tensors with their original (quantized) data, unchanged — using
    # the canonical gguf_new_metadata copy pattern (raw byte shape + dtype).
    for t in r.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype,
                          t.data.nbytes, t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        w.write_tensor_data(t.data)
    w.close()
    sz = os.path.getsize(out_path)
    print(f"wrote {out_path}: {sz/1e6:.1f} MB, {len(r.tensors)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
