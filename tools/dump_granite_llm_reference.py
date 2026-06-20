#!/usr/bin/env python3
"""Granite Vision LLM decode reference for crispembed-diff.

Builds a numerically-faithful reference for the *Granite-3.1-2B language model*
decode path **directly from the converted GGUF weights** (dequantized), so the
C++ engine (granite_vision_ocr.cpp) can be validated layer-by-layer without
needing the 5 GB HF checkpoint.

The forward replicates transformers' GraniteForCausalLM exactly:
  - inputs_embeds = embed[token] * embedding_multiplier
  - per layer:  h = h + attn(rmsnorm(h)) * residual_multiplier
                h = h + mlp(rmsnorm(h))  * residual_multiplier
  - attention scaling = attention_multiplier  (NOT 1/sqrt(head_dim))
  - RoPE = rotate_half (split-half / "NEGHALF"), default rope (theta=300000)
  - final rmsnorm, logits = (h @ embed.T) / logits_scaling   (tied lm_head)

A fixed text-only token sequence is run causally; per-layer hidden states for
the final token and the final-token logits are dumped to a ref.gguf consumable
by crispembed_diff::Ref.

Usage:
  python tools/dump_granite_llm_reference.py <model.gguf> <out_ref.gguf>
"""
import sys, struct
import numpy as np
import gguf
from gguf.quants import dequantize

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
TYPE_STRING = 8
TYPE_F32 = 0

# Fixed, reproducible text-only token sequence (must match the C++ dump hook).
TOKENS = [12, 345, 678, 901, 234, 56, 789]


def main():
    if len(sys.argv) < 3:
        print("usage: dump_granite_llm_reference.py <model.gguf> <out_ref.gguf>")
        return 1
    model_path, out_path = sys.argv[1], sys.argv[2]

    r = gguf.GGUFReader(model_path)
    kv = {f.name: f for f in r.fields.values()}

    def kv_u32(name, default):
        f = kv.get(name)
        if f is None:
            return default
        return int(f.parts[f.data[0]][0])

    def kv_f32(name, default):
        f = kv.get(name)
        if f is None:
            return default
        return float(f.parts[f.data[0]][0])

    D       = kv_u32("granite_vision.llm_dim", 2048)
    n_layer = kv_u32("granite_vision.llm_layers", 40)
    n_head  = kv_u32("granite_vision.llm_heads", 32)
    n_kv    = kv_u32("granite_vision.llm_kv_heads", 8)
    ffn     = kv_u32("granite_vision.llm_ffn_dim", 8192)
    vocab   = kv_u32("granite_vision.vocab_size", 49156)
    emb_mul = kv_f32("granite_vision.embedding_multiplier", 12.0)
    res_mul = kv_f32("granite_vision.residual_multiplier", 0.22)
    log_scl = kv_f32("granite_vision.logits_scaling", 8.0)
    attn_mul = kv_f32("granite_vision.attention_multiplier", 0.015625)
    rms_eps = kv_f32("granite_vision.rms_eps", 1e-5)
    theta   = kv_f32("granite_vision.rope_theta", 300000.0)
    head_dim = D // n_head
    kv_dim = n_kv * head_dim

    print(f"LLM: D={D} L={n_layer} heads={n_head}/{n_kv} hd={head_dim} ffn={ffn} "
          f"vocab={vocab}\n  emb={emb_mul} res={res_mul} logit={log_scl} "
          f"attn={attn_mul} eps={rms_eps} theta={theta}")

    byname = {t.name: t for t in r.tensors}

    def raw(name):
        t = byname[name]
        return dequantize(t.data, t.tensor_type).astype(np.float32)

    # ggml stores 2D weights as ne=[in, out] with `in` the fastest axis, so the
    # flattened dequantized buffer is row-major [out, in] — exactly how the C++
    # gv_linear indexes weight[o*in + j]. Verified against the raw F16 embed
    # bytes: reshape(-1).reshape(out, in) reproduces ggml-flat order; the
    # numpy `.shape` dequantize hands back is NOT a reliable orientation hint.
    def w2d(name, out, inn):
        a = raw(name).reshape(-1)
        assert a.size == out * inn, f"{name}: {a.size} != {out*inn}"
        return a.reshape(out, inn)

    def vec(name):
        return raw(name).reshape(-1)

    # Embedding / tied lm_head: logical [vocab, D].
    embed = w2d("llm.embed.weight", vocab, D)

    def rmsnorm(x, w):
        v = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
        return (x / np.sqrt(v + rms_eps)).astype(np.float32) * w

    def rotate_half(x):
        h = x.shape[-1] // 2
        return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)

    # RoPE tables for positions 0..T-1.
    T = len(TOKENS)
    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))  # [hd/2]
    pos = np.arange(T)[:, None] * inv_freq[None, :]                     # [T, hd/2]
    emb = np.concatenate([pos, pos], axis=-1)                           # [T, hd]
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)

    def apply_rope(x):  # x: [T, n, hd]
        c = cos[:, None, :]
        s = sin[:, None, :]
        return x * c + rotate_half(x) * s

    # ── Forward (full causal prefill over TOKENS) ─────────────────────────
    h = embed[np.array(TOKENS)] * emb_mul        # [T, D]
    refs = {}
    refs["llm_embed_in"] = h[-1].copy()

    for li in range(n_layer):
        p = f"llm.layer.{li}"
        res = h
        x = rmsnorm(h, vec(p + ".norm1.weight"))
        q = x @ w2d(p + ".attn.q.weight", n_head * head_dim, D).T   # [T, n_head*hd]
        k = x @ w2d(p + ".attn.k.weight", kv_dim, D).T             # [T, kv_dim]
        v = x @ w2d(p + ".attn.v.weight", kv_dim, D).T
        q = q.reshape(T, n_head, head_dim)
        k = k.reshape(T, n_kv, head_dim)
        v = v.reshape(T, n_kv, head_dim)
        q = apply_rope(q)
        k = apply_rope(k)
        # GQA expand
        rep = n_head // n_kv
        k = np.repeat(k, rep, axis=1)   # [T, n_head, hd]
        v = np.repeat(v, rep, axis=1)
        # attention per head, causal
        out = np.zeros((T, n_head, head_dim), dtype=np.float32)
        for hh in range(n_head):
            qh = q[:, hh, :]            # [T, hd]
            kh = k[:, hh, :]
            vh = v[:, hh, :]
            scores = (qh @ kh.T) * attn_mul   # [T, T]
            mask = np.triu(np.ones((T, T), dtype=bool), 1)
            scores = np.where(mask, -1e30, scores)
            scores = scores - scores.max(-1, keepdims=True)
            e = np.exp(scores.astype(np.float64))
            a = (e / e.sum(-1, keepdims=True)).astype(np.float32)
            out[:, hh, :] = a @ vh
        out = out.reshape(T, n_head * head_dim)
        attn = out @ w2d(p + ".attn.o.weight", D, n_head * head_dim).T
        h = res + attn * res_mul

        res = h
        x = rmsnorm(h, vec(p + ".norm2.weight"))
        gate = x @ w2d(p + ".ffn.gate.weight", ffn, D).T
        up = x @ w2d(p + ".ffn.up.weight", ffn, D).T
        silu = gate / (1.0 + np.exp(-gate.astype(np.float64))).astype(np.float32)
        down = (silu * up) @ w2d(p + ".ffn.down.weight", D, ffn).T
        h = res + down * res_mul

        if li in (0, 1, n_layer // 2, n_layer - 1):
            refs[f"llm_layer_{li}_out"] = h[-1].copy()

    h = rmsnorm(h, vec("llm.norm.weight"))
    refs["llm_final_norm"] = h[-1].copy()
    logits = (h[-1] @ embed.T) / log_scl
    refs["llm_logits"] = logits.copy()
    print(f"  argmax(logits)={int(np.argmax(logits))} max={logits.max():.4f}")

    # ── Write ref.gguf (crispembed_diff format) ───────────────────────────
    def ws(f, s):
        b = s.encode("utf-8"); f.write(struct.pack("<Q", len(b))); f.write(b)
    items = [(k, np.asarray(v, np.float32)) for k, v in refs.items()]
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC)); f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(items))); f.write(struct.pack("<Q", 1))
        ws(f, "general.architecture"); f.write(struct.pack("<I", TYPE_STRING))
        ws(f, "granite_llm_ref")
        off = 0
        for name, data in items:
            ws(f, name); f.write(struct.pack("<I", 1)); f.write(struct.pack("<Q", data.size))
            f.write(struct.pack("<I", TYPE_F32)); f.write(struct.pack("<Q", off))
            off += data.nbytes; off = (off + 31) & ~31
        pos2 = f.tell(); al = (pos2 + 31) & ~31; f.write(b"\x00" * (al - pos2))
        for name, data in items:
            f.write(data.tobytes())
            pad = ((data.nbytes + 31) & ~31) - data.nbytes
            if pad: f.write(b"\x00" * pad)
    print(f"wrote {out_path}: {len(items)} stages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
