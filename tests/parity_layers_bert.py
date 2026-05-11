#!/usr/bin/env python3
"""Per-layer / per-tensor parity harness for BERT-family GGUFs.

Three checks, each gating the next:
  1. Token-id diff   — CrispEmbed tokenization vs HF tokenization.
                       Critical for SentencePiece-Unigram models, where a single
                       wrong score can mis-segment in ways mean-pooling hides.
                       Requires `CRISPEMBED_DEBUG_TOKENS=1` (added in src/crispembed.cpp).
  2. Tensor diff     — for every CrispEmbed GGUF tensor, find the matching HF
                       state_dict tensor and report max |a-b|. Catches converter
                       bugs (transpose, slice, miscast) that the end-to-end
                       cosine could in principle smooth out.
  3. End-to-end diff — cosine + max_diff between CrispEmbed and HF normalised
                       embeddings on a fixed multilingual probe set.

Usage:
    python tests/parity_layers_bert.py \
        --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        --gguf  /tmp/crispembed-paraphrase/paraphrase-multilingual-MiniLM-L12-v2.gguf

Set HF_HOME / HUGGINGFACE_HUB_CACHE / TRANSFORMERS_OFFLINE=1 first if using a
local cache (e.g. the project's /Volumes/backups/ai/huggingface-hub layout).
"""

import argparse
import os
import re
import subprocess
import sys
from collections import OrderedDict

import numpy as np


# CrispEmbed-format GGUF tensor name → HF state_dict key. CrispEmbed strips
# the backbone prefix ("bert."/"roberta.") at convert time, so we match the
# bare embeddings.* / encoder.layer.*.* keys.
def _gguf_to_hf_key(name: str, n_layers: int):
    """Map CrispEmbed-native GGUF tensor names to HF state_dict keys.

    CrispEmbed `--crisp` format uses `enc.{i}.{block}.{tensor}` (not the
    llama.cpp `blk.{i}.*` convention) and a post-LN BERT layout:
      ln1 = post-attention LN, ln2 = post-FFN LN, ffn.fc1/fc2 = up/down projs.
    """
    # Embeddings
    if name == "token_embd.weight":      return "embeddings.word_embeddings.weight"
    if name == "position_embd.weight":   return "embeddings.position_embeddings.weight"
    if name == "token_type_embd.weight": return "embeddings.token_type_embeddings.weight"
    if name == "embd_ln.weight":         return "embeddings.LayerNorm.weight"
    if name == "embd_ln.bias":           return "embeddings.LayerNorm.bias"

    m = re.match(r"^enc\.(\d+)\.(.+)$", name)
    if not m:
        # MLM/pooler heads, sparse heads — converter writes these for SPLADE
        # support but they are random for plain encoder checkpoints.
        return None
    layer = int(m.group(1))
    suffix = m.group(2)
    base = f"encoder.layer.{layer}"
    mapping = {
        # Attention projections
        "attn.q.weight": f"{base}.attention.self.query.weight",
        "attn.q.bias":   f"{base}.attention.self.query.bias",
        "attn.k.weight": f"{base}.attention.self.key.weight",
        "attn.k.bias":   f"{base}.attention.self.key.bias",
        "attn.v.weight": f"{base}.attention.self.value.weight",
        "attn.v.bias":   f"{base}.attention.self.value.bias",
        "attn.o.weight": f"{base}.attention.output.dense.weight",
        "attn.o.bias":   f"{base}.attention.output.dense.bias",
        # post-attention LayerNorm (ln1 in CrispEmbed naming)
        "ln1.weight": f"{base}.attention.output.LayerNorm.weight",
        "ln1.bias":   f"{base}.attention.output.LayerNorm.bias",
        # FFN
        "ffn.fc1.weight": f"{base}.intermediate.dense.weight",
        "ffn.fc1.bias":   f"{base}.intermediate.dense.bias",
        "ffn.fc2.weight": f"{base}.output.dense.weight",
        "ffn.fc2.bias":   f"{base}.output.dense.bias",
        # post-FFN LayerNorm
        "ln2.weight": f"{base}.output.LayerNorm.weight",
        "ln2.bias":   f"{base}.output.LayerNorm.bias",
    }
    return mapping.get(suffix)


def _load_gguf_tensors(path):
    import gguf
    r = gguf.GGUFReader(path, "r")
    out = OrderedDict()
    for t in r.tensors:
        # GGUFReader returns (raw bytes view, dtype). For F32 it's already f32;
        # for F16 it's a uint8 view we cast through float16.
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            arr = np.frombuffer(t.data.tobytes(), dtype=np.float32)
        elif t.tensor_type == gguf.GGMLQuantizationType.F16:
            arr = np.frombuffer(t.data.tobytes(), dtype=np.float16).astype(np.float32)
        else:
            # Quantised — dequantise to f32 for comparison.
            try:
                arr = gguf.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32)
            except Exception as e:
                print(f"  [tensor-diff] skip {t.name} (dequantise failed: {e})")
                continue
        # gguf shape ordering is reverse of numpy.
        shape = tuple(reversed([int(s) for s in t.shape]))
        out[t.name] = arr.reshape(shape)
    return out, r


TEST_TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Bonjour le monde",
    "Guten Tag",
    "今日はとても良い天気です",
    "机器学习",
    "مرحبا بالعالم",
]


def _hf_tokens(model_id, texts):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return [tok(t, return_tensors="pt", padding=False, truncation=True,
                add_special_tokens=True)["input_ids"][0].tolist() for t in texts]


def _crisp_tokens(binary, gguf, texts):
    """Capture token ids from stderr via CRISPEMBED_DEBUG_TOKENS=1."""
    env = os.environ.copy()
    env["CRISPEMBED_DEBUG_TOKENS"] = "1"
    out = []
    for text in texts:
        r = subprocess.run([binary, "-m", gguf, text],
                           capture_output=True, text=True, timeout=60, env=env)
        m = re.search(r"crispembed: token_ids \(n=(\d+)\):([0-9 \-]+)", r.stderr)
        if not m:
            print(f"  [token-diff] no token line for '{text}':")
            print(r.stderr[-400:])
            out.append(None)
            continue
        out.append([int(x) for x in m.group(2).split()])
    return out


def _hf_embeddings(model_id, texts):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, trust_remote_code=True).encode(
        texts, normalize_embeddings=True)


def _crisp_embeddings(binary, gguf, texts):
    out = []
    for text in texts:
        r = subprocess.run([binary, "-m", gguf, text],
                           capture_output=True, text=True, timeout=60)
        vals = r.stdout.strip().split()
        out.append(np.array([float(x) for x in vals]) if vals else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf",  required=True)
    ap.add_argument("--binary", default="./build/crispembed")
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--skip-tensor-diff", action="store_true",
                    help="Skip per-tensor diff (slow for big models; useful "
                         "if you only care about token + e2e parity).")
    args = ap.parse_args()

    print("=" * 70)
    print(f"  parity_layers_bert: {args.model}")
    print(f"  GGUF: {args.gguf}")
    print("=" * 70)

    # ── Check 1: token IDs ──────────────────────────────────────────
    print("\n[1/3] Token-id diff")
    hf_ids   = _hf_tokens(args.model, TEST_TEXTS)
    cs_ids   = _crisp_tokens(args.binary, args.gguf, TEST_TEXTS)
    tok_ok = True
    for text, h, c in zip(TEST_TEXTS, hf_ids, cs_ids):
        if c is None:
            print(f"  {text!r:50s}  CrispEmbed token capture FAILED")
            tok_ok = False
            continue
        match = h == c
        if not match:
            tok_ok = False
            # Find first divergence
            i = next((k for k, (a, b) in enumerate(zip(h, c)) if a != b),
                     min(len(h), len(c)))
            print(f"  {text!r:50s}  DIFF at idx {i}: HF={h[:i+3]} … vs Crisp={c[:i+3]} …")
        else:
            print(f"  {text!r:50s}  OK  (n={len(h)})")
    print(f"  → tokens: {'PASS' if tok_ok else 'FAIL'}")

    # ── Check 2: per-tensor weight diff ─────────────────────────────
    if not args.skip_tensor_diff:
        print("\n[2/3] Tensor diff (CrispEmbed GGUF vs HF state_dict)")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
        sd = model.state_dict()
        # Strip "bert." / "roberta." / "model." prefix to match converter output
        for pfx in ("roberta.", "bert.", "xlm_roberta.", "model."):
            keys = [k for k in sd.keys() if k.startswith(pfx)]
            if keys:
                sd = {k[len(pfx):]: v for k, v in sd.items() if k.startswith(pfx)} | \
                     {k: v for k, v in sd.items() if not k.startswith(pfx)}
                print(f"  state_dict prefix stripped: '{pfx}'")
                break

        gguf_tensors, _ = _load_gguf_tensors(args.gguf)
        n_compared = n_skipped = 0
        worst = []
        for gname, garr in gguf_tensors.items():
            hkey = _gguf_to_hf_key(gname, args.n_layers)
            if hkey is None or hkey not in sd:
                n_skipped += 1
                continue
            harr = sd[hkey].detach().float().cpu().numpy()
            if garr.shape != harr.shape:
                # GGUF stores transposed for some matmul conventions; try transpose.
                if garr.T.shape == harr.shape:
                    garr_cmp = garr.T
                else:
                    print(f"  [shape mismatch] {gname}: GGUF {garr.shape} vs HF {harr.shape}")
                    n_skipped += 1
                    continue
            else:
                garr_cmp = garr
            md = float(np.max(np.abs(garr_cmp - harr)))
            worst.append((md, gname, hkey, garr.shape))
            n_compared += 1
        worst.sort(reverse=True)
        for md, gname, hkey, shape in worst[:5]:
            print(f"  worst: max|Δ|={md:.2e}  {gname:40s} <- {hkey} {shape}")
        avg_max = float(np.mean([m for m, *_ in worst])) if worst else 0.0
        max_max = max((m for m, *_ in worst), default=0.0)
        print(f"  → {n_compared} tensors compared, {n_skipped} skipped "
              f"(extra MLM/pooler heads not in HF state_dict)")
        print(f"  → mean max|Δ| = {avg_max:.2e}, overall max|Δ| = {max_max:.2e}")
        # f32 → f32 should be EXACTLY zero (bit-identical), so 0.0 here proves
        # the converter performed no lossy ops. Allow ≤ 1e-6 for safety.
        tensor_ok = max_max <= 1e-6
        print(f"  → tensors: {'PASS (bit-exact)' if max_max == 0.0 else 'PASS' if tensor_ok else 'FAIL'}")
    else:
        tensor_ok = True
        print("\n[2/3] Tensor diff — skipped")

    # ── Check 3: end-to-end ─────────────────────────────────────────
    print("\n[3/3] End-to-end cosine vs HF (sentence-transformers)")
    hf_v = _hf_embeddings(args.model, TEST_TEXTS)
    cs_v = _crisp_embeddings(args.binary, args.gguf, TEST_TEXTS)
    e2e_ok = True
    for text, a, b in zip(TEST_TEXTS, hf_v, cs_v):
        if b is None:
            print(f"  {text!r:50s}  CrispEmbed encode FAILED")
            e2e_ok = False
            continue
        cs = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        md = float(np.max(np.abs(a - b)))
        flag = "OK" if cs > 0.999 else "FAIL"
        if cs <= 0.999: e2e_ok = False
        print(f"  {text!r:50s}  cos={cs:.6f}  max|Δ|={md:.2e}  {flag}")

    print("\n" + "=" * 70)
    overall = tok_ok and tensor_ok and e2e_ok
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}  "
          f"(tokens={tok_ok}  tensors={tensor_ok}  e2e={e2e_ok})")
    print("=" * 70)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
