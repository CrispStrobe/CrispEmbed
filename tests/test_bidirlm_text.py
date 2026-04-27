#!/usr/bin/env python3
"""BidirLM-Omni text-path parity test.

Loads only BidirLMOmniTextModel from HF (bypassing the full omni
SentenceTransformer pipeline that drags in librosa/torchvision/etc.) and
compares against CrispEmbed's GGUF inference.

    python tests/test_bidirlm_text.py \
        --model BidirLM/BidirLM-Omni-2.5B-Embedding \
        --gguf  $CRISPEMBED_CACHE_DIR/bidirlm-omni-2.5b-q8_0.gguf \
        --binary ./build/crispembed
"""

import argparse
import subprocess
import sys
import numpy as np


TEST_TEXTS = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming the world of natural language processing",
    "This is a test of the emergency broadcast system",
]


def hf_text_embed(model_id: str, texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json, os

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    text_model_cls = get_class_from_dynamic_module(
        "modeling_bidirlm_omni.BidirLMOmniTextModel", model_id,
    )
    # bf16 to fit a 2.5B model in <8 GB; matches the saved checkpoint dtype.
    text_model = text_model_cls(config.text_config).to(torch.bfloat16)
    text_model.eval()

    # Load weights manually: strip the "language_model." prefix that the
    # full BidirLMOmniModel adds. Try single-shard first, then sharded index.
    sd = {}
    try:
        st_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        raw = load_file(st_path)
    except Exception:
        idx_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
        with open(idx_path) as f:
            idx = json.load(f)["weight_map"]
        shards = set(idx.values())
        raw = {}
        for shard in shards:
            sp = hf_hub_download(repo_id=model_id, filename=shard)
            raw.update(load_file(sp))
    for k, v in raw.items():
        if k.startswith("language_model."):
            sd[k[len("language_model."):]] = v.to(torch.bfloat16)
    missing, unexpected = text_model.load_state_dict(sd, strict=False)
    if missing:
        print(f"WARN: {len(missing)} missing tensors, e.g. {missing[:3]}")
    if unexpected:
        print(f"  ({len(unexpected)} unexpected keys ignored)")

    out = []
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt")
            h = text_model(**enc).last_hidden_state[0]  # [T, H], bf16
            h = h.float()
            mask = enc["attention_mask"][0].to(h.dtype)
            pooled = (h * mask[:, None]).sum(0) / mask.sum().clamp(min=1)
            v = torch.nn.functional.normalize(pooled, dim=-1).cpu().numpy()
            out.append(v)
    return np.stack(out)


def crispembed_embed(binary: str, gguf: str, texts: list[str]) -> np.ndarray:
    # Single CLI call — batches all texts through one model load.
    cmd = [binary, "-m", gguf, "--json"] + texts
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(f"crispembed failed: {r.stderr[-500:]}", file=sys.stderr)
        sys.exit(1)
    import json as _json
    arr = _json.loads(r.stdout)
    return np.asarray([row["embedding"] for row in arr], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gguf", required=True)
    p.add_argument("--binary", default="./build/crispembed")
    args = p.parse_args()

    print(f"Texts: {len(TEST_TEXTS)}")

    print("Computing HF reference (text-only path)...")
    hf = hf_text_embed(args.model, TEST_TEXTS)
    print(f"HF dim: {hf.shape}")

    print("Computing CrispEmbed embeddings...")
    ce = crispembed_embed(args.binary, args.gguf, TEST_TEXTS)
    print(f"CE dim: {ce.shape}")

    if hf.shape != ce.shape:
        print(f"FAIL: shape mismatch")
        return 1

    print()
    print(f"{'Text':<60s} {'MaxDiff':>10s} {'CosSim':>10s} {'Status':>8s}")
    print("-" * 92)
    all_pass = True
    for t, h, c in zip(TEST_TEXTS, hf, ce):
        d = float(np.max(np.abs(h - c)))
        s = float(np.dot(h, c) / (np.linalg.norm(h) * np.linalg.norm(c) + 1e-12))
        ok = s > 0.99
        all_pass &= ok
        label = (t[:57] + "...") if len(t) > 60 else t
        print(f"{label:<60s} {d:>10.6f} {s:>10.6f} {'PASS' if ok else 'FAIL':>8s}")

    print()
    print("ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
