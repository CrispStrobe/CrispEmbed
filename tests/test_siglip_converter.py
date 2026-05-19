#!/usr/bin/env python3
"""Unit test: verify SigLIP GGUF converter produces correct tensors and metadata.

Usage:
    python -u tests/test_siglip_converter.py --gguf /path/to/siglip-base.gguf
    python -u tests/test_siglip_converter.py --gguf /path/to/siglip-base.gguf --hf google/siglip-base-patch16-384
"""

import argparse
import sys

import numpy as np

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


def test_gguf_structure(gguf_path: str):
    """Verify GGUF has expected tensors and metadata."""
    import gguf

    print(f"Loading GGUF: {gguf_path}", flush=True)
    r = gguf.GGUFReader(gguf_path)

    # Check metadata
    fields = {str(f.name): f for f in r.fields.values()}

    required_meta = [
        "vit.hidden_size", "vit.num_hidden_layers", "vit.num_attention_heads",
        "vit.intermediate_size", "vit.image_size", "vit.patch_size",
        "vit.num_patches", "vit.model_type",
    ]
    for key in required_meta:
        assert key in fields, f"Missing metadata: {key}"
    print(f"  Metadata: all {len(required_meta)} required keys present", flush=True)

    # Extract values
    def meta_val(name):
        f = fields[name]
        # GGUF field parts: [key_len, key_bytes, value_type, value]
        # For uint32: parts[3] is the value
        # For string: parts[3] is the string length, parts[4] is the string bytes
        if len(f.parts) >= 4:
            v = f.parts[3].tolist()
            if isinstance(v, list) and len(v) == 1:
                return v[0]
            return v
        return None

    hidden = meta_val("vit.hidden_size")
    layers = meta_val("vit.num_hidden_layers")
    heads = meta_val("vit.num_attention_heads")
    inter = meta_val("vit.intermediate_size")
    img_size = meta_val("vit.image_size")
    patch_size = meta_val("vit.patch_size")
    n_patches = meta_val("vit.num_patches")

    print(f"  hidden={hidden} layers={layers} heads={heads} inter={inter}", flush=True)
    print(f"  image={img_size}x{img_size} patch={patch_size} patches={n_patches}", flush=True)

    assert n_patches == (img_size // patch_size) ** 2, \
        f"n_patches mismatch: {n_patches} != ({img_size}//{patch_size})^2"

    # Check required tensors
    tensor_names = {t.name for t in r.tensors}

    required_tensors = [
        "patch_embed.weight",
        "position_embd.weight",
        "post_ln.weight", "post_ln.bias",
    ]
    for i in range(layers):
        required_tensors.extend([
            f"enc.{i}.ln1.weight", f"enc.{i}.ln1.bias",
            f"enc.{i}.ln2.weight", f"enc.{i}.ln2.bias",
            f"enc.{i}.attn.q.weight", f"enc.{i}.attn.k.weight", f"enc.{i}.attn.v.weight",
            f"enc.{i}.attn.o.weight",
            f"enc.{i}.ffn.fc1.weight", f"enc.{i}.ffn.fc1.bias",
            f"enc.{i}.ffn.fc2.weight", f"enc.{i}.ffn.fc2.bias",
        ])

    missing = [t for t in required_tensors if t not in tensor_names]
    assert len(missing) == 0, f"Missing tensors: {missing}"
    print(f"  Tensors: all {len(required_tensors)} required tensors present", flush=True)

    # Check shapes
    tensor_map = {t.name: t for t in r.tensors}

    patch_w = tensor_map["patch_embed.weight"]
    assert list(patch_w.shape) == [hidden, 3, patch_size, patch_size] or \
           list(patch_w.shape) == [patch_size, patch_size, 3, hidden], \
        f"patch_embed.weight shape: {patch_w.shape}"

    pos_w = tensor_map["position_embd.weight"]
    assert pos_w.shape[0] == n_patches or pos_w.shape[1] == n_patches, \
        f"position_embd.weight shape mismatch: {pos_w.shape} vs n_patches={n_patches}"

    q_w = tensor_map["enc.0.attn.q.weight"]
    assert q_w.shape[0] == hidden and q_w.shape[1] == hidden, \
        f"enc.0.attn.q.weight shape: {q_w.shape}, expected [{hidden}, {hidden}]"

    fc1_w = tensor_map["enc.0.ffn.fc1.weight"]
    assert fc1_w.shape[0] == hidden and fc1_w.shape[1] == inter, \
        f"enc.0.ffn.fc1.weight shape: {fc1_w.shape}, expected [{hidden}, {inter}]"

    print(f"  Shapes: all verified", flush=True)

    # Check attention pooling head (SigLIP)
    if "head.probe" in tensor_names:
        print(f"  Attention pooling head: present", flush=True)
        assert "head.attn.in_proj.weight" in tensor_names
        assert "head.mlp.fc1.weight" in tensor_names
    else:
        print(f"  Attention pooling head: absent (CLIP-style)", flush=True)

    print(f"\nPASS: GGUF structure valid ({len(tensor_names)} tensors)", flush=True)
    return True


def test_weight_parity(gguf_path: str, hf_model: str):
    """Compare GGUF weights against HuggingFace reference."""
    import gguf
    import torch
    from transformers import AutoModel

    print(f"\nWeight parity: {hf_model}", flush=True)
    m = AutoModel.from_pretrained(hf_model, torch_dtype=torch.float32, trust_remote_code=True)
    m.eval()
    sd = m.state_dict()

    r = gguf.GGUFReader(gguf_path)
    tensor_map = {t.name: t for t in r.tensors}

    # Compare a few key tensors
    checks = [
        ("patch_embed.weight", "vision_model.embeddings.patch_embedding.weight"),
        ("position_embd.weight", "vision_model.embeddings.position_embedding.weight"),
        ("enc.0.ln1.weight", "vision_model.encoder.layers.0.layer_norm1.weight"),
        ("enc.0.ffn.fc1.weight", "vision_model.encoder.layers.0.mlp.fc1.weight"),
        ("post_ln.weight", "vision_model.post_layernorm.weight"),
    ]

    all_pass = True
    for gguf_name, hf_name in checks:
        if gguf_name not in tensor_map or hf_name not in sd:
            print(f"  SKIP: {gguf_name} / {hf_name}", flush=True)
            continue

        gguf_data = np.array(tensor_map[gguf_name].data, dtype=np.float32).flatten()[:100]
        hf_data = sd[hf_name].float().numpy().flatten()[:100]

        max_diff = np.max(np.abs(gguf_data - hf_data))
        cos = np.dot(gguf_data, hf_data) / (np.linalg.norm(gguf_data) * np.linalg.norm(hf_data) + 1e-18)

        status = "PASS" if max_diff < 1e-6 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {gguf_name}: max_diff={max_diff:.2e} cos={cos:.6f}", flush=True)

    if all_pass:
        print(f"\nPASS: weight parity verified", flush=True)
    else:
        print(f"\nFAIL: weight parity issues", flush=True)
    return all_pass


def main():
    p = argparse.ArgumentParser(description="Test SigLIP GGUF converter output")
    p.add_argument("--gguf", required=True, help="Path to converted GGUF")
    p.add_argument("--hf", default=None, help="HF model ID for weight parity check")
    args = p.parse_args()

    ok = test_gguf_structure(args.gguf)
    if args.hf:
        ok = test_weight_parity(args.gguf, args.hf) and ok

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
