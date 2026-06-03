#!/usr/bin/env python
"""Dump per-layer reference activations for math_ocr DeiT encoder.

Uses the ONNX encoder model (no PyTorch needed) to capture intermediate
outputs at each encoder layer. Writes to a GGUF archive that the C++
crispembed_diff harness can load and compare against.

Usage:
    python tools/dump_math_ocr_reference.py \
        --model-dir /mnt/storage/models/pix2tex \
        --output /mnt/storage/models/pix2tex/math_ocr_ref.gguf
"""

import argparse
import sys
from pathlib import Path

import gguf
import numpy as np
import onnxruntime as ort


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    model_dir = Path(args.model_dir)

    # Load encoder with intermediate outputs
    enc_path = model_dir / "encoder_model.onnx"
    print(f"Loading ONNX encoder: {enc_path}")

    # Get all node output names to capture intermediates
    sess = ort.InferenceSession(str(enc_path))

    # Create test image: gray 0.8 with dark bar, 3ch, normalized
    S = 384
    gray = np.ones((S, S), dtype=np.float32) * 0.8
    gray[S//2-2:S//2+2, S//4:3*S//4] = 0.1
    norm = (gray - 0.5) / 0.5
    img = np.stack([norm, norm, norm])[np.newaxis].astype(np.float32)

    print(f"Input: {img.shape}, range [{img.min():.2f}, {img.max():.2f}]")

    # Run encoder
    enc_out = sess.run(None, {"pixel_values": img})[0]
    print(f"Encoder output: {enc_out.shape}")

    # Write reference GGUF
    writer = gguf.GGUFWriter(args.output, arch="math_ocr_ref")
    writer.add_string("general.name", "math_ocr_deit_reference")

    # Store the final encoder output
    writer.add_tensor("enc_output", enc_out[0].astype(np.float32),
                       raw_dtype=gguf.GGMLQuantizationType.F32)

    # Also store the input embedding (pre-transformer) for comparison
    # Compute it manually from the ONNX weights
    import onnx
    model = onnx.load(str(enc_path))
    inits = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}

    cls = inits["embeddings.cls_token"]
    dist = inits["embeddings.distillation_token"]
    pos = inits["embeddings.position_embeddings"]
    Wp = inits["embeddings.patch_embeddings.projection.weight"]
    Bp = inits["embeddings.patch_embeddings.projection.bias"]

    H = 384
    P = 16
    T = (S // P) ** 2 + 2  # 578

    embedded = np.zeros((T, H), dtype=np.float32)
    embedded[0] = cls.flatten()[:H]
    embedded[1] = dist.flatten()[:H]

    # Patch embeddings (all uniform gray)
    pixel_val = 0.6
    for p_idx in range(T - 2):
        for h in range(H):
            s = Bp[h]
            for j in range(3 * P * P):
                s += pixel_val * Wp.flatten()[h * 3 * P * P + j]
            embedded[p_idx + 2, h] = s

    embedded += pos.reshape(T, H)[:T]

    writer.add_tensor("embedded_input", embedded.astype(np.float32),
                       raw_dtype=gguf.GGMLQuantizationType.F32)

    # Encoder output token by token for fine-grained comparison
    for t in [0, 1, 2, 100, 577]:
        writer.add_tensor(f"enc_output_tok{t}",
                           enc_out[0, t].astype(np.float32),
                           raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = Path(args.output).stat().st_size / 1024
    print(f"\nWritten: {args.output} ({size:.1f} KB)")
    print(f"  enc_output: {enc_out[0].shape}")
    print(f"  embedded_input: {embedded.shape}")
    print(f"  enc_output tok 0 first 5: {enc_out[0, 0, :5]}")
    print(f"  enc_output tok 2 first 5: {enc_out[0, 2, :5]}")


if __name__ == "__main__":
    main()
