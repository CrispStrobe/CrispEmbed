#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from the TeleOCR model's vision block and FFN equations.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Source revision: b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d
# See NOTICE.teleocr and models/licenses/apache-2.0/LICENSE.
"""Replay TeleOCR vision block 17 to distinguish propagation from local F32 error."""

from pathlib import Path
from types import MethodType, SimpleNamespace
import sys

import numpy as np
import torch

from _common import (arguments, comparison, load_block_state, load_module, native_array,
                     provenance, reference_array, write_json, write_probe)


def main() -> int:
    args = arguments(__doc__)
    model, config = load_module(args, "teleocr_vision_diagnostic")
    cfg = model.Qwen2_5_VLVisionConfig(**config["vision_config"])
    cfg._attn_implementation = "sdpa"
    block = model.Qwen2_5_VLVisionBlock(cfg).eval()
    state, hashes = load_block_state(args.model_dir, "visual.blocks.17.")
    block.load_state_dict(state)
    del state

    before = reference_array(args.reference_dir, "vis_layer_16")
    expected = reference_array(args.reference_dir, "vis_layer_17")
    native_before = native_array(args.native_dir, "vis_layer_16", before.shape)
    native_after = native_array(args.native_dir, "vis_layer_17", expected.shape)
    grid = reference_array(args.reference_dir, "image_grid").long().reshape(-1, 3)
    if before.ndim != 2 or before.shape != expected.shape or before.shape[1] != 1280:
        raise ValueError("Expected matching [patches, 1280] vision boundary arrays")
    if grid.shape != (1, 3) or grid[0, 0] != 1 or int(grid.prod()) != before.shape[0]:
        raise ValueError("Expected one image grid matching the frozen patch count")

    # Invoke original geometry methods without constructing a full encoder.
    geometry = SimpleNamespace(
        spatial_merge_size=2, spatial_merge_unit=4, window_size=112, patch_size=14,
        rotary_pos_emb=model.Qwen2_5_VisionRotaryEmbedding(40),
    )
    geometry.get_window_index = MethodType(
        model.Qwen2_5_VisionTransformerPretrainedModel.get_window_index, geometry)
    geometry.rot_pos_emb = MethodType(model.Qwen2_5_VisionTransformerPretrainedModel.rot_pos_emb, geometry)
    order, cumulative = geometry.get_window_index(grid)
    cumulative = torch.unique_consecutive(torch.tensor(cumulative, dtype=torch.int32))
    rotary = geometry.rot_pos_emb(grid).reshape(-1, 4, 40)[order].reshape(-1, 40)
    embedding = torch.cat([rotary, rotary], dim=-1)
    positions = (embedding.cos(), embedding.sin())

    with torch.inference_mode():
        def replay(value):
            post = value + block.attn(block.norm1(value), cu_seqlens=cumulative,
                                      position_embeddings=positions)
            return post, post + block.mlp(block.norm2(post))

        post, original = replay(before)
        _, source_on_native = replay(native_before)
        np.save(args.out / "post_attention.npy", post.numpy())
        np.save(args.out / "original_output.npy", original.numpy())
        np.save(args.out / "source_native_input_output.npy", source_on_native.numpy())

        norm = block.norm2(post)
        gate = block.mlp.gate_proj(norm)
        up = block.mlp.up_proj(norm)
        silu = block.mlp.act_fn(gate)
        product = silu * up
        down = block.mlp.down_proj(product)
        write_probe(args.out, {
            "input": post, "norm_w": block.norm2.weight,
            "gate_w": block.mlp.gate_proj.weight, "gate_b": block.mlp.gate_proj.bias,
            "up_w": block.mlp.up_proj.weight, "up_b": block.mlp.up_proj.bias,
            "down_w": block.mlp.down_proj.weight, "down_b": block.mlp.down_proj.bias,
            "norm": norm, "gate": gate, "up": up, "silu": silu,
            "product": product, "down": down, "output": original,
        })

        # Controlled double-precision evaluation of the same FFN equations.
        # This is a diagnostic comparison, not a replacement frozen reference.
        value64 = post.double()
        norm64 = value64 * torch.rsqrt(value64.square().mean(-1, keepdim=True) + 1e-6) * block.norm2.weight.double()
        gate64 = torch.nn.functional.linear(norm64, block.mlp.gate_proj.weight.double(),
                                            block.mlp.gate_proj.bias.double())
        up64 = torch.nn.functional.linear(norm64, block.mlp.up_proj.weight.double(), block.mlp.up_proj.bias.double())
        product64 = torch.nn.functional.silu(gate64) * up64
        down64 = torch.nn.functional.linear(product64, block.mlp.down_proj.weight.double(),
                                            block.mlp.down_proj.bias.double())
        output64 = value64 + down64
        np.save(args.out / "float64_output.npy", output64.numpy())
        precision = {}
        for name, high, low in (("norm", norm64, norm), ("gate", gate64, gate), ("up", up64, up),
                               ("product", product64, product), ("down", down64, down),
                               ("output", output64, original)):
            item = comparison(low, high)
            index = tuple(item["index"])
            item.update(float32=float(low[index]), float64=float(high[index]),
                        max_magnitude=float(low.abs().max()),
                        float32_spacing=float(np.spacing(np.float32(abs(float(low[index]))))))
            precision[name] = item
            print(name, item)
        write_json(args.out / "original-f32-vs-f64.json", precision)

        results = {
            "original block, original input vs frozen": comparison(original, expected),
            "original block, native input vs frozen": comparison(source_on_native, expected),
            "native pipeline vs original block with native input": comparison(native_after, source_on_native),
        }
        worst = comparison(native_after, expected)["index"]
        coordinate = tuple(worst)
        results["worst pipeline coordinate"] = {
            "index": worst, "frozen": float(expected[coordinate]),
            "source_original_input": float(original[coordinate]),
            "source_native_input": float(source_on_native[coordinate]),
            "native": float(native_after[coordinate]),
        }
        write_json(args.out / "propagation.json", results)
        for name, item in results.items():
            print(name, item)

    provenance(args, Path(__file__), hashes, ("vis_layer_16", "vis_layer_17", "image_grid"),
               ("vis_layer_16", "vis_layer_17"))
    passed = all(item["pass"] for item in precision.values()) and all(
        item["pass"] for item in results.values() if "pass" in item)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
