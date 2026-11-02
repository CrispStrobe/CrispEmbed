#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from the TeleOCR model's decoder block and FFN equations.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Source revision: b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d
# See NOTICE.teleocr and models/licenses/apache-2.0/LICENSE.
"""Replay TeleOCR decoder block 2 with frozen and native inputs, plus its FFN in F64."""

from pathlib import Path
import sys

import numpy as np
import torch

from _common import (arguments, comparison, load_block_state, load_module, native_array,
                     provenance, reference_array, write_json, write_probe)


def main() -> int:
    args = arguments(__doc__)
    model, config = load_module(args, "teleocr_decoder_diagnostic")
    cfg = model.Qwen2_5_VLTextConfig(**config["text_config"])
    cfg._attn_implementation = "sdpa"
    block = model.Qwen2_5_VLDecoderLayer(cfg, 2).eval()
    state, hashes = load_block_state(args.model_dir, "model.layers.2.")
    block.load_state_dict(state)
    del state

    before = reference_array(args.reference_dir, "llm_layer_1").unsqueeze(0)
    expected = reference_array(args.reference_dir, "llm_layer_2").unsqueeze(0)
    native_before = native_array(args.native_dir, "llm_layer_1", before.shape)
    native_after = native_array(args.native_dir, "llm_layer_2", expected.shape)
    positions_raw = reference_array(args.reference_dir, "mrope_positions")
    if before.ndim != 3 or before.shape != expected.shape or before.shape[2] != 1024:
        raise ValueError("Expected matching [tokens, 1024] decoder boundary arrays")
    if positions_raw.shape != (before.shape[1], 3):
        raise ValueError("Expected frozen [tokens, 3] multimodal position IDs")
    ids = positions_raw.long().T.unsqueeze(1)

    with torch.inference_mode():
        positions = block.self_attn.rotary_emb(before, ids)
        post = before + block.self_attn(block.input_layernorm(before), position_embeddings=positions,
                                        position_ids=ids[0])[0]
        norm = block.post_attention_layernorm(post)
        gate = block.mlp.gate_proj(norm)
        up = block.mlp.up_proj(norm)
        silu = block.mlp.act_fn(gate)
        product = silu * up
        down = block.mlp.down_proj(product)
        original = post + down
        source_on_native = block(native_before, position_embeddings=positions, position_ids=ids[0])[0]
        write_probe(args.out, {
            "input": post[0], "norm_w": block.post_attention_layernorm.weight,
            "gate_w": block.mlp.gate_proj.weight, "gate_b": torch.zeros(cfg.intermediate_size),
            "up_w": block.mlp.up_proj.weight, "up_b": torch.zeros(cfg.intermediate_size),
            "down_w": block.mlp.down_proj.weight, "down_b": torch.zeros(cfg.hidden_size),
            "norm": norm[0], "gate": gate[0], "up": up[0], "silu": silu[0],
            "product": product[0], "down": down[0], "output": original[0],
        })

        value64 = post.double()
        norm64 = (value64 * torch.rsqrt(value64.square().mean(-1, keepdim=True) + 1e-6)
                  * block.post_attention_layernorm.weight.double())
        output64 = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(norm64, block.mlp.gate_proj.weight.double()))
            * torch.nn.functional.linear(norm64, block.mlp.up_proj.weight.double()),
            block.mlp.down_proj.weight.double(),
        ) + value64
        np.save(args.out / "float64_output.npy", output64[0].numpy())
        results = {
            "source original input vs frozen": comparison(original, expected),
            "source native input vs frozen": comparison(source_on_native, expected),
            "native vs source native input": comparison(native_after, source_on_native),
            "source F32 vs F64 MLP": comparison(original.double(), output64),
        }
        write_json(args.out / "propagation.json", results)
        for name, item in results.items():
            print(name, item)

    provenance(args, Path(__file__), hashes, ("llm_layer_1", "llm_layer_2", "mrope_positions"),
               ("llm_layer_1", "llm_layer_2"))
    return 0 if all(item["pass"] for item in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
