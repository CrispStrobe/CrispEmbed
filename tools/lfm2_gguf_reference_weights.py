"""Load exact GGUF storage values into the official FP32 LFM2 reference.

This deliberately removes GGML activation quantization. It is an audit control,
not an emulation of a quantized runtime or a replacement for checkpoint parity.
"""

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize


def tensor_names(name):
    name = name.removeprefix("lfm2.")
    if name == "embed_tokens.weight":
        return "lfm.embed_tokens.weight", "token_embd.weight"
    if name == "embedding_norm.weight":
        return "lfm.embedding_norm.weight", "token_embd_norm.weight"
    _, layer, suffix = name.split(".", 2)
    official = {
        "operator_norm.weight": "attn_norm.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "feed_forward.w1.weight": "ffn_gate.weight",
        "feed_forward.w2.weight": "ffn_down.weight",
        "feed_forward.w3.weight": "ffn_up.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.out_proj.weight": "attn_output.weight",
        "self_attn.q_layernorm.weight": "attn_q_norm.weight",
        "self_attn.k_layernorm.weight": "attn_k_norm.weight",
        "conv.in_proj.weight": "shortconv.in_proj.weight",
        "conv.out_proj.weight": "shortconv.out_proj.weight",
        "conv.conv.weight": "shortconv.conv.weight",
    }[suffix]
    custom = suffix.replace("self_attn.", "attn.").replace("feed_forward.", "ff.")
    return f"lfm.layers.{layer}.{custom}", f"blk.{layer}.{official}"


def load_gguf_weights(model, path):
    reader = GGUFReader(path)
    tensors = {t.name: t for t in reader.tensors}
    used = set()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            candidates = tensor_names(name)
            matches = [n for n in candidates if n in tensors]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one GGUF weight for {name}: {matches}"
                )
            tensor = tensors[matches[0]]
            array = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
            # HF depthwise kernel includes a singleton channel axis.
            shape = tuple(parameter.shape)
            if array.shape != shape and not (
                name.endswith("conv.conv.weight")
                and shape == (array.shape[0], 1, array.shape[-1])
            ):
                raise ValueError(
                    f"Shape mismatch {name}: GGUF {array.shape}, HF {shape}"
                )
            # Replace storage rather than writing into the checkpoint mmap.
            # Both tied-head attributes reference the same Parameter object.
            parameter.data = torch.from_numpy(array.reshape(shape))
            used.add(tensor.name)
    if used != set(tensors):
        raise ValueError(f"Unused GGUF weights: {sorted(set(tensors) - used)}")
    if model.lm_head.weight.data_ptr() != model.lfm2.embed_tokens.weight.data_ptr():
        raise ValueError("The official masked-LM head must be tied to embeddings")
    print(
        f"Loaded {len(used)} exact GGUF weights into FP32 reference from {path}",
        flush=True,
    )
