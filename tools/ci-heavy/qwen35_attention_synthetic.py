#!/usr/bin/env python3
"""Original HF full-attention modules, multi-batch causal/rotary/cache controls."""
import json
import os
import subprocess
from pathlib import Path


def main():
    import gguf
    import numpy as np
    import torch
    import transformers
    from transformers import Qwen3_5TextConfig
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding

    if transformers.__version__ != "5.5.4":
        raise ValueError("Pinned original Transformers implementation required")
    torch.manual_seed(3506)
    torch.set_num_threads(4)
    out, scratch = Path(os.environ["HEAVY_OUT"]), Path(os.environ["HEAVY_SCRATCH"])
    repo = Path(__file__).resolve().parents[2]
    config = Qwen3_5TextConfig(hidden_size=16, num_hidden_layers=24, num_attention_heads=8,
                             num_key_value_heads=2, head_dim=256, full_attention_interval=4,
                             rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                                              "partial_rotary_factor": 0.25,
                                              "mrope_section": [11, 11, 10], "mrope_interleaved": True})
    config._attn_implementation = "eager"
    rotary = Qwen3_5TextRotaryEmbedding(config)
    weights = gguf.GGUFWriter(str(scratch / "synthetic.gguf"), "qwen35_synthetic_weights")
    reference = gguf.GGUFWriter(str(out / "synthetic-reference.gguf"), "qwen35_reference")
    report = {"scope": "synthetic original HF attention; B=2 T=65+1, distinct multimodal axes, not checkpoint OCR",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
    try:
        prefix = torch.randn(2, 65, 16)
        continuation = torch.randn(2, 1, 16)
        # Distinct nonzero axes/batches expose broadcasting and rotary mistakes.
        coords = torch.stack([torch.arange(66).expand(2, -1),
                              torch.arange(66).expand(2, -1) * 3 + 7,
                              torch.arange(66).expand(2, -1) * 5 + 11]).clone()
        coords[:, 1] += 13
        for label, position in [("prefill", coords[:, :, :65]), ("cached", coords[:, :, 65:]),
                                ("full_prefix", coords)]:
            reference.add_tensor(label + ".decoder.position_ids", position.float().numpy().copy())
        for index, kind in enumerate(config.layer_types):
            if kind != "full_attention":
                continue
            block = Qwen3_5Attention(config, index).eval()
            with torch.no_grad():
                block.q_norm.weight.normal_(0, 0.15)
                block.k_norm.weight.normal_(0, 0.15)
            for name, tensor in block.state_dict().items():
                weights.add_tensor(f"l.layers.{index}.self_attn.{name}", tensor.float().numpy().copy())
            cache = DynamicCache(config=config)
            phase = ["prefill"]
            handles = []
            def hook(field, pre=False):
                def capture(module, args, value=None):
                    tensor = args[0] if pre else value
                    reference.add_tensor(f"{phase[0]}.attention.{index}.{field}",
                                         np.ascontiguousarray(tensor.detach().float().numpy()).copy())
                return capture
            for attr, field in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"),
                                ("q_norm", "q_norm"), ("k_norm", "k_norm")]:
                handles.append(getattr(block, attr).register_forward_hook(hook(field)))
            handles.append(block.o_proj.register_forward_pre_hook(hook("pooled", pre=True)))
            with torch.inference_mode():
                for label, hidden, position, past in [
                    ("prefill", prefix, coords[:, :, :65], cache),
                    ("cached", continuation, coords[:, :, 65:], cache),
                    ("full_prefix", torch.cat([prefix, continuation], dim=1), coords, None)
                ]:
                    phase[0] = label
                    reference.add_tensor(f"{label}.attention.{index}.input", hidden.numpy().copy())
                    tokens = hidden.shape[1]
                    total = 66 if label == "cached" else tokens
                    mask = torch.full((tokens, total), torch.finfo(torch.float32).min)
                    mask = torch.triu(mask, diagonal=1 + total - tokens)[None, None]
                    result = block(hidden, position_embeddings=rotary(hidden, position),
                                   attention_mask=mask, past_key_values=past)[0]
                    reference.add_tensor(f"{label}.attention.{index}.output", result.numpy().copy())
                    if past is not None:
                        for field in ["keys", "values"]:
                            reference.add_tensor(f"{label}.cache.{index}.{field}",
                                                 getattr(cache.layers[index], field).numpy().copy())
            for handle in handles:
                handle.remove()
        for writer in [weights, reference]:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
        build = scratch / "attention-build"
        subprocess.run(["cmake", "-S", str(repo), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF", "-DGGML_NATIVE=OFF",
                        "-DCRISPEMBED_BUILD_SHARED=OFF"], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "test-qwen35-attention", "-j", "4"], check=True)
        native = subprocess.run([str(build / "test-qwen35-attention"), str(scratch / "synthetic.gguf"),
                                 str(out / "synthetic-reference.gguf")], capture_output=True, text=True)
        (out / "native-attention.log").write_text(native.stdout + native.stderr)
        print(native.stdout + native.stderr)
        native.check_returncode()
        report["result"] = "passed original-module projections/norms/rotary/gated pooling/output, native KV cache and full prefix"
    except BaseException as error:
        report["result"], report["error"] = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "synthetic.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
