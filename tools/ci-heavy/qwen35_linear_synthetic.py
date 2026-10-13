#!/usr/bin/env python3
"""Checkpoint-free compiled projection/core parity using original HF modules."""

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
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    if transformers.__version__ != "5.5.4":
        raise ValueError("Pinned original Transformers implementation required")
    torch.manual_seed(3518)
    torch.set_num_threads(4)
    out, scratch = Path(os.environ["HEAVY_OUT"]), Path(os.environ["HEAVY_SCRATCH"])
    repo = Path(__file__).resolve().parents[2]
    config = Qwen3_5TextConfig(hidden_size=16, num_hidden_layers=24, linear_num_key_heads=1,
                             linear_num_value_heads=1, linear_key_head_dim=128, linear_value_head_dim=128,
                             linear_conv_kernel_dim=4, full_attention_interval=4)
    weights = gguf.GGUFWriter(str(scratch / "synthetic.gguf"), "qwen35_synthetic_weights")
    reference = gguf.GGUFWriter(str(out / "synthetic-reference.gguf"), "qwen35_reference")
    report = {"scope": "synthetic original HF modules; not OCR/checkpoint parity",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
    try:
        for index, kind in enumerate(config.layer_types):
            if kind != "linear_attention":
                continue
            block = Qwen3_5GatedDeltaNet(config, index).eval()
            for name, tensor in block.state_dict().items():
                weights.add_tensor(f"l.layers.{index}.linear_attn.{name}", tensor.float().numpy())
            cache = DynamicCache(config=config)
            phase = ["prefill"]
            handles = []
            def hook(field):
                def capture(module, args, value):
                    reference.add_tensor(f"{phase[0]}.linear.{index}.{field}",
                                         np.ascontiguousarray(value.detach().float().numpy()))
                return capture
            for attr, field in [("in_proj_qkv", "qkv"), ("in_proj_z", "z"),
                                ("in_proj_a", "a"), ("in_proj_b", "b"), ("norm", "norm")]:
                handles.append(getattr(block, attr).register_forward_hook(hook(field)))
            with torch.inference_mode():
                for label, tokens in [("prefill", 65), ("cached", 1)]:
                    phase[0] = label
                    hidden = torch.randn(2, tokens, 16)
                    reference.add_tensor(f"{label}.linear.{index}.input", hidden.numpy())
                    result = block(hidden, cache_params=cache)
                    reference.add_tensor(f"{label}.linear.{index}.output", result.numpy())
                    layer = cache.layers[index]
                    for name in ["conv_states", "recurrent_states"]:
                        reference.add_tensor(f"{label}.cache.{index}.{name}",
                                             np.ascontiguousarray(getattr(layer, name).numpy()))
            for handle in handles:
                handle.remove()
        for writer in [weights, reference]:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
        build = scratch / "linear-build"
        subprocess.run(["cmake", "-S", str(repo), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF", "-DGGML_NATIVE=OFF",
                        "-DCRISPEMBED_BUILD_SHARED=OFF"], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "test-qwen35-linear", "-j", "4"], check=True)
        native = subprocess.run([str(build / "test-qwen35-linear"), str(scratch / "synthetic.gguf"),
                                 str(out / "synthetic-reference.gguf")], capture_output=True, text=True)
        (out / "native-linear.log").write_text(native.stdout + native.stderr)
        print(native.stdout + native.stderr)
        native.check_returncode()
        report["result"] = "passed original-module projection/core/state/output comparisons"
    except BaseException as error:
        report["result"], report["error"] = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        (out / "synthetic.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
