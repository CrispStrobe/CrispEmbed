#!/usr/bin/env python3
"""Pinned original vision model with non-square multi-image/multi-frame grids."""
import json
import os
import subprocess
from pathlib import Path


def main():
    import gguf
    import numpy as np
    import torch
    import transformers
    from transformers import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    if transformers.__version__ != "5.5.4":
        raise ValueError("Pinned original Transformers implementation required")
    torch.manual_seed(3512)
    torch.set_num_threads(4)
    out, scratch = Path(os.environ["HEAVY_OUT"]), Path(os.environ["HEAVY_SCRATCH"])
    repo = Path(__file__).resolve().parents[2]
    config = Qwen3_5VisionConfig(depth=12, hidden_size=24, intermediate_size=48, num_heads=3,
                               patch_size=2, temporal_patch_size=1, spatial_merge_size=2,
                               out_hidden_size=16, num_position_embeddings=49,
                               hidden_act="gelu_pytorch_tanh")
    config._attn_implementation = "eager"
    model = Qwen3_5VisionModel(config).eval()
    grid = torch.tensor([[1, 4, 6], [2, 6, 4]])
    pixels = torch.randn(72, 12)
    # Distinct frame distributions expose accidental attention across frames.
    pixels[24:48] += 2
    pixels[48:] -= 2
    weights = gguf.GGUFWriter(str(scratch / "synthetic.gguf"), "qwen35_synthetic_weights")
    reference = gguf.GGUFWriter(str(out / "synthetic-reference.gguf"), "qwen35_reference")
    report = {"scope": "synthetic original vision model; 12 layers, non-square grids, two images/three frames; not trained OCR accuracy",
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()}
    arrays = {}
    def save(name, value):
        array = value.detach().float().numpy().copy()
        if not np.isfinite(array).all():
            raise ValueError("Non-finite original reference: " + name)
        arrays[name] = array
    def hook(name):
        def capture(module, args, value):
            save("prefill.vision." + name, value)
        return capture
    handles = []
    try:
        for name, tensor in model.state_dict().items():
            value = tensor.float().numpy().copy()
            if name == "patch_embed.proj.weight":
                value = value.reshape(24, 12)
            weights.add_tensor("v." + name, value)
        save("input.pixel_values", pixels)
        save("input.image_grid_thw", grid)
        handles.append(model.patch_embed.register_forward_hook(hook("patch_embed")))
        for index, block in enumerate(model.blocks):
            handles.append(block.register_forward_hook(hook(f"layer.{index}")))
        for attr, field in [("norm", "norm"), ("linear_fc1", "up"), ("act_fn", "activation")]:
            handles.append(getattr(model.merger, attr).register_forward_hook(hook("merger." + field)))
        handles.append(model.merger.register_forward_hook(hook("merger")))
        def input_hook(module, args, kwargs):
            save("prefill.vision.embedded", args[0])
            save("prefill.vision.rotary_cos", kwargs["position_embeddings"][0])
            save("prefill.vision.rotary_sin", kwargs["position_embeddings"][1])
        handles.append(model.blocks[0].register_forward_pre_hook(input_hook, with_kwargs=True))
        original = model.fast_pos_embed_interpolate
        def position(grid):
            result = original(grid)
            save("prefill.vision.position", result)
            return result
        model.fast_pos_embed_interpolate = position
        with torch.inference_mode():
            result = model(pixels, grid_thw=grid)
            combined = result.pooler_output.clone()
            for handle in handles:
                handle.remove()
            handles.clear()
            model.fast_pos_embed_interpolate = original
            # Independent per-frame reference proves segmentation, rather than
            # assuming the model batch path respects the requested boundaries.
            separated = []
            start = 0
            for t, h, w in grid.tolist():
                for frame in range(t):
                    count = h*w
                    separated.append(model(pixels[start:start+count], grid_thw=torch.tensor([[1, h, w]])).pooler_output)
                    start += count
            error = (combined - torch.cat(separated)).abs().max().item()
            if error > 0.00002:
                raise ValueError(f"Original packed/separate frame discrepancy: {error}")
            report["official_packed_vs_separate_max_abs"] = error
        required = ["input.pixel_values", "input.image_grid_thw"]
        required += ["prefill.vision." + name for name in
                     ["patch_embed", "position", "embedded", "rotary_cos", "rotary_sin",
                      "merger.norm", "merger.up", "merger.activation", "merger"]]
        required += [f"prefill.vision.layer.{index}" for index in range(12)]
        if set(required) != set(arrays):
            raise ValueError("Missing or extra original stages")
        for name in required:
            reference.add_tensor(name, np.ascontiguousarray(arrays[name]))
        for writer in [weights, reference]:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
        build = scratch / "vision-build"
        subprocess.run(["cmake", "-S", str(repo), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF", "-DGGML_NATIVE=OFF",
                        "-DCRISPEMBED_BUILD_SHARED=OFF"], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", "test-qwen35-vision", "-j", "4"], check=True)
        native = subprocess.run([str(build / "test-qwen35-vision"), str(scratch / "synthetic.gguf"),
                                 str(out / "synthetic-reference.gguf"), "--synthetic"], capture_output=True, text=True)
        (out / "native-vision.log").write_text(native.stdout + native.stderr)
        print(native.stdout + native.stderr)
        native.check_returncode()
        report["result"] = "passed original-model native patch/position/12-layer vision/merger, multi-grid/frame and reset/invalid controls"
    except BaseException as error:
        report["result"], report["error"] = "failed", f"{type(error).__name__}: {error}"
        raise
    finally:
        for handle in handles:
            handle.remove()
        (out / "synthetic.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
