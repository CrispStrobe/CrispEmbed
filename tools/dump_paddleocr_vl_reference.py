#!/usr/bin/env python3
"""Dump PaddleOCR-VL reference activations for crispembed-diff.

Manually preprocesses the image (avoiding AutoProcessor which requires
torchvision) and runs the model forward pass to capture intermediate
activations.

Usage:
    PYTHONNOUSERSITE=1 HF_HUB_CACHE=/mnt/storage/huggingface/hub \\
    python3 tools/dump_paddleocr_vl_reference.py \\
        --model PaddlePaddle/PaddleOCR-VL \\
        --image test.png \\
        --output /mnt/storage/gguf-models/paddleocr-vl-ref.gguf
"""

import argparse
import json
import math
import os
import sys
import types
from pathlib import Path

# Block mlx (causes ImportError on non-macOS)
_mlx_stub = types.ModuleType('mlx')
_mlx_stub.__spec__ = types.SimpleNamespace(name='mlx', submodule_search_locations=[])
sys.modules['mlx'] = _mlx_stub
_mlx_core = types.ModuleType('mlx.core')
_mlx_core.__spec__ = types.SimpleNamespace(name='mlx.core', submodule_search_locations=[])
_mlx_core.array = type('array', (), {})  # dummy array class for einops backend check
sys.modules['mlx.core'] = _mlx_core

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "ggml" / "scripts"))
import gguf


# ── Image preprocessing (reimplements PaddleOCRVLImageProcessor) ─────

def smart_resize(height, width, factor=28, min_pixels=147384, max_pixels=2822400):
    """Resize to nearest multiple of factor within pixel budget."""
    if height < factor:
        height = factor
    if width < factor:
        width = factor
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return max(h_bar, factor), max(w_bar, factor)


def preprocess_image(image_path, patch_size=14, merge_size=2,
                     min_pixels=147384, max_pixels=2822400):
    """Load image, resize, normalize, extract patches."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    factor = patch_size * merge_size  # 28
    h_new, w_new = smart_resize(h, w, factor, min_pixels, max_pixels)

    # Bicubic resize
    img = img.resize((w_new, h_new), Image.BICUBIC)

    # To tensor [C, H, W], rescale to [0,1], normalize to [-1,1]
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)  # [C, H, W]

    # Extract patches: [C, H, W] → [N, C, P, P]
    C = 3
    grid_h = h_new // patch_size
    grid_w = w_new // patch_size

    patches = arr.reshape(C, grid_h, patch_size, grid_w, patch_size)
    patches = patches.transpose(1, 3, 0, 2, 4)  # [grid_h, grid_w, C, P, P]
    patches = patches.reshape(grid_h * grid_w, C * patch_size * patch_size)

    return patches, (1, grid_h, grid_w)


def build_position_ids(grid_thw, merge_size=2):
    """Build 3D position IDs for image tokens (temporal, height, width)."""
    t, h, w = grid_thw
    # After merge: h_m = h // merge, w_m = w // merge
    h_m = h // merge_size
    w_m = w // merge_size
    n_tokens = t * h_m * w_m

    # Position IDs: (3, n_tokens)
    pos_ids = np.zeros((3, n_tokens), dtype=np.int32)
    idx = 0
    for ti in range(t):
        for hi in range(h_m):
            for wi in range(w_m):
                pos_ids[0, idx] = ti      # temporal
                pos_ids[1, idx] = hi      # height
                pos_ids[2, idx] = wi      # width
                idx += 1
    return pos_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--prompt", default="OCR:", help="Text prompt")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    args = parser.parse_args()

    cache_dir = os.environ.get("HF_HUB_CACHE",
                               os.path.expanduser("~/.cache/huggingface/hub"))

    # ── Preprocess image ──
    # Use small min_pixels to keep patch count low for VPS RAM
    print(f"Preprocessing image: {args.image}")
    patches, grid_thw = preprocess_image(args.image, min_pixels=784)
    print(f"  Patches: {patches.shape}, grid_thw: {grid_thw}")

    # ── Load tokenizer ──
    print(f"Loading tokenizer: {args.model}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                         cache_dir=cache_dir)

    # Build prompt with image placeholders
    # Template: <|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|>...<|IMAGE_END|>\nOCR:\nAssistant:
    n_image_tokens = (grid_thw[1] // 2) * (grid_thw[2] // 2)  # after 2x2 merge
    image_placeholder_id = 100295
    vision_start_id = 101305
    vision_end_id = 101306
    bos_id = 100273  # <|begin_of_sentence|>

    # Encode text parts
    user_prefix = tok.encode("User: ", add_special_tokens=False)
    prompt_tokens = tok.encode(f"\n{args.prompt}\n", add_special_tokens=False)
    assistant_tokens = tok.encode("Assistant:", add_special_tokens=False)

    # Build full sequence
    input_ids = ([bos_id] + user_prefix +
                 [vision_start_id] +
                 [image_placeholder_id] * n_image_tokens +
                 [vision_end_id] +
                 prompt_tokens +
                 assistant_tokens)
    print(f"  Input IDs: {len(input_ids)} tokens ({n_image_tokens} image placeholders)")

    # ── Load model ──
    print(f"Loading model: {args.model}")
    # Monkey-patch to avoid torchvision import in custom code
    import types
    import importlib
    try:
        # Create a minimal torchvision.transforms stub
        tv_module = types.ModuleType("torchvision")
        tv_transforms = types.ModuleType("torchvision.transforms")

        class InterpolationMode:
            BICUBIC = 3
            BILINEAR = 2
            NEAREST = 0
            NEAREST_EXACT = 0
            LANCZOS = 1
            BOX = 4
            HAMMING = 5

        tv_transforms.InterpolationMode = InterpolationMode
        tv_module.transforms = tv_transforms

        # Also create torchvision.io, models etc. as empty modules
        for submod in ["io", "models", "ops", "datasets", "utils",
                       "_meta_registrations"]:
            setattr(tv_module, submod, types.ModuleType(f"torchvision.{submod}"))

        sys.modules["torchvision"] = tv_module
        sys.modules["torchvision.transforms"] = tv_transforms
        for submod in ["io", "models", "ops", "datasets", "utils",
                       "_meta_registrations"]:
            sys.modules[f"torchvision.{submod}"] = getattr(tv_module, submod)
    except Exception as e:
        print(f"  Warning: torchvision stub failed: {e}")

    # Monkey-patch create_causal_mask for transformers compat
    # (model code uses 'inputs_embeds', newer transformers renamed to 'input_embeds')
    import transformers.masking_utils as _mask_utils
    if hasattr(_mask_utils, 'create_causal_mask'):
        _orig_ccm = _mask_utils.create_causal_mask
        def _patched_ccm(*args, **kwargs):
            if 'inputs_embeds' in kwargs and 'input_embeds' not in kwargs:
                kwargs['input_embeds'] = kwargs.pop('inputs_embeds')
            return _orig_ccm(*args, **kwargs)
        _mask_utils.create_causal_mask = _patched_ccm

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, cache_dir=cache_dir,
        torch_dtype=torch.float16, device_map="cpu")
    model.eval()

    # ── Hook activations ──
    captured = {}

    def hook_post_layernorm(module, inp, out):
        captured['vision_output'] = out.detach().float().cpu()

    def hook_projector(module, inp, out):
        if isinstance(out, (list, tuple)):
            captured['projector_output'] = torch.cat([o.detach().float().cpu() for o in out], dim=0)
        else:
            captured['projector_output'] = out.detach().float().cpu()

    model.visual.vision_model.post_layernorm.register_forward_hook(hook_post_layernorm)
    model.mlp_AR.register_forward_hook(hook_projector)

    # ── Forward pass ──
    print("Running forward pass...")
    N = patches.shape[0]
    # Model's forward does unsqueeze(0) internally, so pass [N, C, H, W] (4D)
    pixel_values = torch.from_numpy(patches).reshape(N, 3, 14, 14).to(torch.float16)
    # pixel_values: [N, 3, 14, 14]

    input_ids_t = torch.tensor([input_ids], dtype=torch.long)
    image_grid_thw = torch.tensor([list(grid_thw)], dtype=torch.long)  # [1, 3]

    # Attention mask
    attn_mask = torch.ones_like(input_ids_t)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_t,
            attention_mask=attn_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=False,
        )

    logits = outputs.logits.detach().float().cpu().numpy()
    print(f"  Logits shape: {logits.shape}")

    # ── Write reference GGUF ──
    writer = gguf.GGUFWriter(args.output, "crispembed_diff")

    if 'vision_output' in captured:
        v = captured['vision_output'].numpy().astype(np.float32)
        if v.ndim == 3:
            v = v[0]
        writer.add_tensor("vision_output", v)
        print(f"  vision_output: {v.shape}")

    if 'projector_output' in captured:
        p = captured['projector_output'].numpy().astype(np.float32)
        if p.ndim == 3:
            p = p[0]
        writer.add_tensor("projector_output", p)
        print(f"  projector_output: {p.shape}")

    # Last-position logits
    last_logits = logits[0, -1:, :].astype(np.float32)
    writer.add_tensor("logits", last_logits)
    print(f"  logits: {last_logits.shape}")

    # Input token IDs
    ids_np = np.array(input_ids, dtype=np.int32)
    writer.add_tensor("input_ids", ids_np)
    print(f"  input_ids: {ids_np.shape}")

    # Image grid THW
    grid_np = np.array(list(grid_thw), dtype=np.int32)
    writer.add_tensor("image_grid_thw", grid_np)
    print(f"  image_grid_thw: {grid_np}")

    # Pixel values shape
    pv_shape = np.array([1, N, 3, 14, 14], dtype=np.int32)
    writer.add_tensor("pixel_values_shape", pv_shape)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nOutput: {args.output} ({out_size:.1f} MB)")


if __name__ == "__main__":
    main()
