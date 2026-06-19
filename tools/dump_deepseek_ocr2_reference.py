#!/usr/bin/env python3
"""Dump DeepSeek-OCR-2 reference vision intermediates for crispembed-diff parity.

The novel part of the CrispEmbed port is the vision pathway:

    image_ori (1,3,1024,1024)
      -> sam_model        (SAM-ViT-B)            -> "sam_output"
      -> qwen2_model      (Qwen2 24L encoder)    -> "qwen2_enc_output"
      -> projector        (linear 896->1280)     -> "projector_output"  (256 tokens)

This script reproduces the reference preprocessing exactly (ImageOps.pad to
1024x1024 + ToTensor + Normalize(0.5,0.5)) and runs *only* those three modules,
in float32 on CPU. That avoids the model's CUDA-hardcoded `infer()` path and the
3.4B MoE decoder, so it runs on a 16 GB Mac without a GPU.

The full LLM/logits stages need the MoE forward (CUDA-only in upstream); they are
out of scope here — vision parity is the part the C++ runtime had to re-derive.

The output is a GGUF archive (what `crispembed_diff::Ref::load()` reads): each
stage is stored as a named F32 tensor.

Usage:
    python tools/dump_deepseek_ocr2_reference.py \
        --model-dir /path/to/DeepSeek-OCR-2 \
        --image test.png \
        --output /tmp/deepseek-ocr2-ref.gguf
"""

import argparse
from pathlib import Path
import numpy as np


def squeeze_leading(data: np.ndarray) -> np.ndarray:
    """Drop leading singleton dims (e.g. batch) so element count matches C++."""
    data = np.ascontiguousarray(data, dtype=np.float32)
    while data.ndim > 2 and data.shape[0] == 1:
        data = data[0]
    return data


def as_numpy(x):
    """bf16/fp16/fp32 tensor (or tuple[0]) -> float32 numpy."""
    import torch
    if isinstance(x, (tuple, list)):
        x = x[0]
    if hasattr(x, "last_hidden_state"):
        x = x.last_hidden_state
    assert isinstance(x, torch.Tensor), f"unexpected hook output: {type(x)}"
    return x.detach().to(torch.float32).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="HF model dir (safetensors + config + *.py)")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--output", required=True, help="Output binary reference file")
    p.add_argument("--base-size", type=int, default=1024, help="Global-view square size")
    args = p.parse_args()

    import importlib.util
    import sys
    import torch
    from PIL import Image, ImageOps
    from torchvision import transforms
    from safetensors import safe_open

    torch.manual_seed(0)
    model_dir = Path(args.model_dir)

    # Instantiate ONLY the vision modules from deepencoderv2.py. We deliberately
    # avoid AutoModel/DeepseekOCR2: its modeling_deepseekv2.py (the MoE LLM) is
    # written for an old transformers and fails to import on current versions
    # (e.g. LlamaFlashAttention2 was removed). deepencoderv2.py is self-contained
    # (torch + transformers Qwen2Model), so the vision tower loads cleanly.
    print("Importing deepencoderv2 (vision modules only)...", flush=True)
    spec = importlib.util.spec_from_file_location(
        "deepencoderv2", str(model_dir / "deepencoderv2.py"))
    dev2 = importlib.util.module_from_spec(spec)
    sys.modules["deepencoderv2"] = dev2
    spec.loader.exec_module(dev2)

    # Dict-like cfg: MlpProjector uses both attribute access and .get().
    class Cfg(dict):
        def __getattr__(self, k):
            return self[k]

    sam = dev2.build_sam_vit_b()
    qwen2 = dev2.build_qwen2_decoder_as_encoder()  # 24L/896d/14H/2KV/4864, defaults
    projector = dev2.MlpProjector(
        Cfg(projector_type="linear", input_dim=896, n_embed=1280))

    # Load weights by prefix from the safetensors (fp32, strict to verify mapping).
    st_path = str(model_dir / "model-00001-of-000001.safetensors")

    def load_prefix(module, prefix):
        sd = {}
        with safe_open(st_path, framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    sd[k[len(prefix):]] = f.get_tensor(k).float()
        module.load_state_dict(sd, strict=True)
        module.eval().float()
        print(f"  loaded {len(sd)} tensors into {prefix}", flush=True)

    load_prefix(sam, "model.sam_model.")
    load_prefix(qwen2, "model.qwen2_model.")
    load_prefix(projector, "model.projector.")
    with safe_open(st_path, framework="pt") as f:
        vsep = f.get_tensor("model.view_seperator").float()

    # --- Preprocess exactly like infer(): pad to base_size, ToTensor, Normalize ---
    img = Image.open(args.image).convert("RGB")
    img = ImageOps.exif_transpose(img)
    print(f"Image: {img.size[0]}x{img.size[1]}", flush=True)
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    global_view = ImageOps.pad(
        img, (args.base_size, args.base_size),
        color=tuple(int(x * 255) for x in mean),
    )
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    image_ori = tf(global_view).unsqueeze(0).float()  # (1,3,base,base)
    print(f"image_ori: {tuple(image_ori.shape)}", flush=True)

    captures = {}

    def hook(name):
        def fn(module, inp, out):
            captures[name] = as_numpy(out)
        return fn

    sam.register_forward_hook(hook("sam_output"))
    qwen2.register_forward_hook(hook("qwen2_enc_output"))
    projector.register_forward_hook(hook("projector_output"))

    # Per-layer qwen2 hidden states (full [vis+query] sequence, pre-final-norm)
    # for bisecting the encoder. qwen2.model.model = the inner Qwen2Model.
    def pre_hook(name):
        # Captures a module's INPUT (args[0]); used on post_attention_layernorm
        # to grab the post-attention hidden state (residual + attn, pre-FFN).
        def fn(module, args):
            captures[name] = as_numpy(args[0])
        return fn
    try:
        qlayers = qwen2.model.model.layers
        for i, layer in enumerate(qlayers):
            layer.register_forward_hook(hook(f"qwen2_layer_{i}"))
            layer.post_attention_layernorm.register_forward_pre_hook(
                pre_hook(f"qwen2_layer_{i}_postattn"))
        print(f"  hooked {len(qlayers)} qwen2 layers (+ postattn)", flush=True)
    except AttributeError as e:
        print(f"  (no per-layer qwen2 hooks: {e})", flush=True)

    print("Running vision pathway (sam -> qwen2 -> projector)...", flush=True)
    with torch.no_grad():
        g1 = sam(image_ori)
        g2 = qwen2(g1)
        g3 = projector(g2)

    # The C++ runtime stores every stage token-major ([token, channel], token
    # outer, row-major over the HxW grid). SAM's module output is channel-major
    # (C, H, W); transpose it to (H*W, C) so the flat order matches and cosine
    # is meaningful. qwen2/projector are already (tokens, channel).
    so = np.squeeze(captures["sam_output"])  # drop batch -> (C, H, W)
    if so.ndim == 3:  # (C, H, W) -> (H*W, C), token-major to match C++
        so = so.transpose(1, 2, 0).reshape(-1, so.shape[0])
    captures["sam_output"] = so

    # Final 256 image features + the learned view separator (what gets spliced
    # into the LLM prompt). Stored for completeness; C++ "projector_output" is g3.
    feats = as_numpy(g3).reshape(-1, as_numpy(g3).shape[-1])
    captures["vision_features"] = feats
    captures["view_separator"] = vsep.detach().to(torch.float32).cpu().numpy()

    print(f"Writing reference GGUF to {args.output}...", flush=True)
    import gguf
    writer = gguf.GGUFWriter(args.output, arch="deepseek_ocr2_ref")
    for name in sorted(captures.keys()):
        data = squeeze_leading(captures[name])
        writer.add_tensor(name, data, raw_dtype=gguf.GGMLQuantizationType.F32)
        print(f"  {name}: shape={data.shape}, "
              f"range=[{data.min():.4f}, {data.max():.4f}]", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {len(captures)} stages to {args.output}", flush=True)


if __name__ == "__main__":
    main()
