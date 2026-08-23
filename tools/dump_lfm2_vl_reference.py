#!/usr/bin/env python3
"""Dump per-stage LFM2.5-VL-3B reference activations for crispembed parity testing.

Architecture summary:
  Vision encoder: SigLIP2 NaFlex — 27L, 1152d, patches flat [n_patches, 768],
                  position embeddings bilinear-interpolated from 16x16 grid
  Projector: pixel_unshuffle(factor=2) → Linear(4608,2048) → GELU →
             Linear(2048,2048)
  LFM2 LLM: hybrid conv+attention backbone, 30L, 2048d
             mixed layer_types [conv,conv,attn,...] (22 conv, 8 attn)
             ShortConv: in_proj → split(B,C,x) → B*x → causal depthwise
             conv1d(k=3) → C*conv_out → out_proj
             Attention: QK RMSNorm (per-head, dim=64, eps=1e-5) before RoPE,
             GQA 32/8 heads, SwiGLU MLP

Stages captured (written to reference GGUF):
  Vision encoder:
    vis_patch_embed        (n_patches, 1152)   after patch Linear + pos embed
    vis_layer_{i}          (n_patches, 1152)   first --max-vis-layers blocks
    vis_post_ln            (n_patches, 1152)   after final vision LayerNorm
  Projector:
    projector_out          (n_merged, 2048)    after pixel_unshuffle + MLP
  LLM:
    llm_embed              (T, 2048)           text+image spliced embedding
    llm_layer_{i}          (T, 2048)           first --max-llm-layers outputs
    llm_logits_last        (1, vocab_size)     logits at last position
  Metadata:
    generated_text         (string KV)         greedy-decoded first token text

Uses transformers forward hooks (NOT pure numpy) because the ShortConv
architecture is too complex to reimplement in numpy reliably.

Usage:
    python tools/dump_lfm2_vl_reference.py \\
        --model LiquidAI/LFM2.5-VL-3B \\
        --image /tmp/test.png \\
        --output /tmp/lfm2-vl-ref.gguf \\
        --max-vis-layers 4 \\
        --max-llm-layers 4

Requires: transformers, torch, gguf, Pillow
Memory: 3B in f32 = ~12 GB; needs GPU ≥16 GB (Kaggle T4) or CPU with enough RAM.
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "ggml" / "scripts"))
try:
    import gguf
except ImportError:
    print("pip install gguf", file=sys.stderr)
    sys.exit(1)


# ── Hook helpers ──────────────────────────────────────────────────────

def _capture_output(store, key):
    """Return a forward hook that stores the module output under store[key]."""
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, torch.Tensor):
            store[key] = out.detach().float().cpu()
    return hook


def _capture_input(store, key):
    """Return a forward hook that stores the first input tensor under store[key]."""
    def hook(module, inp, out):
        if isinstance(inp, (list, tuple)):
            t = inp[0]
        else:
            t = inp
        if isinstance(t, torch.Tensor):
            store[key] = t.detach().float().cpu()
    return hook


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dump LFM2.5-VL-3B per-stage reference activations")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-VL-3B",
                        help="HF model ID or local path")
    parser.add_argument("--image", required=True,
                        help="Path to test image")
    parser.add_argument("--output", default="lfm2-vl-ref.gguf",
                        help="Output GGUF path")
    parser.add_argument("--max-vis-layers", type=int, default=4,
                        help="Number of vision layers to dump (default 4)")
    parser.add_argument("--max-llm-layers", type=int, default=4,
                        help="Number of LLM layers to dump (default 4)")
    parser.add_argument("--prompt", default="OCR this image. Output the text content.",
                        help="Text prompt to use")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Image:  {args.image}")

    # ── Load processor ───────────────────────────────────────────────
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image

    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(args.model)

    # ── Load model ───────────────────────────────────────────────────
    # Use bf16 to keep memory manageable (f32 OOMs on a 16 GB GPU for 3B params).
    # If CUDA is available but the installed PyTorch doesn't support the GPU's
    # compute capability (e.g. P100/sm_60 + torch built for sm_70+), fall back
    # to CPU rather than crashing on the first kernel launch.
    if device == "cuda":
        try:
            torch.zeros(1, device="cuda")
        except Exception as e:
            print(f"CUDA probe failed ({e}), falling back to CPU")
            device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading model ({dtype}, device={device})...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print(f"  Model class: {type(model).__name__}")

    # ── Preprocess image + prompt ────────────────────────────────────
    print(f"\nPreprocessing image: {args.image}")
    img = Image.open(args.image).convert("RGB")
    print(f"  Size: {img.size}")

    # Build the chat-style messages list that the processor expects
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": args.prompt},
            ],
        }
    ]

    # apply_chat_template is the standard path for LFM2.5-VL
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback: some processors use a different signature
        text = args.prompt

    inputs = processor(
        text=[text],
        images=[img],
        return_tensors="pt",
    ).to(device)

    print(f"  Input IDs shape: {inputs['input_ids'].shape}")
    if "pixel_values" in inputs:
        print(f"  Pixel values shape: {inputs['pixel_values'].shape}")

    # ── Discover model structure ─────────────────────────────────────
    # LFM2.5-VL: model.vision_tower / model.vision_model for vision encoder
    #            model.multi_modal_projector for projector
    #            model.language_model / model.model for LLM
    # The exact attribute names depend on the transformers version; probe them.
    def find_attr(obj, *candidates):
        for name in candidates:
            if hasattr(obj, name):
                return getattr(obj, name), name
        return None, None

    # LFM2.5-VL wraps everything in model.model (Lfm2VlModel).
    # Try model.model.vision_tower first, then model.vision_tower, etc.
    inner_model = getattr(model, "model", model)
    vision_enc, vis_attr = find_attr(
        inner_model,
        "vision_tower", "vision_model", "visual_encoder", "encoder")
    if vision_enc is None:
        vision_enc, vis_attr = find_attr(
            model,
            "vision_tower", "vision_model", "visual_encoder", "encoder")
    projector, proj_attr = find_attr(
        inner_model,
        "multi_modal_projector", "projector", "vision_projector",
        "mm_projector", "connector")
    if projector is None:
        projector, proj_attr = find_attr(
            model,
            "multi_modal_projector", "projector", "vision_projector",
            "mm_projector", "connector")
    llm, llm_attr = find_attr(
        inner_model,
        "language_model", "model", "decoder", "lm")
    if llm is None:
        llm, llm_attr = find_attr(
            model,
            "language_model", "model", "decoder", "lm")

    print(f"\nModel structure (inner_model={type(inner_model).__name__}):")
    print(f"  Top-level attrs: {[a for a in dir(inner_model) if not a.startswith('_') and isinstance(getattr(inner_model, a, None), torch.nn.Module)][:20]}")
    print(f"  vision encoder: {vis_attr} ({type(vision_enc).__name__ if vision_enc else 'NOT FOUND'})")
    print(f"  projector:      {proj_attr} ({type(projector).__name__ if projector else 'NOT FOUND'})")
    print(f"  LLM:            {llm_attr} ({type(llm).__name__ if llm else 'NOT FOUND'})")

    # ── Register hooks ───────────────────────────────────────────────
    captured = {}
    hooks = []

    # Vision encoder: patch embed output
    vis_embed_module, _ = find_attr(
        vision_enc if vision_enc else model,
        "patch_embedding", "patch_embed", "embeddings",
        "patch_projection", "conv_patch_embedding")
    if vis_embed_module is not None:
        hooks.append(vis_embed_module.register_forward_hook(
            _capture_output(captured, "vis_patch_embed")))
        print(f"  Hook: vis_patch_embed on {type(vis_embed_module).__name__}")

    # Vision encoder: per-layer outputs
    # Try to find the transformer layers list
    vis_layers_module = None
    for candidate_path in [
        "encoder.layers", "layers", "encoder.blocks", "blocks",
        "model.encoder.layers", "model.layers",
    ]:
        parts = candidate_path.split(".")
        cur = vision_enc
        ok = True
        for part in parts:
            if cur is not None and hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and cur is not None and hasattr(cur, "__len__"):
            vis_layers_module = cur
            print(f"  Vision layers found at: vision_enc.{candidate_path} "
                  f"({len(vis_layers_module)} layers)")
            break

    if vis_layers_module is not None:
        n_vis_dump = min(args.max_vis_layers, len(vis_layers_module))
        for i in range(n_vis_dump):
            hooks.append(vis_layers_module[i].register_forward_hook(
                _capture_output(captured, f"vis_layer_{i}")))
        print(f"  Hooks: vis_layer_0..{n_vis_dump-1}")

    # Vision encoder: post LayerNorm (final vision output before projector)
    # SigLIP2 typically has a post_layernorm or layernorm at the end
    vis_post_ln, _ = find_attr(
        vision_enc if vision_enc else model,
        "post_layernorm", "final_layer_norm", "norm", "layernorm",
        "model.post_layernorm", "encoder.post_layernorm")
    if vis_post_ln is not None:
        hooks.append(vis_post_ln.register_forward_hook(
            _capture_output(captured, "vis_post_ln")))
        print(f"  Hook: vis_post_ln on {type(vis_post_ln).__name__}")

    # Projector output
    if projector is not None:
        hooks.append(projector.register_forward_hook(
            _capture_output(captured, "projector_out")))
        print(f"  Hook: projector_out on {type(projector).__name__}")

    # LLM: embedding output (the spliced embedding after image tokens are inserted)
    # We hook the embed_tokens module to get the raw text embedding,
    # then separately capture the model's forward input to get the spliced version.
    llm_embed_module = None
    if llm is not None:
        llm_embed_module, _ = find_attr(
            llm, "embed_tokens", "wte", "tok_embeddings", "embedding")
    if llm_embed_module is not None:
        # We want the post-splice input, so capture inputs_embeds at the LLM model level.
        # Hook the first LLM decoder layer's input instead (which sees the spliced embeds).
        pass  # handled below after finding LLM layers

    # LLM layers
    llm_layers_module = None
    if llm is not None:
        for candidate_path in [
            "model.layers", "layers", "decoder.layers",
            "transformer.h", "blocks",
        ]:
            parts = candidate_path.split(".")
            cur = llm
            ok = True
            for part in parts:
                if cur is not None and hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    ok = False
                    break
            if ok and cur is not None and hasattr(cur, "__len__"):
                llm_layers_module = cur
                print(f"  LLM layers found at: llm.{candidate_path} "
                      f"({len(llm_layers_module)} layers)")
                break

    if llm_layers_module is not None:
        n_llm_dump = min(args.max_llm_layers, len(llm_layers_module))
        # Capture llm_embed from the first layer's input (sees spliced embeds)
        hooks.append(llm_layers_module[0].register_forward_hook(
            _capture_input(captured, "llm_embed")))
        for i in range(n_llm_dump):
            hooks.append(llm_layers_module[i].register_forward_hook(
                _capture_output(captured, f"llm_layer_{i}")))
        print(f"  Hooks: llm_embed (from layer 0 input), llm_layer_0..{n_llm_dump-1}")

    # ── Forward pass (prefill only, no generation) ───────────────────
    print("\nRunning prefill forward pass...")
    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, T, vocab_size)

    # Capture llm_logits_last: logits at last position
    last_logits = logits[0, -1:, :].detach().float().cpu().numpy()  # (1, V)
    captured_numpy = {k: v.numpy() if isinstance(v, torch.Tensor) else v
                      for k, v in captured.items()}

    print(f"\nCaptured stages: {sorted(captured_numpy.keys())}")
    print(f"Last logits shape: {last_logits.shape}")

    # Greedy decode the first generated token
    next_token_id = int(last_logits[0].argmax())
    try:
        generated_text = processor.tokenizer.decode(
            [next_token_id], skip_special_tokens=True)
    except Exception:
        generated_text = f"<token_{next_token_id}>"
    print(f"Next token: {next_token_id} → {generated_text!r}")

    # Remove hooks
    for h in hooks:
        h.remove()
    hooks.clear()

    # ── Memory management: free model before writing large GGUF ─────
    del model
    del outputs
    del logits
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Model freed, writing GGUF...")

    # ── Write reference GGUF ─────────────────────────────────────────
    print(f"\nWriting reference GGUF: {args.output}")
    writer = gguf.GGUFWriter(str(args.output), "lfm2vl_ref")

    # Metadata
    writer.add_string("general.name", "lfm2vl_reference")
    writer.add_string("lfm2vl.model_id", args.model)
    writer.add_string("lfm2vl.image_path", str(args.image))
    writer.add_string("lfm2vl.prompt", args.prompt)
    writer.add_string("lfm2vl.generated_text", generated_text)
    writer.add_uint32("lfm2vl.max_vis_layers", args.max_vis_layers)
    writer.add_uint32("lfm2vl.max_llm_layers", args.max_llm_layers)

    # Emit tensors in a canonical order
    stage_order = (
        ["vis_patch_embed"]
        + [f"vis_layer_{i}" for i in range(args.max_vis_layers)]
        + ["vis_post_ln", "projector_out", "llm_embed"]
        + [f"llm_layer_{i}" for i in range(args.max_llm_layers)]
    )

    n_written = 0

    # Write ordered stages first
    for name in stage_order:
        if name not in captured_numpy:
            print(f"  (skipped {name}: not captured)")
            continue
        arr = captured_numpy[name]
        # Squeeze batch dim if present: (1, T, D) → (T, D)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        n_written += 1
        shape_str = "x".join(str(d) for d in arr.shape)
        print(f"  {name}: {shape_str} ({arr.nbytes / 1024:.1f} KB)")

    # llm_logits_last
    arr = np.ascontiguousarray(last_logits, dtype=np.float32)
    writer.add_tensor("llm_logits_last", arr,
                      raw_dtype=gguf.GGMLQuantizationType.F32)
    n_written += 1
    shape_str = "x".join(str(d) for d in arr.shape)
    print(f"  llm_logits_last: {shape_str} ({arr.nbytes / 1024:.1f} KB)")

    # Write any extra captured tensors not already emitted
    ordered_set = set(stage_order) | {"llm_logits_last"}
    for name, arr in sorted(captured_numpy.items()):
        if name in ordered_set:
            continue
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        writer.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        n_written += 1
        shape_str = "x".join(str(d) for d in arr.shape)
        print(f"  {name} (extra): {shape_str} ({arr.nbytes / 1024:.1f} KB)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    fsize = Path(args.output).stat().st_size
    print(f"\nWrote {n_written} tensors to {args.output} "
          f"({fsize / 1024 / 1024:.1f} MB)")
    print(f"Generated text (first token): {generated_text!r}")


if __name__ == "__main__":
    main()
