#!/usr/bin/env python3
"""Dump SmolDocling reference activations for crispembed-diff parity testing.

Memory-efficient: loads model with bf16, captures per-stage activations
via forward hooks. Dumps to GGUF archive.

Usage:
    PYTHONNOUSERSITE=1 python tools/dump_smoldocling_reference.py \
        --model /mnt/storage/models/SmolDocling-256M-preview \
        --image test.png \
        --output /mnt/storage/gguf-models/smoldocling-ref.gguf \
        --max-vis-layers 4 --max-llm-layers 2
"""
import argparse, sys, os, gc, types, importlib
from pathlib import Path

# Workaround: torchvision NMS operator registration crashes on this env.
# Block torchvision before transformers loads, then provide a fake module.
sys.modules['torchvision'] = None  # block import during transformers init
import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model ID or local path")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--output", "-o", required=True, help="Output reference GGUF")
    p.add_argument("--max-vis-layers", type=int, default=4,
                   help="Max vision layers to dump (0=none)")
    p.add_argument("--max-llm-layers", type=int, default=2,
                   help="Max LLM layers to dump (0=none)")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Max tokens to generate")
    args = p.parse_args()

    import torch
    from PIL import Image

    # Provide fake torchvision so transformers image_processing can import
    del sys.modules['torchvision']
    fake_tv = types.ModuleType('torchvision')
    fake_tv.__version__ = '0.0.0'
    fake_tv.__spec__ = importlib.util.find_spec('torch')
    fake_transforms = types.ModuleType('torchvision.transforms')
    class _IM:
        BILINEAR = 2; BICUBIC = 3; NEAREST = 0; LANCZOS = 1
        NEAREST_EXACT = 0; BOX = 4; HAMMING = 5
    fake_transforms.InterpolationMode = _IM
    fake_tv.transforms = fake_transforms
    fake_io = types.ModuleType('torchvision.io')
    fake_tv.io = fake_io
    sys.modules['torchvision'] = fake_tv
    sys.modules['torchvision.transforms'] = fake_transforms
    sys.modules['torchvision.io'] = fake_io
    sys.modules['torchvision._meta_registrations'] = types.ModuleType('torchvision._meta_registrations')

    from transformers import AutoTokenizer
    from transformers.models.idefics3.modeling_idefics3 import Idefics3ForConditionalGeneration

    # Load model in bf16 (256M fits in 8GB)
    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Load and preprocess image manually (single 512x512 patch)
    image = Image.open(args.image).convert("RGB")
    print(f"Image: {image.size[0]}x{image.size[1]}")

    # Resize to 512x512 and normalize: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1
    img_tensor = torch.tensor(np.array(image.resize((512, 512), Image.BILINEAR)), dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # [3, 512, 512]
    img_tensor = (img_tensor - 0.5) / 0.5  # normalize
    # Shape: [1, 1, 3, 512, 512] — single image, single patch
    pixel_values = img_tensor.unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    print(f"Pixel values: {pixel_values.shape}, range=[{pixel_values.float().min():.4f}, {pixel_values.float().max():.4f}]")

    # Build prompt: <|im_start|>User:<image>Convert this page to docling.<end_of_utterance>\nAssistant:
    # Image gets 64 placeholder tokens (1024 patches / scale_factor^2 = 64)
    image_token_id = 49190
    n_image_tokens = 64  # after pixel shuffle with scale=4: 1024/16=64
    prompt_text = "User:" + "".join(["<image>"] * n_image_tokens) + "Convert this page to docling.<end_of_utterance>\nAssistant:"
    input_ids = tokenizer.encode("<|im_start|>" + prompt_text, add_special_tokens=False, return_tensors="pt")
    print(f"Input IDs: {input_ids.shape}")

    # Build inputs dict
    inputs = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": torch.ones_like(input_ids),
    }

    # Capture activations via hooks
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            elif hasattr(output, 'last_hidden_state'):
                out = output.last_hidden_state
            else:
                out = output
            captured[name] = out.detach().float().cpu().numpy()
        return hook

    hooks = []

    # Vision encoder hooks
    if args.max_vis_layers > 0:
        # After embeddings
        hooks.append(model.model.vision_model.embeddings.register_forward_hook(
            make_hook("vis_embeddings")))
        # Per-layer
        for i in range(min(args.max_vis_layers,
                          len(model.model.vision_model.encoder.layers))):
            hooks.append(model.model.vision_model.encoder.layers[i].register_forward_hook(
                make_hook(f"vis_layer_{i}")))
        # Post layernorm
        hooks.append(model.model.vision_model.post_layernorm.register_forward_hook(
            make_hook("vis_post_ln")))

    # Connector hook
    hooks.append(model.model.connector.register_forward_hook(
        make_hook("connector_output")))

    # LLM layer hooks
    if args.max_llm_layers > 0:
        for i in range(min(args.max_llm_layers,
                          len(model.model.text_model.layers))):
            hooks.append(model.model.text_model.layers[i].register_forward_hook(
                make_hook(f"llm_layer_{i}")))

    # Run prefill (forward pass with all inputs, no generation yet)
    print("\nRunning prefill...")
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

    # Capture logits from prefill
    logits = outputs.logits[:, -1, :].float().cpu().numpy()
    captured["logits_step0"] = logits
    print(f"Logits step0: shape={logits.shape}, "
          f"top5={np.argsort(logits[0])[-5:][::-1].tolist()}")

    # Remove hooks before generation
    for h in hooks:
        h.remove()
    hooks.clear()

    # Generate full text
    print(f"\nGenerating up to {args.max_new_tokens} tokens...")
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    # Decode generated text
    prompt_len = inputs['input_ids'].shape[1]
    gen_only = gen_ids[0, prompt_len:]
    text = tokenizer.decode(gen_only, skip_special_tokens=False)
    print(f"Generated ({len(gen_only)} tokens): {text[:200]}")

    # Save generated IDs
    captured["generated_ids"] = gen_only.cpu().numpy().astype(np.float32)
    captured["input_ids"] = inputs['input_ids'][0].cpu().numpy().astype(np.float32)

    # Write GGUF reference
    print(f"\nWriting reference to {args.output}...")
    writer = gguf.GGUFWriter(args.output, "smoldocling-ref")

    # Metadata
    writer.add_uint32("ref.prompt_len", prompt_len)
    writer.add_uint32("ref.n_generated", len(gen_only))
    writer.add_string("ref.generated_text", text[:512])

    # Write captured tensors
    for name, data in sorted(captured.items()):
        if data.ndim == 0:
            data = data.reshape(1)
        flat = data.flatten().astype(np.float32)
        print(f"  {name}: shape={data.shape} range=[{flat.min():.4f}, {flat.max():.4f}]")
        writer.add_tensor(name, flat, raw_dtype=gguf.GGMLQuantizationType.F32)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_kb = Path(args.output).stat().st_size / 1024
    print(f"\nReference: {args.output} ({size_kb:.0f} KB, {len(captured)} tensors)")


if __name__ == "__main__":
    main()
