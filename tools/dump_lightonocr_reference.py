#!/usr/bin/env python3
"""Dump LightOnOCR-2-1B per-layer activations to GGUF for parity testing.

Usage:
    python tools/dump_lightonocr_reference.py \
        --model lightonai/LightOnOCR-2-1B \
        --image test.png \
        --output /tmp/lightonocr-ref.gguf \
        --max-vis-layers 2 --max-llm-layers 2
"""

import argparse
import gc
import os
import numpy as np

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


def write_ref_gguf(path, tensors):
    import gguf
    writer = gguf.GGUFWriter(path, arch="lightonocr-ref")
    for name, arr in tensors.items():
        writer.add_tensor(name, arr.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True, help="Test image (PNG/JPG)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-vis-layers", type=int, default=2)
    parser.add_argument("--max-llm-layers", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image

    print(f"Loading: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # LightOnOCR checkpoint uses 'vision_encoder'/'vision_projection' but
    # transformers' Mistral3 class expects 'vision_tower'/'multi_modal_projector'.
    # Load with name remapping.
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig
    from pathlib import Path

    model_path = Path(args.model)
    sf_path = str(model_path / "model.safetensors") if model_path.is_dir() else \
              hf_hub_download(args.model, "model.safetensors",
                  cache_dir='/mnt/akademie_storage/huggingface/hub/')

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_config(config)

    sd = {}
    with safe_open(sf_path, framework="pt") as sf:
        for name in sf.keys():
            t = sf.get_tensor(name)
            mapped = name.replace("model.vision_encoder.", "model.vision_tower.") \
                         .replace("model.vision_projection.", "model.multi_modal_projector.")
            sd[mapped] = t

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Loaded {len(sd)-len(unexpected)} tensors, {len(missing)} missing, {len(unexpected)} unexpected")
    if missing:
        for m in missing[:3]: print(f"    missing: {m}")

    model = model.to(torch.bfloat16)
    model.eval()

    # Load image
    image = Image.open(args.image).convert("RGB")
    print(f"  Image: {image.size}")

    # Process
    prompt = "<|im_start|>user\n<image>OCR this document.<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    print(f"  input_ids: {inputs['input_ids'].shape}")
    print(f"  pixel_values: {inputs['pixel_values'].shape if 'pixel_values' in inputs else 'N/A'}")

    tensors = {}
    captured = {}

    # Hook vision encoder layers
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[name] = output[0].detach().float().cpu().numpy()
            else:
                captured[name] = output.detach().float().cpu().numpy()
        return hook

    vis_model = model.vision_encoder if hasattr(model, 'vision_encoder') else \
                model.model.vision_encoder if hasattr(model.model, 'vision_encoder') else None

    if vis_model:
        max_vl = min(args.max_vis_layers, len(vis_model.transformer.layers))
        for i in range(max_vl):
            vis_model.transformer.layers[i].register_forward_hook(make_hook(f"vis_layer_{i}"))
        print(f"  Hooking {max_vl} vision layers")

    # Hook LM decoder layers
    lm_model = model.language_model if hasattr(model, 'language_model') else \
               model.model.language_model if hasattr(model.model, 'language_model') else None

    if lm_model:
        max_ll = min(args.max_llm_layers, len(lm_model.layers))
        for i in range(max_ll):
            lm_model.layers[i].register_forward_hook(make_hook(f"llm_layer_{i}"))
        print(f"  Hooking {max_ll} LLM layers")

    # Generate
    print("  Generating...")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)

    # Decode
    generated = output_ids[0][inputs['input_ids'].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True)
    print(f"  Generated: {text[:200]}")

    # Save captured tensors
    for name, arr in captured.items():
        arr = arr.squeeze(0) if arr.ndim > 2 else arr
        tensors[name] = arr
        print(f"  {name}: {arr.shape}")

    # Save input/output metadata
    tensors["input_ids"] = inputs['input_ids'][0].numpy().astype(np.float32)
    tensors["generated_ids"] = generated.numpy().astype(np.float32)

    write_ref_gguf(args.output, tensors)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")
    print(f"Generated text: {text}")


if __name__ == "__main__":
    main()
