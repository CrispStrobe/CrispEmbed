#!/usr/bin/env python3
"""Capture the official dense Qwen3.5 OCR implementation, never a manual clone.

Stores inputs, every vision/decoder layer, actual positional IDs, logits, cache
states and official greedy generated IDs. NPZ/JSON are diagnostic artifacts;
they are never inserted into production GGUF weights. Run on a hosted runner.
"""

import argparse
import hashlib
import json
from pathlib import Path


def dump_reference(model_dir, image_path, output, prompt="", max_new_tokens=12, threads=4):
    import numpy as np
    import torch
    import transformers
    import gguf
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    root, output = Path(model_dir), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    if prompt:
        messages[0]["content"].append({"type": "text", "text": prompt})
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt",
                       return_mm_token_type_ids=True)
    if "mm_token_type_ids" not in inputs or inputs["mm_token_type_ids"].shape != inputs["input_ids"].shape:
        raise ValueError("Official processor must supply aligned multimodal token types")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        root, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager"
    ).eval()
    arrays = {}
    phase = ["prefill"]

    def save(name, tensor):
        if torch.is_tensor(tensor):
            value = tensor.detach().cpu()
            if value.is_floating_point():
                value = value.float()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite reference: {name}")
            arrays[name] = value.numpy().copy()

    for name, value in inputs.items():
        save("input." + name, value)

    def hook(name):
        def capture(module, args, result):
            if phase[0] is not None:
                if torch.is_tensor(result):
                    save(f"{phase[0]}.{name}", result)
                elif isinstance(result, tuple):
                    save(f"{phase[0]}.{name}", result[0])
                else:
                    save(f"{phase[0]}.{name}", result.pooler_output)
        return capture

    handles = []
    visual = model.model.visual
    for index, block in enumerate(visual.blocks):
        handles.append(block.register_forward_hook(hook(f"vision.layer.{index}")))
    for name in ["patch_embed", "merger"]:
        handles.append(getattr(visual, name).register_forward_hook(hook("vision." + name)))
    handles.append(visual.register_forward_hook(hook("vision.output")))
    language = model.model.language_model
    # Capture the official recurrence boundary after projection/convolution,
    # before Q/K normalization. Keep the original upstream functions intact.
    delta_layer = language.layers[0].linear_attn
    originals = {}
    for method_name in ["chunk_gated_delta_rule", "recurrent_gated_delta_rule"]:
        original = getattr(delta_layer, method_name)
        originals[method_name] = original

        def wrap(fn):
            def capture(query, key, value, *args, **kwargs):
                if phase[0] is not None:
                    if args or not kwargs.get("use_qk_l2norm_in_kernel", False):
                        raise ValueError("Unaudited official delta call convention")
                    for field, tensor in [("q", query), ("k", key), ("v", value),
                                          ("g", kwargs["g"]), ("beta", kwargs["beta"])]:
                        save(f"{phase[0]}.delta.{field}", tensor)
                    initial = kwargs.get("initial_state")
                    if initial is None:
                        initial = torch.zeros(query.shape[0], value.shape[2], query.shape[-1], value.shape[-1])
                    save(f"{phase[0]}.delta.initial_state", initial)
                result = fn(query, key, value, *args, **kwargs)
                if phase[0] is not None:
                    save(f"{phase[0]}.delta.output", result[0])
                    save(f"{phase[0]}.delta.final_state", result[1])
                return result
            return capture
        setattr(delta_layer, method_name, wrap(original))
    for index, block in enumerate(language.layers):
        handles.append(block.register_forward_hook(hook(f"decoder.layer.{index}")))
    handles.append(language.norm.register_forward_hook(hook("decoder.norm")))

    def language_input(module, args, kwargs):
        if phase[0] is not None:
            for name in ["position_ids", "inputs_embeds", "input_ids"]:
                save(f"{phase[0]}.decoder.{name}", kwargs.get(name))

    handles.append(language.register_forward_pre_hook(language_input, with_kwargs=True))

    def save_cache(cache, label):
        for index, layer in enumerate(cache.layers):
            for name in ["conv_states", "recurrent_states", "keys", "values"]:
                save(f"{label}.cache.{index}.{name}", getattr(layer, name, None))

    with torch.inference_mode():
        prefill = model(**inputs, use_cache=True, logits_to_keep=1)
        save("prefill.logits", prefill.logits[:, -1])
        save_cache(prefill.past_key_values, "prefill")
        first_id = prefill.logits[:, -1].argmax(-1, keepdim=True)
        # One actual cached continuation, compared with an independent full-prefix
        # forward of the same token. Retain both logits even if this check fails.
        phase[0] = "cached"
        continued_mask = torch.cat([inputs["attention_mask"], torch.ones_like(first_id)], dim=-1)
        cached = model(input_ids=first_id, attention_mask=continued_mask,
                       past_key_values=prefill.past_key_values, use_cache=True,
                       logits_to_keep=1)
        save("cached.logits", cached.logits[:, -1])
        save_cache(cached.past_key_values, "cached")
        phase[0] = "full_prefix"
        complete = dict(inputs)
        complete["input_ids"] = torch.cat([inputs["input_ids"], first_id], dim=-1)
        complete["attention_mask"] = continued_mask
        complete["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], torch.zeros_like(first_id)], dim=-1
        )
        full = model(**complete, use_cache=False, logits_to_keep=1)
        save("full_prefix.logits", full.logits[:, -1])
        delta = (cached.logits[:, -1] - full.logits[:, -1]).abs()
        max_error = delta.max().item()
        same_argmax = bool(torch.equal(cached.logits[:, -1].argmax(-1), full.logits[:, -1].argmax(-1)))
        phase[0] = None
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                   use_cache=True, return_dict_in_generate=True, output_scores=True)
        ids = generated.sequences[:, inputs["input_ids"].shape[1]:]
        save("generated.ids", ids)
        for index, score in enumerate(generated.scores):
            save(f"generated.logits.{index}", score)
    for handle in handles:
        handle.remove()
    for name, original in originals.items():
        setattr(delta_layer, name, original)
    vision_depth = model.config.vision_config.depth
    decoder_depth = model.config.text_config.num_hidden_layers
    required = [f"prefill.vision.layer.{index}" for index in range(vision_depth)]
    for label in ["prefill", "cached", "full_prefix"]:
        required += [f"{label}.decoder.layer.{index}" for index in range(decoder_depth)]
        required += [f"{label}.decoder.position_ids", f"{label}.logits"]
        for index, kind in enumerate(model.config.text_config.layer_types):
            if label != "full_prefix":
                fields = ["conv_states", "recurrent_states"] if kind == "linear_attention" else ["keys", "values"]
                required += [f"{label}.cache.{index}.{field}" for field in fields]
    missing = sorted(set(required) - set(arrays))
    np.savez(output, **arrays)
    # Small standalone native-op probe; never part of exported model weights.
    probe = gguf.GGUFWriter(str(output.with_name("official-delta.gguf")), "qwen35_delta_reference")
    for name, value in arrays.items():
        if ".delta." in name and name.startswith(("prefill.", "cached.")):
            probe.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))
    probe.write_header_to_file()
    probe.write_kv_data_to_file()
    probe.write_tensors_to_file()
    probe.close()
    report = {
        "transformers": transformers.__version__, "torch": torch.__version__,
        "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        "image_size": list(image.size), "prompt": prompt, "rendered_prompt": rendered,
        "generation_config": model.generation_config.to_dict(),
        "source_config": model.config.to_dict(), "vision_layers": vision_depth,
        "decoder_layers": decoder_depth, "stages": {name: list(value.shape) for name, value in arrays.items()},
        "generated_ids": ids.tolist(),
        "decoded": processor.batch_decode(ids, skip_special_tokens=True),
        "cached_vs_full_prefix_max_abs": max_error, "cached_vs_full_prefix_same_argmax": same_argmax,
        "native_parity": "not implemented", "transcription_accuracy": "unscored",
        "missing_stages": missing,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if missing:
        raise ValueError(f"Missing reference stages: {missing}")
    if not same_argmax or max_error > 0.002:
        raise ValueError(f"Official cache/full-prefix inconsistency: abs={max_error}, argmax={same_argmax}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    report = dump_reference(args.model_dir, args.image, args.output, args.prompt,
                            args.max_new_tokens, args.threads)
    print(json.dumps({key: report[key] for key in ["decoded", "cached_vs_full_prefix_max_abs",
                                                 "cached_vs_full_prefix_same_argmax"]}, indent=2))


if __name__ == "__main__":
    main()
