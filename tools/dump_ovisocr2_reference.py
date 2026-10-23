#!/usr/bin/env python3
"""Capture the pinned original OvisOCR2 Hugging Face F32 graph.

Stores inputs, every vision/decoder layer, actual positional IDs, logits, cache
states and official greedy generated IDs. NPZ/JSON are diagnostic artifacts;
they are never inserted into production GGUF weights. Run on a hosted runner.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
import qwen35_ovis_profile as ovis_profile


VERSIONS = {"transformers": "5.16.1", "tokenizers": "0.23.3", "torch": "2.14.0"}


def validate_request(model_dir, prompt):
    """Perform all source checks before any framework model/processor reader."""
    profile = ovis_profile.load_profile(model_dir, ovis_profile.MODEL, ovis_profile.REVISION)
    ovis_profile.verify_checkpoint(Path(model_dir) / "model.safetensors", profile)
    config = json.loads((Path(model_dir) / "config.json").read_text())
    ovis_profile.validate_config(config, profile)
    if prompt and prompt != profile["rendered_prompt"]:
        raise ValueError("OvisOCR2 custom prompt override is forbidden")
    if len(profile["rendered_prompt"].encode("utf-8")) != 625:
        raise ValueError("OvisOCR2 original prompt byte count differs")
    return profile


def check_framework_sources(profile, transformers, tokenizers, torch, processor, model_class):
    for name, module in [("transformers", transformers), ("tokenizers", tokenizers), ("torch", torch)]:
        if module.__version__.split("+")[0] != VERSIONS[name]:
            raise ValueError(f"OvisOCR2 requires {name}=={VERSIONS[name]}")
    from transformers import GenerationConfig
    classes = {"generation": GenerationConfig, "model": model_class,
               "processor": type(processor), "image_processor": type(processor.image_processor),
               "tokenizer": type(processor.tokenizer)}
    actual = {name: ovis_profile._file_sha256(inspect.getfile(cls)) for name, cls in classes.items()}
    if actual != profile["qualification"]["loaded_source_sha256"]:
        raise ValueError("OvisOCR2 loaded original framework source hashes differ")
    return actual


def observed_visual_positions(frame, event, arg, visual, save, phase):
    """Read the real local variable at the original forward return boundary."""
    if event == "return" and frame.f_locals.get("self") is visual:
        positions = frame.f_locals.get("pos_embeds")
        if positions is not None and phase[0] is not None:
            save(f"{phase[0]}.vision.position", positions)


def restore_observers(handles, module, originals, profiler):
    """Restore every temporary observation even when the original model fails."""
    sys.setprofile(profiler)
    for handle in handles:
        handle.remove()
    for name, original in originals.items():
        setattr(module, "torch_" + name, original)


def required_stages(config, score_count=0):
    required = [f"prefill.vision.layer.{index}" for index in range(config.vision_config.depth)]
    required += [f"prefill.vision.{name}" for name in
                 ["patch_embed", "position", "embedded", "rotary_cos", "rotary_sin",
                  "merger.norm", "merger.up", "merger.activation", "merger"]]
    required += ["input.pixel_values", "input.image_grid_thw", "input.input_ids",
                 "input.mm_token_type_ids", "generated.ids", "prefill.logits",
                 "cached.logits", "full_prefix.logits"]
    required += ["linear.conv_weights", "linear.a_log", "linear.dt_bias", "linear.norm_weights",
                 "prefill.vision.output"]
    required += [f"prefill.vision.probe.{index}.{field}" for index in [5, 11] for field in
                 ["norm1", "qkv", "pooled", "attention", "norm2", "up", "activation", "down"]]
    for label in ["prefill", "cached", "full_prefix"]:
        required += [f"{label}.decoder.layer.{index}" for index in range(config.text_config.num_hidden_layers)]
        required += [f"{label}.decoder.position_ids", f"{label}.decoder.inputs_embeds",
                     f"{label}.decoder.norm", f"{label}.logits", f"{label}.argmax_id"]
        required += [f"{label}.delta.{field}" for field in
                     ["q", "k", "v", "g", "beta", "output"]]
        if label == "cached":
            required.append(f"{label}.delta.initial_state")
        if label != "full_prefix":
            required.append(f"{label}.delta.final_state")
        for index, kind in enumerate(config.text_config.layer_types):
            if kind == "linear_attention":
                required += [f"{label}.linear.{index}.{field}"
                             for field in ["input", "qkv", "z", "a", "b", "norm", "output"]]
            else:
                required += [f"{label}.attention.{index}.{field}" for field in
                             ["input", "q", "k", "v", "q_norm", "k_norm", "pooled", "output"]]
            if label != "full_prefix":
                fields = ["conv_states", "recurrent_states"] if kind == "linear_attention" else ["keys", "values"]
                required += [f"{label}.cache.{index}.{field}" for field in fields]
    required += [f"generated.{index}.decoder.position_ids" for index in range(score_count)]
    return sorted(set(required))


def dump_reference(model_dir, image_path, output, prompt="", max_new_tokens=12, threads=4):
    profile = validate_request(model_dir, prompt)
    import numpy as np
    import torch
    import transformers
    import tokenizers
    import gguf
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    for name, module in [("transformers", transformers), ("tokenizers", tokenizers), ("torch", torch)]:
        if module.__version__.split("+")[0] != VERSIONS[name]:
            raise ValueError(f"OvisOCR2 requires {name}=={VERSIONS[name]}")
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    root, output = Path(model_dir), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    source_hashes = check_framework_sources(profile, transformers, tokenizers, torch,
                                            processor, Qwen3_5ForConditionalGeneration)
    if (processor.tokenizer.eos_token_id != profile["primary_eos"]
            or type(processor).__module__ + "." + type(processor).__name__ != profile["processor"]["class_name"]
            or type(processor.image_processor).__module__ + "." + type(processor.image_processor).__name__
            != profile["processor"]["image_class"]
            or getattr(processor.image_processor, "backend", None) != profile["processor"]["image_backend"]
            or processor.image_processor.to_dict() != profile["processor"]["image_config"]):
        raise ValueError("OvisOCR2 original processor or EOS differs")
    image = Image.open(image_path).convert("RGB")
    rendered = profile["rendered_prompt"]
    prompt = rendered
    inputs = processor(text=[rendered], images=[image], return_tensors="pt",
                       return_mm_token_type_ids=True,
                       images_kwargs={"min_pixels": profile["image_min_pixels"],
                                      "max_pixels": profile["image_max_pixels"]})
    if "mm_token_type_ids" not in inputs or inputs["mm_token_type_ids"].shape != inputs["input_ids"].shape:
        raise ValueError("Official processor must supply aligned multimodal token types")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        root, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager"
    ).eval()
    if model.config.vision_config.depth != 12 or model.config.text_config.num_hidden_layers != 24:
        raise ValueError("OvisOCR2 pinned vision/text layer count differs")
    if model.generation_config.to_diff_dict() != profile["generation"]:
        raise ValueError("OvisOCR2 resolved original generation config differs")
    arrays = {}
    phase = ["prefill"]
    generation_step = [None]

    def save(name, tensor):
        if name.startswith("generated.") and ".decoder.position_ids" not in name and ".scores." not in name and name != "generated.ids":
            return
        if torch.is_tensor(tensor):
            value = tensor.detach().cpu()
            if value.is_floating_point():
                value = value.float()
                if not torch.isfinite(value).all():
                    raise ValueError(f"Non-finite reference: {name}")
            arrays[name] = value.numpy().copy()

    def save_partial(exc):
        np.savez(output, **arrays)
        output.with_suffix(".json").write_text(json.dumps({
            "status": "partial_original_reference_error", "error": str(exc),
            "source_model": profile["source_model"], "source_revision": profile["source_revision"],
            "checkpoint_sha256": profile["checkpoint_sha256"], "loaded_source_sha256": source_hashes,
            "stages": {name: list(value.shape) for name, value in arrays.items()},
            "missing_stages": sorted(set(required_stages(model.config)) - set(arrays)),
        }, indent=2) + "\n")

    save("input.rgb", torch.from_numpy(np.array(image, dtype=np.uint8).copy()))
    save("input.prompt_utf8", torch.tensor(list(prompt.encode("utf-8")), dtype=torch.int32) if prompt else torch.zeros(1, dtype=torch.int32))
    save("input.prompt_length", torch.tensor([len(prompt.encode("utf-8"))], dtype=torch.int32))
    save("input.decoded_utf8", torch.zeros(1))  # overwritten after official generate
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
        if index in [5, 11]:
            for module_name, field in [("norm1", "norm1"), ("attn.qkv", "qkv"),
                                       ("attn", "attention"), ("norm2", "norm2"),
                                       ("mlp.linear_fc1", "up"), ("mlp.act_fn", "activation"),
                                       ("mlp.linear_fc2", "down")]:
                submodule = block
                for part in module_name.split("."):
                    submodule = getattr(submodule, part)
                handles.append(submodule.register_forward_hook(hook(f"vision.probe.{index}.{field}")))
            def pooled_capture(label):
                def capture(module, args):
                    if phase[0] is not None:
                        save(f"{phase[0]}.vision.probe.{label}.pooled", args[0])
                return capture
            handles.append(block.attn.proj.register_forward_pre_hook(pooled_capture(index)))
    def vision_input(module, args, kwargs):
        if phase[0] is not None:
            save(f"{phase[0]}.vision.embedded", args[0])
            cosine, sine = kwargs["position_embeddings"]
            save(f"{phase[0]}.vision.rotary_cos", cosine)
            save(f"{phase[0]}.vision.rotary_sin", sine)
    handles.append(visual.blocks[0].register_forward_pre_hook(vision_input, with_kwargs=True))
    prior_profiler = sys.getprofile()
    def position_capture(frame, event, arg):
        observed_visual_positions(frame, event, arg, visual, save, phase)
        if prior_profiler is not None:
            prior_profiler(frame, event, arg)
    for name, label in [("norm", "norm"), ("linear_fc1", "up"), ("act_fn", "activation")]:
        handles.append(getattr(visual.merger, name).register_forward_hook(hook("vision.merger." + label)))
    for name in ["patch_embed", "merger"]:
        handles.append(getattr(visual, name).register_forward_hook(hook("vision." + name)))
    handles.append(visual.register_forward_hook(hook("vision.output")))
    language = model.model.language_model
    # Capture the official recurrence boundary after projection/convolution,
    # before Q/K normalization. Keep the original upstream functions intact.
    delta_layer = language.layers[0].linear_attn
    for name, value in [("conv_weights", delta_layer.conv1d.weight),
                        ("a_log", delta_layer.A_log), ("dt_bias", delta_layer.dt_bias),
                        ("norm_weights", delta_layer.norm.weight)]:
        save("linear." + name, value)
    for name, field in [("in_proj_qkv", "qkv"), ("in_proj_z", "z"),
                        ("in_proj_a", "a"), ("in_proj_b", "b"), ("norm", "norm")]:
        handles.append(getattr(delta_layer, name).register_forward_hook(hook("linear." + field)))
    def projection_input(module, args):
        if phase[0] is not None:
            save(f"{phase[0]}.linear.input", args[0])
    handles.append(delta_layer.in_proj_qkv.register_forward_pre_hook(projection_input))
    handles.append(delta_layer.register_forward_hook(hook("linear.output")))
    # Full trained linear-attention coverage, not just one operator's parameters.
    def block_input(label):
        def capture(module, args):
            if phase[0] is not None:
                save(f"{phase[0]}.{label}", args[0])
        return capture
    for index, block in enumerate(language.layers):
        if model.config.text_config.layer_types[index] != "linear_attention":
            attention = block.self_attn
            label = f"attention.{index}"
            for name, field in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"),
                                ("q_norm", "q_norm"), ("k_norm", "k_norm")]:
                handles.append(getattr(attention, name).register_forward_hook(hook(f"{label}.{field}")))
            handles.append(attention.q_proj.register_forward_pre_hook(block_input(f"{label}.input")))
            handles.append(attention.o_proj.register_forward_pre_hook(block_input(f"{label}.pooled")))
            handles.append(attention.register_forward_hook(hook(f"{label}.output")))
            continue
        linear = block.linear_attn
        label = f"linear.{index}"
        for name, field in [("in_proj_qkv", "qkv"), ("in_proj_z", "z"),
                            ("in_proj_a", "a"), ("in_proj_b", "b"), ("norm", "norm")]:
            handles.append(getattr(linear, name).register_forward_hook(hook(f"{label}.{field}")))
        handles.append(linear.in_proj_qkv.register_forward_pre_hook(block_input(f"{label}.input")))
        handles.append(linear.register_forward_hook(hook(f"{label}.output")))
    model_module = sys.modules[type(delta_layer).__module__]
    originals = {}
    delta_wrappers = {}
    delta_active = [False]
    delta_none_initial = set()
    for method_name in ["chunk_gated_delta_rule", "recurrent_gated_delta_rule"]:
        original = getattr(model_module, "torch_" + method_name)
        originals[method_name] = original

        def wrap(fn):
            def capture(query, key, value, *args, **kwargs):
                if phase[0] is not None and delta_active[0]:
                    if args or not kwargs.get("use_qk_l2norm_in_kernel", False):
                        raise ValueError("Unaudited official delta call convention")
                    for field, tensor in [("q", query), ("k", key), ("v", value),
                                          ("g", kwargs["g"]), ("beta", kwargs["beta"])]:
                        save(f"{phase[0]}.delta.{field}", tensor)
                    initial = kwargs.get("initial_state")
                    if initial is None:
                        delta_none_initial.add(phase[0])
                    else:
                        save(f"{phase[0]}.delta.initial_state", initial)
                result = fn(query, key, value, *args, **kwargs)
                if phase[0] is not None and delta_active[0]:
                    save(f"{phase[0]}.delta.output", result[0])
                    save(f"{phase[0]}.delta.final_state", result[1])
                return result
            return capture
        delta_wrappers[method_name] = wrap(original)
    def delta_enter(module, args):
        for name, wrapper in delta_wrappers.items():
            setattr(model_module, "torch_" + name, wrapper)
        delta_active[0] = True
    def delta_leave(module, args, result):
        delta_active[0] = False
        for name, original in originals.items():
            setattr(model_module, "torch_" + name, original)
    handles.append(delta_layer.register_forward_pre_hook(delta_enter))
    handles.append(delta_layer.register_forward_hook(delta_leave))
    for index, block in enumerate(language.layers):
        handles.append(block.register_forward_hook(hook(f"decoder.layer.{index}")))
    handles.append(language.norm.register_forward_hook(hook("decoder.norm")))

    def language_input(module, args, kwargs):
        if generation_step[0] is not None:
            phase[0] = f"generated.{generation_step[0]}"
            generation_step[0] += 1
        if phase[0] is not None:
            for name in ["position_ids", "inputs_embeds", "input_ids"]:
                save(f"{phase[0]}.decoder.{name}", kwargs.get(name))

    handles.append(language.register_forward_pre_hook(language_input, with_kwargs=True))

    def save_cache(cache, label):
        for index, layer in enumerate(cache.layers):
            for name in ["conv_states", "recurrent_states", "keys", "values"]:
                value = getattr(layer, name, None)
                if name in ("conv_states", "recurrent_states") and isinstance(value, (list, tuple)):
                    if len(value) != 1:
                        raise ValueError(f"Unexpected OvisOCR2 {name} cache arity")
                    value = value[0]
                save(f"{label}.cache.{index}.{name}", value)

    try:
        sys.setprofile(position_capture)
        with torch.inference_mode():
            # Use the same official positional-input preparation as generate(). A
            # raw cached forward with a full attention mask produces full-length
            # rotary positions for a one-token query in this pinned implementation.
            prefill_positions = model._prepare_position_ids_for_generation(inputs["input_ids"], dict(inputs))
            prefill_inputs = model.prepare_inputs_for_generation(
                **inputs, position_ids=prefill_positions, use_cache=True,
                is_first_iteration=True, logits_to_keep=1
            )
            prefill = model(**prefill_inputs)
            save("prefill.logits", prefill.logits[:, -1])
            save("prefill.argmax_id", prefill.logits[:, -1].argmax(-1))
            save_cache(prefill.past_key_values, "prefill")
            first_id = prefill.logits[:, -1].argmax(-1, keepdim=True)
            # One actual cached continuation, compared with an independent full-prefix
            # forward of the same token. Retain both logits even if this check fails.
            phase[0] = "cached"
            continued_mask = torch.cat([inputs["attention_mask"], torch.ones_like(first_id)], dim=-1)
            cached_kwargs = dict(input_ids=first_id, attention_mask=continued_mask,
                                 past_key_values=prefill.past_key_values, use_cache=True)
            cached_positions = model._prepare_position_ids_for_generation(first_id, cached_kwargs)
            cached_inputs = model.prepare_inputs_for_generation(
                **cached_kwargs, position_ids=cached_positions,
                is_first_iteration=False, logits_to_keep=1
            )
            cached = model(**cached_inputs)
            save("cached.logits", cached.logits[:, -1])
            save("cached.argmax_id", cached.logits[:, -1].argmax(-1))
            save_cache(cached.past_key_values, "cached")
            phase[0] = "full_prefix"
            complete = dict(inputs)
            complete["input_ids"] = torch.cat([inputs["input_ids"], first_id], dim=-1)
            complete["attention_mask"] = continued_mask
            complete["mm_token_type_ids"] = torch.cat(
                [inputs["mm_token_type_ids"], torch.zeros_like(first_id)], dim=-1
            )
            full_positions = model._prepare_position_ids_for_generation(complete["input_ids"], complete)
            full_inputs = model.prepare_inputs_for_generation(
                **complete, position_ids=full_positions, use_cache=False,
                is_first_iteration=True, logits_to_keep=1
            )
            full = model(**full_inputs)
            save("full_prefix.logits", full.logits[:, -1])
            save("full_prefix.argmax_id", full.logits[:, -1].argmax(-1))
            phase[0] = "full_prefix"
            delta = (cached.logits[:, -1] - full.logits[:, -1]).abs()
            max_error = delta.max().item()
            same_argmax = bool(torch.equal(cached.logits[:, -1].argmax(-1), full.logits[:, -1].argmax(-1)))
            phase[0] = None
            generation_step[0] = 0
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                       use_cache=True, eos_token_id=profile["stop_ids"],
                                       return_dict_in_generate=True, output_scores=True)
            phase[0] = None
            generation_step[0] = None
            ids = generated.sequences[:, inputs["input_ids"].shape[1]:]
            save("generated.ids", ids)
            for index, score in enumerate(generated.scores):
                save(f"generated.scores.{index}", score)
    except Exception as exc:
        save_partial(exc)
        raise
    finally:
        restore_observers(handles, model_module, originals, prior_profiler)
        delta_active[0] = False
    phase[0] = None
    try:
        with torch.inference_mode():
            unobserved = model(**prefill_inputs)
            unobserved_generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                eos_token_id=profile["stop_ids"], return_dict_in_generate=True)
    except Exception as exc:
        save_partial(exc)
        raise
    observational_control = {
        "prefill_logits_exact": bool(torch.equal(prefill.logits[:, -1], unobserved.logits[:, -1])),
        "greedy_ids_exact": bool(torch.equal(ids, unobserved_generated.sequences[:, inputs["input_ids"].shape[1]:])),
    }
    vision_depth = model.config.vision_config.depth
    decoder_depth = model.config.text_config.num_hidden_layers
    required = required_stages(model.config, len(generated.scores))
    missing = sorted(set(required) - set(arrays))
    decoded = processor.batch_decode(ids, skip_special_tokens=True)[0]
    raw_decoded = processor.batch_decode(ids, skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)
    save("input.decoded_utf8", torch.tensor(list(decoded.encode("utf-8")), dtype=torch.int32))
    save("input.raw_decoded_utf8", torch.tensor(list(raw_decoded[0].encode("utf-8")), dtype=torch.int32))
    np.savez(output, **arrays)
    report = {
        "transformers": transformers.__version__, "tokenizers": tokenizers.__version__, "torch": torch.__version__,
        "source_model": profile["source_model"], "source_revision": profile["source_revision"],
        "checkpoint_sha256": profile["checkpoint_sha256"], "loaded_source_sha256": source_hashes,
        "metadata_sha256": {name: ovis_profile._file_sha256(root / name)
                            for name in ovis_profile.metadata_files()},
        "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        "image_size": list(image.size), "prompt": prompt, "rendered_prompt": rendered,
        "resolved_processor": type(processor).__module__ + "." + type(processor).__name__,
        "resolved_image_processor": type(processor.image_processor).__module__ + "." + type(processor.image_processor).__name__,
        "resolved_image_backend": getattr(processor.image_processor, "backend", None),
        "generation_config": model.generation_config.to_dict(),
        "generation_override": {"eos_token_id": profile["stop_ids"],
                                "reason": "pinned original vLLM effective stop set; missing generation_config.json"},
        "delta_initial_state_none_phases": sorted(delta_none_initial),
        "torch_build": torch.__config__.show(), "observational_control": observational_control,
        "source_config": model.config.to_dict(), "vision_layers": vision_depth,
        "decoder_layers": decoder_depth, "stages": {name: list(value.shape) for name, value in arrays.items()},
        "generated_ids": ids.tolist(),
        "generated_stop_id": ids[0, -1].item() if ids.numel() and ids[0, -1].item() in profile["stop_ids"] else None,
        "raw_decoded": raw_decoded,
        "decoded": processor.batch_decode(ids, skip_special_tokens=True),
        "cached_vs_full_prefix_max_abs": max_error, "cached_vs_full_prefix_same_argmax": same_argmax,
        "native_parity": "not implemented", "transcription_accuracy": "unscored",
        "missing_stages": missing,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if missing:
        raise ValueError(f"Missing reference stages: {missing}")
    if not all(observational_control.values()):
        raise ValueError("OvisOCR2 observation changed original output")
    if not same_argmax or max_error > 0.002:
        raise ValueError(f"Official cache/full-prefix inconsistency: abs={max_error}, argmax={same_argmax}")
    # Small standalone native-op probe; never part of exported model weights.
    probe = gguf.GGUFWriter(str(output.with_name("official-delta.gguf")), "qwen35_delta_reference")
    for name, value in arrays.items():
        if ((".delta." in name or ".linear." in name or ".cache." in name
             or ".attention." in name or ".decoder." in name or ".vision." in name or name.endswith(".logits"))
                and name.startswith(("prefill.", "cached.", "full_prefix."))) or name.startswith(("linear.", "generated.", "input.")):
            probe.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))
    probe.write_header_to_file()
    probe.write_kv_data_to_file()
    probe.write_tensors_to_file()
    probe.close()
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
