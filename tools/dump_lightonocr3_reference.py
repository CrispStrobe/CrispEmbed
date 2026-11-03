#!/usr/bin/env python3
"""Observe the actual pinned LightOnOCR-3-1B implementation on hosted compute.

No reconstructed PyTorch equations. This deliberately uses deterministic greedy
F32 generation, not the model card's stochastic quality recipe. Full vocabulary
logits at every generated position, all prefill layers and decoded output are
retained. Model/card/processor/tokenizer/library sources are hash checked first.
"""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from lightonocr3_profile import (manifest, prompt_ids, sha256, verify_source, validate_config,
                                trace_shapes, validate_generation_ids, STOP_IDS, REFERENCE_PACKAGES)
from lightonocr3_output import parse_grounding


def verify_transformers():
    pins = manifest()
    for module, expected in pins["transformers_modules"].items():
        imported = importlib.import_module(module)
        if sha256(imported.__file__) != expected:
            raise ValueError(f"Unpinned original implementation: {module}")


def verify_environment():
    versions = {name: importlib.metadata.version(name) for name in REFERENCE_PACKAGES}
    if versions != REFERENCE_PACKAGES:
        raise ValueError(f"Original package versions differ from the frozen recipe: {versions}")
    verify_transformers()
    return versions


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["transcription", "grounding"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("bounded reference cap must be in [1, 256]")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("reference output directory must be empty; never overwrite a frozen receipt")
    pins = verify_source(args.model_dir)
    cfg = json.loads((args.model_dir / "config.json").read_text())
    validate_config(cfg, json.loads((args.model_dir / "processor_config.json").read_text()),
                    json.loads((args.model_dir / "generation_config.json").read_text()))
    import numpy as np
    import torch
    import transformers
    import torchvision
    from PIL import Image
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor
    versions = verify_environment()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    arrays = {}

    def put(name, tensor):
        if name in arrays:
            return
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        if hasattr(tensor, "last_hidden_state"):
            tensor = tensor.last_hidden_state
        a = tensor.detach().cpu().float().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor, dtype=np.float32)
        a = np.ascontiguousarray(a)
        if not np.isfinite(a).all():
            raise ValueError(f"Nonfinite original stage {name}")
        np.save(args.output / f"{name}.npy", a, allow_pickle=False)
        arrays[name] = {"shape": list(a.shape), "sha256": sha256(args.output / f"{name}.npy")}

    model, loading = LightOnOcrForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False, output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("unexpected_keys") or loading.get("mismatched_keys"):
        raise ValueError(f"Original checkpoint did not load completely: {loading}")
    model.eval()
    processor = LightOnOcrProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    prompts = prompt_ids(processor.tokenizer, (args.model_dir / "chat_template.jinja").read_text())
    image = Image.open(args.image).convert("RGB")
    content = [{"type": "image", "image": image}]
    if args.mode == "grounding":
        content.append({"type": "text", "text": "grounding"})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(dtype=torch.float32) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
              for k, v in inputs.items()}
    ids = inputs["input_ids"][0].tolist()
    prefix, suffix = prompts[args.mode]
    count = ids.count(151655)
    if not count or ids != prefix + [151655] * count + suffix:
        raise ValueError("Processor prompt does not match the candidate's single-image contract")
    put("pixel_values", inputs["pixel_values"])
    put("prompt_ids", ids)
    hooks = []
    visual, language = model.model.vision_encoder, model.model.language_model

    def output(name):
        return lambda _m, _a, value: put(name, value)

    def patch(_module, _args, value):
        put("patch_conv", value.permute(0, 2, 3, 1).reshape(-1, 1024))

    hooks.append(visual.patch_conv.register_forward_hook(patch))
    hooks.append(visual.ln_pre.register_forward_hook(output("vis_ln_pre")))
    for index, layer in enumerate(visual.transformer.layers):
        hooks.append(layer.register_forward_hook(output(f"vis_layer_{index}")))
    hooks.append(model.model.vision_projection.register_forward_hook(output("proj_out")))

    def decoder_input(_module, args_, kwargs):
        put("decoder_embed", args_[0] if args_ else kwargs["hidden_states"])

    hooks.append(language.layers[0].register_forward_pre_hook(decoder_input, with_kwargs=True))
    for index, layer in enumerate(language.layers):
        hooks.append(layer.register_forward_hook(output(f"llm_layer_{index}")))
    hooks.append(language.norm.register_forward_hook(output("llm_final_norm")))
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=False, num_beams=1, use_cache=True,
            max_new_tokens=args.max_new_tokens, eos_token_id=[151645, 151643], pad_token_id=151643,
            forced_eos_token_id=None, forced_bos_token_id=None,
            return_dict_in_generate=True, output_logits=True, logits_to_keep=1,
        )
    seconds = time.perf_counter() - start
    for hook in hooks:
        hook.remove()
    output_ids = generated.sequences[0, len(ids):].tolist()
    validate_generation_ids(output_ids, args.max_new_tokens)
    if len(generated.logits) != len(output_ids):
        raise ValueError("Original generate did not return every decisive logit")
    for index, logits in enumerate(generated.logits):
        if logits.shape != (1, 151936) or int(logits.argmax(dim=-1)[0]) != output_ids[index]:
            raise ValueError("Original IDs do not match full-vocabulary greedy raw logits")
        put(f"logits_step{index}", logits)
    put("generated_ids", output_ids)
    shapes, expected_prompt = trace_shapes(list(image.size), args.mode, len(output_ids))
    if set(arrays) != set(shapes) or ids != expected_prompt:
        raise ValueError("Incomplete original capture or unexpected processor prompt")
    for name, shape in shapes.items():
        if arrays[name]["shape"] != list(shape):
            raise ValueError(f"Unexpected original stage shape: {name}")
    text = processor.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    (args.output / "decoded.txt").write_text(text, encoding="utf-8")
    (args.output / "forced-ids.txt").write_text(" ".join(map(str, output_ids)) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": 2, "model_id": pins["model_id"], "revision": pins["revision"],
        "source_manifest_sha256": sha256(Path(__file__).resolve().parents[1] / "models/lightonocr3-1b-source.json"),
        "mode": args.mode, "image_sha256": sha256(args.image), "image_size": list(image.size),
        "dtype": "float32", "device": "cpu", "attention": "eager", "do_sample": False,
        "use_cache": True, "max_new_tokens": args.max_new_tokens, "token_ids": output_ids,
        "stop_ids": list(STOP_IDS), "forced_eos_token_ids": [], "hit_cap": output_ids[-1] not in STOP_IDS,
        "decoded_sha256": sha256(args.output / "decoded.txt"), "forced_ids_sha256": sha256(args.output / "forced-ids.txt"),
        "seconds": seconds, "versions": {**versions, "transformers": transformers.__version__,
        "python": platform.python_version()},
        "arrays": arrays, "grounding": parse_grounding(text) if args.mode == "grounding" else None,
        "qualification": "reference only; not a benchmark score",
    }
    (args.output / "reference.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
