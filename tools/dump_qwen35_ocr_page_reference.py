"""Bounded original-model page capture; never used by production inference.

The full component dumper remains unchanged. This captures its 77 quantization
prefill boundaries and official generate() scores without storing unused probes.
The hosted driver checks every captured array against the full anchor reference.
"""
import hashlib
import json
import time
from pathlib import Path


def dump_reference(model_dir, image_path, output, max_new_tokens=64, anchor=None, prefill_controls=False):
    import gguf
    import numpy as np
    import torch
    import transformers
    import tokenizers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if transformers.__version__ != "5.5.4" or tokenizers.__version__ != "0.22.2":
        raise ValueError("Reference requires transformers==5.5.4/tokenizers==0.22.2")
    if not 1 <= max_new_tokens <= 512:
        raise ValueError("Page reference token cap outside1..512")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    root, output = Path(model_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
    if inputs["mm_token_type_ids"].shape != inputs["input_ids"].shape:
        raise ValueError("Unaligned official modality labels")
    patches = int(torch.prod(inputs["image_grid_thw"][0]).item())
    if patches > 1024:
        raise ValueError("Prepared fixture exceeds existing native1024-patch bound")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        root, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").eval()
    arrays = {}
    phase = ["prefill"]

    def save(name, tensor):
        if name.startswith("prefill."):
            name = phase[0] + name[len("prefill"):]
        value = tensor.detach().cpu()
        if value.is_floating_point():
            value = value.float()
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite original reference: " + name)
        arrays[name] = value.numpy().copy()

    for name in ("input_ids", "pixel_values", "image_grid_thw"):
        save("input." + name, inputs[name])
    handles = []

    def hook(name):
        def capture(module, args, result):
            save(name, result[0] if isinstance(result, tuple) else result)
        return capture

    language = model.model.language_model
    for index, block in enumerate(language.layers):
        handles.append(block.register_forward_hook(hook(f"prefill.decoder.layer.{index}")))
    handles.append(language.norm.register_forward_hook(hook("prefill.decoder.norm")))
    handles.append(model.model.visual.merger.register_forward_hook(hook("prefill.vision.merger")))

    def language_input(module, args, kwargs):
        save("prefill.decoder.inputs_embeds", kwargs["inputs_embeds"])
    handles.append(language.register_forward_pre_hook(language_input, with_kwargs=True))
    with torch.inference_mode():
        positions = model._prepare_position_ids_for_generation(inputs["input_ids"], dict(inputs))
        prepared = model.prepare_inputs_for_generation(
            **inputs, position_ids=positions, use_cache=True, is_first_iteration=True, logits_to_keep=1)
        def capture_prefill():
            prefill = model(**prepared)
            save("prefill.logits", prefill.logits[:, -1])
            for index, layer in enumerate(prefill.past_key_values.layers):
                for name in ("conv_states", "recurrent_states", "keys", "values"):
                    value = getattr(layer, name, None)
                    if value is not None:
                        save(f"prefill.cache.{index}.{name}", value)
        capture_prefill()
        if prefill_controls:
            phase[0] = "backend_off"
            with torch.backends.mkldnn.flags(enabled=False):
                capture_prefill()
            original_patch = model.model.visual.patch_embed.forward
            def chunked_patch(value):
                return torch.cat([original_patch(value[i:i+1]) for i in range(value.shape[0])], dim=0)
            try:
                phase[0] = "patch_chunked"
                model.model.visual.patch_embed.forward = chunked_patch
                capture_prefill()
            finally:
                model.model.visual.patch_embed.forward = original_patch
            phase[0] = "prefill"
        for handle in handles:
            handle.remove()
        del prepared
        started = time.perf_counter()
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                   use_cache=True, return_dict_in_generate=True, output_scores=True)
        generation_seconds = time.perf_counter() - started
        ids = generated.sequences[:, inputs["input_ids"].shape[1]:]
        save("generated.ids", ids)
        for index, score in enumerate(generated.scores):
            save(f"generated.scores.{index}", score)
    decoded = processor.batch_decode(ids, skip_special_tokens=True)[0]
    arrays["input.decoded_utf8"] = np.frombuffer(decoded.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    # Same model, processor, F32 eager backend, image and generation parameters
    # as the full dumper. No tolerance change or reference substitution.
    if anchor is not None:
        with np.load(anchor) as full:
            for name, value in arrays.items():
                if name not in full or not np.array_equal(value, full[name]):
                    raise ValueError("Compact/full original capture differs: " + name)
    arrays["input.decoded_length"] = np.asarray([len(decoded.encode("utf-8"))], dtype=np.int32)
    if not arrays["input.decoded_utf8"].size:
        arrays["input.decoded_utf8"] = np.zeros(1, dtype=np.int32)
    writer = gguf.GGUFWriter(str(output / "official-delta.gguf"), "qwen35_delta_reference")
    for name, value in arrays.items():
        writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    generated_ids = ids.reshape(-1).tolist()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    result = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
              "torch": torch.__version__, "dtype": "float32", "attention": "eager",
              "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
              "image_size": list(image.size), "grid": inputs["image_grid_thw"].tolist(), "patches": patches,
              "prompt": "", "rendered_prompt": rendered, "max_new_tokens": max_new_tokens,
              "generated_ids": generated_ids, "decoded": decoded,
              "eos_ids": eos, "reached_eos": bool(generated_ids and generated_ids[-1] in eos),
              "stop": "eos" if generated_ids and generated_ids[-1] in eos else "max_tokens",
              "captured_arrays": len(arrays), "anchor_arrays_exact": anchor is not None}
    result["generation_seconds"] = generation_seconds
    if prefill_controls:
        result["cpu_controls"] = {}
        for label in ("backend_off", "patch_chunked"):
            records = []
            for name, value in arrays.items():
                if name.startswith(label + "."):
                    original = arrays["prefill" + name[len(label):]]
                    delta = value.astype(np.float64) - original.astype(np.float64)
                    records.append(dict(stage="prefill" + name[len(label):], elements=value.size,
                                        max_abs=float(np.max(np.abs(delta))), mean_abs=float(np.mean(np.abs(delta))),
                                        rms=float(np.sqrt(np.mean(delta * delta))),
                                        control_norm=float(np.linalg.norm(value.astype(np.float64))),
                                        original_norm=float(np.linalg.norm(original.astype(np.float64)))))
            result["cpu_controls"][label] = records
    result["timing_scope"] = "official generate including vision/prefill/greedy; excludes processor/model load, includes score collection"
    (output / "official-reference.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result
