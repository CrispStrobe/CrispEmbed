"""Bounded original-model page capture; never used by production inference.

The full component dumper remains unchanged. This captures its 77 quantization
prefill boundaries and official generate() scores without storing unused probes.
The hosted driver checks every captured array against the full anchor reference.
"""
import hashlib
import json
import time
from pathlib import Path


def validate_token_cap(cap, long_document=False):
    limit = 1024 if long_document else 512
    if type(cap) is not int or not 1 <= cap <= limit:
        raise ValueError(f"Page reference token cap outside1..{limit}")


def dump_reference(model_dir, image_path, output, max_new_tokens=64, anchor=None, prefill_controls=False,
                   vision_probes=False, norm_statistics=False, native_geometry=None, attention_trace=False,
                   prompt="", long_document=False):
    import gguf
    import numpy as np
    import torch
    import transformers
    import tokenizers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    if transformers.__version__ != "5.5.4" or tokenizers.__version__ != "0.22.2":
        raise ValueError("Reference requires transformers==5.5.4/tokenizers==0.22.2")
    validate_token_cap(max_new_tokens, long_document=long_document)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    root, output = Path(model_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    if prompt:
        messages[0]["content"].append({"type": "text", "text": prompt})
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
    if inputs["mm_token_type_ids"].shape != inputs["input_ids"].shape:
        raise ValueError("Unaligned official modality labels")
    patches = int(torch.prod(inputs["image_grid_thw"][0]).item())
    if patches > 1024:
        raise ValueError("Prepared fixture exceeds existing native1024-patch bound")
    if attention_trace and not vision_probes:
        raise ValueError("Attention capture requires original vision probes")
    if norm_statistics and not vision_probes:
        raise ValueError("Norm statistics require original vision probes")
    geometry = None
    if native_geometry is not None:
        import subprocess
        executable, weights = native_geometry
        grid = inputs["image_grid_thw"].tolist()
        if len(grid)!=1 or grid[0][0]!=1:
            raise ValueError("Native geometry control requires one frame")
        process = subprocess.run([str(executable), "--geometry", str(weights), str(grid[0][1]), str(grid[0][2]), str(output)], capture_output=True, text=True)
        (output / "geometry.log").write_text(process.stdout + process.stderr)
        process.check_returncode()
        if "GEOMETRY finite=1 controls=1" not in process.stdout:
            raise ValueError("Native geometry independence controls missing")
        geometry = [np.fromfile(output / ("native-rotary-" + name + ".bin"), dtype=np.float32) for name in ("cos","sin")]
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

    # The same original modules/boundaries as the full dumper, expanded to all
    # twelve blocks for localization. Hooks observe original calls; no native
    # values or reconstructed reference arithmetic enter this capture.
    visual = model.model.visual
    original_position = visual.fast_pos_embed_interpolate
    statistics_controls = {}
    if vision_probes:
        def prehook(name):
            def capture(module, args):
                save(name, args[0])
            return capture
        for index, block in enumerate(visual.blocks):
            prefix = f"prefill.vision.probe.{index}."
            handles.append(block.register_forward_pre_hook(prehook(prefix + "input")))
            handles.append(block.register_forward_hook(hook(f"prefill.vision.layer.{index}")))
            for module_name, field in [("norm1", "norm1"), ("attn.qkv", "qkv"),
                                       ("attn", "attention"), ("norm2", "norm2"),
                                       ("mlp.linear_fc1", "up"), ("mlp.act_fn", "activation"),
                                       ("mlp.linear_fc2", "down")]:
                submodule = block
                for part in module_name.split("."):
                    submodule = getattr(submodule, part)
                handles.append(submodule.register_forward_hook(hook(prefix + field)))
            handles.append(block.norm2.register_forward_pre_hook(prehook(prefix + "norm2_input")))
            handles.append(block.attn.proj.register_forward_pre_hook(prehook(prefix + "pooled")))
        def vision_input(module, args, kwargs):
            save("prefill.vision.embedded", args[0])
            cosine, sine = kwargs["position_embeddings"]
            save("prefill.vision.rotary_cos", cosine)
            save("prefill.vision.rotary_sin", sine)
        handles.append(visual.blocks[0].register_forward_pre_hook(vision_input, with_kwargs=True))
        def position_capture(grid):
            value = original_position(grid)
            save("prefill.vision.position", value)
            return value
        visual.fast_pos_embed_interpolate = position_capture
        handles.append(visual.patch_embed.register_forward_hook(hook("prefill.vision.patch_embed")))
        handles.append(visual.merger.norm.register_forward_pre_hook(prehook("prefill.vision.merger.input")))
        for name, label in [("norm", "norm"), ("linear_fc1", "up"), ("act_fn", "activation")]:
            handles.append(getattr(visual.merger, name).register_forward_hook(hook("prefill.vision.merger." + label)))

    if norm_statistics:
        def statistics_hook(name):
            def capture(module, args, result):
                if args[0].dtype != torch.float32 or module.eps != 1e-6:
                    raise ValueError("Statistics oracle requires F32")
                value, mean, rstd = torch.ops.aten.native_layer_norm.default(
                    args[0], module.normalized_shape, module.weight, module.bias, module.eps)
                if not torch.equal(value, result):
                    raise ValueError("Native LayerNorm oracle differs from original module: " + name)
                statistics_controls[phase[0]] = statistics_controls.get(phase[0], 0) + 1
                save(name + ".mean", mean.reshape(1,-1))
                save(name + ".rstd", rstd.reshape(1,-1))
            return capture
        for i, block in enumerate(visual.blocks):
            for label in ("norm1","norm2"):
                handles.append(getattr(block,label).register_forward_hook(statistics_hook(f"prefill.vision.probe.{i}.{label}")))
        handles.append(visual.merger.norm.register_forward_hook(statistics_hook("prefill.vision.merger.norm")))

    attention_result = None
    if attention_trace:
        import importlib.util
        spec = importlib.util.spec_from_file_location('qwen35_attention_reference', Path(__file__).with_name('qwen35_attention_reference.py'))
        observer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(observer)
        check_attention_pool, finish_attention, restore_attention = observer.install(visual, phase, arrays, save, patches)

    def language_input(module, args, kwargs):
        save("prefill.decoder.inputs_embeds", kwargs["inputs_embeds"])
    handles.append(language.register_forward_pre_hook(language_input, with_kwargs=True))
    from contextlib import ExitStack
    with torch.inference_mode(), ExitStack() as cleanup:
        if attention_trace: cleanup.callback(restore_attention)
        positions = model._prepare_position_ids_for_generation(inputs["input_ids"], dict(inputs))
        prepared = model.prepare_inputs_for_generation(
            **inputs, position_ids=positions, use_cache=True, is_first_iteration=True, logits_to_keep=1)
        def capture_prefill():
            prefill = model(**prepared)
            if attention_trace: check_attention_pool()
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
        if attention_trace: attention_result = finish_attention(prefill_controls)
        for handle in handles:
            handle.remove()
        visual.fast_pos_embed_interpolate = original_position
        del prepared
        if geometry is not None:
            head_width = visual.config.hidden_size // visual.config.num_heads
            if any(v.size!=patches*head_width or not np.isfinite(v).all() for v in geometry):
                raise ValueError("Malformed native rotary arrays")
            original_geometry = tuple(torch.from_numpy(arrays["prefill.vision.rotary_"+name]) for name in ("cos","sin"))
            native_positions = tuple(torch.from_numpy(v.reshape(patches,head_width)) for v in geometry)
            cu = torch.tensor([0,patches], dtype=torch.int32)
            for i, block in enumerate(visual.blocks):
                block_input = torch.from_numpy(arrays[f"prefill.vision.probe.{i}.input"])
                original_output = block(block_input,cu_seqlens=cu,position_embeddings=original_geometry)
                if not torch.equal(original_output,torch.from_numpy(arrays[f"prefill.vision.layer.{i}"])):
                    raise ValueError("Original block replay differs from unchanged hook")
                controlled = block(block_input,cu_seqlens=cu,position_embeddings=native_positions)
                save(f"native_geometry.vision.layer.{i}",controlled)
        started = time.perf_counter()
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                   use_cache=True, return_dict_in_generate=True, output_scores=True)
        generation_seconds = time.perf_counter() - started
        ids = generated.sequences[:, inputs["input_ids"].shape[1]:]
        save("generated.ids", ids)
        for index, score in enumerate(generated.scores):
            save(f"generated.scores.{index}", score)
    # Diagnostic confidence from the actual original generate scores, including
    # EOS. Normalize in F64 as the native public confidence API does; generation
    # and every captured score stay unchanged. This is not an accuracy measure.
    confidence = []
    for index, score in enumerate(generated.scores):
        values = score[0].double()
        selected = int(ids[0, index])
        confidence.append(float((1.0 / torch.exp(values - values[selected]).sum()).item()))
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
              "prompt": prompt, "rendered_prompt": rendered, "max_new_tokens": max_new_tokens,
              "generated_ids": generated_ids, "decoded": decoded,
              "eos_ids": eos, "reached_eos": bool(generated_ids and generated_ids[-1] in eos),
              "stop": "eos" if generated_ids and generated_ids[-1] in eos else "max_tokens",
              "captured_arrays": len(arrays), "anchor_arrays_exact": anchor is not None}
    result["selected_token_confidences"] = confidence
    result["mean_confidence"] = sum(confidence) / len(confidence)
    result["confidence_scope"] = "Original generated F32 scores, F64 exponential normalization of selected token, includes EOS; no calibration claim"
    result["generation_seconds"] = generation_seconds
    if attention_trace: result["attention_trace"] = attention_result
    if norm_statistics:
        if any(v!=25 for v in statistics_controls.values()) or len(statistics_controls)!=(3 if prefill_controls else 1):
            raise ValueError("Incomplete original mean/rstd controls")
        result["norm_statistics"] = dict(original_kernel="aten.native_layer_norm", module_output_exact_controls=statistics_controls,
            cpu_capability=torch.backends.cpu.get_cpu_capability(), torch_git_version=torch.version.git_version,
            reference_vector_width=None, vector_width_scope="not exposed by Python; candidate widths are tested, not assumed")
    if geometry is not None:
        result["native_geometry"] = dict(original_block_replay_exact=12, pixel_independence_exact=True,
            maximum_original_rotary_error=max(float(np.max(np.abs(v.reshape(patches,-1)-arrays["prefill.vision.rotary_"+name]))) for v,name in zip(geometry,("cos","sin"))),
            arrays_sha256={name:hashlib.sha256(v.tobytes()).hexdigest() for name,v in zip(("cos","sin"),geometry)})
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
