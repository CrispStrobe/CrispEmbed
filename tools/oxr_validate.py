#!/usr/bin/env python3
"""Capture the original pinned OXR Python model and compare native requests.

Run model work on hosted CI or an authorized machine. This driver calls the
checkpoint's original modeling_oxr.OXR.preprocess_inputs/generate methods. It
never substitutes a rewritten reference graph. The separately timed request
excludes file decoding/model loading and includes preprocessing, vision, cached
greedy generation, transfer and detokenization. Stage capture is untimed and
must preserve completion IDs/text. The original HF generation path considers
completion IDs for repetition penalties; the published vLLM path additionally
considers prompt IDs. These are named distinct profiles, not assumed identical.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import subprocess
import sys
import time

MODEL_ID = "SoMarkAI/OXR-1.0"
MODEL_REVISION = "c708407f9d2b4df658f2543d9132b62bb7221df4"
BLUEPRINT_REVISION = "c1a1dcb312ee9f691404f127785510453c0d9350"
EOS = {151643, 151645}
PROMPTS = {
    "text": "Recognize the content of the image.",
    "formula": "Recognize the formula of the image into KaTex format.",
    "table": "Recognize the table of the image into html format.",
    "code": "Recognize the code of the image into markdown code format.",
    "page": "Recognize the whole page content of the image.",
    "chemical": "Recognize the chemical structure of the image into SMILES.",
}
STAGE_NAMES = (
    ["input.patches", "vision.patch_embed", "vision.position", "vision.embedding"] +
    [f"vision.layer.{i}" for i in range(27)] +
    ["vision.merge", "visual.logits", "visual.probs", "visual.embeddings", "llm.input"] +
    [f"llm.layer.{i}" for i in range(28)] + ["llm.norm", "llm.logits"]
)
REQUEST_FIELDS = [
    "prompt", "max_tokens", "effective_max_tokens", "min_pixels", "max_pixels",
    "repetition_penalty", "hf_generation", "grid_thw", "input_ids", "token_ids",
    "text", "raw_text", "eos", "capped", "finish_reason", "image_width", "image_height",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def converter_module():
    path = Path(__file__).resolve().parents[1] / "models/convert-oxr-to-gguf.py"
    spec = importlib.util.spec_from_file_location("oxr_converter_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def edit_distance(left, right):
    if len(left) > len(right):
        left, right = right, left
    row = list(range(len(left) + 1))
    for i, value in enumerate(right, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(left, 1):
            old = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, previous + (value != other))
            previous = old
    return row[-1]


def text_metrics(expected, actual):
    words = expected.split()
    char_errors, word_errors = edit_distance(expected, actual), edit_distance(words, actual.split())
    return {"exact": expected == actual, "reference_chars": len(expected), "native_chars": len(actual),
            "reference_lines": len(expected.splitlines()), "native_lines": len(actual.splitlines()),
            "character_errors": char_errors, "word_errors": word_errors,
            "cer": char_errors / max(1, len(expected)), "wer": word_errors / max(1, len(words))}


def median_warm(report):
    runs = report["timing"]["runs_ms"]
    if len(runs) < 2 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in runs):
        raise ValueError("timing requires a cold run and at least one finite positive warm run")
    return statistics.median(runs[1:])


def repeat_stable(report):
    runs = report.get("runs", [])
    return bool(runs) and len(runs) == len(report["timing"]["runs_ms"]) and all(
        all(field in run and field in report and run[field] == report[field] for field in REQUEST_FIELDS)
        for run in runs
    )


def request_value_equal(field, left, right):
    if field == "repetition_penalty":
        try:
            return math.isfinite(left) and math.isfinite(right) and struct.pack("<f", left) == struct.pack("<f", right)
        except (TypeError, OverflowError, struct.error):
            return False
    return left == right


def comparison(reference, native, max_slowdown=1.0):
    if not math.isfinite(max_slowdown) or max_slowdown <= 0:
        raise ValueError("max_slowdown must be finite and positive")
    checks = {field: field in reference and field in native and request_value_equal(field, reference[field], native[field])
              for field in REQUEST_FIELDS}
    checks["reference_repeat_stability"] = repeat_stable(reference)
    checks["source_default_cache_neutrality"] = reference.get("cache_neutrality_pass") is True
    checks["confidence_observer_neutrality"] = reference.get("confidence_observer_neutral") is True
    checks["native_repeat_stability"] = repeat_stable(native)
    request_pass = all(checks.values()) and native.get("eos") is True and native.get("capped") is False
    cpu_comparable = reference.get("device") == native.get("device") == "cpu"
    cpu_comparable &= reference.get("threads") == native.get("threads")
    precision_match = reference.get("dtype") == "f32" and native.get("conversion_recipe") == "f32"
    cpu_comparable &= precision_match
    cpu_comparable &= bool(reference.get("machine_fingerprint")) and reference.get("machine_fingerprint") == native.get("machine_fingerprint")
    reference_ms, native_ms = median_warm(reference), median_warm(native)
    ratio = native_ms / reference_ms
    return {
        "schema": "crispembed.oxr.request-comparison.v1", "checks": checks,
        "request_parity_pass": request_pass, "speed_comparable": bool(cpu_comparable),
        "reference_dtype": reference.get("dtype"), "native_conversion_recipe": native.get("conversion_recipe"),
        "primary_precision_profile_match": precision_match,
        "python_warm_median_ms": reference_ms, "native_warm_median_ms": native_ms,
        "native_over_python": ratio, "max_slowdown": max_slowdown,
        "speed_pass": bool(cpu_comparable) and request_pass and ratio <= max_slowdown,
        "text_metrics": text_metrics(reference["text"], native["text"]),
        "confidence": {"reference": reference.get("confidence"), "native": native.get("confidence"),
                       "absolute_error": abs(reference["confidence"] - native["confidence"])
                       if all(isinstance(x, (int, float)) and math.isfinite(x) for x in
                              [reference.get("confidence"), native.get("confidence")]) else None},
        "qualification": {"representative_document_quality": False,
                          "intermediate_numerical_parity": "requires separate stage result",
                          "vllm_throughput_parity": False, "other_hardware_speed": False},
    }


def machine_fingerprint():
    # Avoid publishing a machine hostname in retained public reports.
    identity = platform.node() + "\n" + platform.machine() + "\n" + platform.processor()
    return hashlib.sha256(identity.encode()).hexdigest()


def package_versions():
    result = {}
    for name in ["torch", "torchvision", "transformers", "tokenizers", "Pillow", "numpy", "gguf", "safetensors"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def validate_runtime(versions):
    expected = {"torch": "2.11.0", "torchvision": "0.26.0", "transformers": "5.13.1",
                "tokenizers": "0.22.2", "Pillow": "12.3.0", "numpy": "1.26.4", "gguf": "0.17.1",
                "safetensors": "0.8.0"}
    for name, wanted in expected.items():
        if (versions.get(name) or "").split("+")[0] != wanted:
            raise ValueError(f"reference package drift: {name}={versions.get(name)}, required {wanted}")


def dump_stages(path, stage, request, dtype, image_hash):
    import gguf
    import numpy as np
    missing = sorted(set(STAGE_NAMES) - set(stage))
    extra = sorted(set(stage) - set(STAGE_NAMES))
    if missing or extra:
        raise RuntimeError(f"stage inventory mismatch missing={missing}, extra={extra}")
    for name, value in stage.items():
        if not np.isfinite(value).all():
            raise RuntimeError(f"nonfinite original stage: {name}")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen stage reference {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(path), "oxr_ref")
    try:
        writer.add_string("oxr.model", MODEL_ID)
        writer.add_string("oxr.model_revision", MODEL_REVISION)
        writer.add_string("oxr.blueprint_revision", BLUEPRINT_REVISION)
        writer.add_string("oxr.dtype", dtype)
        writer.add_string("oxr.prompt", request["prompt"])
        writer.add_string("oxr.image_sha256", image_hash)
        writer.add_string("oxr.stage_names", json.dumps(STAGE_NAMES))
        writer.add_bool("oxr.hf_generation", request["hf_generation"])
        writer.add_float32("oxr.repetition_penalty", request["repetition_penalty"])
        for key in ["min_pixels", "max_pixels", "max_tokens", "effective_max_tokens"]:
            writer.add_uint32("oxr." + key, request[key])
        for axis, value in zip("thw", request["grid_thw"]):
            writer.add_uint32("oxr.grid_" + axis, value)
        tensors = {**stage, "input_ids": np.asarray(request["input_ids"], dtype=np.float32),
                   "grid_thw": np.asarray(request["grid_thw"], dtype=np.float32),
                   "output_ids": np.asarray(request["token_ids"], dtype=np.float32)}
        for name, value in tensors.items():
            writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()


def reference(args):
    if args.output.exists():
        raise FileExistsError(f"refusing frozen reference output {args.output}")
    if args.dump_reference and args.dump_reference.exists():
        raise FileExistsError(f"refusing frozen stage reference {args.dump_reference}")
    versions = package_versions()
    validate_runtime(versions)
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
    converter = converter_module()
    checkpoint = converter.resolve_model(args.model)
    hashes = converter.verify_source_files(checkpoint)
    converter.validate_config(*[json.loads((checkpoint / name).read_text(encoding="utf-8")) for name in [
        "config.json", "preprocessor_config.json", "generation_config.json"]])
    entries = converter.read_safetensors_header(checkpoint / "model.safetensors")
    converter.validate_tensors(entries)
    converter.verify_tied_weights(checkpoint / "model.safetensors", entries)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested CUDA reference is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    # modeling_oxr registers Siglip2NavitConfig from its own copy of configuration_oxr.
    # AutoConfig loads that file again as a separately hashed dynamic module, so its
    # vit_config class would not be the registered one (AutoModel.from_config then
    # raises). Build the config from the modeling module's own configuration module.
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    model_class = get_class_from_dynamic_module("modeling_oxr.OXR", str(checkpoint), local_files_only=True)
    config_module = sys.modules[model_class.__module__.rsplit(".", 1)[0] + ".configuration_oxr"]
    config = config_module.OXRConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.llm_config._attn_implementation = args.attention
    start = time.perf_counter()
    model = model_class.from_pretrained(str(checkpoint), config=config,
        trust_remote_code=True, local_files_only=True, dtype=dtype,
        attn_implementation={"": "eager", "llm_config": args.attention, "vit_config": "eager"}).to(device=args.device, dtype=dtype).eval()
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - start) * 1000
    # Transformers 5 builds the model on the meta device and loads checkpoint
    # tensors; non-persistent buffers computed in __init__ are left uninitialised
    # unless _init_weights rebuilds them, and modeling_oxr's does not. Rebuild them
    # with the checkpoint module's own constants/classes (no formula is retyped).
    remote = sys.modules[model_class.__module__]
    restored = []
    vocab = model.config.visual_vocab_size
    indicator = torch.arange(vocab - len(remote.INDICATOR_IDS), vocab, dtype=torch.long, device=args.device)
    model.indicator_token_indices = indicator
    restored.append("indicator_token_indices")
    for name, module in model.named_modules():
        if isinstance(module, remote.VisionRotaryEmbedding):
            module.inv_freq = remote.VisionRotaryEmbedding(module.dim, module.theta).inv_freq.to(args.device)
            restored.append(name + ".inv_freq")
    if int(model.indicator_token_indices.min()) < 0 or int(model.indicator_token_indices.max()) >= model.vte.num_embeddings:
        raise ValueError("visual indicator indices fall outside the visual embedding table")
    # Reject silently substituted model classes/processors. Check source identity,
    # then execute the original checkpoint functions without a mathematical patch.
    model_source = Path(sys.modules[type(model).__module__].__file__)
    if sha256(model_source) != hashes["modeling_oxr.py"]:
        raise ValueError("loaded OXR model code differs from authenticated checkpoint source")
    if model.llm.config._attn_implementation != args.attention:
        raise ValueError("reference attention implementation differs from requested profile")
    processor = model.visual_tokenizer.image_processor
    with Image.open(args.image) as source_image:
        image = source_image.convert("RGB")
    prompt = args.prompt if args.prompt is not None else PROMPTS[args.task]
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    timings, runs, confidence_scores = [], [], []
    generation_options = {
        "do_sample": False, "repetition_penalty": args.repetition_penalty,
        "eos_token_id": sorted(EOS), "pad_token_id": 151643,
    }
    if model.llm.generation_config.use_cache is not False:
        raise ValueError("source-default generation cache setting changed from pinned use_cache=False")

    class ConfidenceObserver(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # The registered observer is appended after original processors. It
            # returns the identical score tensor and retains one scalar only.
            selected = int(torch.argmax(scores[0]).item())
            confidence_scores.append(float(torch.softmax(scores[0].double(), dim=-1)[selected].item()))
            return scores

    def predict(use_cache=True, observe_scores=False):
        if args.device == "cuda":
            torch.cuda.synchronize()
        begin = time.perf_counter()
        ids, pixels, grid = model.preprocess_inputs(messages, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        input_ids = ids[0].tolist()
        max_new = args.max_tokens
        if max_new <= 0 or len(input_ids) + max_new > 32768:
            raise ValueError("requested generation exceeds the native 32768-token profile")
        ids = ids.to(args.device)
        pixels = pixels.to(device=args.device, dtype=dtype)
        grid = grid.to(args.device)
        kwargs = {**generation_options, "max_new_tokens": max_new}
        if use_cache is not None:
            kwargs["use_cache"] = use_cache
        if observe_scores:
            kwargs["logits_processor"] = LogitsProcessorList([ConfidenceObserver()])
        output = model.generate(ids, pixel_values=pixels, grid_thws=grid, **kwargs)
        if hasattr(output, "sequences"):
            output = output.sequences
        token_ids = output[0].detach().cpu().tolist()
        text = model.text_tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        raw_text = model.text_tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000
        eos = bool(token_ids and token_ids[-1] in EOS)
        result = {
            "prompt": prompt, "max_tokens": args.max_tokens, "effective_max_tokens": max_new,
            "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
            "repetition_penalty": args.repetition_penalty, "hf_generation": True,
            "input_ids": input_ids, "prompt_ids": input_ids, "token_ids": token_ids,
            "grid_thw": grid.detach().cpu()[0].tolist(), "text": text, "raw_text": raw_text,
            "eos": eos, "capped": not eos and len(token_ids) >= max_new,
            "finish_reason": "stop" if eos else "length", "image_width": image.width,
            "image_height": image.height,
        }
        return result, elapsed

    with torch.inference_mode():
        for _ in range(args.repeat):
            result, elapsed = predict()
            runs.append(result)
            timings.append(elapsed)
    if any(run != runs[0] for run in runs[1:]):
        raise RuntimeError("original Python repeated resident requests changed output or request IDs")
    # Compare a separate unobserved source-default request with no cache override.
    # Although checkpoint configuration says use_cache=False, Transformers5.13.1
    # forces caching for decoder-only inputs_embeds. Both requests should thus
    # use the same effective cache behavior; retain failures instead of assuming it.
    try:
        with torch.inference_mode():
            source_default, source_default_ms = predict(use_cache=None)
    except Exception as error:
        write_json(args.output, {"schema": "crispembed.oxr.reference-failure.v1",
            "status": "source_default_failed", "model_revision": MODEL_REVISION,
            "cached_runs": runs, "cached_timings_ms": timings,
            "source_default_error": str(error), "speed_qualified": False})
        raise
    cache_neutral = source_default == runs[0]
    if not cache_neutral:
        write_json(args.output, {"schema": "crispembed.oxr.reference-failure.v1",
            "status": "cache_neutrality_failed", "model_revision": MODEL_REVISION,
            "cached_runs": runs, "cached_timings_ms": timings,
            "source_default": source_default, "source_default_ms": source_default_ms,
            "speed_qualified": False})
        return 2
    stage = {}
    if args.dump_reference:
        vision = model.visual_tokenizer.vit.vision_model
        grid = torch.tensor([runs[0]["grid_thw"]], device=args.device)
        window_index, _ = vision.encoder.get_window_index(grid)
        reverse = torch.argsort(window_index)
        handles = []

        def save(name, value):
            if name not in stage:
                if isinstance(value, (tuple, list)):
                    value = value[0]
                stage[name] = value.detach().float().cpu().numpy().copy()

        def hook(name, reshape=None):
            def callback(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                if reshape is not None:
                    value = reshape(value)
                save(name, value)
            return callback

        def vision_input(_module, args_, kwargs):
            save("input.patches", args_[0] if args_ else kwargs["pixel_values"])
        handles.append(model.visual_tokenizer.register_forward_pre_hook(vision_input, with_kwargs=True))
        handles.append(vision.embeddings.patch_embedding.register_forward_hook(hook("vision.patch_embed", lambda x: x.reshape(-1, 1152))))
        handles.append(vision.embeddings.register_forward_hook(hook("vision.embedding")))
        for i, layer in enumerate(vision.encoder.layers):
            handles.append(layer.register_forward_hook(hook(f"vision.layer.{i}",
                lambda x: x.reshape(-1, 4, 1152)[reverse].reshape(-1, 1152))))
        def merge_input(_module, args_):
            save("vision.merge", args_[0])
        handles.append(model.visual_tokenizer.head.register_forward_pre_hook(merge_input))
        handles.append(model.visual_tokenizer.head.register_forward_hook(hook("visual.logits")))
        handles.append(model.visual_tokenizer.register_forward_hook(hook("visual.probs", lambda x: x[:, :65532])))
        def visual_embeddings(_module, args_, output):
            if args_[0].is_floating_point():
                save("visual.embeddings", output)
        handles.append(model.vte.register_forward_hook(visual_embeddings))
        def llm_input(_module, _args, kwargs):
            value = kwargs.get("inputs_embeds")
            if value is not None:
                save("llm.input", value[0])
        handles.append(model.llm.model.register_forward_pre_hook(llm_input, with_kwargs=True))
        for i, layer in enumerate(model.llm.model.layers):
            handles.append(layer.register_forward_hook(hook(f"llm.layer.{i}", lambda x: x[0])))
        handles.append(model.llm.model.norm.register_forward_hook(hook("llm.norm", lambda x: x[0])))
        handles.append(model.llm.lm_head.register_forward_hook(hook("llm.logits", lambda x: x[0, -1:, :])))
        # Capture the actual original function-local positional tensor. This is
        # observation only: the source forward, operands and arithmetic stay intact.
        embedding_code = type(vision.embeddings).forward.__code__
        previous_profile = sys.getprofile()
        def observe_position(frame, event, _arg):
            if event == "return" and frame.f_code is embedding_code:
                save("vision.position", frame.f_locals["pos_embed_new"])
        try:
            sys.setprofile(observe_position)
            with torch.inference_mode():
                observed, _ = predict(observe_scores=True)
            if observed != runs[0]:
                raise RuntimeError("stage observer changed original completion or request metadata")
        finally:
            sys.setprofile(previous_profile)
            for handle in handles:
                handle.remove()
        dump_stages(args.dump_reference, stage, runs[0], args.dtype, sha256(args.image))
    else:
        with torch.inference_mode():
            observed, _ = predict(observe_scores=True)
        if observed != runs[0]:
            raise RuntimeError("confidence observer changed original completion or request metadata")
    if len(confidence_scores) != len(runs[0]["token_ids"]) or not confidence_scores:
        raise RuntimeError("confidence observer did not visit every generated token")
    confidence = sum(confidence_scores) / len(confidence_scores)
    report = {
        "schema": "crispembed.oxr.request.v1", "engine": "original-transformers",
        "profile": "original-hf-cached-with-explicit-pixel-budget", "model": MODEL_ID,
        "model_revision": MODEL_REVISION, "blueprint_revision": BLUEPRINT_REVISION,
        "source_sha256": hashes, "packages": versions,
        "modeling_source_sha256": sha256(model_source),
        "qwen3_source_sha256": sha256(Path(sys.modules[type(model.llm).__module__].__file__)),
        "generation_source_sha256": sha256(Path(sys.modules["transformers.generation.utils"].__file__)),
        "processor": type(processor).__name__, "dtype": args.dtype,
        "device": args.device, "attention": args.attention, "threads": args.threads,
        "machine_fingerprint": machine_fingerprint(), "platform": platform.platform(),
        "image_sha256": sha256(args.image), **runs[0], "runs": runs,
        "generation_kwargs": {**generation_options, "max_new_tokens": args.max_tokens, "use_cache": True},
        "source_default_cache_control_generation_kwargs": {**generation_options, "max_new_tokens": args.max_tokens},
        "source_default_configured_cache": False, "source_default_effective_cache": True,
        "cache_default_reason": "Transformers5.13.1 forces caching for decoder-only inputs_embeds",
        "source_default": source_default,
        "source_default_ms": source_default_ms, "cache_neutrality_pass": cache_neutral,
        "confidence": confidence, "confidence_token_probabilities": confidence_scores,
        "confidence_observer_neutral": True,
        "restored_nonpersistent_buffers": restored,
        "timing": {"load_ms": load_ms, "runs_ms": timings,
                   "scope": "resident image preprocessing+vision+prefill+cached greedy decode+transfer+detokenize"},
        "stage_count": len(stage), "stages": list(stage),
        "stage_reference_sha256": sha256(args.dump_reference) if args.dump_reference else None,
        "qualification": {"native_parity": False, "vllm_parity": False, "full_document_pipeline": False},
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "generated_tokens": len(runs[0]["token_ids"]),
                      "eos": runs[0]["eos"], "warm_median_ms": median_warm(report), "stage_count": len(stage)}))
    return 0


def compare(args):
    ref = json.loads(args.reference.read_text(encoding="utf-8"))
    if ref.get("model_revision") != MODEL_REVISION or ref.get("blueprint_revision") != BLUEPRINT_REVISION:
        raise ValueError("reference source pin does not match OXR integration")
    if sha256(args.image) != ref["image_sha256"]:
        raise ValueError("source image bytes differ from frozen reference")
    if ref.get("engine") != "original-transformers" or ref.get("hf_generation") is not True:
        raise ValueError("this comparator requires the named original HF reference profile")
    env = {**os.environ, "CRISPEMBED_OXR": "1", "CRISPEMBED_ACCEPT_OXR_LICENSE": "1", "CRISPEMBED_OXR_FORCE_CPU": "1"}
    command = [str(args.native), "--model", str(args.gguf), "--image", str(args.image),
        "--prompt", ref["prompt"], "--max-tokens", str(ref["max_tokens"]),
        "--threads", str(ref["threads"]), "--repeat", str(len(ref["timing"]["runs_ms"])),
        "--repetition-penalty", str(ref["repetition_penalty"]), "--hf-generation", "1",
        "--min-pixels", str(ref["min_pixels"]), "--max-pixels", str(ref["max_pixels"])]
    start = time.perf_counter()
    process = subprocess.run(command, env=env, text=True, encoding="utf-8", capture_output=True)
    elapsed = (time.perf_counter() - start) * 1000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.output) + ".native.stdout.json").write_text(process.stdout, encoding="utf-8")
    Path(str(args.output) + ".native.stderr.log").write_text(process.stderr, encoding="utf-8")
    if process.returncode:
        write_json(args.output, {"status": "native_failed", "returncode": process.returncode,
                                "native_subprocess_ms": elapsed})
        return 2
    native = json.loads(process.stdout)
    native.setdefault("input_ids", native.get("prompt_ids"))
    native["machine_fingerprint"] = machine_fingerprint()
    native["image_sha256"] = sha256(args.image)
    report = comparison(ref, native, args.max_slowdown)
    report.update({"reference": ref, "native": native, "reference_sha256": sha256(args.reference),
        "gguf_sha256": sha256(args.gguf), "native_executable_sha256": sha256(args.native),
        "native_subprocess_ms": elapsed,
        "native_environment": {key: env.get(key) for key in ["CRISPEMBED_OXR", "CRISPEMBED_OXR_FORCE_CPU",
            "CRISPEMBED_OXR_REUSE_DECODE_GRAPH", "CRISPEMBED_OXR_FLASH_ATTN"]}})
    if args.gold:
        gold = args.gold.read_text(encoding="utf-8")
        report["ground_truth"] = {"sha256": sha256(args.gold), "python": text_metrics(gold, ref["text"]),
                                  "native": text_metrics(gold, native["text"])}
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ["request_parity_pass", "speed_comparable", "speed_pass", "native_over_python"]}))
    return 0 if report["request_parity_pass"] and report["speed_pass"] else 1


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def token_limit(value):
    value = int(value)
    if not 1 <= value <= 32768:
        raise argparse.ArgumentTypeError("must be in [1,32768]")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("--model", default=MODEL_ID)
    ref.add_argument("--image", type=Path, required=True)
    prompt = ref.add_mutually_exclusive_group()
    prompt.add_argument("--task", choices=list(PROMPTS), default="text")
    prompt.add_argument("--prompt")
    ref.add_argument("--max-tokens", type=token_limit, default=1024)
    ref.add_argument("--min-pixels", type=positive_int, default=200704)
    ref.add_argument("--max-pixels", type=positive_int, default=921600,
                     help="published plugin budget; use2408448 to reproduce original HF default budget")
    ref.add_argument("--repetition-penalty", type=positive_float, default=1.0)
    ref.add_argument("--threads", type=positive_int, default=4)
    ref.add_argument("--repeat", type=positive_int, default=3)
    ref.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ref.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="f32")
    ref.add_argument("--attention", choices=["eager", "sdpa"], default="sdpa")
    ref.add_argument("--output", type=Path, required=True)
    ref.add_argument("--dump-reference", type=Path)
    ref.set_defaults(run=reference)
    comp = commands.add_parser("compare")
    comp.add_argument("--reference", type=Path, required=True)
    comp.add_argument("--native", type=Path, required=True)
    comp.add_argument("--gguf", type=Path, required=True)
    comp.add_argument("--image", type=Path, required=True)
    comp.add_argument("--gold", type=Path)
    comp.add_argument("--max-slowdown", type=positive_float, default=1.0)
    comp.add_argument("--output", type=Path, required=True)
    comp.set_defaults(run=compare)
    args = parser.parse_args(argv)
    if args.command == "reference":
        if args.repeat < 2:
            parser.error("--repeat needs cold and warm runs (at least2)")
        if args.max_pixels < args.min_pixels or args.max_pixels > 2408448:
            parser.error("pixel budget must satisfy min_pixels <= max_pixels <=2408448")
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
