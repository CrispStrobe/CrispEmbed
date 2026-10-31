#!/usr/bin/env python3
"""Xiaomi-OCR-0 phase 1: freeze the original blueprint, never a manual clone.

Produces, from the actual pinned Transformers 5.8.1 implementation:

* the verified full checkpoint digest and the audited metadata bytes,
* the live meta-model tensor schema compared with the safetensors header,
* the rendered document prompt, stops and resolved generation config,
* processor inputs for three licensed upstream example pages,
* GPU BF16 greedy generations with ids, raw text and wall times. This run uses
  SDPA as a memory-saving alternative and compares the first two raw texts to
  the preserved eager-attention run; it is not an eager-equivalent assertion.

No conversion, no native build, no quantizer call, no publication, no alias.
Everything is written to /kaggle/working/xiaomi-ocr-0-blueprint.
"""

import hashlib
import inspect
import json
import os
import platform
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODEL = "xiaomi-research/Xiaomi-OCR-0"
REVISION = "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30"
CHECKPOINT_BYTES = 1746937856
CHECKPOINT_SHA256 = "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348"
TRANSFORMERS = "5.8.1"
TOKENIZERS = "0.22.2"
# Whole-page document task, verbatim from the upstream evaluation entry point.
DOCUMENT_PROMPT = (
    "Extract all information from the main body of the document image and represent it "
    "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
    "format, formulas in the document should be represented using LATEX format, and the "
    "parsing should be organized according to the reading order."
)
MAX_EDGE = 2048
PAGES = ["qianziwen.jpeg", "receipt.jpg", "dense-equations.jpg"]
MAX_NEW_TOKENS = 512
EAGER_V1_TEXT_SHA256 = {
    "qianziwen": "8db51e73e76a4f407a2c4195c0bf3f11cef57033df78e8b188d43c333fb0a702",
    "receipt": "e680f24e8b5fccebac7442ef16cb9a2e47e7542215f95422f8e38399ed162cb1",
}
EAGER_V1_PIXELS_SHA256 = {
    "qianziwen": "9cb5fa12514f6288a84edf0a0a12dd289a3aa977df6a1aea46feab1396b1de77",
    "receipt": "28e5e67462e3d0feb85a6c5d0fefed57ac7008bdeb81a59b80048264df32f112",
}
WORK = Path("/kaggle/working/xiaomi-ocr-0-blueprint")
SCRATCH = Path("/kaggle/temp/xiaomi")
REPORT = {"scope": "original blueprint only; no native inference or quantization",
          "publication": "none", "native_alias_qualified": False}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def save():
    (WORK / "evidence.json").write_text(json.dumps(REPORT, indent=2, allow_nan=False) + "\n")


def step(name):
    print(f"=== {name}", flush=True)
    REPORT["phase"] = name
    save()


def fit_image(image):
    """Downscale to the supported bound, preserving aspect ratio (upstream rule)."""
    from PIL import Image
    if image.width > MAX_EDGE or image.height > MAX_EDGE:
        image = image.copy()
        image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
    return image


def resolve_generation(processor, config):
    from transformers import GenerationConfig
    generation = GenerationConfig.from_model_config(config)
    stops = sorted({int(processor.tokenizer.eos_token_id), int(generation.eos_token_id),
                    int(config.text_config.eos_token_id)})
    return generation, stops


def capture_qian_prefill(model, inputs):
    """Retain original-model activations; these are not native parity evidence."""
    import gguf
    import numpy as np
    import torch

    expected = {
        "input_ids": ([1, 568], "int64", "c7efff031d804325c4ad6ddeb60dc981e876d158ef52b1cd1b02cec37806b740"),
        "attention_mask": ([1, 568], "int64", "ba5efd3ed40e2ed362169f1e454ba379932c1e72c304944b27a6d29878ef1b18"),
        "mm_token_type_ids": ([1, 568], "int64", "ab91cedb160c44d473733fa5d33e457e46bff9a533e1f4bc7c6c902b72181d24"),
        "pixel_values": ([1984, 1536], "float32", "9cb5fa12514f6288a84edf0a0a12dd289a3aa977df6a1aea46feab1396b1de77"),
        "image_grid_thw": ([1, 3], "int64", "dbde42b7690a521a292f75e8143f71b37d486c09640437835cb4033b67acc471"),
    }
    if set(inputs) != set(expected):
        raise ValueError("qianziwen processor fields differ from v2")
    arrays, metadata, handles = {}, {}, []

    def record(name, tensor):
        if not torch.is_tensor(tensor):
            return
        metadata[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        value = tensor.detach().cpu()
        if value.is_floating_point():
            value = value.float()
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite original prefill: {name}")
        arrays[name] = value.numpy().copy()

    for name, tensor in inputs.items():
        record("input." + name, tensor)
        value = arrays["input." + name]
        if (list(value.shape), str(value.dtype), hashlib.sha256(value.tobytes()).hexdigest()) != expected[name]:
            raise ValueError(f"qianziwen processor field differs from v2: {name}")

    def hook(name):
        def capture(_module, _args, value):
            if isinstance(value, tuple):
                value = value[0]
            elif not torch.is_tensor(value):
                value = value.pooler_output
            record(name, value)
        return capture

    def vision_input(_module, args, kwargs):
        record("prefill.vision.embedded", args[0])
        cosine, sine = kwargs["position_embeddings"]
        record("prefill.vision.rotary_cos", cosine)
        record("prefill.vision.rotary_sin", sine)

    def language_input(_module, _args, kwargs):
        for name in ("position_ids", "inputs_embeds"):
            record("prefill.decoder." + name, kwargs.get(name))

    visual, language = model.model.visual, model.model.language_model
    original_position = visual.fast_pos_embed_interpolate

    def capture_position(grid):
        value = original_position(grid)
        record("prefill.vision.position", value)
        return value

    try:
        visual.fast_pos_embed_interpolate = capture_position
        handles.append(visual.blocks[0].register_forward_pre_hook(vision_input, with_kwargs=True))
        for index, block in enumerate(visual.blocks):
            handles.append(block.register_forward_hook(hook(f"prefill.vision.layer.{index}")))
        for name, label in [("norm", "norm"), ("linear_fc1", "up"), ("act_fn", "activation")]:
            handles.append(getattr(visual.merger, name).register_forward_hook(hook("prefill.vision.merger." + label)))
        for name in ("patch_embed", "merger"):
            handles.append(getattr(visual, name).register_forward_hook(hook("prefill.vision." + name)))
        handles.append(language.register_forward_pre_hook(language_input, with_kwargs=True))
        handles.append(language.layers[0].register_forward_hook(hook("prefill.decoder.layer.0")))
        handles.append(language.norm.register_forward_hook(hook("prefill.decoder.norm")))
        with torch.inference_mode():
            positions = model._prepare_position_ids_for_generation(inputs["input_ids"], dict(inputs))
            prepared = model.prepare_inputs_for_generation(**inputs, position_ids=positions, use_cache=True,
                                                           is_first_iteration=True, logits_to_keep=1)
            prefill = model(**prepared)
            record("prefill.logits", prefill.logits[:, -1])
        required = (["input." + name for name in expected] +
                    ["prefill.vision." + name for name in ("patch_embed", "position", "embedded", "rotary_cos",
                                                           "rotary_sin", "merger.norm", "merger.up",
                                                           "merger.activation", "merger")] +
                    [f"prefill.vision.layer.{i}" for i in range(len(visual.blocks))] +
                    ["prefill.decoder.position_ids", "prefill.decoder.inputs_embeds", "prefill.decoder.layer.0",
                     "prefill.decoder.norm", "prefill.logits"])
        missing = sorted(set(required) - set(arrays))
        if missing:
            raise ValueError(f"Missing original prefill stages: {missing}")
        npz = WORK / "qianziwen-prefill.npz"
        reference = WORK / "qianziwen-prefill.gguf"
        np.savez_compressed(npz, **arrays)
        writer = gguf.GGUFWriter(str(reference), "qwen35_delta_reference")
        for name, value in arrays.items():
            writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        REPORT["original_prefill"] = {"status": "complete", "scope": "original only; not native parity",
                                     "case": "qianziwen", "stages": metadata,
                                     "first_token_id": int(np.argmax(arrays["prefill.logits"])),
                                     "npz_sha256": sha256_file(npz), "gguf_sha256": sha256_file(reference)}
        save()
    finally:
        for handle in handles:
            handle.remove()
        visual.fast_pos_embed_interpolate = original_position


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    REPORT["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    REPORT["source_model"], REPORT["source_revision"] = MODEL, REVISION
    REPORT["checkpoint_bytes"], REPORT["checkpoint_sha256"] = CHECKPOINT_BYTES, CHECKPOINT_SHA256
    REPORT["host"] = {"python": platform.python_version(), "machine": platform.machine(),
                      "cpu_count": os.cpu_count()}
    save()

    step("install")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                    f"transformers=={TRANSFORMERS}", f"tokenizers=={TOKENIZERS}",
                    "gguf>=0.19.0", "safetensors>=0.4.3", "pillow", "numpy"], check=True)

    import numpy as np
    import torch
    import tokenizers
    import transformers
    from PIL import Image
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3_5ForConditionalGeneration

    if transformers.__version__ != TRANSFORMERS or tokenizers.__version__ != TOKENIZERS:
        raise ValueError("pinned reference environment required")
    REPORT["versions"] = {"transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
                          "torch": torch.__version__, "numpy": np.__version__}
    REPORT["gpu"] = (subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
                     or None)
    if not torch.cuda.is_available():
        raise RuntimeError("This GPU blueprint run requires CUDA; CPU work uses hosted CPU CI")
    capability = torch.cuda.get_device_capability()
    REPORT["cuda_capability"] = list(capability)
    if capability[0] < 7:
        raise RuntimeError("GPU capability below sm_70; preinstalled PyTorch does not support P100")
    torch.set_num_threads(4)
    save()

    step("source-files")
    metadata_names = ["config.json", "processor_config.json", "tokenizer_config.json",
                      "chat_template.jinja", "tokenizer.json", "README.md"]
    local = Path(snapshot_download(MODEL, revision=REVISION,
                                   allow_patterns=metadata_names + ["model.safetensors"]))
    REPORT["checkpoint_files_sha256"] = {name: sha256_file(local / name) for name in metadata_names}
    REPORT["missing_source_files"] = [name for name in
                                      ("generation_config.json", "preprocessor_config.json")
                                      if not (local / name).exists()]
    config = json.loads((local / "config.json").read_text())
    REPORT["config_canonical_sha256"] = canonical(config)
    save()

    step("checkpoint-digest")
    digest = sha256_file(local / "model.safetensors")
    REPORT["checkpoint_verified_sha256"] = digest
    if digest != CHECKPOINT_SHA256 or (local / "model.safetensors").stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("checkpoint digest differs from the Hub declaration")

    step("header")
    with (local / "model.safetensors").open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    main_names = sorted(n for n in tensors if not n.startswith("mtp."))
    auxiliary = sorted(n for n in tensors if n.startswith("mtp."))
    REPORT["header"] = {"tensors": len(tensors), "main": len(main_names), "auxiliary": len(auxiliary),
                        "auxiliary_names": auxiliary, "header_bytes": header_length,
                        "dtypes": sorted({v["dtype"] for v in tensors.values()})}
    if len(main_names) != 473 or len(auxiliary) != 15:
        raise ValueError("unexpected checkpoint tensor census")

    step("official-schema")
    hf_config = AutoConfig.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    hf_config._attn_implementation = "eager"
    with torch.device("meta"):
        meta = Qwen3_5ForConditionalGeneration(hf_config)
    expected = {name: list(value.shape) for name, value in meta.state_dict().items()}
    del meta
    missing = sorted(set(expected) - set(tensors))
    extra = sorted(set(tensors) - set(expected))
    wrong = [{"name": n, "official": expected[n], "checkpoint": list(tensors[n]["shape"])}
             for n in sorted(set(expected) & set(tensors)) if expected[n] != list(tensors[n]["shape"])]
    REPORT["schema"] = {"official_tensors": len(expected), "missing": missing, "extra": extra,
                        "shape_differences": wrong,
                        "lm_head_absent": "lm_head.weight" not in tensors,
                        "mtp_in_official": sorted(n for n in expected if n.startswith("mtp.")),
                        "official_main": len([n for n in expected if not n.startswith("mtp.")]),
                        "shapes": expected}
    if wrong:
        raise ValueError("checkpoint/official shape mismatch")

    step("processor")
    processor = AutoProcessor.from_pretrained(local, local_files_only=True, trust_remote_code=False)
    REPORT["processor"] = {"class": type(processor).__name__,
                           "image_class": type(processor.image_processor).__name__,
                           "image_config": processor.image_processor.to_dict(),
                           "tokenizer_class": type(processor.tokenizer).__name__}
    generation, stops = resolve_generation(processor, hf_config)
    REPORT["resolved_generation"] = generation.to_diff_dict()
    REPORT["stop_ids"], REPORT["tokenizer_eos"] = stops, int(processor.tokenizer.eos_token_id)
    messages = [{"role": "user", "content": [{"type": "text", "text": DOCUMENT_PROMPT},
                                             {"type": "image"}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    (WORK / "rendered-prompt.txt").write_text(rendered)
    REPORT["task"], REPORT["task_prompt"] = "document", DOCUMENT_PROMPT
    REPORT["rendered_prompt"] = rendered
    REPORT["rendered_prompt_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
    REPORT["rendered_prompt_repr"] = rendered.replace("\n", "\\n")
    REPORT["main_tensor_count"], REPORT["auxiliary_tensor_count"] = 473, 15
    REPORT["transformers_version"] = TRANSFORMERS
    save()

    step("page-inputs")
    cases = []
    for name in PAGES:
        source = Path(hf_hub_download(MODEL, f"example_pics/{name}", revision=REVISION))
        image = fit_image(Image.open(source).convert("RGB"))
        (WORK / f"{source.stem}.png").write_bytes(b"")  # placeholder, replaced below
        image.save(WORK / f"{source.stem}.png", format="PNG")
        inputs = processor(text=[rendered], images=[image], return_tensors="pt",
                           return_mm_token_type_ids=True)
        arrays = {key: value.detach().cpu().numpy() for key, value in inputs.items()}
        if not np.isfinite(arrays["pixel_values"]).all():
            raise ValueError("non-finite processor output")
        pixel_sha256 = hashlib.sha256(arrays["pixel_values"].tobytes()).hexdigest()
        if source.stem in EAGER_V1_PIXELS_SHA256 and pixel_sha256 != EAGER_V1_PIXELS_SHA256[source.stem]:
            raise ValueError("processor pixels differ from the preserved eager v1 control")
        np.savez_compressed(WORK / f"{source.stem}-processor.npz", **arrays)
        cases.append({"name": source.stem, "rgb_shape": [image.height, image.width, 3],
                      "grid": arrays["image_grid_thw"].tolist(),
                      "input_ids": len(arrays["input_ids"][0]),
                      "fields": {k: [list(v.shape), str(v.dtype),
                                     hashlib.sha256(v.tobytes()).hexdigest()] for k, v in arrays.items()},
                      "processor_sha256": sha256_file(WORK / f"{source.stem}-processor.npz")})
    REPORT["cases"] = cases
    save()

    step("original-generation")
    device = "cuda"
    dtype = torch.bfloat16
    REPORT["attention_implementation"] = "sdpa"
    REPORT["eager_control_scope"] = "exact decoded text and input pixel hashes for two pages, not token/logit parity"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        local, local_files_only=True, trust_remote_code=False, dtype=dtype,
        attn_implementation="sdpa").eval()
    if device == "cuda":
        model = model.to(device)
    # Identify the implementation actually loaded, not just the package tag.
    implementations = {}
    for name, obj in [("model", model), ("base", model.model), ("vision", model.model.visual),
                      ("vision_block", model.model.visual.blocks[0]), ("language", model.model.language_model)]:
        cls = type(obj)
        path = inspect.getsourcefile(cls)
        if not path or not Path(path).is_file():
            raise ValueError(f"Cannot resolve loaded implementation: {name}")
        source = Path(path).read_bytes()
        implementations[name] = {"class": f"{cls.__module__}.{cls.__qualname__}",
                                 "filename": Path(path).name,
                                 "sha256": hashlib.sha256(source).hexdigest(),
                                 "normalized_sha256": hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest()}
    embedding = model.model.language_model.embed_tokens.weight
    head = model.lm_head.weight
    REPORT["loaded_implementations"] = implementations
    REPORT["tied_head_observation"] = {
        "same_parameter": embedding is head,
        "same_storage": embedding.untyped_storage().data_ptr() == head.untyped_storage().data_ptr(),
        "embedding_offset": embedding.storage_offset(), "head_offset": head.storage_offset(),
        "embedding_shape": list(embedding.shape), "head_shape": list(head.shape),
        "embedding_stride": list(embedding.stride()), "head_stride": list(head.stride()),
        "embedding_dtype": str(embedding.dtype), "head_dtype": str(head.dtype),
        "embedding_device": str(embedding.device), "head_device": str(head.device),
        "values_equal": bool(torch.equal(embedding, head)),
    }
    save()
    step("original-qianziwen-prefill")
    with np.load(WORK / "qianziwen-processor.npz") as frozen:
        prefill_inputs = {key: torch.from_numpy(frozen[key]).to(device) for key in frozen.files}
    capture_qian_prefill(model, prefill_inputs)
    del prefill_inputs
    step("original-generation")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    outputs = []
    for case in cases:
        name = case["name"]
        inputs = dict(np.load(WORK / f"{name}-processor.npz"))
        tensors_in = {key: torch.from_numpy(value).to(device) for key, value in inputs.items()}
        # Warm-up then two measured repeats keep the first-call graph setup out.
        with torch.inference_mode():
            model.generate(**tensors_in, max_new_tokens=8, do_sample=False)
            if device == "cuda":
                torch.cuda.synchronize()
            timings = []
            for _ in range(2):
                started = time.perf_counter()
                generated = model.generate(**tensors_in, max_new_tokens=MAX_NEW_TOKENS,
                                           do_sample=False)
                if device == "cuda":
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
        prompt_length = int(tensors_in["input_ids"].shape[1])
        new_ids = generated[0, prompt_length:].tolist()
        if name == "qianziwen" and (not new_ids or new_ids[0] != REPORT["original_prefill"]["first_token_id"]):
            raise ValueError("original prefill argmax differs from greedy generation")
        text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        (WORK / f"{name}-original.txt").write_text(text)
        text_sha256 = hashlib.sha256(text.encode()).hexdigest()
        outputs.append({"name": name, "prompt_tokens": prompt_length, "generated_ids": new_ids,
                        "stop_hit": bool(set(new_ids) & set(stops)),
                        "capped": len(new_ids) >= MAX_NEW_TOKENS,
                        "text_sha256": text_sha256,
                        "matches_eager_v1_text": (text_sha256 == EAGER_V1_TEXT_SHA256[name]
                                                  if name in EAGER_V1_TEXT_SHA256 else None),
                        "text_bytes": len(text.encode()),
                        "seconds": [round(value, 3) for value in timings],
                        "device": device, "dtype": str(dtype)})
        REPORT["original_gpu"] = outputs
        save()
        print(f"{name}: {prompt_length} prompt, {len(new_ids)} new, {timings[-1]:.2f}s", flush=True)
    REPORT["original_gpu"] = outputs
    REPORT["original_gpu_peak_vram_gib"] = (round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                                            if device == "cuda" else None)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    save()

    step("complete")
    REPORT["status"] = "completed original blueprint capture; no native parity claimed"
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    REPORT["evidence_canonical_sha256"] = canonical(
        {k: v for k, v in REPORT.items() if k != "evidence_canonical_sha256"})
    save()
    print("BLUEPRINT OK", REPORT["evidence_canonical_sha256"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        REPORT["status"] = "failed blueprint capture"
        REPORT["error_type"] = type(error).__name__
        REPORT["error"] = str(error)[:1200]
        REPORT["traceback"] = traceback.format_exc()[-2500:]
        save()
        raise
