"""Hosted Xiaomi-OCR-0 metadata or bounded original/native F32/F16 experiment.

No execution at import. Metadata runs first and must succeed before any weights.
The frozen two-image protocol is an integration experiment, never a benchmark.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "tests/fixtures/xiaomi0-bounded/protocol.json"
PROTOCOL_SHA256 = "4fa314d26c4c2372f929fc7ada8b8aa0188bbc9148ec82a50b509d9eabb56c30"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


policy = module("xiaomi0_hosted_policy", ROOT / "models/qwen35_xiaomi0_profile.py")
converter = module("xiaomi0_hosted_export", ROOT / "models/convert-qwen35-ocr-to-gguf.py")


def preflight():
    if policy.file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("Frozen Xiaomi integration protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["dtype_arms"] != ["f32", "f16"] or len(protocol["cases"]) != 2:
        raise ValueError("Xiaomi initial execution budget differs")
    for case in protocol["cases"]:
        if policy.file_sha256(ROOT / case["image"]) != case["image_sha256"] or case["max_new_tokens"] != 16:
            raise ValueError("Xiaomi frozen image/cap differs")
    policy.evidence()
    return protocol


def native_env():
    # Do not inherit old diagnostic/precision gates into a new source experiment.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CRISPEMBED_QWEN35_", "CRISPEMBED_OVISOCR", "CRISPEMBED_XIAOMI_"))}
    env.update({"CRISPEMBED_QWEN35_OCR": "1", "CRISPEMBED_XIAOMI_OCR0_FRONTEND": "1",
                "CRISPEMBED_XIAOMI_OCR0_INFERENCE": "1", "CRISPEMBED_BUILD_MODE": "portable"})
    return env


def run(command, path, *, env=None):
    result = subprocess.run(list(map(str, command)), cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    path.write_text(result.stdout)
    return result


def metadata(directory, out):
    import gguf
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration

    pinned = policy.load_profile(directory, policy.MODEL, policy.REVISION)
    runtime = policy.validate_runtime_metadata(directory)
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(config)
    expected = {name: list(value.shape) for name, value in model.state_dict().items()}
    del model
    if len(expected) != 474 or any(name.startswith("mtp.") for name in expected):
        raise ValueError("Xiaomi HF ordinary-generation meta schema differs")
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    writer = gguf.GGUFWriter(str(out / "frontend.gguf"), "qwen35_ocr")
    writer.add_string("general.source.huggingface.repository", policy.MODEL)
    writer.add_string("general.source.huggingface.revision", policy.REVISION)
    writer.add_uint32("qwen35_ocr.format_version", 4)
    converter.write_tokenizer_metadata(writer, converter.read_tokenizer_metadata(
        Path(directory), 248320, pinned["generation"]))
    converter.write_xiaomi0_metadata(writer, pinned)

    def save(name, values):
        value = np.asarray(values)
        if not value.size or not np.isfinite(value).all():
            raise ValueError("Empty/nonfinite frontend reference: " + name)
        writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))

    def text(name, value):
        save(name, list(value.encode("utf-8")))

    height, width = 257, 359
    pixels = ((np.arange(height * width * 3, dtype=np.uint32) * 73 + 29) % 256).astype(np.uint8).reshape(height, width, 3)
    image = Image.fromarray(pixels)
    resized = []
    original = processor.image_processor.resize

    def capture_resize(*args, **kwargs):
        result = original(*args, **kwargs)
        resized.append(result.detach().cpu().numpy().copy())
        return result

    processor.image_processor.resize = capture_resize
    tasks = list(policy.evidence()["prompts"])
    writer.add_uint32("test.xiaomi_cases", len(tasks))
    try:
        for index, task in enumerate(tasks):
            prompt = pinned["prompts"][task]
            rendered = policy.rendered_prompt(task)
            inputs = processor(text=[rendered], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
            label = f"case.{index}"
            writer.add_uint32(label + ".image_tokens", int((inputs["input_ids"] == 248056).sum()))
            text(label + ".prompt", prompt)
            text(label + ".rendered", rendered)
            save(label + ".ids", inputs["input_ids"].numpy())
            save(label + ".modalities", inputs["mm_token_type_ids"].numpy())
            if index == 0:
                if len(resized) != 1 or resized[0].dtype != np.uint8 or resized[0].shape[0] != 1:
                    raise ValueError("Xiaomi uint8 torchvision resize capture differs")
                save("image.rgb", pixels)
                save("image.resized", resized[0][0].transpose(1, 2, 0))
                save("image.patches", inputs["pixel_values"].numpy())
                save("image.grid", inputs["image_grid_thw"].numpy())
    finally:
        processor.image_processor.resize = original
    texts = ["Hello OCR 42", "e\u0301lan", "किरण ١٢३ 中文", "\u0301\u0308abc", "x^2+y^2=25", "<think>ok</think><|im_end|>"]
    writer.add_uint32("test.xiaomi_texts", len(texts))
    for index, value in enumerate(texts):
        ids = processor.tokenizer.encode(value, add_special_tokens=False)
        text(f"text.{index}.input", value)
        save(f"text.{index}.ids", ids)
        text(f"text.{index}.decoded", processor.tokenizer.decode(ids, skip_special_tokens=True))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"runtime": runtime, "meta_tensor_count": len(expected),
            "meta_schema_sha256": hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
            "source_payload_verified": False, "model_forward_executed": False}


def original_reference(directory, case, output, model, processor):
    import numpy as np
    import torch
    from PIL import Image

    image = Image.open(ROOT / case["image"]).convert("RGB")
    if hashlib.sha256(image.tobytes()).hexdigest() != case["rgb_sha256"]:
        raise ValueError("Frozen Xiaomi RGB changed")
    prompt = policy.evidence()["prompts"][case["task"]]
    rendered = policy.rendered_prompt(case["task"])
    inputs = processor(text=[rendered], images=[image], return_tensors="pt", return_mm_token_type_ids=True)
    arrays, handles = {}, []

    def capture(name):
        def observer(_module, _args, result):
            if name in arrays:
                return  # Retain the first, actual prefill only.
            tensor = result[0] if isinstance(result, tuple) else result
            if not torch.is_tensor(tensor):
                raise ValueError("Unexpected original stage: " + name)
            arrays[name] = tensor.detach().float().cpu().numpy().copy()
        return observer

    def before(name):
        def observer(_module, args, kwargs):
            if name not in arrays:
                tensor = kwargs.get("hidden_states", kwargs.get("inputs_embeds", args[0] if args else None))
                if not torch.is_tensor(tensor):
                    raise ValueError("Missing original stage input: " + name)
                arrays[name] = tensor.detach().float().cpu().numpy().copy()
        return observer

    visual, language = model.model.visual, model.model.language_model
    handles.append(visual.blocks[0].register_forward_pre_hook(before("vision.embedded"), with_kwargs=True))
    for index, layer in enumerate(visual.blocks):
        handles.append(layer.register_forward_hook(capture(f"vision.layer.{index}")))
    handles.append(visual.merger.register_forward_hook(capture("vision.merger")))
    handles.append(language.register_forward_pre_hook(before("decoder.input"), with_kwargs=True))
    for index, layer in enumerate(language.layers):
        handles.append(layer.register_forward_hook(capture(f"decoder.layer.{index}")))
    handles.append(language.norm.register_forward_hook(capture("decoder.norm")))
    try:
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=case["max_new_tokens"], do_sample=False,
                                       return_dict_in_generate=True, output_scores=True, output_logits=True)
    finally:
        for handle in handles:
            handle.remove()
    ids = generated.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    logits = np.stack([tensor[0].detach().float().cpu().numpy() for tensor in generated.logits])
    if logits.shape != (len(ids), 248320) or len(arrays) != 40 or not np.isfinite(logits).all():
        raise ValueError("Incomplete Xiaomi actual logits or prefill stages")
    arrays["generated.logits"] = logits
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite original stage: " + name)
    np.savez(output / (case["id"] + "-original.npz"), **arrays)
    record = {"ids": ids, "text": processor.tokenizer.decode(ids, skip_special_tokens=True),
              "stop": "eos" if ids and ids[-1] == 248046 else "max_tokens",
              "cap": case["max_new_tokens"], "image_sha256": case["image_sha256"],
              "input_ids": inputs["input_ids"][0].tolist(), "grid": inputs["image_grid_thw"].tolist(),
              "stage_shapes": {name: list(value.shape) for name, value in arrays.items()}}
    if record["stop"] == "max_tokens" and len(ids) != record["cap"]:
        raise ValueError("Unexplained original generation termination")
    return record


def metrics(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite or shape-mismatched comparison")
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    norms_a = np.linalg.norm(rows_a, axis=1)
    norms_b = np.linalg.norm(rows_b, axis=1)
    den = norms_a * norms_b
    cos = np.divide((rows_a * rows_b).sum(1), den, out=np.zeros_like(den), where=den > 0)
    cos[(norms_a == 0) & (norms_b == 0)] = 1
    maximum = float(np.max(np.abs(a - b)))
    minimum = float(cos.min())
    return {"cosine_min": minimum, "max_abs": maximum,
            "rms_actual": float(np.sqrt((a * a).mean())), "rms_original": float(np.sqrt((b * b).mean())),
            "strict_pass": minimum >= 0.99999 and maximum <= 0.002}


def compare_native(log, logits_path, original_path, reference, dtype):
    import numpy as np
    rows = [json.loads(line.removeprefix("XIAOMI0_RESULT ")) for line in log.splitlines() if line.startswith("XIAOMI0_RESULT ")]
    if len(rows) != 1:
        raise ValueError("Missing unique actual Xiaomi generation result")
    row = rows[0]
    stage_rows = [json.loads(line.removeprefix("XIAOMI0_STAGE ")) for line in log.splitlines() if line.startswith("XIAOMI0_STAGE ")]
    if len(stage_rows) != 40 or len({x["name"] for x in stage_rows}) != 40:
        raise ValueError("Incomplete native stage capture")
    with np.load(original_path) as arrays:
        raw = np.fromfile(str(logits_path) + ".stages", dtype=np.float32)
        stages = {}
        offset = 0
        for stage in stage_rows:
            if stage["byte_offset"] != offset * 4 or stage["count"] <= 0 or stage["name"] not in arrays:
                raise ValueError("Invalid native stage range/name")
            expected = arrays[stage["name"]]
            if expected.size != stage["count"]:
                raise ValueError("Native stage shape mismatch")
            actual = raw[offset:offset + stage["count"]].reshape(expected.shape)
            stages[stage["name"]] = metrics(actual, expected)
            offset += stage["count"]
        if offset != raw.size:
            raise ValueError("Native stage tail or truncation")
        native_logits = np.fromfile(logits_path, dtype=np.float32).reshape(-1, 248320)
        if native_logits.shape[0] != len(row["token_ids"]):
            raise ValueError("Native logits/IDs length differs")
        complete = native_logits.shape == arrays["generated.logits"].shape
        final = metrics(native_logits, arrays["generated.logits"]) if complete else None
    native_text = bytes.fromhex(row["raw_text_utf8_hex"]).decode("utf-8")
    ids_equal = row["token_ids"] == reference["ids"]
    stop_equal = row["stop_kind"] == (1 if reference["stop"] == "eos" else 2)
    worst = min(stages, key=lambda name: stages[name]["cosine_min"])
    return {"native": row, "ids_equal": ids_equal, "text_equal": native_text == reference["text"],
            "stop_equal": stop_equal, "prefixes_equal": ids_equal, "final_logits": final,
            "owner_bar": 0.99 if dtype == "f32" else 0.98,
            "owner_final_logit_bar_pass": final is not None and final["cosine_min"] > (0.99 if dtype == "f32" else 0.98),
            "stage_metrics": stages, "worst_intermediate_stage": worst,
            "precision_qualified": False, "full_document_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--inference", action="store_true", help="Run exactly two frozen original cases and four native arms on hosted CI")
    args = parser.parse_args()
    protocol = preflight()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("This recipe builds/captures only on hosted CI; lightweight source tests run locally")
    if os.environ.get("CRISPEMBED_BUILD_MODE", "portable") != "portable":
        parser.error("Xiaomi precision evidence requires the portable CPU build")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "xiaomi0-hosted-result/v1", "phase": "metadata", "protocol_sha256": PROTOCOL_SHA256,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_audit_sha256": policy.AUDIT_SHA256, "build_mode": "portable", "original": {}, "native": {},
              "full_document_qualified": False, "publication_qualified": False}

    def save():
        (out / "xiaomi0-result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        from huggingface_hub import snapshot_download
        import torch
        import torchvision
        if torch.__version__.split("+")[0] != "2.14.0" or torchvision.__version__.split("+")[0] != "0.29.0":
            raise ValueError("Xiaomi original verified torch/torchvision versions differ")
        torch.set_num_threads(protocol["threads"])
        torch.manual_seed(0)
        directory = args.model_dir or Path(snapshot_download(policy.MODEL, revision=policy.REVISION,
                                                             allow_patterns=list(policy.metadata_files())))
        report["metadata"] = metadata(directory, out)
        build = out / "build"
        env = native_env()
        result = run(["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
                      "-DCRISPEMBED_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"],
                     out / "configure.log", env=env)
        if result.returncode:
            raise ValueError("Native configure failed")
        result = run(["cmake", "--build", build, "--target", "test-xiaomi0-frontend", "test-xiaomi0-run", "-j", "2"],
                     out / "build.log", env=env)
        if result.returncode:
            raise ValueError("Native build failed")
        result = run([build / "test-xiaomi0-frontend", out / "frontend.gguf"], out / "frontend.log", env=env)
        report["frontend_pass"] = result.returncode == 0
        save()
        if result.returncode:
            raise ValueError("Original/native Xiaomi frontend mismatch")
        if args.inference:
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
            if args.model_dir is None:
                directory = Path(snapshot_download(policy.MODEL, revision=policy.REVISION,
                                                    allow_patterns=list(policy.metadata_files()) + ["model.safetensors"]))
            policy.verify_checkpoint(directory / "model.safetensors", policy.expected_profile())
            report["phase"] = "original"; save()
            processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
            model = Qwen3_5ForConditionalGeneration.from_pretrained(directory, dtype=torch.float32,
                        attn_implementation="eager", local_files_only=True, trust_remote_code=False).eval()
            if model.generation_config.eos_token_id != 248046:
                raise ValueError("Loaded Xiaomi model EOS differs from metadata")
            for case in protocol["cases"]:
                report["original"][case["id"]] = original_reference(directory, case, out, model, processor)
                save()
            del model, processor
            gc.collect()
            report["phase"] = "native"; save()
            for dtype in protocol["dtype_arms"]:
                output = out / ("xiaomi0-" + dtype + ".gguf")
                converter.convert(directory, output, policy.MODEL, policy.REVISION, dtype=dtype, profile="xiaomi-ocr-0")
                for case in protocol["cases"]:
                    key = dtype + "-" + case["id"]
                    prompt = out / (case["id"] + ".prompt.txt")
                    prompt.write_text(policy.evidence()["prompts"][case["task"]])
                    logits = out / (key + ".logits.f32")
                    result = run([build / "test-xiaomi0-run", output, ROOT / case["image"], case["max_new_tokens"],
                                  prompt, logits, protocol["threads"]], out / (key + ".log"), env=env)
                    if result.returncode:
                        raise ValueError("Native Xiaomi execution failed: " + key)
                    report["native"][key] = compare_native(result.stdout, logits,
                        out / (case["id"] + "-original.npz"), report["original"][case["id"]], dtype)
                    save()
            report["bounded_final_logit_bars_pass"] = all(x["owner_final_logit_bar_pass"] for x in report["native"].values())
        report["phase"] = "complete"
        report["scope"] = protocol["scope"]
        save()
        if args.inference and not report["bounded_final_logit_bars_pass"]:
            raise ValueError("Bounded final-logit precision bars failed; actual artifacts retained")
    except Exception as error:
        report["failure"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
