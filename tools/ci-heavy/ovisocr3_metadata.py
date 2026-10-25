"""Hosted OvisOCR3 weight-free source, processor, schema and stop probe."""

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import types
import typing
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "models"))
import qwen35_ovis3_profile as policy

VLLM = "0decac0d96c42b49572498019f0a0e3600f50398"
VLLM_SAMPLING_SHA256 = "c5f9be229115230eccc33788d30fb2b399d71d7c1093962e9bf8f125fceb19a6"
MODEL_SOURCE_SHA256 = "788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263"
GENERATION_SOURCE_SHA256 = "427f7e57922fb58c3b321f5e5837ee782bd640cd5b22833029b16f9d4b9c4cdb"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def failure_detail(error):
    text = str(error)
    text = re.sub(r"https?://\S+", "<url>", text)
    text = re.sub(r"(?:hf_|gh[pousr]_)[A-Za-z0-9]+", "<credential>", text)
    text = re.sub(r"(?<![A-Za-z0-9])/(?:[^\s\]\)]+)", "<path>", text)
    return text[:800]


def original_stop_probe(source, generation, tokenizer_eos):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SamplingParams")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "update_from_generation_config")
    namespace = {"Any": typing.Any}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "pinned-vllm-stop-method", "exec"), namespace)
    instance = types.SimpleNamespace(ignore_eos=False, _eos_token_id=None,
                                     _all_stop_token_ids=set(), stop_token_ids=[])
    namespace["update_from_generation_config"](instance, generation, tokenizer_eos)
    return {"primary_eos": instance._eos_token_id,
            "additional_stop_ids": sorted(instance.stop_token_ids),
            "all_stop_ids": sorted(instance._all_stop_token_ids)}


def check_schema(expected, actual, tied):
    missing = set(expected) - set(actual)
    if tied and missing == {"lm_head.weight"}:
        missing.clear()
    extra = set(actual) - set(expected)
    wrong = [name for name in actual if name in expected and actual[name] != expected[name]]
    if missing or extra or wrong:
        raise ValueError("Official/header schema differs: " + str((sorted(missing), sorted(extra), wrong[:8])))
    return {"expected_tensors": len(expected), "checkpoint_tensors": len(actual),
            "tied_head_absent": "lm_head.weight" not in actual}


def header_shapes():
    """Read only the audited 58,680-byte header using a bounded HTTP range."""
    report, _, _ = policy._evidence()
    header_bytes = report["checkpoint_header"]["header_bytes"]
    url = "https://huggingface.co/" + policy.MODEL + "/resolve/" + policy.REVISION + "/model.safetensors"
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{header_bytes + 7}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith("bytes 0-"):
            raise ValueError("Pinned checkpoint header range was not honored")
        payload = response.read(header_bytes + 9)
    if len(payload) != header_bytes + 8 or int.from_bytes(payload[:8], "little") != header_bytes:
        raise ValueError("Checkpoint header range/length differs")
    if sha_bytes(payload[8:]) != report["checkpoint_header"]["header_sha256"]:
        raise ValueError("Checkpoint header digest differs")
    header = json.loads(payload[8:])
    header.pop("__metadata__", None)
    if len(header) != 473 or any(entry["dtype"] != "BF16" for entry in header.values()):
        raise ValueError("Checkpoint header tensor count/dtype differs")
    offsets = sorted(tuple(entry["data_offsets"]) for entry in header.values())
    if offsets[0][0] != 0 or any(a[1] != b[0] for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Checkpoint header offsets not contiguous")
    if offsets[-1][1] != report["checkpoint_header"]["payload_bytes_from_header"]:
        raise ValueError("Checkpoint header payload declaration differs")
    return {name: entry["shape"] for name, entry in header.items()}


def preflight():
    report, _, _ = policy._evidence()
    if (len(report["prompt_files"]) != 18 or len(policy.metadata_files()) != 31
            or report["original_example"]["vllm_tag_commit"] != VLLM
            or report["original_example"]["nested_images_kwargs"] != {
                "min_pixels": 448 * 448, "max_pixels": 2880 * 2880}):
        raise ValueError("OvisOCR3 audited source/README contract differs")
    return report


def main():
    if "--preflight" in sys.argv:
        preflight()
        return
    out = Path(os.environ["HEAVY_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    report = {"source_commit": os.environ["GITHUB_SHA"], "driver_sha256": policy._file_sha256(__file__),
              "source_report_sha256": policy.SOURCE_REPORT_SHA256, "source_model": policy.MODEL,
              "source_revision": policy.REVISION, "scope": "weight-free original HF5.16.1 metadata and processor; no model forward or native qualification",
              "phase": "preflight", "status": "started"}
    def save():
        (out / "ovisocr3-metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        source_report = preflight()
        report["phase"] = "environment"; save()
        import numpy as np
        import torch
        import torchvision
        import transformers
        from PIL import Image
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3_5ForConditionalGeneration
        if (transformers.__version__ != "5.16.1" or torch.__version__.split("+")[0] != "2.14.0"
                or torchvision.__version__.split("+")[0] != "0.29.0"):
            raise ValueError("Pinned metadata stack differs")
        torch.set_num_threads(2)
        report["environment"] = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                                 "transformers": transformers.__version__, "numpy": np.__version__}
        report["phase"] = "source-download"; save()
        directory = Path(snapshot_download(policy.MODEL, revision=policy.REVISION,
                                           allow_patterns=list(policy.metadata_files())))
        if any(directory.glob("*.safetensors")):
            raise ValueError("Unexpected weight payload in metadata download")
        pinned = policy.load_profile(directory, policy.MODEL, policy.REVISION)
        report["source_files_sha256"] = {name: policy._file_sha256(directory / name)
                                         for name in policy.metadata_files()}
        report["profile_source_only"] = {key: value for key, value in pinned.items() if key != "task_prompt"}
        report["task_prompt_sha256"] = sha_bytes(pinned["task_prompt"].encode())
        report["phase"] = "loaded-source"; save()
        report["loaded_source_sha256"] = {}
        for label, cls in (("generation", GenerationConfig), ("model", Qwen3_5ForConditionalGeneration)):
            path = Path(inspect.getfile(cls))
            digest = sha_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
            expected = GENERATION_SOURCE_SHA256 if label == "generation" else MODEL_SOURCE_SHA256
            if digest != expected:
                raise ValueError("Pinned HF implementation differs: " + label)
            report["loaded_source_sha256"][label] = digest
        report["phase"] = "generation-and-stops"; save()
        config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        generation = GenerationConfig.from_pretrained(directory, local_files_only=True)
        report["resolved_generation"] = generation.to_diff_dict()
        if report["resolved_generation"].get("eos_token_id") != policy.SOURCE_EOS_IDS:
            raise ValueError("Resolved explicit EOS list differs")
        source = urllib.request.urlopen("https://raw.githubusercontent.com/vllm-project/vllm/" + VLLM +
                                        "/vllm/sampling_params.py", timeout=30).read()
        if sha_bytes(source) != VLLM_SAMPLING_SHA256:
            raise ValueError("Pinned vLLM stop implementation differs")
        report["vllm_sampling_source_sha256"] = sha_bytes(source)
        report["phase"] = "processor-load"; save()
        processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        report["processor"] = {"class_name": type(processor).__module__ + "." + type(processor).__name__,
                               "image_class": type(processor.image_processor).__module__ + "." + type(processor.image_processor).__name__,
                               "image_backend": getattr(processor.image_processor, "backend", None),
                               "image_config": processor.image_processor.to_dict()}
        for label, cls in (("processor", type(processor)), ("image_processor", type(processor.image_processor)),
                           ("tokenizer", type(processor.tokenizer))):
            report["loaded_source_sha256"][label] = policy._file_sha256(inspect.getfile(cls))
        report["tokenizer_eos"] = processor.tokenizer.eos_token_id
        report["effective_stops_source_probe"] = original_stop_probe(source.decode(),
                                                                      report["resolved_generation"],
                                                                      report["tokenizer_eos"])
        if (report["tokenizer_eos"] != 248046
                or report["effective_stops_source_probe"]["all_stop_ids"] != policy.SOURCE_EOS_IDS):
            raise ValueError("OvisOCR3 derived vLLM stop contract differs")
        save()
        report["phase"] = "header-and-official-meta-schema"; save()
        actual = header_shapes()
        config._attn_implementation = "eager"
        with torch.device("meta"):
            model = Qwen3_5ForConditionalGeneration(config)
        if not all(parameter.device.type == "meta" for parameter in model.parameters()):
            raise ValueError("Meta schema model allocated checkpoint payload")
        expected = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
        report["schema"] = check_schema(expected, actual, bool(config.tie_word_embeddings))
        report["schema"]["official_state_dict"] = expected
        report["schema"]["header_canonical_sha256"] = source_report["checkpoint_header"]["tensor_metadata_canonical_sha256"]
        del model
        report["phase"] = "prompt"; save()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pinned["task_prompt"]}]}]
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)
        (out / "rendered-prompt.txt").write_text(text)
        report["rendered_prompt_sha256"] = policy._file_sha256(out / "rendered-prompt.txt")
        report["inputs"] = []; save()
        for label, width, height in (("square", 448, 448), ("odd", 449, 417), ("small-control", 96, 64)):
            report["phase"] = "processor-" + label; save()
            y, x = np.indices((height, width))
            pixels = np.stack(((x * 3 + y) % 256, (y * 5 + x) % 256,
                               (x + y * 7) % 256), axis=-1).astype(np.uint8)
            image = Image.fromarray(pixels, "RGB")
            def process(instance):
                return instance(text=[text], images=[image], return_tensors="pt",
                                images_kwargs={"min_pixels": 448 * 448, "max_pixels": 2880 * 2880})
            first, repeated = process(processor), process(processor)
            arrays = {key: value.detach().cpu().numpy() for key, value in first.items()}
            np.savez_compressed(out / (label + "-processor.npz"), **arrays)
            entry = {"case": label, "rgb_shape": list(pixels.shape),
                     "rgb_sha256": sha_bytes(pixels.tobytes()),
                     "fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                      "sha256": sha_bytes(value.tobytes())} for key, value in arrays.items()},
                     "ids": arrays["input_ids"].tolist(), "grid": arrays["image_grid_thw"].tolist(),
                     "attention_mask": arrays["attention_mask"].tolist(),
                     "archive_sha256": policy._file_sha256(out / (label + "-processor.npz")),
                     "repeat_exact": set(first) == set(repeated) and
                     all(torch.equal(value, repeated[key]) for key, value in first.items())}
            report["inputs"].append(entry); save()
            if not entry["repeat_exact"] or not np.isfinite(arrays["pixel_values"]).all():
                raise ValueError("Original OvisOCR3 processor repeat or finite check differs")
            grid = entry["grid"]
            ids = entry["ids"]
            if (len(grid) != 1 or len(grid[0]) != 3 or grid[0][0] != 1
                    or any(type(dimension) is not int or dimension <= 0 for dimension in grid[0])
                    or grid[0][1] % 2 or grid[0][2] % 2
                    or list(arrays["pixel_values"].shape) != [grid[0][1] * grid[0][2], 1536]
                    or len(ids) != 1 or len(entry["attention_mask"]) != 1
                    or len(ids[0]) != len(entry["attention_mask"][0])
                    or sum(token == 248056 for token in ids[0]) != grid[0][1] * grid[0][2] // 4):
                raise ValueError("Original OvisOCR3 image placeholder/patch geometry differs")
            if label == "small-control":
                fresh = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
                defaults = fresh(text=[text], images=[image], return_tensors="pt")
                default_arrays = {key: value.detach().cpu().numpy() for key, value in defaults.items()}
                np.savez_compressed(out / "small-default-processor.npz", **default_arrays)
                report["pixel_override_control"] = {
                    "default_grid": default_arrays["image_grid_thw"].tolist(),
                    "explicit_grid": arrays["image_grid_thw"].tolist(),
                    "default_archive_sha256": policy._file_sha256(out / "small-default-processor.npz"),
                    "default_fields": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                            "sha256": sha_bytes(value.tobytes())} for key, value in default_arrays.items()}}
                save()
                def area(grid):
                    if len(grid) != 1 or len(grid[0]) != 3 or grid[0][0] != 1:
                        raise ValueError("Single-image grid differs")
                    return grid[0][1] * grid[0][2] * 16 * 16
                before, after = area(report["pixel_override_control"]["default_grid"]), area(entry["grid"])
                report["pixel_override_control"].update({"default_pixels": before, "explicit_pixels": after,
                    "min_override_effect_proved": before < 448 * 448 <= after <= 2880 * 2880
                                                  and report["pixel_override_control"]["default_grid"] != entry["grid"]})
                save()
                if not report["pixel_override_control"]["min_override_effect_proved"]:
                    raise ValueError("README min_pixels override had no demonstrated effect")
        report["phase"] = "complete"
        report["status"] = "completed weight-free OvisOCR3 HF metadata/source probe; no model forward or native qualification"
        save()
    except BaseException as error:
        report["status"] = "failed OvisOCR3 metadata probe"
        report["error_type"] = type(error).__name__
        report["error_message"] = failure_detail(error)
        save()
        raise


if __name__ == "__main__":
    main()
