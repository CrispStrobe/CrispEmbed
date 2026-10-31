#!/usr/bin/env python3
"""Capture the original Python MinerU request and compare native output/timing.

Run expensive model work on hosted CI or an authorized machine. References use
the original pinned TransformersVlmClient, Qwen2VL model and FAST image processor;
the script does not reimplement the reference forward graph. Completion IDs,
terminal EOS, structural tokens, actual input IDs, dimensions, package versions,
source hashes and cold/warm timings are retained. A stage GGUF is optional and is
captured in a separate, untimed pass so instrumentation does not contaminate speed.

Examples (see docs/mineru25-pro.md for installation):
  python tools/mineru25_validate.py reference --model HF_SNAPSHOT \
    --blueprint MINERU_VL_UTILS_CHECKOUT --image page.png --task text \
    --max-tokens 1024 --output reference.json --dump-reference reference.gguf
  python tools/mineru25_validate.py compare --reference reference.json \
    --native build/test-mineru25-run --gguf mineru25-f32.gguf \
    --image page.png --output comparison.json
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
import subprocess
import sys
import time

MODEL_ID = "opendatalab/MinerU2.5-Pro-2604-1.2B"
MODEL_REVISION = "d3f5e08d073c21466bbabe21c71bb1e9c2e595da"
BLUEPRINT_REVISION = "c9797cde8f54cc48306196fc6aebe73ea31dd4a3"
PROMPTS = {"layout": "\nLayout Detection:", "text": "\nText Recognition:",
           "table": "\nTable Recognition:", "formula": "\nFormula Recognition:",
           "image": "\nImage Analysis:"}
EOS = {151645, 151643}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


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
    char_errors = edit_distance(expected, actual)
    word_errors = edit_distance(words, actual.split())
    return {
        "exact": expected == actual,
        "reference_chars": len(expected), "native_chars": len(actual),
        "reference_lines": len(expected.splitlines()), "native_lines": len(actual.splitlines()),
        "character_errors": char_errors, "word_errors": word_errors,
        "cer": char_errors / max(1, len(expected)), "wer": word_errors / max(1, len(words)),
    }


def median_warm(report):
    runs = report["timing"]["runs_ms"]
    if len(runs) < 2 or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in runs):
        raise ValueError("need a cold run and at least one positive finite warm timing")
    return statistics.median(runs[1:])


def comparison(reference, native, max_slowdown):
    """Strict request parity, separate from corpus quality or numerical proof."""
    keys = ["task", "prompt", "max_tokens", "effective_max_tokens", "grid_thw", "input_ids", "token_ids",
            "eos", "capped", "text", "raw_text"]
    checks = {key: key in reference and key in native and reference[key] == native[key] for key in keys}
    repeated = native.get("runs", [])
    repeat_keys = ["token_ids", "grid_thw", "eos", "capped", "text", "raw_text"]
    checks["resident_repeat_stability"] = (
        len(repeated) == len(native["timing"]["runs_ms"]) and
        bool(repeated) and
        all(all(key in run and run[key] == native[key] for key in repeat_keys) and
            run.get("input_ids", run.get("prompt_ids")) == native.get("input_ids")
            for run in repeated)
    )
    ref_ms, native_ms = median_warm(reference), median_warm(native)
    request_pass = all(checks.values()) and bool(native.get("eos")) and not native.get("capped", True)
    comparable = reference.get("device") == native.get("device") == "cpu" and reference.get("threads") == native.get("threads")
    # Source reference and native result must be captured on the same host.
    comparable = comparable and reference.get("machine_fingerprint") == native.get("machine_fingerprint")
    ratio = native_ms / ref_ms
    return {
        "schema": "crispembed.mineru25.request-comparison.v1",
        "checks": checks, "text_metrics": text_metrics(reference["text"], native["text"]),
        "request_parity_pass": request_pass,
        "speed_comparable": comparable, "python_warm_median_ms": ref_ms,
        "native_warm_median_ms": native_ms, "native_over_python": ratio,
        "max_slowdown": max_slowdown,
        "speed_pass": comparable and request_pass and ratio <= max_slowdown,
        "qualification": {
            "representative_corpus_quality": "not established by one request",
            "intermediate_numerical_parity": "requires the stage comparison report",
            "all_backends_speed": "not established by one host/backend",
        },
    }


def machine_fingerprint():
    # Host identity is hashed, not published. Retain concrete hardware separately.
    value = platform.node() + "\n" + platform.machine() + "\n" + platform.processor()
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def packages():
    result = {}
    for name in ("torch", "torchvision", "transformers", "Pillow", "numpy", "gguf"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def verify_blueprint(path):
    path = Path(path).resolve()
    revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if revision != BLUEPRINT_REVISION:
        raise ValueError(f"blueprint must be {BLUEPRINT_REVISION}, got {revision}")
    changed = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--", "mineru_vl_utils"], text=True)
    if changed:
        raise ValueError("reference Python sources are modified; restore the pinned checkout")
    sys.path.insert(0, str(path))
    return path


def reference(args):
    blueprint = verify_blueprint(args.blueprint)
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from PIL import Image
    from mineru_vl_utils.mineru_client import (
        MinerUSamplingParams, MinerUClientHelper, DEFAULT_PROMPTS, DEFAULT_SAMPLING_PARAMS,
    )
    from mineru_vl_utils.vlm_client.transformers_client import TransformersVlmClient

    if transformers.__version__ != "4.57.3":
        raise ValueError("reference protocol requires transformers==4.57.3; do not silently change the blueprint")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    source = Path(args.model)
    if not source.is_dir() and args.model != MODEL_ID:
        raise ValueError("only the pinned model repository or a local snapshot is supported")
    converter_file = Path(__file__).resolve().parents[1] / "models/convert-mineru25-pro-to-gguf.py"
    module_spec = importlib.util.spec_from_file_location("mineru25_reference_source", converter_file)
    converter = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = converter
    module_spec.loader.exec_module(converter)
    source, _ = converter.resolve_model(args.model)
    source_hashes = converter.verify_source_files(source)
    dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(source, use_fast=True, trust_remote_code=False, local_files_only=True)
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("modelcard FAST image processor is required")
    model = Qwen2VLForConditionalGeneration.from_pretrained(source, torch_dtype=dtype,
        attn_implementation=args.attention, trust_remote_code=False, local_files_only=True).to(args.device).eval()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - start) * 1000
    client = TransformersVlmClient(model=model, processor=processor, use_tqdm=False,
        sampling_params=MinerUSamplingParams(max_new_tokens=args.max_tokens or None))
    image = Image.open(args.image).convert("RGB")
    original_size = list(image.size)
    helper = MinerUClientHelper(
        backend="transformers", prompts=DEFAULT_PROMPTS, sampling_params=DEFAULT_SAMPLING_PARAMS,
        layout_image_size=(1036, 1036), min_image_edge=28, max_image_edge_ratio=50,
        simple_post_process=False, handle_equation_block=True, abandon_list=False,
        abandon_paratext=False, image_analysis=False, debug=False,
    )
    def predict():
        prepared = helper.prepare_for_layout(image) if args.task == "layout" else helper.resize_by_need(image)
        return client.predict(prepared, PROMPTS[args.task])

    original_generate = model.generate
    captures = []
    stage = {}
    def capture_generate(*positional, **kwargs):
        result = original_generate(*positional, **kwargs)
        inputs = kwargs["input_ids"].detach().cpu()[0].tolist()
        output = result.detach().cpu()[0].tolist()[len(inputs):]
        captures.append({"input_ids": inputs, "token_ids": output,
                         "grid_thw": kwargs["image_grid_thw"].detach().cpu()[0].tolist()})
        return result
    runs, texts = [], []
    try:
        with torch.inference_mode():
            for _ in range(args.repeat):
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                text = predict()
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                runs.append((time.perf_counter() - start) * 1000)
                texts.append(text)
        if any(text != texts[0] for text in texts):
            raise RuntimeError("original model was not deterministic across measured runs")

        # ID/CPU transfer instrumentation is kept outside every timed run.
        model.generate = capture_generate
        with torch.inference_mode():
            captured_text = predict()
        if captured_text != texts[0]:
            raise RuntimeError("untimed token observer changed generated text")

        if args.dump_reference:
            handles = []
            def output_hook(name):
                def hook(_module, _args, output):
                    if name not in stage:
                        value = output[0] if isinstance(output, tuple) else output
                        if name == "llm_logits_last":
                            value = value[:, -1:, :]
                        stage[name] = value.detach().float().cpu().squeeze(0).numpy().copy()
                return hook
            visual = model.model.visual
            language = model.model.language_model
            handles.append(visual.patch_embed.register_forward_hook(output_hook("vis_patch_embed")))
            handles.extend(block.register_forward_hook(output_hook(f"vis_layer_{i}")) for i, block in enumerate(visual.blocks))
            handles.append(visual.merger.register_forward_hook(output_hook("vis_merger_output")))
            handles.extend(block.register_forward_hook(output_hook(f"llm_layer_{i}")) for i, block in enumerate(language.layers))
            handles.append(language.norm.register_forward_hook(output_hook("llm_final_norm")))
            handles.append(model.lm_head.register_forward_hook(output_hook("llm_logits_last")))
            def language_input(_module, _args, kwargs):
                if "llm_embed" not in stage:
                    stage["llm_embed"] = kwargs["inputs_embeds"].detach().float().cpu()[0].numpy().copy()
                    pos = kwargs.get("position_ids")
                    if pos is not None:
                        # v4.57.3 generation prepends the packed-mask text
                        # positions; Qwen2VLTextModel strips them before RoPE.
                        if pos.ndim == 3 and pos.shape[0] == 4:
                            pos = pos[1:]
                        if pos.ndim != 3 or pos.shape[0] != 3:
                            raise RuntimeError("unexpected reference mRoPE position layout")
                        stage["mrope_positions"] = pos[:, 0, :].T.detach().float().cpu().numpy().copy()
            handles.append(language.register_forward_pre_hook(language_input, with_kwargs=True))
            def visual_input(_module, positional, _kwargs):
                if "input_patches" not in stage:
                    stage["input_patches"] = positional[0].detach().float().cpu().numpy().copy()
            handles.append(visual.register_forward_pre_hook(visual_input, with_kwargs=True))
            try:
                with torch.inference_mode():
                    observed_text = predict()
                if captures[-1] != captures[0] or observed_text != texts[0]:
                    raise RuntimeError("reference stage observer changed generated output")
            finally:
                for handle in handles:
                    handle.remove()
            import gguf
            import numpy as np
            args.dump_reference.parent.mkdir(parents=True, exist_ok=True)
            writer = gguf.GGUFWriter(str(args.dump_reference), "mineru25_ref")
            writer.add_string("mineru25.model", MODEL_ID)
            writer.add_string("mineru25.model_revision", MODEL_REVISION)
            writer.add_string("mineru25.blueprint_revision", BLUEPRINT_REVISION)
            writer.add_string("mineru25.processor", type(processor.image_processor).__name__)
            writer.add_string("mineru25.task", args.task)
            writer.add_string("mineru25.dtype", args.dtype)
            writer.add_string("mineru25.image_sha256", sha256(args.image))
            for index, axis in enumerate("thw"):
                writer.add_uint32("qwen2vl.grid_" + axis, captures[0]["grid_thw"][index])
            stage["input_ids"] = np.asarray(captures[0]["input_ids"], dtype=np.float32)
            stage["grid_thw"] = np.asarray(captures[0]["grid_thw"], dtype=np.float32)
            for name, value in stage.items():
                writer.add_tensor(name, np.ascontiguousarray(value, dtype=np.float32))
            writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    finally:
        model.generate = original_generate

    output = captures[0]["token_ids"]
    eos = bool(output and output[-1] in EOS)
    effective_max_tokens = args.max_tokens or max(0, client.model_max_length - len(captures[0]["input_ids"]))
    report = {
        "schema": "crispembed.mineru25.request.v1", "engine": "original-python",
        "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "expected_model_revision": MODEL_REVISION, "source_sha256": source_hashes,
        "blueprint_revision": BLUEPRINT_REVISION,
        "blueprint_source_sha256": {str(p.relative_to(blueprint)): sha256(p) for p in [
            blueprint / "mineru_vl_utils/vlm_client/transformers_client.py",
            blueprint / "mineru_vl_utils/vlm_client/base_client.py",
            blueprint / "mineru_vl_utils/mineru_client.py"]},
        "transformers_model_source_sha256": sha256(Path(sys.modules[type(model).__module__].__file__)),
        "processor": type(processor.image_processor).__name__, "packages": packages(),
        "image_sha256": sha256(args.image), "original_size": original_size,
        "task": args.task, "prompt": PROMPTS[args.task], "dtype": args.dtype,
        "device": args.device, "attention": args.attention, "threads": args.threads,
        "machine_fingerprint": machine_fingerprint(), "platform": platform.platform(),
        "max_tokens": args.max_tokens, "effective_max_tokens": effective_max_tokens, **captures[0],
        "raw_text": processor.tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        "text": texts[0], "eos": eos, "capped": not eos and len(output) >= effective_max_tokens,
        "timing": {"load_ms": load_ms, "runs_ms": runs,
                   "scope": "resident helper resize+processor+input transfer+vision+prefill+decode+output transfer+detokenize"},
        "stages": sorted(stage), "stage_reference_sha256": sha256(args.dump_reference) if args.dump_reference else None,
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "generated_tokens": len(output), "eos": eos,
                      "warm_median_ms": median_warm(report), "stage_count": len(stage)}))
    return 0


def compare(args):
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    if sha256(args.image) != ref["image_sha256"]:
        raise ValueError("native image bytes differ from the frozen reference")
    if ref["blueprint_revision"] != BLUEPRINT_REVISION:
        raise ValueError("reference belongs to a different blueprint")
    env = {**os.environ, "CRISPEMBED_MINERU25_PRO": "1", "QWEN2VL_OCR_FORCE_CPU": "1"}
    command = [str(args.native), "--model", str(args.gguf), "--image", str(args.image),
               "--task", ref["task"], "--max-tokens", str(ref["max_tokens"]),
               "--threads", str(ref["threads"]), "--repeat", str(len(ref["timing"]["runs_ms"]))]
    start = time.perf_counter()
    process = subprocess.run(command, env=env, text=True, encoding="utf-8", capture_output=True)
    elapsed = (time.perf_counter() - start) * 1000
    # Retain failure evidence before reporting a nonzero runner exit.
    Path(str(args.output) + ".native.stderr.log").write_text(process.stderr, encoding="utf-8")
    Path(str(args.output) + ".native.stdout.json").write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        write_json(args.output, {"status": "native_failed", "returncode": process.returncode,
                                "command": command, "subprocess_ms": elapsed})
        return 2
    native = json.loads(process.stdout)
    if "input_ids" not in native and "prompt_ids" in native:
        native["input_ids"] = native["prompt_ids"]
    native["machine_fingerprint"] = machine_fingerprint()
    native["image_sha256"] = sha256(args.image)
    report = comparison(ref, native, args.max_slowdown)
    report.update({"reference_sha256": sha256(args.reference), "gguf_sha256": sha256(args.gguf),
                   "native_executable_sha256": sha256(args.native), "command": command,
                   "native_environment": {key: env.get(key) for key in [
                       "CRISPEMBED_MINERU25_PRO", "CRISPEMBED_MINERU25_PRO_REUSE_DECODE_GRAPH", "CRISPEMBED_MINERU25_PRO_EXACT_GELU",
                       "CRISPEMBED_MINERU25_PRO_VISION_F32_ACT",
                       "QWEN2VL_OCR_FORCE_CPU", "CRISPEMBED_NO_KV_CACHE",
                       "CRISPEMBED_QWEN2VL_REF", "CRISPEMBED_BPE_LEGACY_WHITESPACE"]},
                   "native": native, "reference": ref, "native_subprocess_ms": elapsed})
    if args.gold:
        gold = Path(args.gold).read_text(encoding="utf-8")
        report["ground_truth"] = {"sha256": sha256(args.gold), "python": text_metrics(gold, ref["text"]),
                                   "native": text_metrics(gold, native["text"])}
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ["request_parity_pass", "speed_comparable", "speed_pass", "native_over_python"]}))
    return 0 if report["request_parity_pass"] and report["speed_pass"] else 1


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def token_limit(value):
    number = int(value)
    if number < 0 or number > 8192:
        raise argparse.ArgumentTypeError("must be in [0,8192]; zero selects the original total-context limit")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("--model", default=MODEL_ID)
    ref.add_argument("--blueprint", type=Path, required=True)
    ref.add_argument("--image", type=Path, required=True)
    ref.add_argument("--task", choices=list(PROMPTS), default="text")
    ref.add_argument("--max-tokens", type=token_limit, default=1024,
                     help="explicit new-token limit, or 0 for original max_length=8192")
    ref.add_argument("--threads", type=positive_int, default=4)
    ref.add_argument("--repeat", type=positive_int, default=3)
    ref.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ref.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="f32")
    ref.add_argument("--attention", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    ref.add_argument("--output", type=Path, required=True)
    ref.add_argument("--dump-reference", type=Path)
    ref.set_defaults(run=reference)
    comp = commands.add_parser("compare")
    comp.add_argument("--reference", type=Path, required=True)
    comp.add_argument("--native", type=Path, required=True)
    comp.add_argument("--gguf", type=Path, required=True)
    comp.add_argument("--image", type=Path, required=True)
    comp.add_argument("--gold", type=Path, help="optional human-transcribed expected text")
    comp.add_argument("--max-slowdown", type=float, default=1.0,
                      help="frozen native/python warm-time ratio budget; default demands no slowdown")
    comp.add_argument("--output", type=Path, required=True)
    comp.set_defaults(run=compare)
    args = parser.parse_args(argv)
    if args.command == "reference" and args.repeat < 2:
        parser.error("--repeat needs at least 2 (cold and warm)")
    if args.command == "compare" and (not math.isfinite(args.max_slowdown) or args.max_slowdown <= 0):
        parser.error("--max-slowdown must be finite and positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
