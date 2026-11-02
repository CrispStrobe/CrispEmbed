#!/usr/bin/env python3
"""Capture actual pinned TeleOCR modules and greedy output for native parity.

This test tool uses the Apache-2.0 Hugging Face model, not a reimplementation
of its equations. It never downloads or executes an unpinned model module.
Production inference and GGUF conversion do not depend on this program.

Example:
  python tools/dump_teleocr_reference.py --model-dir teleocr-hf --out refs

Dependencies: torch==2.8.0, transformers==4.57.1, Pillow, numpy, gguf.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import gguf
import numpy as np
from PIL import Image
import PIL
import torch
import transformers
from transformers import AutoModel, AutoProcessor


REVISION = "b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d"
# Filled from the publicly pinned file, verified before trust_remote_code.
MODEL_MODULE_SHA256 = "7adac1cc17009f9f1b8e0c58284d7c0b9989a590f017b9e5964cc42270207922"
CASES = {
    "text": ("text.png", "Please output the text content from the image.", 0),
    "formula": (
        "formula.png",
        "Please write out the expression of the formula in the image using LaTeX format.",
        0,
    ),
    "table": (
        "table.png",
        "This is the image of a table. Please output the table in OTSL format.",
        0,
    ),
    "code": (
        "code.png",
        "The image contains a code snippet, please output the parsing result.",
        0,
    ),
    "layout": ("layout.jpg", "Analyze the image layout.", 1036),
    "synthetic": ("synthetic-text.png", "Please output the text content from the image.", 0),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Capture:
    """Keep the first (prefill) value of each stage, immediately on disk."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.arrays: dict[str, Path] = {}

    def put(self, name: str, tensor, width: int | None = None):
        if name in self.arrays:
            return
        if isinstance(tensor, tuple):
            tensor = tensor[0]
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            array = np.asarray(tensor, dtype=np.float32)
        if width is not None:
            array = array.reshape(-1, width)
        elif array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        array = np.ascontiguousarray(array)
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite reference stage: {name}")
        path = self.directory / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        self.arrays[name] = path

    def archive(self, output: Path, metadata: dict):
        writer = gguf.GGUFWriter(str(output), "teleocr-reference", use_temp_file=True)
        writer.add_string("teleocr.reference.metadata", json.dumps(metadata, ensure_ascii=False))
        for name, path in self.arrays.items():
            writer.add_tensor(name, np.load(path, mmap_mode="r", allow_pickle=False))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()


def attach_hooks(model, capture: Capture):
    handles = []
    visual = model.model.visual
    language = model.model.language_model

    def output_hook(name, width=None):
        return lambda _m, _a, output: capture.put(name, output, width)

    handles.append(visual.patch_embed.register_forward_hook(output_hook("vis_patch_embed")))
    for index, block in enumerate(visual.blocks):
        handles.append(block.register_forward_hook(output_hook(f"vis_layer_{index}")))
    handles.append(visual.merger.ln_q.register_forward_hook(output_hook("vis_pre_merger")))
    handles.append(visual.register_forward_hook(output_hook("merger_output")))

    def first_layer_input(_module, args, kwargs):
        capture.put("llm_embed", args[0] if args else kwargs["hidden_states"])

    handles.append(language.layers[0].register_forward_pre_hook(first_layer_input, with_kwargs=True))
    q_width = model.config.text_config.num_attention_heads * model.config.text_config.head_dim
    handles.append(language.layers[0].self_attn.q_norm.register_forward_hook(
        output_hook("llm_L0_Q_pre_rope", q_width)
    ))
    handles.append(language.layers[0].self_attn.o_proj.register_forward_hook(
        output_hook("llm_L0_attn_out")
    ))

    def post_attention_input(_module, args):
        capture.put("llm_L0_post_attn", args[0])

    handles.append(language.layers[0].post_attention_layernorm.register_forward_pre_hook(post_attention_input))
    for index, layer in enumerate(language.layers):
        handles.append(layer.register_forward_hook(output_hook(f"llm_layer_{index}")))
    handles.append(language.norm.register_forward_hook(output_hook("llm_final_norm")))
    handles.append(model.lm_head.register_forward_hook(output_hook("logits_step0")))

    # Observe the original rotation function's return value without changing it.
    module = importlib.import_module(type(model).__module__)
    original_rope = module.apply_multimodal_rotary_pos_emb

    def observed_rope(*args, **kwargs):
        result = original_rope(*args, **kwargs)
        q = result[0].transpose(1, 2).reshape(-1, q_width)
        capture.put("llm_L0_Q_rope", q)
        return result

    module.apply_multimodal_rotary_pos_emb = observed_rope

    def remove():
        for handle in handles:
            handle.remove()
        module.apply_multimodal_rotary_pos_emb = original_rope

    return remove


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-neutrality-check", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.max_new_tokens < 1:
        parser.error("threads and max-new-tokens must be positive")
    model_dir = args.model_dir.resolve()
    source_hash = sha256(model_dir / "modeling_naviocr.py")
    if source_hash != MODEL_MODULE_SHA256:
        raise ValueError("Refusing to execute a model module that differs from the pinned Apache-2.0 source")
    if transformers.__version__ != "4.57.1":
        raise ValueError("The frozen reference protocol requires transformers==4.57.1")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_fast=False
    )
    started = time.monotonic()
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, local_files_only=True,
        dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    print(f"Loaded reference in {time.monotonic() - started:.2f}s", flush=True)
    environment = {
        "model": "XingChen-AGI/TeleOCR", "revision": REVISION,
        "module_sha256": source_hash, "torch": torch.__version__,
        "transformers": transformers.__version__, "numpy": np.__version__,
        "pillow": PIL.__version__, "python": platform.python_version(),
        "device": "cpu", "dtype": "float32", "attention": "sdpa",
        "processor": type(processor.image_processor).__name__, "threads": args.threads,
        "max_new_tokens": args.max_new_tokens, "do_sample": False,
        "repetition_penalty": 1.05, "eos_token_ids": [151645, 151643],
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    results = []
    with torch.inference_mode():
        for case in args.cases:
            filename, prompt, resize = CASES[case]
            source_image = (Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "teleocr" / filename
                            if case == "synthetic" else model_dir / "assets" / filename)
            image = Image.open(source_image).convert("RGB")
            if resize:
                image = image.resize((resize, resize), Image.Resampling.BICUBIC)
            image_path = args.out / f"{case}.png"
            image.save(image_path)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            ]
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat], images=[image], padding=True, return_tensors="pt")
            capture = Capture(args.out / f"{case}-stages")
            capture.put("input_patches", inputs.pixel_values)
            capture.put("image_grid", inputs.image_grid_thw)
            capture.put("prompt_ids", inputs.input_ids[0])
            positions, _delta = model.model.get_rope_index(
                inputs.input_ids, inputs.image_grid_thw, None, None, inputs.attention_mask
            )
            capture.put("mrope_positions", positions[:, 0].T)
            remove = attach_hooks(model, capture)
            print(f"Starting {case}: grid={inputs.image_grid_thw.tolist()} prompt={inputs.input_ids.shape[1]}", flush=True)
            started = time.monotonic()
            try:
                sequence = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    repetition_penalty=1.05, no_repeat_ngram_size=0, use_cache=True,
                    logits_to_keep=1,
                )
            finally:
                remove()
            elapsed = time.monotonic() - started
            generated = sequence[0, inputs.input_ids.shape[1]:].cpu()
            capture.put("generated_ids", generated)
            text = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ids = generated.tolist()
            reached_eos = bool(ids and ids[-1] in (151645, 151643))
            result = {
                "case": case, "prompt": prompt, "image": image_path.name,
                "source_image_sha256": sha256(source_image),
                "image_sha256": sha256(image_path), "image_size": list(image.size),
                "grid_thw": inputs.image_grid_thw[0].tolist(),
                "prompt_ids": inputs.input_ids[0].tolist(), "generated_ids": ids,
                "text": text, "tokens": len(ids), "reached_eos": reached_eos,
                "hit_cap": not reached_eos and len(ids) >= args.max_new_tokens,
                "seconds": elapsed, "stages": list(capture.arrays),
                "environment": environment,
            }
            if case == args.cases[0] and not args.no_neutrality_check:
                control = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                    repetition_penalty=1.05, no_repeat_ngram_size=0, use_cache=True,
                    logits_to_keep=1,
                )
                result["observer_neutral"] = torch.equal(sequence, control)
                if not result["observer_neutral"]:
                    raise RuntimeError("Reference observer changed generated token IDs")
            capture.archive(args.out / f"{case}.gguf", result)
            result["reference_sha256"] = sha256(args.out / f"{case}.gguf")
            (args.out / f"{case}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            (args.out / f"{case}.txt").write_text(text, encoding="utf-8")
            results.append(result)
            print(json.dumps({k: result[k] for k in ("case", "tokens", "reached_eos", "seconds", "text")}, ensure_ascii=False), flush=True)
            (args.out / "reference-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(r["reached_eos"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
