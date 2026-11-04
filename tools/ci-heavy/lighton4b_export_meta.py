#!/usr/bin/env python3
"""Hosted official LightOnOCR-3-4B meta-schema probe; no model data or forward.

Use an isolated, pinned Transformers 5.5.4/PyTorch environment on hosted CI.
The original classes are constructed from the acquired config on the meta device.
No from_pretrained call, tokenizer load, checkpoint access or GGUF writer is used.
"""

import argparse
from collections import Counter
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
CONVERTER_PATH = "models/convert-qwen35-ocr-to-gguf.py"
PROFILE_PATH = "models/lightonocr3_4b_profile.py"
PROBE_PATH = "tools/ci-heavy/lighton4b_export_meta.py"
CODE_PATHS = (CONVERTER_PATH, PROFILE_PATH, PROBE_PATH)
SCOPE = ("Official Transformers meta-device construction and schema only; no model forward, "
         "checkpoint or full-tokenizer acquisition, GGUF emission, native execution or model qualification")


def canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def load_converter(root=ROOT):
    spec = importlib.util.spec_from_file_location("lighton4b_export_meta_converter", Path(root) / CONVERTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_provenance(root=ROOT):
    root = Path(root)
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()
    revision = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Probe checkout does not have an immutable commit")
    git("ls-files", "--error-unmatch", "--", *CODE_PATHS)
    if git("status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS):
        raise ValueError("Probe source closure has uncommitted changes")
    return {"repository": "CrispStrobe/CrispEmbed", "revision": revision,
            "source_files_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                    for name in CODE_PATHS}}


@contextlib.contextmanager
def offline_imports():
    """Prevent Hub access and socket connections in this metadata-only process."""
    def deny(*_args, **_kwargs):
        raise RuntimeError("Network access is forbidden in the official meta-schema probe")
    environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    with patch.dict(os.environ, environment), patch.object(socket, "getaddrinfo", deny), \
            patch.object(socket, "create_connection", deny), patch.object(socket.socket, "connect", deny), \
            patch.object(socket.socket, "connect_ex", deny):
        yield


def _meta_shape(name, tensor):
    if not getattr(tensor, "is_meta", False) or str(tensor.device) != "meta":
        raise ValueError("Materialized tensor in official meta schema: " + name)
    shape = list(tensor.shape)
    if any(type(size) is not int or size < 1 for size in shape):
        raise ValueError("Invalid official meta tensor shape: " + name)
    return shape


def inspect_meta_model(model, converter, config):
    """Inspect metadata and parameter ownership; never read a tensor value."""
    state = model.state_dict()
    shapes = {name: _meta_shape(name, tensor) for name, tensor in state.items()}
    schema = converter.validate_lighton4b_meta_schema(shapes, config)
    source = converter._lighton4b_profile()
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("Official meta model does not tie output head to the input embedding")
    named = list(model.named_parameters())
    if len({name for name, _ in named}) != len(named) or len({id(tensor) for _, tensor in named}) != len(named):
        raise ValueError("Official unique-parameter enumeration contains duplicates")
    unique = {name: _meta_shape(name, tensor) for name, tensor in named}
    if unique != source.expected_tensor_schema(config):
        raise ValueError("Official unique parameters differ from the 723 stored source tensors")
    count = sum(math.prod(shape) for shape in unique.values())
    if count != source.PARAMETER_COUNT:
        raise ValueError("Official unique parameter count differs from source header")
    buffers = {name: _meta_shape(name, tensor) for name, tensor in model.named_buffers()}
    return {"schema_validation": schema, "state_dict_shapes": shapes,
            "state_dict_shapes_sha256": canonical_sha256(shapes),
            "state_dict_tensor_count": len(shapes), "unique_parameter_tensor_count": len(unique),
            "unique_parameter_count": count,
            "state_dict_elements_including_tied_alias": sum(math.prod(shape) for shape in shapes.values()),
            "output_head_is_input_embedding": True, "all_parameters_and_buffers_on_meta": True,
            "meta_parameter_dtype_counts": dict(sorted(Counter(str(tensor.dtype) for _, tensor in named).items())),
            "nonpersistent_buffer_shapes": buffers, "model_forward_observed": False}


def export_plan_summaries(converter, config):
    result = {}
    for dtype, recipe in (("f32", "original-f16"), ("f16", "original-f16"),
                          ("f16", converter.LIGHTON4B_VISION_RECIPE)):
        plan = converter.lighton4b_tensor_plan(config, dtype=dtype, recipe=recipe)
        result[dtype + ":" + recipe] = {
            "plan_sha256": canonical_sha256(plan), "format_version": plan["format_version"],
            "tensor_count": len(plan["records"]), "retained_f32_matrix_count": len(plan["keep_f32"]),
            "planned_dtype_counts": dict(sorted(Counter(item["dtype"] for item in plan["records"]).items())),
            "metadata_only": True, "payload_export_executed": False,
        }
    return result


def execute_official_meta(converter, config):
    """The only framework-dependent function; called exclusively by the hosted CLI."""
    with offline_imports():
        import torch
        import transformers
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

        hashes = {key: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                  for key, cls in (("model", Qwen3_5ForConditionalGeneration), ("config", Qwen3_5Config))}
        implementation = converter.validate_lighton4b_meta_sources(transformers.__version__, hashes)
        hf_config = Qwen3_5Config.from_dict(copy.deepcopy(config))
        hf_config._attn_implementation = "eager"

        def forbidden(*_args, **_kwargs):
            raise RuntimeError("Model forward or state loading is forbidden in the official meta-schema probe")
        with patch.object(torch.nn.Module, "_call_impl", forbidden), \
                patch.object(torch.nn.Module, "load_state_dict", forbidden), torch.device("meta"):
            model = Qwen3_5ForConditionalGeneration(hf_config)
            result = inspect_meta_model(model, converter, config)
            del model
        return {"implementation": implementation,
                "packages": {"python": platform.python_version(), "torch": torch.__version__,
                             "transformers": transformers.__version__},
                "network_access_disabled": True, "forward_and_state_loading_disabled": True,
                "model": result}


def run_probe(root=ROOT):
    root = Path(root)
    report = {"schema": "lightonocr3-4b-official-meta/v1", "status": "failed", "scope": SCOPE,
              "official_meta_observed": False, "original_inference_observed": False,
              "checkpoint_payload_acquired": False, "full_tokenizer_acquired": False,
              "payload_export_executed": False, "native_supported": False, "native_qualified": False,
              "export_qualified": False, "native_alias_qualified": False,
              "image_max_qualified": False, "full_document_qualified": False}
    try:
        report["producer"] = source_provenance(root)
        converter = load_converter(root)
        source = converter._lighton4b_profile()
        preflight = source.load_profile()
        audit = source.evidence()
        config_bytes = source._verified_bytes(source.FIXTURE / "config.json", audit["source_files"]["config.json"])
        config = source._json(config_bytes)
        source.validate_config(config)
        converter.validate_config(config)  # Exercise the preceding generic exporter guards too.
        report["source"] = {"repository": source.MODEL, "revision": source.REVISION,
                            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                            "audit_manifest_sha256": source.AUDIT_SHA256,
                            "tokenizer_prefix_audit_sha256": source.PREFIX_AUDIT_SHA256,
                            "checkpoint_header_sha256": source.HEADER_SHA256,
                            "checkpoint_tensor_count": preflight["checkpoint"]["tensor_contract"]["tensor_count"],
                            "declared_checkpoint_dtype": "BF16", "metadata_verified": True}
        report["official_meta"] = execute_official_meta(converter, config)
        report["official_meta_observed"] = True
        report["export_plans"] = export_plan_summaries(converter, config)
        report["status"] = "passed"
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error).replace(str(root), "[repository]")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Scoped JSON evidence; no model artifact is written")
    args = parser.parse_args(argv)
    report = run_probe()
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".partial")
    pending.write_bytes(raw)
    pending.replace(args.output)
    print(json.dumps({"status": report["status"], "scope": SCOPE, "report": args.output.name,
                      "report_sha256": hashlib.sha256(raw).hexdigest(),
                      "official_meta_observed": report["official_meta_observed"],
                      "payload_export_executed": False, "native_supported": False, "export_qualified": False}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
