#!/usr/bin/env python3
"""Hosted, checkpoint-free original/native GDN head-group comparison.

Execute the unmodified Transformers 5.5.4 module and DynamicCache on bounded
synthetic parameters. Freeze every original case before building or executing
the native component. These results cannot qualify a checkpoint, OCR output,
full 4B hidden width, model export, quantization, or a public model alias.
Imports above main are stdlib-only so contract controls need no frameworks.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "lighton4b-gdn-components/v1"
TRANSFORMERS_VERSION = "5.5.4"
TRANSFORMERS_COMMIT = "75d3bdcd4b3ba70cee21287218d9764f33da41f0"
SOURCE_FILES = {
    "models/qwen3_5/modeling_qwen3_5.py": "aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048",
    "models/qwen3_5/configuration_qwen3_5.py": "cbea1f42a949edb6b9b001109df68cbf7cff04c96d3fe8a8e96602cb017b5a36",
    "cache_utils.py": "a51d2cd525f4458941a20cb5b3b8a8d989d8ca5bcd451855a27bb41743fed586",
}
FUNCTION_HASHES = {
    "torch_causal_conv1d_update": "3e69946e91492dea1d3650f62d88566ba654c3f954894d6079e0ed0cb6ef106c",
    "torch_chunk_gated_delta_rule": "3cf91cd1d7af6970672f5b78dd7889b54b24568c16326e73259e57bc32a4a361",
    "torch_recurrent_gated_delta_rule": "0764d99b2091eb050b0d8009d97167a4b0f67f7452a3530b6c8cbc49f530e28e",
    "l2norm": "6bfc53d1a491d0ae3856b85cb7fde1ea2198d5e3ed427953652cf2ede696e6fd",
    "Qwen3_5GatedDeltaNet.forward": "7c437c5a9ee098edde2f984305f843ec927c5222448411487dd5155ed610f5e8",
    "Qwen3_5RMSNormGated.forward": "9e4a919ddd7afcb4301dd2bdaab79f193dd5c0732482e8ac1dae510c3bf2ded9",
}
PACKAGES = {
    "torch": "2.13.0+cpu", "transformers": TRANSFORMERS_VERSION,
    "numpy": "2.3.5", "gguf": "0.19.0", "huggingface-hub": "1.31.0",
    "tokenizers": "0.22.2", "safetensors": "0.8.0", "pillow": "12.0.0",
    "jinja2": "3.1.6",
}
PHASES = ("prefill", "cached0", "cached1", "full_prefix")
# Width 128 and the 16:32 head geometry match the reviewed 4B GDN source.
# Hidden size 16, random weights, one layer, and short sequences are synthetic.
CASES = (
    {"name": "equal16-t65", "key_heads": 16, "value_heads": 16, "batch": 1, "prefill": 65, "seed": 35401},
    {"name": "grouped16-32-t65", "key_heads": 16, "value_heads": 32, "batch": 1, "prefill": 65, "seed": 35402},
    {"name": "grouped16-32-t1", "key_heads": 16, "value_heads": 32, "batch": 1, "prefill": 1, "seed": 35403},
    {"name": "grouped2-4-t3-b2", "key_heads": 2, "value_heads": 4, "batch": 2, "prefill": 3, "seed": 35404},
)
HIDDEN, WIDTH, KERNEL = 16, 128, 4
MAX_ABS, MIN_COS = 0.002, 0.99999
MAX_ARTIFACT_BYTES = 768 * 1024 * 1024
NATIVE_SOURCES = (
    "src/qwen35_delta.cpp", "src/qwen35_delta.h", "src/qwen35_linear.cpp", "src/qwen35_linear.h",
    "src/qwen35_math.h", "src/qwen35_weight_cache.h", "src/crispembed_diff.h",
    "src/qwen35_profile.cpp", "src/qwen35_profile.h",
    "src/core/env_gate.h", "src/core/env_value.h", "src/core/ggml_cpu_backend.h",
    "src/core/clean_exit.h", "src/core/gguf_loader.h", "src/core/gguf_loader.cpp",
    "tests/test_qwen35_gdn_heads.cpp", "CMakeLists.txt",
    "tools/ci-heavy/lighton4b_gdn_reference.py", "tools/lighton4b-gdn-requirements.txt",
)
FALSE_QUALIFICATIONS = (
    "checkpoint_loaded", "checkpoint_parity", "native_model_supported", "native_model_qualified",
    "full_hidden_width_qualified", "ocr_quality_qualified", "export_qualified",
    "quantization_qualified", "public_alias_qualified",
)
COMMON_CONTROLS = (
    "source_contract", "gate_group_unset", "gate_group_empty", "gate_group_zero", "gate_ocr_zero",
    "reject_key_zero", "reject_key_nondivisor", "reject_tokens_over_limit", "reject_width_over_limit",
    "reject_missing_qkv", "reject_bad_norm_epsilon", "reject_nonfinite_norm",
    "reject_missing_hidden", "reject_nonfinite_hidden", "reject_quantized_weight",
    "execute_prefill", "execute_cached0", "execute_cached1", "execute_full_prefix",
    "reset_projected", "reset_core", "comparator_rejects_corruption", "dump_preserved",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    path = Path(path)
    return {"bytes": path.stat().st_size, "sha256": sha256(path.read_bytes())}


def verify_source_files(package_root, expected=SOURCE_FILES):
    root = Path(package_root).resolve()
    result = {}
    for name, digest in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or escaping original source: " + name)
        record = file_record(path)
        if record["sha256"] != digest:
            raise ValueError("Installed original source differs: " + name)
        result["transformers/" + name] = record
    return result


def checked_case(case):
    if set(case) != {"name", "key_heads", "value_heads", "batch", "prefill", "seed"}:
        raise ValueError("Unexpected synthetic case fields")
    if not re.fullmatch(r"[a-z0-9-]+", case["name"]):
        raise ValueError("Unsafe synthetic case name")
    for name in ("key_heads", "value_heads", "batch", "prefill", "seed"):
        if type(case[name]) is not int or case[name] < 1:
            raise ValueError("Synthetic dimensions must be positive integers")
    k, v = case["key_heads"], case["value_heads"]
    if k > v or v % k or v > 32 or case["batch"] > 2 or case["prefill"] > 65:
        raise ValueError("Synthetic dimensions exceed the reviewed component scope")
    return dict(case)


def expected_stages():
    """Exact native payload contract; attribution must not overwrite primary."""
    rows = []
    for phase in PHASES:
        projected = [f"{phase}.linear.0.{name}" for name in ("qkv", "z", "a", "b", "norm", "output")]
        caches = [f"{phase}.cache.0.{name}" for name in ("conv_states", "recurrent_states")]
        rows.extend(("projected", name, name + ".f32") for name in projected + caches)
        rows.extend(("core", name, "core/" + name + ".f32")
                    for name in [f"{phase}.linear.0.norm"] + caches)
        rows.extend(("delta", name, "delta/" + name + ".f32")
                    for name in [f"{phase}.delta.output", f"{phase}.delta.final_state"])
    return tuple(rows)


def metric_pass(record):
    if record.get("finite") is not True:
        return False
    for name in ("max_abs", "cos_min"):
        value = record.get(name)
        if type(value) not in (float, int) or not math.isfinite(value):
            return False
    return 0 <= record["max_abs"] <= MAX_ABS and MIN_COS <= record["cos_min"] <= 1.000000000001


def validate_stage_records(records):
    expected = [(lane, name) for lane, name, _ in expected_stages()]
    if [(item.get("lane"), item.get("stage")) for item in records] != expected:
        raise ValueError("Missing, repeated, reordered, or unexpected native stage")
    return all(metric_pass(item) for item in records)


def expected_controls(equal_heads):
    extra = ([f"equal_{lane}_{phase}" for phase in PHASES for lane in ("projected", "core")]
             if equal_heads else ["legacy_rejects_grouped_zero", "legacy_rejects_grouped_one"])
    return set(COMMON_CONTROLS) | set(extra)


def parse_controls(log, equal_heads):
    result = []
    for line in log.splitlines():
        if not line.startswith("CONTROL "):
            continue
        match = re.fullmatch(r"CONTROL name=([a-z0-9_]+) pass=([01])", line)
        if match is None:
            raise ValueError("Malformed native control result")
        result.append({"name": match[1], "passed": match[2] == "1"})
    names = [item["name"] for item in result]
    if len(names) != len(set(names)) or set(names) != expected_controls(equal_heads):
        raise ValueError("Missing, repeated, or unexpected native control")
    return result


def report_qualification(report):
    if any(report.get(name) is not False for name in FALSE_QUALIFICATIONS):
        raise ValueError("Component evidence cannot qualify a checkpoint or OCR model")
    return True


def metrics(actual, reference, np):
    if actual.shape != reference.shape or actual.size == 0:
        raise ValueError("Comparison shape or element coverage differs")
    finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
    result = {"shape": list(reference.shape), "elements": int(reference.size), "finite": finite}
    if not finite:
        result.update(max_abs=None, cos_min=None, passed=False)
        return result
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    difference = a - b
    rows_a, rows_b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    na, nb = (rows_a * rows_a).sum(-1), (rows_b * rows_b).sum(-1)
    product = (rows_a * rows_b).sum(-1)
    cos = np.zeros_like(na)
    both_nonzero = (na > 1e-18) & (nb > 1e-18)
    cos[both_nonzero] = product[both_nonzero] / np.sqrt(na[both_nonzero] * nb[both_nonzero])
    cos[(na <= 1e-18) & (nb <= 1e-18)] = 1.0
    total_a, total_b = float(na.sum()), float(nb.sum())
    global_cos = (float(product.sum()) / math.sqrt(total_a * total_b)
                  if total_a > 1e-18 and total_b > 1e-18
                  else float(total_a <= 1e-18 and total_b <= 1e-18))
    result.update(
        max_abs=float(np.abs(difference).max()), mean_abs=float(np.abs(difference).mean()),
        rms=float(np.sqrt((difference * difference).mean())),
        cos_min=float(cos.min()), cos_global=global_cos,
        actual_max_abs=float(np.abs(a).max()), reference_max_abs=float(np.abs(b).max()),
        actual_l2=math.sqrt(total_a), reference_l2=math.sqrt(total_b),
    )
    result["passed"] = metric_pass(result)
    return result


def load_original():
    versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    if versions != PACKAGES:
        raise ValueError("Installed reference package versions differ: " + json.dumps(versions, sort_keys=True))
    import transformers
    package_root = Path(transformers.__file__).resolve().parent
    source_files = verify_source_files(package_root)
    import numpy as np
    import gguf
    import torch
    from transformers.models.qwen3_5 import modeling_qwen3_5 as original
    from transformers import Qwen3_5TextConfig
    from transformers.cache_utils import DynamicCache
    if transformers.__version__ != TRANSFORMERS_VERSION or torch.__version__ != PACKAGES["torch"]:
        raise ValueError("Imported package version differs from installed metadata")
    if (Qwen3_5TextConfig is not original.Qwen3_5TextConfig or DynamicCache is not original.DynamicCache
            or Path(inspect.getsourcefile(DynamicCache)).resolve() != package_root / "cache_utils.py"):
        raise ValueError("Original configuration or cache class was replaced")
    if (original.causal_conv1d_fn is not None or original.causal_conv1d_update is not None
            or original.chunk_gated_delta_rule is not None or original.fused_recurrent_gated_delta_rule is not None
            or original.FusedRMSNormGated is not None or original.is_fast_path_available):
        raise ValueError("Reference requires the original Torch fallback without optional fast kernels")
    functions = {}
    for name, digest in FUNCTION_HASHES.items():
        obj = original
        for component in name.split("."):
            obj = getattr(obj, component)
        if (Path(inspect.getsourcefile(obj)).resolve()
                != package_root / "models/qwen3_5/modeling_qwen3_5.py"):
            raise ValueError("Original function was replaced: " + name)
        actual = sha256(inspect.getsource(obj).encode())
        if actual != digest:
            raise ValueError("Original function source differs: " + name)
        functions[name] = actual
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mkldnn.enabled = False
    original_info = {
        "repository": "huggingface/transformers", "commit": TRANSFORMERS_COMMIT,
        "version": TRANSFORMERS_VERSION, "packages": versions, "source_files": source_files,
        "executed_function_sha256": functions,
        "cache_class": DynamicCache.__module__ + "." + DynamicCache.__name__,
        "device": "cpu", "dtype": "float32", "threads": 2, "interop_threads": 1,
        "deterministic_algorithms": True, "mkldnn_enabled": False,
        "float32_matmul_precision": "highest", "optional_fast_kernels": False,
        "installed_distributions": sorted(
            {item.metadata["Name"] + "==" + item.version for item in importlib.metadata.distributions()}),
    }
    return torch, np, gguf, original, Qwen3_5TextConfig, DynamicCache, original_info


def write_gguf(path, values, gguf):
    writer = gguf.GGUFWriter(str(path), "qwen35_gdn_component")
    for name, value in values.items():
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def generate_case(case, folder, runtime):
    torch, np, gguf, original, Config, Cache, _ = runtime
    checked_case(case)
    folder.mkdir(parents=True)
    raw_dir = folder / "reference"
    raw_dir.mkdir()
    config = Config(hidden_size=HIDDEN, num_hidden_layers=1, layer_types=["linear_attention"],
                    linear_num_key_heads=case["key_heads"], linear_num_value_heads=case["value_heads"],
                    linear_key_head_dim=WIDTH, linear_value_head_dim=WIDTH,
                    linear_conv_kernel_dim=KERNEL, rms_norm_eps=1e-6)
    torch.manual_seed(case["seed"])
    block = original.Qwen3_5GatedDeltaNet(config, 0).eval().float()
    if (block.causal_conv1d_fn is not None
            or block.causal_conv1d_update is not original.torch_causal_conv1d_update
            or block.chunk_gated_delta_rule is not original.torch_chunk_gated_delta_rule
            or block.recurrent_gated_delta_rule is not original.torch_recurrent_gated_delta_rule
            or type(block.norm) is not original.Qwen3_5RMSNormGated):
        raise ValueError("Original component selected an unexpected implementation")
    refs, phase, calls, handles, originals = {}, ["prefill"], [], [], {}

    def array(value):
        if value.dtype != torch.float32 or value.device.type != "cpu":
            raise ValueError("Reference unexpectedly changed device or dtype")
        result = np.ascontiguousarray(value.detach().numpy()).copy()
        if not np.isfinite(result).all():
            raise ValueError("Non-finite original component result")
        return result

    def put(name, value):
        if name in refs:
            raise ValueError("Repeated original capture: " + name)
        refs[name] = array(value)

    for attr, field in (("in_proj_qkv", "qkv"), ("in_proj_z", "z"),
                        ("in_proj_a", "a"), ("in_proj_b", "b"), ("norm", "norm")):
        def hook(module, args, output, field=field):
            value = output
            if field == "norm":
                value = output.reshape(case["batch"], -1, case["value_heads"], WIDTH)
            put(f"{phase[0]}.linear.0.{field}", value)
        handles.append(getattr(block, attr).register_forward_hook(hook))

    def wrap_delta(attr, fn):
        def run(query, key, value, **kwargs):
            label = phase[0]
            expected = (case["batch"], query.shape[1], case["value_heads"], WIDTH)
            if tuple(query.shape) != expected or tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise ValueError("Original expanded Q/K/V shapes differ")
            ratio = case["value_heads"] // case["key_heads"]
            for name, item in (("q", query), ("k", key)):
                groups = item.reshape(case["batch"], item.shape[1], case["key_heads"], ratio, WIDTH)
                if not torch.equal(groups, groups[:, :, :, :1, :].expand_as(groups)):
                    raise ValueError("Original Q/K repeat-interleave ordering differs")
                put(f"{label}.delta.{name}", item)
            put(f"{label}.delta.v", value)
            for name in ("g", "beta"):
                put(f"{label}.delta.{name}", kwargs[name])
            initial = kwargs["initial_state"]
            if (initial is None) != (label in ("prefill", "full_prefix")):
                raise ValueError("Original prefill/cached recurrence route differs")
            if kwargs["output_final_state"] is not True or kwargs["use_qk_l2norm_in_kernel"] is not True:
                raise ValueError("Original recurrence flags differ")
            if initial is not None:
                prior = "prefill" if label == "cached0" else "cached0"
                if not np.array_equal(array(initial), refs[f"{prior}.cache.0.recurrent_states"]):
                    raise ValueError("Original recurrence did not consume the previous cache")
            output, state = fn(query, key, value, **kwargs)
            put(f"{label}.delta.output", output)
            put(f"{label}.delta.final_state", state)
            calls.append({"phase": label, "function": fn.__name__, "qk_repeat_factor": ratio,
                          "initial_state_present": initial is not None})
            return output, state
        return run

    for attr in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
        originals[attr] = getattr(block, attr)
        setattr(block, attr, wrap_delta(attr, originals[attr]))
    original_conv = block.causal_conv1d_update

    def conv_update(hidden, state, weight, bias, activation):
        label = phase[0]
        prior = "prefill" if label == "cached0" else "cached0"
        if label not in ("cached0", "cached1") or not np.array_equal(
                array(state), refs[f"{prior}.cache.0.conv_states"]):
            raise ValueError("Original convolution did not consume the previous cache")
        output = original_conv(hidden, state, weight, bias, activation)
        calls.append({"phase": label, "function": original_conv.__name__})
        return output

    block.causal_conv1d_update = conv_update
    generator = torch.Generator().manual_seed(case["seed"] + 1000)
    inputs = {name: torch.randn(case["batch"], length, HIDDEN, generator=generator)
              for name, length in (("prefill", case["prefill"]), ("cached0", 1), ("cached1", 1))}
    inputs["full_prefix"] = torch.cat([inputs[name] for name in PHASES[:3]], dim=1)
    weights = {"l.layers.0.linear_attn." + name: array(value) for name, value in block.state_dict().items()}
    cache = Cache(config=config)
    try:
        with torch.inference_mode():
            for label in PHASES:
                phase[0] = label
                if label == "full_prefix":
                    cache = Cache(config=config)
                put(f"{label}.linear.0.input", inputs[label])
                result = block(inputs[label], cache_params=cache)
                put(f"{label}.linear.0.output", result)
                for name in ("conv_states", "recurrent_states"):
                    put(f"{label}.cache.0.{name}", getattr(cache.layers[0], name))
    finally:
        for handle in handles:
            handle.remove()
        for attr, fn in originals.items():
            setattr(block, attr, fn)
        block.causal_conv1d_update = original_conv
    expected_calls = [
        ("prefill", "torch_chunk_gated_delta_rule"),
        ("cached0", "torch_causal_conv1d_update"), ("cached0", "torch_recurrent_gated_delta_rule"),
        ("cached1", "torch_causal_conv1d_update"), ("cached1", "torch_recurrent_gated_delta_rule"),
        ("full_prefix", "torch_chunk_gated_delta_rule"),
    ]
    if [(item["phase"], item["function"]) for item in calls] != expected_calls:
        raise ValueError("Original function execution coverage differs")
    channels = (2 * case["key_heads"] + case["value_heads"]) * WIDTH
    for label in PHASES:
        length = inputs[label].shape[1]
        if refs[f"{label}.linear.0.qkv"].shape != (case["batch"], length, channels):
            raise ValueError("Original compact QKV width differs")
        if refs[f"{label}.cache.0.conv_states"].shape != (case["batch"], channels, KERNEL):
            raise ValueError("Original compact convolution cache width differs")
        if refs[f"{label}.cache.0.recurrent_states"].shape != (case["batch"], case["value_heads"], WIDTH, WIDTH):
            raise ValueError("Original recurrent cache geometry differs")
    # Re-run the same original instance after removing all observation hooks.
    # This proves observation did not change the module's output or caches.
    neutral = []
    cache = Cache(config=config)
    with torch.inference_mode():
        for label in PHASES:
            if label == "full_prefix":
                cache = Cache(config=config)
            result = block(inputs[label], cache_params=cache)
            for name, value in [(f"{label}.linear.0.output", result)] + [
                    (f"{label}.cache.0.{name}", getattr(cache.layers[0], name))
                    for name in ("conv_states", "recurrent_states")]:
                exact = array(value).tobytes() == refs[name].tobytes()
                neutral.append({"stage": name, "byte_exact": exact})
    for name, value in (("conv_weights", block.conv1d.weight), ("a_log", block.A_log),
                        ("dt_bias", block.dt_bias), ("norm_weights", block.norm.weight)):
        put("linear." + name, value)
    self_checks = []
    for name, actual, expected in [
        ("incremental_vs_full_output", np.concatenate([refs[f"{p}.linear.0.output"] for p in PHASES[:3]], axis=1), refs["full_prefix.linear.0.output"]),
        *[("incremental_vs_full_" + name, refs[f"cached1.cache.0.{name}"], refs[f"full_prefix.cache.0.{name}"])
          for name in ("conv_states", "recurrent_states")],
    ]:
        result = metrics(actual, expected, np)
        self_checks.append({"stage": name, **result})
    manifest = {}
    for name, value in refs.items():
        path = raw_dir / (name + ".f32")
        path.write_bytes(value.astype("<f4", copy=False).tobytes())
        manifest[name] = {"shape": list(value.shape), "dtype": "float32", **file_record(path)}
    write_gguf(folder / "weights.gguf", weights, gguf)
    write_gguf(folder / "reference.gguf", refs, gguf)
    # Verify the exact arrays written by GGUFWriter before native use.
    for filename, values in (("weights.gguf", weights), ("reference.gguf", refs)):
        reader = gguf.GGUFReader(str(folder / filename))
        actual = {tensor.name: tensor for tensor in reader.tensors}
        if set(actual) != set(values):
            raise ValueError("GGUF tensor coverage differs: " + filename)
        for name, value in values.items():
            stored = actual[name].data
            if stored.shape != value.shape or stored.dtype != np.float32 or stored.tobytes() != value.tobytes():
                raise ValueError("GGUF tensor bytes or shape differ: " + name)
        del reader
    result = {
        **case, "hidden_size": HIDDEN, "head_width": WIDTH, "conv_kernel": KERNEL,
        "compact_qkv_channels": channels, "weights": file_record(folder / "weights.gguf"),
        "reference": file_record(folder / "reference.gguf"), "reference_tensors": manifest,
        "original_function_calls": calls, "observer_neutrality": neutral,
        "original_incremental_checks": self_checks,
        "original_checks_passed": all(item["byte_exact"] for item in neutral)
                                   and all(item["passed"] for item in self_checks),
    }
    (folder / "reference-manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def compare_native(case, folder, binary, np):
    for filename, field in (("weights.gguf", "weights"), ("reference.gguf", "reference")):
        if file_record(folder / filename) != case[field]:
            raise ValueError("Frozen original artifact changed before native use: " + filename)
    dump = folder / "native"
    for subdir in (dump, dump / "core", dump / "delta"):
        subdir.mkdir()
    env = dict(os.environ, CRISPEMBED_DIFF_DUMP_DIR=str(dump))
    command = [str(binary), str(folder / "weights.gguf"), str(folder / "reference.gguf"), str(case["key_heads"])]
    timeout_error = None
    try:
        native = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
        exit_code, log = native.returncode, native.stdout + native.stderr
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        exit_code, log = None, decoded(error.stdout) + decoded(error.stderr)
        timeout_error = "Native component exceeded the 180-second case limit"
    log = log.replace(str(folder), "$CASE").replace(str(ROOT), "$REPO")
    (folder / "native.log").write_text(log)
    rows = []
    for lane, name, relative in expected_stages():
        manifest = case["reference_tensors"][name]
        source = folder / "reference" / (name + ".f32")
        target = dump / relative
        error = None
        if file_record(source) != {key: manifest[key] for key in ("bytes", "sha256")}:
            error = "Frozen reference changed before native comparison"
        elif not target.is_file() or target.stat().st_size != manifest["bytes"]:
            error = "Missing or incomplete native payload"
        if error:
            rows.append({"lane": lane, "stage": name, "finite": False, "passed": False,
                         "error": error, "payload": "native/" + relative})
            continue
        reference = np.fromfile(source, dtype="<f4").reshape(manifest["shape"])
        actual = np.fromfile(target, dtype="<f4").reshape(manifest["shape"])
        rows.append({"lane": lane, "stage": name, **metrics(actual, reference, np),
                     "native_payload": {"path": "native/" + relative, **file_record(target)}})
    stages_pass = validate_stage_records(rows)
    control_error = None
    try:
        controls = parse_controls(log, case["key_heads"] == case["value_heads"])
    except ValueError as error:
        controls, control_error = [], str(error)
    controls_pass = control_error is None and all(item["passed"] for item in controls)
    frozen_intact = all(file_record(folder / filename) == case[field]
                        for filename, field in (("weights.gguf", "weights"), ("reference.gguf", "reference")))
    return {"exit_code": exit_code, "timeout_error": timeout_error,
            "stage_count": len(rows), "stages": rows,
            "first_failing_stage": next(({"lane": r["lane"], "stage": r["stage"]} for r in rows if not r["passed"]), None),
            "controls": controls, "control_error": control_error, "controls_passed": controls_pass,
            "frozen_original_artifacts_unchanged": frozen_intact,
            "log": "native.log", "passed": exit_code == 0 and stages_pass and controls_pass and frozen_intact}


def portable_build(build, output):
    command = ["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
               "-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_BLAS=OFF",
               "-DGGML_METAL=OFF", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF",
               "-DCRISPEMBED_BUILD_SHARED=OFF"]
    build_command = ["cmake", "--build", str(build), "--target", "test-qwen35-gdn-heads", "-j", "2"]
    for name, args in (("configure", command), ("build", build_command)):
        result = subprocess.run(args, capture_output=True, text=True, timeout=600)
        log = (result.stdout + result.stderr).replace(str(ROOT), "$REPO").replace(str(build), "$BUILD")
        (output / (name + ".log")).write_text(log)
        if result.returncode:
            raise ValueError("Portable native " + name + " failed; see " + name + ".log")
    binary = build / "test-qwen35-gdn-heads"
    if not binary.is_file():
        raise ValueError("Native target did not produce the component runner")
    return binary


def artifact_size(output):
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("Framework execution and native builds belong on hosted CI")
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        parser.error("The checkpoint-free execution requires both Hugging Face offline flags")
    if args.output.exists() or args.build_dir.exists():
        parser.error("Fresh output and build directories are required")
    output, build = args.output.resolve(), args.build_dir.resolve()
    output.mkdir(parents=True)
    report = {"schema": SCHEMA, "scope": "synthetic single-layer CPU/F32 components; no model weights or OCR",
              **{name: False for name in FALSE_QUALIFICATIONS}, "component_passed": False,
              "max_abs_budget": MAX_ABS, "min_row_cosine_budget": MIN_COS,
              "cases": [], "status": "started"}
    code = 1
    try:
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report["github_sha"] = os.environ.get("GITHUB_SHA")
        report["github_run_id"] = os.environ.get("GITHUB_RUN_ID")
        if os.environ.get("GITHUB_EVENT_PATH"):
            event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
            report["pull_request_head_sha"] = event.get("pull_request", {}).get("head", {}).get("sha")
            report["pull_request_base_sha"] = event.get("pull_request", {}).get("base", {}).get("sha")
        report["native_sources"] = {name: file_record(ROOT / name) for name in NATIVE_SOURCES}
        report["ggml_commit"] = subprocess.check_output(["git", "-C", str(ROOT / "ggml"), "rev-parse", "HEAD"], text=True).strip()
        runtime = load_original()
        report["original"] = runtime[-1]
        for case in CASES:
            print("Generating original component:", case["name"], flush=True)
            report["cases"].append(generate_case(case, output / case["name"], runtime))
        report["original_references_frozen_before_native"] = True
        report["reference_artifact_bytes"] = artifact_size(output)
        (output / "original-reference.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if report["reference_artifact_bytes"] > MAX_ARTIFACT_BYTES:
            raise ValueError("Component reference artifact byte bound exceeded")
        if not all(case["original_checks_passed"] for case in report["cases"]):
            raise ValueError("Original observation or incremental/full-prefix controls failed; original arrays retained")
        binary = portable_build(build, output)
        report["native_binary"] = file_record(binary)
        for case in report["cases"]:
            print("Comparing native component:", case["name"], flush=True)
            case["native"] = compare_native(case, output / case["name"], binary, runtime[1])
            print(case["name"], "stages", case["native"]["stage_count"], "passed", case["native"]["passed"], flush=True)
        for name, before in report["native_sources"].items():
            if file_record(ROOT / name) != before:
                raise ValueError("Native or harness source changed during execution: " + name)
        report["native_binary_unchanged"] = file_record(binary) == report["native_binary"]
        if not report["native_binary_unchanged"]:
            raise ValueError("Native component binary changed during execution")
        import transformers
        if verify_source_files(Path(transformers.__file__).resolve().parent) != report["original"]["source_files"]:
            raise ValueError("Original source changed during execution")
        report["component_passed"] = all(case["native"]["passed"] for case in report["cases"])
        report["status"] = "passed" if report["component_passed"] else "failed"
        code = 0 if report["component_passed"] else 1
    except Exception as error:
        report["status"] = "failed"
        report["error"] = (f"{type(error).__name__}: {error}".replace(str(output), "$OUT")
                           .replace(str(build), "$BUILD").replace(str(ROOT), "$REPO"))
        print(report["error"], file=sys.stderr)
    finally:
        report_qualification(report)
        report["artifact_bytes_before_report"] = artifact_size(output)
        if report["artifact_bytes_before_report"] > MAX_ARTIFACT_BYTES:
            report.update(status="failed", component_passed=False, error="Component artifact byte bound exceeded")
            code = 1
        (output / "component-report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        summary = ["# LightOnOCR-3-4B synthetic GDN components", "", "Status: " + report["status"],
                   "", "CPU/F32 synthetic one-layer evidence only; checkpoint, OCR, export and public-alias qualification remain false.", ""]
        for case in report["cases"]:
            native = case.get("native", {})
            summary.append(f"- {case['name']}: {native.get('stage_count', 0)} native stages; pass={native.get('passed', False)}")
        if "error" in report:
            summary += ["", report["error"]]
        (output / "summary.md").write_text("\n".join(summary) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
