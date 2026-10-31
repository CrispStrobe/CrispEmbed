#!/usr/bin/env python3
"""Hosted CPU-only position attribution of the Xiaomi-OCR-0 vision prefill.

Question: the retained hosted vision run (kernel ``vision-prefill`` v3) shows
``prefill.vision.position`` failing the unchanged strict gate against the frozen
reference (max_abs 0.0188269615, row cos_min 0.999989152). Is that gap the
reference's BF16 compute regime, or an independent deviation of the native CPU
position interpolation from the pinned Transformers implementation?

The replay executes the *actual* pinned method
``transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionModel.fast_pos_embed_interpolate``
(transformers 5.8.1) on CPU, on a minimal real Qwen3_5VisionModel instance
(depth 0, real ``nn.Embedding`` sized from the checkpoint config), with the
checkpoint's own BF16 ``model.visual.pos_embed.weight`` and with the F32
widening of that weight, and compares:

1. BF16 module CPU vs the frozen GPU-produced reference activation,
2. F32 module CPU vs the mounted native F32 position trace,
3. native trace vs the frozen reference (reproducing the retained report),
4. a permuted-row negative control.

It does not run the native graph, build anything, touch the decoder, OCR text,
quantization, speed, or claim decoded quality. The strict gate is recorded
unchanged and is never relaxed for dtype.

Only the standard library is imported at module scope so the repository test
can import the staged script without torch/numpy on the host.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

SCRIPT_VERSION = "xiaomi-position-replay-v1"
RUN_ID = "xiaomi-position-replay-v1"

BASE_COMMIT = "15adb672f71f8470a23db1284dafa102515adae9"
MODEL_REPO = "xiaomi-research/Xiaomi-OCR-0"
REVISION = "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30"
CHECKPOINT_BYTES = 1746937856
CHECKPOINT_SHA256 = "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348"
CONFIG_SHA256 = "2eb95982069a24336fe66b3a0869ea5a711aba7566787d4a599654907ecc070c"
EXPORT_SHA256 = "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235"
REFERENCE_SHA256 = "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334"
TRANSFORMERS = "5.8.1"
TOKENIZERS = "0.22.2"
GGUF = "0.19.0"
SAFETENSORS = "0.8.0"

# Pinned upstream source. The hosted run recomputes this digest from the
# installed module and aborts when it differs: the replay is only the audited
# implementation when the bytes are the audited bytes.
SOURCE_MODULE = "transformers/models/qwen3_5/modeling_qwen3_5.py"
SOURCE_MODULE_SHA256 = "641da34484f6b9769e28615c680cc4e3202afbf5bdbf1da7d56d5a3d678918c7"
SOURCE_CLASS = "Qwen3_5VisionModel"
SOURCE_METHOD = "fast_pos_embed_interpolate"
METHOD_TEXT_PINS = (
    "torch.linspace(0, self.num_grid_per_side - 1, h)",
    "dtype=self.pos_embed.weight.dtype",
    "pos_embeds = self.pos_embed(idx_tensor)",
    ".permute(0, 1, 3, 2, 4, 5)",
)

# Retained hosted evidence (vision-prefill kernel v3 artefact set).
VISION_PREFILL_RUNNER_SHA256 = "18bfc3e880032e00e7ceadd6f7eea5e2cc2ef3cce28d6723561e2abd155e15d2"
VISION_PREFILL_REPORT_SHA256 = "eae8d973500b2e3890e3dd408f051d67edc26ffa7b5de71f9b3429949470816d"
NATIVE_TRACE_NAME = "prefill.vision.position.f32"
NATIVE_POSITION_SHA256 = "7e257d92b4acfd424684fe57c716e8e077661962cf68c9f91ddb52656af9ce1c"
NATIVE_POSITION_BYTES = 6094848
NATIVE_POSITION_FNV1A = "5c8c4c38aa47f346"
NATIVE_STAGE_PINS = {
    "n": 1523712,
    "shape": [768, 1984],
    "max_abs": 0.0188269615,
    "mean_abs": 0.00014037863,
    "rms": 0.000310289237,
    "cos_min": 0.999989152,
    "cos_mean": 0.999996603,
    "cos_global": 0.999995828,
    "native_norm": 132.798233,
    "reference_norm": 132.780319,
    "input_fnv1a": NATIVE_POSITION_FNV1A,
}
NATIVE_STAGE_GRID_TEXT = "grid=1,62,32 tokens=1984"

# Geometry of the qianziwen page as frozen by the hosted blueprint run.
GRID = [1, 62, 32]
TOKENS = 1984
WIDTH = 768
POSITION_ROWS = 2304
SPATIAL_MERGE = 2
BREAKING_REPORT_TOLERANCE = 5e-7
GGUF_POSITION = "v.pos_embed.weight"
GGUF_EXPORT_POSITION_SHA256 = "c8ef73745e573cfb3a36aebfeefc1ca38a2afeb2ca60b2f0488b6f861a429ae0"
CHECKPOINT_POSITION = "model.visual.pos_embed.weight"
REFERENCE_POSITION = "prefill.vision.position"

# Unchanged native gate. Never edit these thresholds in this lane.
STRICT_MAX_ABS = 0.002
STRICT_COS_MIN = 0.99999
# Diagnostic budget only: it is neither a parity gate nor evidence of exact
# reproduction or of a causal explanation for the position discrepancy.
BF16_EXACT_MIN = 0.99
BF16_RMS_MAX = 1e-4

OUT = Path("/kaggle/working/xiaomi-position-replay")
SCRATCH = Path("/kaggle/temp/xiaomi-position-replay")
REPORT = {
    "script_version": SCRIPT_VERSION,
    "run_id": RUN_ID,
    "scope": "hosted CPU-only attribution of the prefill.vision.position stage only; "
             "not a native parity run, no native graph construction, no decoder, "
             "no decoded text, no quantization, no speed measurement and no quality claim",
    "strict_gate": {"max_abs": STRICT_MAX_ABS, "row_cos_min": STRICT_COS_MIN, "unchanged": True,
                    "relaxed_for_dtype": False},
    "bf16_attribution_budget": {
        "exact_fraction_min": BF16_EXACT_MIN, "rms_max": BF16_RMS_MAX,
        "purpose": "diagnostic only; neither exact reproduction nor a parity pass"},
    "geometry": {"grid_thw": GRID, "tokens": TOKENS, "width": WIDTH,
                 "position_dimensions": [TOKENS, WIDTH], "position_table_rows": POSITION_ROWS,
                 "spatial_merge_size": SPATIAL_MERGE},
    "claims": {"decoded_parity_claimed": False, "decoded_quality_claimed": False,
               "speed_claimed": False, "native_graph_claimed": False,
               "native_strict_gate_waived": False},
    "not_tested": ["native graph construction or execution", "native CPU/CUDA stage reproduction",
                   "full vision tower", "decoder", "decoded OCR text", "quantization", "speed"],
    "status": "running",
}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def save(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["phase"] = phase
    REPORT["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=2, allow_nan=False) + "\n")
    print("===", phase, flush=True)


def bf16_round(values):
    """Round F32 to BF16 (round-to-nearest-even) held in F32 storage."""
    import numpy as np

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def bf16_representable_fraction(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    return float(np.mean(bf16_round(values) == values))


def metrics(actual, reference):
    """Pairwise comparison of equal-shaped 2-D arrays (rows are compared by cosine)."""
    import numpy as np

    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape or a.ndim != 2:
        raise ValueError(f"comparison shapes differ: {a.shape} vs {r.shape}")
    if not (np.isfinite(a).all() and np.isfinite(r).all()):
        raise ValueError("comparison inputs contain non-finite values")
    delta = a - r
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1)
    if (denominator <= 0).any():
        raise ValueError("zero-magnitude row in comparison")
    row_cos = (a * r).sum(axis=1) / denominator
    flat = np.abs(delta).reshape(-1)
    worst = int(np.argmax(flat))
    row, column = divmod(worst, a.shape[1])
    exact = delta == 0
    return {
        "shape": list(a.shape),
        "max_abs": float(np.abs(delta).max()),
        "mean_abs": float(np.abs(delta).mean()),
        "rms": float(np.sqrt(np.square(delta).mean())),
        "row_cos_min": float(row_cos.min()),
        "row_cos_mean": float(row_cos.mean()),
        "row_cos_max": float(row_cos.max()),
        "cos_global": float((a * r).sum() / float(np.linalg.norm(a) * np.linalg.norm(r))),
        "exact_fraction": float(exact.mean()),
        "mismatch_count": int((~exact).sum()),
        "elements": int(a.size),
        "worst_index": [row, column],
        "worst_abs": float(flat[worst]),
        "worst_actual": float(a[row, column]),
        "worst_reference": float(r[row, column]),
        "actual_norm": float(np.linalg.norm(a)),
        "reference_norm": float(np.linalg.norm(r)),
    }


def strict_pass(metric):
    return metric["max_abs"] <= STRICT_MAX_ABS and metric["row_cos_min"] >= STRICT_COS_MIN


def bf16_budget_pass(metric):
    return (metric["exact_fraction"] >= BF16_EXACT_MIN and metric["rms"] <= BF16_RMS_MAX
            and metric["row_cos_min"] >= STRICT_COS_MIN)


def position_tokens(grid):
    """Token count implied by grid_thw rows; rejects malformed grids."""
    rows = [list(row) for row in grid]
    total = 0
    for row in rows:
        if len(row) != 3 or any(int(value) <= 0 for value in row):
            raise ValueError(f"grid_thw row is not three positive integers: {row}")
        total += int(row[0]) * int(row[1]) * int(row[2])
    if not rows:
        raise ValueError("grid_thw is empty")
    return total


def num_grid_per_side(rows):
    import math as _math

    root = int(round(_math.sqrt(int(rows))))
    if root * root != int(rows):
        raise ValueError(f"{rows} is not a perfect square; num_grid_per_side is undefined")
    return root


def native_trace_matrix(raw, tokens, width):
    """Read a native diff dump with contiguous features for each token."""
    import numpy as np

    expected = int(tokens) * int(width) * 4
    if len(raw) != expected:
        raise ValueError(f"native trace bytes {len(raw)} != expected {expected}")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("native trace contains non-finite values")
    return np.ascontiguousarray(values.reshape(int(tokens), int(width)))


def fnv1a(raw):
    value = 0xCBF29CE484222325
    for byte in raw:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def attribution_conditions(**measured):
    """Assemble the fail-closed verdict from measurements; no defaults, no assumptions."""
    required = ("source_module_pinned", "method_is_pinned", "geometry_ok", "weights_widening_equal",
                "reference_bf16_representable", "bf16_repeat_identical", "bf16_vs_reference",
                "f32_vs_native", "native_vs_reference", "report_reproduction_ok",
                "negative_control_rejected", "report_strict_failure_reproduced")
    missing = [key for key in required if key not in measured]
    if missing:
        raise ValueError("missing measurement: " + ", ".join(missing))
    unexpected = sorted(set(measured) - set(required))
    if unexpected:
        raise ValueError("unexpected measurement: " + ", ".join(unexpected))
    for key in ("bf16_vs_reference", "f32_vs_native", "native_vs_reference"):
        for field in ("max_abs", "rms", "row_cos_min", "exact_fraction"):
            if field not in measured[key]:
                raise ValueError(f"missing measurement: {key}.{field}")
            value = measured[key][field]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"non-finite measurement: {key}.{field}={value!r}")
    checks = {
        "pinned_module_source_verified": bool(measured["source_module_pinned"] and measured["method_is_pinned"]),
        "geometry_matches_grid": bool(measured["geometry_ok"]),
        "checkpoint_bf16_widening_matches_export": bool(measured["weights_widening_equal"]),
        "frozen_reference_is_bf16_representable": bool(measured["reference_bf16_representable"]),
        "bf16_module_repeat_is_bit_identical": bool(measured["bf16_repeat_identical"]),
        "bf16_cpu_reproduces_frozen_reference": measured["bf16_vs_reference"]["exact_fraction"] == 1.0,
        "f32_module_matches_native_trace": strict_pass(measured["f32_vs_native"]),
        "native_trace_reproduces_retained_report": bool(measured["report_reproduction_ok"]),
        "retained_report_strict_failure_reproduced": bool(measured["report_strict_failure_reproduced"]),
        "negative_control_fails_both_gates": bool(measured["negative_control_rejected"]),
    }
    return {
        "conditions": checks,
        "confirmed": all(checks.values()),
        "cpu_gpu_bf16_ambiguity": not checks["bf16_cpu_reproduces_frozen_reference"],
        "native_deviation_beyond_precision": not checks["f32_module_matches_native_trace"],
    }


def attribution_statement(bf16_metric, f32_metric, native_metric, verdict, reproduction_ok):
    """Human-readable statement built only from measured values."""
    return (
        "measured on CPU: BF16 pinned-module output vs frozen GPU reference "
        f"max_abs={bf16_metric['max_abs']:.9g} rms={bf16_metric['rms']:.9g} "
        f"exact_fraction={bf16_metric['exact_fraction']:.6f}; "
        "F32 pinned-module output vs mounted native F32 trace "
        f"max_abs={f32_metric['max_abs']:.9g} cos_min={f32_metric['row_cos_min']:.9g}; "
        + ("native trace vs frozen reference reproduces the retained report "
           if reproduction_ok else
           "native trace vs frozen reference does not reproduce the retained report ")
        + f"(max_abs={native_metric['max_abs']:.9g}, cos_min={native_metric['row_cos_min']:.9g})"
        + ("; BF16 CPU and the frozen GPU activation are not bit-identical, "
           "so device/backend differences remain possible"
           if verdict["cpu_gpu_bf16_ambiguity"] else "")
        + ("; the pinned F32 module does not match the native F32 trace within the "
           "unchanged strict gate" if verdict["native_deviation_beyond_precision"] else "")
        + ".")


def find_unique_input(name, expected_sha256, expected_size=None):
    matches = []
    for path in Path("/kaggle/input").rglob(name):
        if path.is_file() and digest(path) == expected_sha256:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"{name}: expected exactly one mounted input with the pinned sha256, "
                         f"found {len(matches)}")
    if expected_size is not None and matches[0].stat().st_size != expected_size:
        raise ValueError(f"{name}: mounted size differs from the pin")
    return matches[0]


def tensor_from_gguf(reader, name, shape):
    import numpy as np

    found = [tensor for tensor in reader.tensors if tensor.name == name]
    if len(found) != 1:
        raise ValueError(f"gguf tensor {name}: expected exactly one, found {len(found)}")
    values = np.ascontiguousarray(np.asarray(found[0].data, dtype=np.float32).reshape(shape))
    if values.size != int(np.prod(shape)):
        raise ValueError(f"gguf tensor {name}: element count differs")
    return values


def close(actual, expected, tolerance=BREAKING_REPORT_TOLERANCE):
    return abs(float(actual) - float(expected)) <= tolerance * max(1.0, abs(float(expected)))


def compare_outputs(bf16_values, bf16_repeat_values, f32_values, negative_values, saved, native):
    """All pairwise comparisons plus the retained-report reproduction check.

    ``saved`` is the frozen reference activation (tokens, width) and ``native``
    the mounted native trace in the same orientation; crispembed_diff groups
    rows by the contiguous 768-wide feature axis, which is that orientation.
    """
    comparisons = {
        "bf16_vs_frozen_reference": metrics(bf16_values, saved),
        "bf16_repeat_vs_first": metrics(bf16_repeat_values, bf16_values),
        "f32_vs_native_trace": metrics(f32_values, native),
        "f32_vs_frozen_reference": metrics(f32_values, saved),
        "native_vs_frozen_reference": metrics(native, saved),
        "negative_shifted_rows_vs_frozen_reference": metrics(negative_values, saved),
    }
    for metric in comparisons.values():
        metric["strict_pass"] = strict_pass(metric)
        metric["bf16_budget_pass"] = bf16_budget_pass(metric)
    reproduced = comparisons["native_vs_frozen_reference"]
    fields = {"max_abs": "max_abs", "mean_abs": "mean_abs", "rms": "rms",
              "cos_min": "row_cos_min", "cos_mean": "row_cos_mean",
              "cos_global": "cos_global", "native_norm": "actual_norm",
              "reference_norm": "reference_norm"}
    checks = {key: close(reproduced[field], NATIVE_STAGE_PINS[key]) for key, field in fields.items()}
    reproduction = {
        "orientation": "1984 token rows of 768 contiguous features, as crispembed_diff groups them",
        "tolerance_relative": BREAKING_REPORT_TOLERANCE,
        "checks": checks,
        "ok": all(checks.values()),
        "computed": {key: reproduced[field] for key, field in fields.items()},
        "pinned": NATIVE_STAGE_PINS,
    }
    return {"comparisons": comparisons, "reproduction": reproduction, "reproduced": reproduced}


def verify_report(report, native_path):
    """Authenticate the mounted v3 artefact set against the retained report."""
    stages = [row for row in report.get("stages", []) if row.get("name") == "prefill.vision.position"]
    if len(stages) != 1:
        raise ValueError("mounted report does not carry exactly one prefill.vision.position stage")
    stage = stages[0]
    for key, pinned in NATIVE_STAGE_PINS.items():
        actual = stage.get(key)
        if actual is None:
            raise ValueError(f"mounted report stage {key} is missing")
        if key == "shape":
            if list(actual or []) != pinned:
                raise ValueError(f"mounted report stage shape {actual} != {pinned}")
        elif key == "input_fnv1a":
            if actual != pinned:
                raise ValueError(f"mounted report stage {key} {actual} != {pinned}")
        elif isinstance(pinned, int):
            if int(actual) != pinned:
                raise ValueError(f"mounted report stage {key} {actual} != {pinned}")
        elif not close(actual, pinned):
            raise ValueError(f"mounted report stage {key} {actual} != {pinned}")
    trace = report.get("native_trace_sha256", {}).get(NATIVE_TRACE_NAME)
    if not trace or trace.get("sha256") != NATIVE_POSITION_SHA256 or trace.get("bytes") != NATIVE_POSITION_BYTES:
        raise ValueError("mounted report native trace entry differs from the pin")
    if stage.get("pass") is not False or stage.get("strict_absolute_pass") is not False:
        raise ValueError("mounted report claims a strict pass for prefill.vision.position; "
                         "the retained evidence records a failure and this lane does not waive it")
    closure = report.get("source_closure", {})
    if closure.get("staged_runner_sha256") != VISION_PREFILL_RUNNER_SHA256:
        raise ValueError("mounted report was produced by a different vision-prefill runner")
    backend = report.get("backend_evidence", "")
    if NATIVE_STAGE_GRID_TEXT not in backend or "CPU" not in backend:
        raise ValueError(f"mounted report backend evidence differs from the pinned CPU grid: {backend!r}")
    if native_path.parent.name != "native-trace":
        raise ValueError("mounted native trace is not from the vision-prefill native-trace directory")
    return {"stage": stage, "report_sha256": None}


def main():
    save("admission")
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("this CPU replay must run on Kaggle")
    free = shutil.disk_usage("/kaggle/working").free
    REPORT["initial_free_bytes"] = free
    if free < 8 * 1024 ** 3:
        raise RuntimeError("insufficient hosted workspace")
    import numpy as np
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("GPU is visible; this CPU-only lane refuses to run its CPU comparison here")
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    except (RuntimeError, ValueError):
        pass
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(False)
    REPORT["device"] = "CPU"
    REPORT["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                       "threads": torch.get_num_threads(), "device": "cpu"}
    save("install")
    install = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                              f"transformers=={TRANSFORMERS}", f"tokenizers=={TOKENIZERS}",
                              f"gguf=={GGUF}", f"safetensors=={SAFETENSORS}"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / "install.log").write_text(install.stdout)
    print(install.stdout[-4000:], flush=True)
    if install.returncode != 0:
        raise RuntimeError(f"pinned dependency install failed with exit {install.returncode}")

    import gguf
    import importlib.metadata
    import inspect
    import transformers
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    REPORT["versions"] = {"transformers": transformers.__version__, "torch": torch.__version__,
                          "numpy": np.__version__,
                          **{name: importlib.metadata.version(name) for name in
                             ("gguf", "safetensors", "huggingface-hub")}}
    if transformers.__version__ != TRANSFORMERS:
        raise ValueError("pinned transformers version required")

    save("source-closure")
    module_file = Path(inspect.getsourcefile(Qwen3_5VisionModel))
    module_sha = digest(module_file)
    method = Qwen3_5VisionModel.fast_pos_embed_interpolate
    method_source = inspect.getsource(method)
    source_pins = {pin: (pin in method_source) for pin in METHOD_TEXT_PINS}
    class_declares_method = SOURCE_METHOD in vars(Qwen3_5VisionModel)
    REPORT["original_module"] = {
        "module_file": str(module_file),
        "module_file_sha256": module_sha,
        "module_sha_matches_pin": module_sha == SOURCE_MODULE_SHA256,
        "class": f"{Qwen3_5VisionModel.__module__}.{SOURCE_CLASS}",
        "method": SOURCE_METHOD,
        "method_qualname": method.__qualname__,
        "class_declares_method": bool(class_declares_method),
        "method_source_sha256": digest_bytes(method_source.encode()),
        "method_source_pins": source_pins,
        "method_source": method_source,
    }
    if module_sha != SOURCE_MODULE_SHA256:
        raise RuntimeError("installed modeling_qwen3_5.py differs from the pinned audited module digest")
    if method.__qualname__ != f"{SOURCE_CLASS}.{SOURCE_METHOD}" or not all(source_pins.values()):
        raise RuntimeError("pinned fast_pos_embed_interpolate no longer matches the audited implementation")
    REPORT["source_closure"] = {
        "script_version": SCRIPT_VERSION,
        "base_commit": BASE_COMMIT,
        "runner_sha256": digest(__file__),
        "pinned": {
            "transformers": TRANSFORMERS, "tokenizers": TOKENIZERS, "gguf": GGUF,
            "safetensors": SAFETENSORS, "model_repo": MODEL_REPO, "revision": REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256, "config_sha256": CONFIG_SHA256,
            "export_sha256": EXPORT_SHA256, "reference_sha256": REFERENCE_SHA256,
            "source_module": SOURCE_MODULE, "source_module_sha256": SOURCE_MODULE_SHA256,
            "vision_prefill_runner_sha256": VISION_PREFILL_RUNNER_SHA256,
            "vision_prefill_report_sha256": VISION_PREFILL_REPORT_SHA256,
            "native_position_sha256": NATIVE_POSITION_SHA256,
            "native_position_bytes": NATIVE_POSITION_BYTES,
            "grid_thw": GRID, "tokens": TOKENS, "width": WIDTH,
        },
    }
    (OUT / "source-closure.json").write_text(json.dumps(REPORT["source_closure"], indent=2) + "\n")
    module_ok = True
    method_ok = True

    save("inputs")
    reference_path = find_unique_input("qianziwen-prefill.gguf", REFERENCE_SHA256)
    export_path = find_unique_input("xiaomi-ocr-0-f32-experimental.gguf", EXPORT_SHA256)
    native_path = find_unique_input(NATIVE_TRACE_NAME, NATIVE_POSITION_SHA256, NATIVE_POSITION_BYTES)
    report_path = native_path.parent.parent / "report.json"
    if not report_path.is_file():
        raise ValueError(f"mounted vision-prefill report missing next to the native trace: {report_path}")
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    verify_report(report, native_path)
    REPORT["input_paths"] = {"reference": str(reference_path), "export": str(export_path),
                             "native_trace": str(native_path), "vision_prefill_report": str(report_path)}
    REPORT["vision_prefill_report"] = {
        "byte_sha256": digest_bytes(report_bytes),
        "byte_sha256_matches_retained_copy": digest_bytes(report_bytes) == VISION_PREFILL_REPORT_SHA256,
        "staged_runner_sha256": report["source_closure"]["staged_runner_sha256"],
        "backend_evidence": report.get("backend_evidence"),
        "stage_pin_verified": True,
    }
    raw_native = native_path.read_bytes()
    native = native_trace_matrix(raw_native, TOKENS, WIDTH)
    REPORT["native_trace"] = {
        "name": NATIVE_TRACE_NAME, "bytes": len(raw_native),
        "sha256": digest_bytes(raw_native), "fnv1a": fnv1a(raw_native),
        "fnv1a_matches_report": fnv1a(raw_native) == NATIVE_POSITION_FNV1A,
        "gguf_extents": [WIDTH, TOKENS], "shape_token_major": [TOKENS, WIDTH],
        "absmax": float(np.abs(native).max()), "norm": float(np.linalg.norm(native.astype(np.float64))),
    }

    reference_reader = gguf.GGUFReader(str(reference_path))
    saved = tensor_from_gguf(reference_reader, REFERENCE_POSITION, (TOKENS, WIDTH))
    if not np.isfinite(saved).all():
        raise ValueError("frozen reference position activation contains non-finite values")
    REPORT["reference_activation"] = {
        "name": REFERENCE_POSITION, "shape": [TOKENS, WIDTH],
        "sha256": digest_bytes(saved.tobytes()),
        "absmax": float(np.abs(saved).max()), "norm": float(np.linalg.norm(saved.astype(np.float64))),
        "bf16_representable_fraction": bf16_representable_fraction(saved),
    }
    geometry_ok = (position_tokens([GRID]) == TOKENS and saved.shape == (TOKENS, WIDTH)
                   and native.shape == (TOKENS, WIDTH)
                   and NATIVE_STAGE_PINS["n"] == TOKENS * WIDTH)
    REPORT["geometry_checks"] = {
        "grid_thw": GRID, "grid_product": position_tokens([GRID]),
        "reference_shape": list(saved.shape), "native_shape": list(native.shape),
        "elements": int(saved.size), "ok": bool(geometry_ok),
    }
    if not geometry_ok:
        raise ValueError("position geometry does not match the pinned 1984x768 grid")
    save("checkpoint")
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(MODEL_REPO, revision=REVISION, token=False,
                                   allow_patterns=["*.safetensors", "config.json"],
                                   local_dir=str(SCRATCH / "checkpoint")))
    candidates = [path for path in local.glob("*.safetensors")]
    if len(candidates) != 1:
        raise ValueError(f"checkpoint shards: expected exactly one, found {len(candidates)}")
    checkpoint = candidates[0]
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("downloaded checkpoint bytes differ from the pinned revision digest")
    config_path = local / "config.json"
    if digest(config_path) != CONFIG_SHA256:
        raise ValueError("downloaded config.json differs from the pinned blueprint digest")
    REPORT["checkpoint"] = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size,
                            "sha256": CHECKPOINT_SHA256, "config_sha256": CONFIG_SHA256,
                            "anonymous": True, "token": False}
    with safe_open(str(checkpoint), framework="pt") as handle:
        position_bf16 = handle.get_tensor(CHECKPOINT_POSITION)
    if tuple(position_bf16.shape) != (POSITION_ROWS, WIDTH):
        raise ValueError(f"checkpoint position table shape differs: {tuple(position_bf16.shape)}")
    if position_bf16.dtype is not torch.bfloat16:
        raise ValueError("checkpoint position table is not BF16")
    REPORT["checkpoint_position"] = {"name": CHECKPOINT_POSITION, "shape": list(position_bf16.shape),
                                     "dtype": str(position_bf16.dtype)}

    save("weights-widening")
    export_reader = gguf.GGUFReader(str(export_path))
    position_export = tensor_from_gguf(export_reader, GGUF_POSITION, (POSITION_ROWS, WIDTH))
    widened = position_bf16.float()
    weight_equal = bool(torch.equal(widened, torch.from_numpy(position_export)))
    export_digest = digest_bytes(position_export.tobytes())
    REPORT["weights_widening"] = {
        "weight_bitwise_equal": weight_equal,
        "checkpoint_widened_f32_sha256": digest_bytes(widened.numpy().tobytes()),
        "export_tensor_sha256": export_digest,
        "export_tensor_sha256_matches_manifest": export_digest == GGUF_EXPORT_POSITION_SHA256,
        "export_tensor_absmax": float(np.abs(position_export).max()),
    }
    if not weight_equal:
        raise ValueError("exported F32 position table is not the exact BF16 widening of the checkpoint")

    save("module")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(local))
    vision_config = config.vision_config
    patch_size = vision_config.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    observed = {"hidden_size": int(vision_config.hidden_size),
                "num_position_embeddings": int(vision_config.num_position_embeddings),
                "spatial_merge_size": int(vision_config.spatial_merge_size),
                "patch_size": int(patch_size),
                "temporal_patch_size": int(vision_config.temporal_patch_size),
                "in_channels": int(vision_config.in_channels),
                "depth": int(vision_config.depth)}
    REPORT["vision_config"] = observed
    if (observed["hidden_size"], observed["num_position_embeddings"], observed["spatial_merge_size"],
            observed["patch_size"], observed["temporal_patch_size"], observed["in_channels"]) != (
            WIDTH, POSITION_ROWS, SPATIAL_MERGE, 16, 2, 3):
        raise ValueError("vision config geometry differs from the audited page prefill")
    if num_grid_per_side(observed["num_position_embeddings"]) != 48:
        raise ValueError("num_grid_per_side is not the audited 48")

    # Minimal real instance: the pinned class with zero vision blocks. Only
    # pos_embed (a real nn.Embedding), num_grid_per_side and config are read by
    # fast_pos_embed_interpolate; the unused patch/merger parameters keep their
    # initialisation and are recorded, never executed.
    vision_config.depth = 0
    model_bf16 = Qwen3_5VisionModel(vision_config).to(torch.bfloat16).eval()
    model_f32 = Qwen3_5VisionModel(vision_config).to(torch.float32).eval()
    for model in (model_bf16, model_f32):
        if len(model.blocks) != 0:
            raise ValueError("minimal vision instance unexpectedly carries blocks")
    state_bf16 = {"pos_embed.weight": position_bf16}
    state_f32 = {"pos_embed.weight": widened}
    missing_bf16, unexpected_bf16 = model_bf16.load_state_dict(state_bf16, strict=False)
    missing_f32, unexpected_f32 = model_f32.load_state_dict(state_f32, strict=False)
    if unexpected_bf16 or unexpected_f32:
        raise ValueError(f"unexpected state keys: {list(unexpected_bf16)} / {list(unexpected_f32)}")
    if not (torch.equal(model_bf16.pos_embed.weight, position_bf16)
            and torch.equal(model_f32.pos_embed.weight, widened)):
        raise ValueError("position table was not accepted bit-exactly by the pinned class")
    method_identity = (type(model_bf16).fast_pos_embed_interpolate is Qwen3_5VisionModel.fast_pos_embed_interpolate
                       and type(model_f32).fast_pos_embed_interpolate is Qwen3_5VisionModel.fast_pos_embed_interpolate
                       and "fast_pos_embed_interpolate" not in vars(model_bf16)
                       and "fast_pos_embed_interpolate" not in vars(model_f32))
    method_ok = bool(method_identity)
    if not method_ok:
        raise ValueError("fast_pos_embed_interpolate is not the unmodified pinned class method")
    REPORT["module_instance"] = {
        "class": f"{type(model_bf16).__module__}.{type(model_bf16).__name__}",
        "depth_forced": 0, "checkpoint_depth": observed["depth"],
        "pos_embed_is_nn_embedding": isinstance(model_bf16.pos_embed, torch.nn.Embedding),
        "num_grid_per_side": int(model_bf16.num_grid_per_side),
        "spatial_merge_size": int(model_bf16.config.spatial_merge_size),
        "bf16_weight_dtype": str(model_bf16.pos_embed.weight.dtype),
        "f32_weight_dtype": str(model_f32.pos_embed.weight.dtype),
        "missing_unused_keys": sorted(missing_bf16),
        "unexpected_keys": sorted(unexpected_bf16),
        "method_is_pinned_class_method": method_ok,
        "random_but_unused_parameters": sorted(
            name for name, _ in model_bf16.named_parameters() if not name.startswith("pos_embed.")),
    }

    save("execute")
    grid = torch.tensor([GRID], dtype=torch.long)
    with torch.inference_mode():
        bf16_first = model_bf16.fast_pos_embed_interpolate(grid)
        bf16_second = model_bf16.fast_pos_embed_interpolate(grid)
        f32_out = model_f32.fast_pos_embed_interpolate(grid)
    for name, value in (("bf16", bf16_first), ("bf16_repeat", bf16_second), ("f32", f32_out)):
        if tuple(value.shape) != (TOKENS, WIDTH) or not torch.isfinite(value).all():
            raise ValueError(f"{name} module output shape/non-finite failure: {tuple(value.shape)}")
    repeat_identical = bool(torch.equal(bf16_first, bf16_second))
    def numpy_of(tensor):
        return np.ascontiguousarray(tensor.detach().float().cpu().numpy())

    bf16_values = numpy_of(bf16_first)
    f32_values = numpy_of(f32_out)
    negative_values = np.ascontiguousarray(np.roll(bf16_values, shift=TOKENS // 2 + 1, axis=0))
    REPORT["outputs"] = {
        "bf16": {"sha256": digest_bytes(bf16_values.tobytes()),
                 "bf16_representable_fraction": bf16_representable_fraction(bf16_values),
                 "absmax": float(np.abs(bf16_values).max())},
        "f32": {"sha256": digest_bytes(f32_values.tobytes()),
                "bf16_representable_fraction": bf16_representable_fraction(f32_values),
                "absmax": float(np.abs(f32_values).max())},
        "bf16_repeat_bit_identical": repeat_identical,
        "negative_control_shift_rows": TOKENS // 2 + 1,
    }
    for name, values in (("bf16", bf16_values), ("f32", f32_values), ("negative", negative_values)):
        np.ascontiguousarray(values, dtype="<f4").tofile(str(OUT / (name + ".position.f32")))

    save("compare")
    compared = compare_outputs(bf16_values, numpy_of(bf16_second), f32_values, negative_values,
                               saved, native)
    comparisons = compared["comparisons"]
    reproduced = compared["reproduced"]
    REPORT["report_reproduction"] = compared["reproduction"]
    negative = comparisons["negative_shifted_rows_vs_frozen_reference"]
    verdict = attribution_conditions(
        source_module_pinned=module_ok and REPORT["original_module"]["module_sha_matches_pin"],
        method_is_pinned=method_ok,
        geometry_ok=geometry_ok,
        weights_widening_equal=weight_equal,
        reference_bf16_representable=REPORT["reference_activation"]["bf16_representable_fraction"] == 1.0,
        bf16_repeat_identical=repeat_identical,
        bf16_vs_reference=comparisons["bf16_vs_frozen_reference"],
        f32_vs_native=comparisons["f32_vs_native_trace"],
        native_vs_reference=reproduced,
        report_reproduction_ok=REPORT["report_reproduction"]["ok"],
        negative_control_rejected=(not strict_pass(negative)) and (not bf16_budget_pass(negative)),
        report_strict_failure_reproduced=not strict_pass(reproduced),
    )
    (OUT / "comparisons.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n")
    REPORT["comparisons_sha256"] = digest(OUT / "comparisons.json")
    REPORT["key_comparisons"] = {
        "bf16_cpu_vs_frozen_gpu_reference": "bf16_vs_frozen_reference",
        "f32_module_cpu_vs_native_f32_trace": "f32_vs_native_trace",
        "native_trace_vs_frozen_reference": "native_vs_frozen_reference",
        "negative_control": "negative_shifted_rows_vs_frozen_reference",
    }
    REPORT["conditions"] = verdict["conditions"]
    REPORT["attribute"] = {
        "bf16": {key: comparisons["bf16_vs_frozen_reference"][key] for key in
                 ("max_abs", "mean_abs", "rms", "row_cos_min", "row_cos_max", "exact_fraction",
                  "mismatch_count", "elements", "worst_index", "worst_abs", "worst_actual",
                  "worst_reference")},
        "f32_vs_native": {key: comparisons["f32_vs_native_trace"][key] for key in
                          ("max_abs", "mean_abs", "rms", "row_cos_min", "row_cos_max",
                           "exact_fraction", "mismatch_count", "elements", "worst_index",
                           "worst_abs", "worst_actual", "worst_reference")},
        "native_vs_reference": {key: reproduced[key] for key in
                                ("max_abs", "mean_abs", "rms", "row_cos_min", "cos_global",
                                 "exact_fraction", "mismatch_count", "elements")},
        "negative_control": {key: negative[key] for key in ("max_abs", "rms", "row_cos_min",
                                                            "exact_fraction")},
    }
    REPORT["attribution"] = {
        "confirmed": bool(verdict["confirmed"]),
        "cpu_gpu_bf16_ambiguity": bool(verdict["cpu_gpu_bf16_ambiguity"]),
        "native_deviation_beyond_precision": bool(verdict["native_deviation_beyond_precision"]),
        "position_failure_consistent_with_reference_precision_regime": bool(
            verdict["confirmed"] and not verdict["native_deviation_beyond_precision"]
            and not verdict["cpu_gpu_bf16_ambiguity"]),
        "statement": attribution_statement(comparisons["bf16_vs_frozen_reference"],
                                           comparisons["f32_vs_native_trace"], reproduced,
                                           verdict, REPORT["report_reproduction"]["ok"]),
        "strict_gate_waived": False,
        "decoded_parity_claimed": False,
    }
    if verdict["confirmed"]:
        REPORT["status"] = ("completed: pinned BF16 CPU output is bit-identical to the frozen reference; "
                            "F32 output matches the native trace within the unchanged strict gate. "
                            "This is consistent with a precision effect, not proof of sole causation; "
                            "no native graph, decode, quality or speed claim")
        save("complete")
        return 0
    REPORT["status"] = ("completed with an unconfirmed position attribution; see conditions and "
                        "attribution flags; the strict gate is unchanged")
    REPORT["error_type"] = "AttributionNotConfirmed"
    REPORT["error"] = json.dumps(verdict["conditions"], sort_keys=True)
    save("attribution-not-confirmed")
    return 1


if __name__ == "__main__":
    START = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    REPORT["started_utc"] = START
    try:
        CODE = main()
    except BaseException as error:
        REPORT["status"] = "failed"
        REPORT["error_type"] = type(error).__name__
        REPORT["error"] = str(error)[:1500]
        REPORT["traceback"] = traceback.format_exc()[-6000:]
        save(REPORT.get("phase", "startup"))
        CODE = 2
    REPORT["exit_code"] = CODE
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save(REPORT.get("phase", "startup"))
    print("POSITION_REPLAY_RESULT", json.dumps(
        {"status": REPORT["status"], "confirmed": REPORT.get("attribution", {}).get("confirmed"),
         "ambiguity": REPORT.get("attribution", {}).get("cpu_gpu_bf16_ambiguity"),
         "exit_code": CODE}, sort_keys=True), flush=True)
    raise SystemExit(CODE)
