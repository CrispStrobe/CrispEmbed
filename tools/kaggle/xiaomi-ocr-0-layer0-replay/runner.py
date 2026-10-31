#!/usr/bin/env python3
"""Hosted layer-0 attribution replay for the Xiaomi-OCR-0 vision prefill.

Question: the retained hosted vision run (kernel ``vision-prefill`` v3) records
``prefill.vision.layer.0`` failing the unchanged strict gate against the frozen
reference (max_abs 0.0336941481, row cos_min 0.999947548). How much of that gap
is already present at the block input, and how much is produced by the block?

This replay executes the *actual, unmodified* pinned
``transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionBlock``
(transformers 5.8.1) with the checkpoint's own BF16 block tensors, on the
archived original input and on the mounted native input, and decomposes

    D = N - A          native output minus archive output (the retained failure)
    B = N - R3         native output minus CPU-F32 original module on native input
    I = R3 - R2        input/rotary path: CPU-F32 module on native vs archive input
    G = R2 - A         block+device path: CPU-F32 module vs archive output

with the exact vector identity D = B + I + G. The cross terms, energies and
cancellation are reported, because a pass or failure of any single term does
not identify a single cause: B passing does not establish that the block has no
defect, and a failing B does not prove all error is inherited. The verdict is
therefore explicitly a *comparison consistency* statement, never a causal one.

No decoder, no decoded text, no image frontend, no quantization and no speed
claim are in scope.

It measures four named runs plus separately captured controls:
  R1  checkpoint BF16 block on CUDA/T4 with SDPA, archive input + archive rotary
      (baseline archive closure: bitwise exact is the only acceptable closure)
  R2  CPU eager F32 block on the archive input
  R3  CPU eager F32 block on the native input
  R4  CPU eager BF16 block on the native input (optional control)

Eager-vs-SDPA and CPU-vs-GPU BF16 rounding are captured separately, never
folded into the strict gate. A permuted-row negative control must fail both the
magnitude and the row-cosine gate. Pins, widening and identity checks fail
closed.

Only the standard library is imported at module scope so the repository test can
import the staged script without torch/numpy on the host. No licensed upstream
code is copied: the original installed module is imported and called.
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

SCRIPT_VERSION = "xiaomi-layer0-replay-v1"
RUN_ID = "xiaomi-layer0-replay-v1"

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
SOURCE_CLASS = "Qwen3_5VisionBlock"
CLASS_TEXT_PINS = (
    "hidden_states = hidden_states + self.attn(",
    "self.norm1(hidden_states)",
    "hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))",
    "position_embeddings=position_embeddings",
)
ATTENTION_TEXT_PINS = (
    "self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)",
    "apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)",
    "self.config._attn_implementation",
    "attn_output.reshape(seq_length, -1).contiguous()",
)

# Retained hosted evidence (kernel vision-prefill v3 artefact set).
VISION_PREFILL_RUNNER_SHA256 = "18bfc3e880032e00e7ceadd6f7eea5e2cc2ef3cce28d6723561e2abd155e15d2"
VISION_PREFILL_REPORT_SHA256 = "eae8d973500b2e3890e3dd408f051d67edc26ffa7b5de71f9b3429949470816d"
NATIVE_EMBEDDED_TRACE = "prefill.vision.embedded.f32"
NATIVE_EMBEDDED_SHA256 = "0fa3aa5a49166ae8467e582a9919dc36d4b26bda14775df9984fb95cd43f0d3b"
NATIVE_LAYER0_TRACE = "prefill.vision.layer.0.f32"
NATIVE_LAYER0_SHA256 = "dd1d3fa45703dfd92f790c7be8dd947b8c525bcb5bfc6544a911f00a567ddc37"
TRACE_BYTES = 6094848
NATIVE_EMBEDDED_FNV1A = "0f61758a8c967ae3"
LAYER0_STAGE_PINS = {
    "n": 1523712,
    "shape": [768, 1984],
    "max_abs": 0.0336941481,
    "mean_abs": 0.00107735128,
    "rms": 0.00161888264,
    "cos_min": 0.999947548,
    "cos_mean": 0.999983907,
    "cos_global": 0.999984026,
    "native_norm": 353.476929,
    "reference_norm": 353.509766,
}
EMBEDDED_STAGE_PINS = {
    "max_abs": 0.0178086758,
    "rms": 0.000957455137,
    "cos_min": 0.999993145,
}
BACKEND_TEXT = "grid=1,62,32 tokens=1984"
REPORT_TOLERANCE = 5e-7

# Geometry frozen by the hosted blueprint/prefill runs.
TOKENS = 1984
WIDTH = 768
HEADS = 12
HEAD_DIM = 64
ROTARY = 64
LAYERS = 12
GRID = [1, 62, 32]
CU_SEQLENS = [0, 1984]
SPATIAL_MERGE = 2
INTERMEDIATE = 3072
SEMANTIC = "prefill.vision.layer.0"

REFERENCE_EMBEDDED = "prefill.vision.embedded"
REFERENCE_LAYER0 = "prefill.vision.layer.0"
REFERENCE_COS = "prefill.vision.rotary_cos"
REFERENCE_SIN = "prefill.vision.rotary_sin"
CHECKPOINT_PREFIX = "model.visual.blocks.0."
EXPORT_PREFIX = "v.blocks.0."
BLOCK_TENSORS = (
    ("norm1.weight", (WIDTH,)),
    ("norm1.bias", (WIDTH,)),
    ("norm2.weight", (WIDTH,)),
    ("norm2.bias", (WIDTH,)),
    ("attn.qkv.weight", (3 * WIDTH, WIDTH)),
    ("attn.qkv.bias", (3 * WIDTH,)),
    ("attn.proj.weight", (WIDTH, WIDTH)),
    ("attn.proj.bias", (WIDTH,)),
    ("mlp.linear_fc1.weight", (INTERMEDIATE, WIDTH)),
    ("mlp.linear_fc1.bias", (INTERMEDIATE,)),
    ("mlp.linear_fc2.weight", (WIDTH, INTERMEDIATE)),
    ("mlp.linear_fc2.bias", (WIDTH,)),
)

# Unchanged native gate. Never edit these thresholds in this lane.
STRICT_MAX_ABS = 0.002
STRICT_COS_MIN = 0.99999
IDENTITY_TOLERANCE = 1e-9
VERDICT_MEASUREMENTS = (
    "source_pins_ok", "weights_widening_ok", "geometry_ok",
    "reference_bf16_representable", "archive_embedded_bf16_representable",
    "repeat_identical", "device_matches_archive", "decomposition",
    "r1_vs_archive", "r2_vs_archive", "r3_vs_native",
    "negative_control_rejected", "report_reproduction_ok",
    "strict_failure_reproduced", "r4_vs_native",
)

OUT = Path("/kaggle/working/xiaomi-layer0-replay")
SCRATCH = Path("/kaggle/temp/xiaomi-layer0-replay")
REPORT = {
    "script_version": SCRIPT_VERSION,
    "run_id": RUN_ID,
    "scope": "hosted original-module layer-0 attribution of the Xiaomi-OCR-0 vision "
             "prefill; single block, single page prefill; not native parity, not full "
             "vision, not the decoder, not decoded text, not quantization, not speed",
    "strict_gate": {"max_abs": STRICT_MAX_ABS, "row_cos_min": STRICT_COS_MIN,
                    "unchanged": True, "relaxed_for_dtype": False},
    "decomposition": {"D": "N - A", "B": "N - R3", "I": "R3 - R2", "G": "R2 - A",
                      "identity": "D = B + I + G"},
    "runs": {"R1": "checkpoint BF16 block, CUDA/T4, SDPA, archive input + archive rotary",
             "R2": "F32 CPU module, eager, archive input",
             "R3": "F32 CPU module, eager, native input",
             "R4": "BF16 CPU module, eager, native input (optional control)",
             "rotary": "every run uses the archive cos/sin from the reference GGUF, so the "
                       "I term isolates the embedded input only; the retained native rotary "
                       "differs from the archive by at most 5.96e-8 (v3 rotary stages pass)"},
    "claims": {"decoded_parity_claimed": False, "decoded_quality_claimed": False,
               "speed_claimed": False, "native_graph_claimed": False,
               "native_strict_gate_waived": False, "block_defect_ruled_out": False,
               "sole_causal_label": None},
    "caveats": [
        "B passing does not establish that the block has no defect.",
        "A failing B does not prove that all error is inherited from the input.",
        "Any R1 difference from the archive leaves a CPU/GPU/backend ambiguity.",
        "No sole causal label is chosen; interactions are quantified, not apportioned.",
        "Row cosine is reported only for actual-output comparisons, never for error terms.",
        "The retained native rotary differs from the archive rotary by <=5.96e-8; all runs "
        "use the archive rotary so that the I term isolates the embedded input.",
    ],
    "not_tested": ["native graph construction or execution", "full vision tower",
                   "decoder", "decoded OCR text", "quantization", "speed",
                   "other layers", "other pages or grids"],
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


def close(actual, expected, tolerance=REPORT_TOLERANCE):
    return abs(float(actual) - float(expected)) <= tolerance * max(1.0, abs(float(expected)))


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


def metrics(actual, reference, cosines=True):
    """Pairwise comparison of equal-shaped 2-D arrays.

    ``cosines`` is True only for actual-output comparisons. Error terms use
    ``cosines=False`` so a row cosine can never be read as a gate on an error.
    """
    import numpy as np

    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape or a.ndim != 2:
        raise ValueError(f"comparison shapes differ: {a.shape} vs {r.shape}")
    if not (np.isfinite(a).all() and np.isfinite(r).all()):
        raise ValueError("comparison inputs contain non-finite values")
    delta = a - r
    flat = np.abs(delta).reshape(-1)
    worst = int(np.argmax(flat))
    row, column = divmod(worst, a.shape[1])
    exact = delta == 0
    result = {
        "shape": list(a.shape),
        "max_abs": float(np.abs(delta).max()),
        "mean_abs": float(np.abs(delta).mean()),
        "rms": float(np.sqrt(np.square(delta).mean())),
        "exact_fraction": float(exact.mean()),
        "mismatch_count": int((~exact).sum()),
        "elements": int(a.size),
        "worst_index": [row, column],
        "worst_abs": float(flat[worst]),
        "worst_actual": float(a[row, column]),
        "worst_reference": float(r[row, column]),
        "actual_norm": float(np.linalg.norm(a)),
        "reference_norm": float(np.linalg.norm(r)),
        "cosines_reported": bool(cosines),
    }
    if cosines:
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1)
        if (denominator <= 0).any():
            raise ValueError("zero-magnitude row in comparison")
        row_cos = (a * r).sum(axis=1) / denominator
        result.update({
            "row_cos_min": float(row_cos.min()),
            "row_cos_mean": float(row_cos.mean()),
            "row_cos_max": float(row_cos.max()),
            "cos_global": float((a * r).sum() / float(np.linalg.norm(a) * np.linalg.norm(r))),
        })
    return result


def strict_pass(metric):
    if not metric.get("cosines_reported", True):
        raise ValueError("strict gate needs a row cosine; error terms are not gated")
    return metric["max_abs"] <= STRICT_MAX_ABS and metric["row_cos_min"] >= STRICT_COS_MIN


def read_trace(raw, tokens=TOKENS, width=WIDTH, name="trace"):
    """Read a native diff dump with contiguous features for each token."""
    import numpy as np

    expected = int(tokens) * int(width) * 4
    if len(raw) != expected:
        raise ValueError(f"{name} bytes {len(raw)} != expected {expected}")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(values.reshape(int(tokens), int(width)))


def cu_seqlens_ok(values):
    import numpy as np

    array = np.asarray(values)
    return (array.dtype.kind == "i" and array.dtype.itemsize == 4 and array.ndim == 1
            and array.tolist() == CU_SEQLENS and int(array[-1]) == TOKENS)


def rotary_ok(cosine, sine):
    import numpy as np

    for name, value in (("cos", cosine), ("sin", sine)):
        if tuple(value.shape) != (TOKENS, ROTARY):
            raise ValueError(f"rotary {name} shape {tuple(value.shape)} != {(TOKENS, ROTARY)}")
        if not np.isfinite(value).all():
            raise ValueError(f"rotary {name} contains non-finite values")
    return True


def tensor_from_gguf(reader, name, shape):
    import numpy as np

    found = [tensor for tensor in reader.tensors if tensor.name == name]
    if len(found) != 1:
        raise ValueError(f"gguf tensor {name}: expected exactly one, found {len(found)}")
    values = np.ascontiguousarray(np.asarray(found[0].data, dtype=np.float32).reshape(shape))
    if values.size != int(np.prod(shape)):
        raise ValueError(f"gguf tensor {name}: element count differs")
    if not np.isfinite(values).all():
        raise ValueError(f"gguf tensor {name}: non-finite values")
    return values


def block_tensor_layout():
    """The twelve named block tensors, checkpoint and export names and shapes."""
    return {
        "names": [name for name, _ in BLOCK_TENSORS],
        "shapes": {name: list(shape) for name, shape in BLOCK_TENSORS},
        "checkpoint": {name: CHECKPOINT_PREFIX + name for name, _ in BLOCK_TENSORS},
        "export": {name: EXPORT_PREFIX + name for name, _ in BLOCK_TENSORS},
        "count": len(BLOCK_TENSORS),
    }


def widening_report(checkpoint_tensors, export_tensors):
    """Bytewise check that every export block tensor is the BF16 widening."""
    import numpy as np

    layout = block_tensor_layout()
    per_name = {}
    missing = []
    for name in layout["names"]:
        checkpoint = checkpoint_tensors.get(name)
        export = export_tensors.get(name)
        expected_shape = tuple(layout["shapes"][name])
        if checkpoint is None or export is None:
            missing.append(name)
            continue
        checkpoint = np.asarray(checkpoint)
        export = np.asarray(export, dtype=np.float32)
        if tuple(checkpoint.shape) != expected_shape or tuple(export.shape) != expected_shape:
            raise ValueError(f"{name}: shape {checkpoint.shape}/{export.shape} != {expected_shape}")
        widened = np.ascontiguousarray(checkpoint.astype(np.float32))
        per_name[name] = {
            "widened_dtype": "float32 (exact BF16 widening)",
            "shape": list(expected_shape),
            "elements": int(export.size),
            "byte_equal": bool(widened.tobytes() == np.ascontiguousarray(export, dtype="<f4").tobytes()),
            "checkpoint_sha256": digest_bytes(widened.tobytes()),
            "export_sha256": digest_bytes(np.ascontiguousarray(export, dtype="<f4").tobytes()),
        }
    if missing:
        raise ValueError("block tensors missing: " + ", ".join(missing))
    ok = all(row["byte_equal"] for row in per_name.values())
    return {"per_name": per_name, "all_byte_equal": bool(ok), "count": len(per_name),
            "mismatched": sorted(name for name, row in per_name.items() if not row["byte_equal"])}


def decompose(native, archive, r2, r3):
    """Exact algebra D=N-A, B=N-R3, I=R3-R2, G=R2-A with interaction accounting.

    All arithmetic is float64 over float32 measurements, so every subtraction is
    exact. The identity residual is reported, not assumed. Row cosines are never
    computed for the error terms.
    """
    import numpy as np

    arrays = {"N": native, "A": archive, "R2": r2, "R3": r3}
    reference_shape = np.asarray(native).shape
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.shape != reference_shape or value.ndim != 2:
            raise ValueError(f"{name}: shape {value.shape} != {reference_shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name}: non-finite values")
    n = np.asarray(native, dtype=np.float64)
    a = np.asarray(archive, dtype=np.float64)
    two = np.asarray(r2, dtype=np.float64)
    three = np.asarray(r3, dtype=np.float64)
    terms = {"D": n - a, "B": n - three, "I": three - two, "G": two - a}
    reconstructed = terms["B"] + terms["I"] + terms["G"]
    residual = terms["D"] - reconstructed
    identity_max_abs = float(np.abs(residual).max())
    labels = ("D", "B", "I", "G")
    norms = {name: float(np.linalg.norm(terms[name])) for name in labels}
    energy = {name: float(np.square(terms[name]).sum()) for name in labels}
    cross = {}
    pair_order = ("B", "I", "G")
    for index, first in enumerate(pair_order):
        for second in pair_order[index + 1:]:
            dot = float((terms[first] * terms[second]).sum())
            denominator = norms[first] * norms[second]
            cross[f"{first}_{second}"] = {
                "dot": dot,
                "normalized_dot": float(dot / denominator) if denominator > 0 else None,
            }
    denominator_total = norms["B"] + norms["I"] + norms["G"]
    interaction_total = 2.0 * sum(row["dot"] for row in cross.values())
    magnitude = {
        name: {"max_abs": float(np.abs(terms[name]).max()),
               "mean_abs": float(np.abs(terms[name]).mean()),
               "rms": float(np.sqrt(np.square(terms[name]).mean())),
               "norm": norms[name]}
        for name in labels
    }
    return {
        "terms": terms,
        "report": {
            "labels": {"D": "N - A", "B": "N - R3", "I": "R3 - R2", "G": "R2 - A"},
            "shape": list(reference_shape),
            "identity_max_abs": identity_max_abs,
            "identity_exact": bool(identity_max_abs == 0.0),
            "identity_tolerance": IDENTITY_TOLERANCE,
            "magnitudes": magnitude,
            "energy": energy,
            "energy_sum_terms": energy["B"] + energy["I"] + energy["G"],
            "interaction_energy": interaction_total,
            "energy_reconstructed": energy["B"] + energy["I"] + energy["G"] + interaction_total,
            "cancellation": {
                "norm_D": norms["D"],
                "norm_B_plus_I_plus_G": denominator_total,
                "ratio": float(norms["D"] / denominator_total) if denominator_total > 0 else None,
                "amplification": float(denominator_total / norms["D"]) if norms["D"] > 0 else None,
            },
            "cross_terms": cross,
            "interaction_fraction_of_D": (
                float(interaction_total / energy["D"]) if energy["D"] > 0 else None),
            "row_cosines_reported": False,
            "note": "error terms are magnitudes/interactions only; no cosine is a gate here",
        },
    }


def negative_control(values, shift=TOKENS // 2 + 1):
    import numpy as np

    values = np.asarray(values)
    shifted = np.ascontiguousarray(np.roll(values, shift=shift, axis=0))
    if shifted.shape != values.shape or np.array_equal(shifted, values):
        raise ValueError("negative control did not change the rows")
    return shifted


def reproduction_checks(measured, pins=None, tolerance=REPORT_TOLERANCE):
    """Compare a measured layer-0 error metric with the retained v3 stage row."""
    pins = LAYER0_STAGE_PINS if pins is None else pins
    fields = {"max_abs": "max_abs", "mean_abs": "mean_abs", "rms": "rms",
              "cos_min": "row_cos_min", "cos_mean": "row_cos_mean",
              "cos_global": "cos_global", "native_norm": "actual_norm",
              "reference_norm": "reference_norm"}
    checks = {key: close(measured[field], pins[key]) for key, field in fields.items()}
    return {"tolerance_relative": tolerance, "checks": checks, "ok": bool(all(checks.values())),
            "computed": {key: measured[field] for key, field in fields.items()},
            "pinned": dict(pins)}


def verdict(**measured):
    """Fail-closed comparison-consistency verdict; no defaults, no causal label."""
    required = VERDICT_MEASUREMENTS
    missing = [key for key in required if key not in measured]
    if missing:
        raise ValueError("missing measurement: " + ", ".join(missing))
    unexpected = sorted(set(measured) - set(required))
    if unexpected:
        raise ValueError("unexpected measurement: " + ", ".join(unexpected))
    for key in ("r1_vs_archive", "r2_vs_archive", "r3_vs_native"):
        metric = measured[key]
        for field in ("max_abs", "rms", "row_cos_min", "exact_fraction"):
            if field not in metric:
                raise ValueError(f"missing measurement: {key}.{field}")
            value = metric[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"non-finite measurement: {key}.{field}={value!r}")
    if not isinstance(measured["r4_vs_native"], dict):
        raise ValueError("r4_vs_native must be a measurement or an explicit not-run record")
    report = measured["decomposition"]
    for field in ("identity_max_abs", "identity_tolerance", "energy", "cross_terms",
                  "cancellation", "magnitudes"):
        if field not in report:
            raise ValueError(f"missing measurement: decomposition.{field}")
    if not math.isfinite(float(report["identity_max_abs"])) or report["identity_max_abs"] < 0:
        raise ValueError("non-finite decomposition identity residual")
    r1 = measured["r1_vs_archive"]
    bit_exact = r1["exact_fraction"] == 1.0 and r1["max_abs"] == 0.0
    checks = {
        "pinned_source_verified": bool(measured["source_pins_ok"]),
        "twelve_block_tensors_are_exact_bf16_widening": bool(measured["weights_widening_ok"]),
        "geometry_matches_1984x768_int32_cu_seqlens": bool(measured["geometry_ok"]),
        "archive_activations_are_bf16_representable": bool(
            measured["reference_bf16_representable"]),
        "archive_input_is_bf16_representable": bool(
            measured["archive_embedded_bf16_representable"]),
        "repeated_run_is_bit_identical": bool(measured["repeat_identical"]),
        "hosted_device_matches_retained_t4": bool(measured["device_matches_archive"]),
        "decomposition_identity_within_tolerance": bool(
            report["identity_max_abs"] <= float(report["identity_tolerance"])),
        "archive_closure_is_bitwise_exact": bool(bit_exact),
        "retained_strict_failure_reproduced": bool(measured["strict_failure_reproduced"]),
        "retained_stage_metrics_reproduced": bool(measured["report_reproduction_ok"]),
        "negative_control_fails_both_gates": bool(measured["negative_control_rejected"]),
    }
    ambiguity = (not checks["archive_closure_is_bitwise_exact"]
                 or not checks["archive_input_is_bf16_representable"]
                 or not checks["hosted_device_matches_retained_t4"]
                 or not checks["repeated_run_is_bit_identical"])
    return {
        "conditions": checks,
        "confirmed": bool(all(checks.values())),
        "comparison_consistency_only": True,
        "cpu_gpu_backend_ambiguity": bool(ambiguity),
        "block_defect_ruled_out": False,
        "b_pass_does_not_prove_no_block_defect": True,
        "sole_causal_label": None,
        "interaction_fraction_of_D": report.get("interaction_fraction_of_D"),
        "cancellation": report.get("cancellation"),
        "statement": consistency_statement(measured, checks, ambiguity),
    }


def consistency_statement(measured, checks, ambiguity):
    r1 = measured["r1_vs_archive"]
    r2 = measured["r2_vs_archive"]
    r3 = measured["r3_vs_native"]
    report = measured["decomposition"]
    parts = [
        f"R1 archive closure max_abs={r1['max_abs']:.9g} rms={r1['rms']:.9g} "
        f"exact_fraction={r1['exact_fraction']:.9f}",
        f"R2 (CPU F32, archive input) vs archive max_abs={r2['max_abs']:.9g} "
        f"row_cos_min={r2['row_cos_min']:.9g}",
        f"R3 (CPU F32, native input) vs native max_abs={r3['max_abs']:.9g} "
        f"row_cos_min={r3['row_cos_min']:.9g}",
        f"identity residual {report['identity_max_abs']:.3g} "
        f"(tolerance {report['identity_tolerance']:.3g})",
        f"cancellation ratio {report['cancellation']['ratio']}",
    ]
    if ambiguity:
        parts.append("archive closure is not bitwise exact, so CPU/GPU/backend and BF16 "
                     "rounding differences remain possible and no single cause is labelled")
    if not checks["negative_control_fails_both_gates"]:
        parts.append("negative control was not rejected; the comparison is invalid")
    if measured["r4_vs_native"].get("status") == "not_run":
        parts.append("R4 BF16 CPU control was not run")
    parts.append("B passing does not establish that the block has no defect and does not "
                 "prove all error is inherited; this is comparison consistency only")
    return "; ".join(parts) + "."


def attention_interfaces(modeling):
    """Resolve exactly as the pinned vision attention does, including eager fallback."""
    registry = modeling.ALL_ATTENTION_FUNCTIONS
    eager = modeling.eager_attention_forward
    functions = {name: registry.get_interface(name, eager) for name in ("eager", "sdpa")}
    if functions["eager"] is not eager or functions["sdpa"] is eager:
        raise ValueError("attention interface resolution differs from the pinned model")
    return {name: f"{fn.__module__}.{fn.__name__}" for name, fn in functions.items()}


def negative_control_rejected(metric):
    return metric["max_abs"] > STRICT_MAX_ABS and metric["row_cos_min"] < STRICT_COS_MIN


def release_cuda_cache(torch_module):
    """Free the CUDA allocator through the module.

    ``torch.device`` has no ``empty_cache``; only the ``torch.cuda`` module does.
    Kept as a helper so the call site is testable without a GPU.
    """
    empty_cache = getattr(getattr(torch_module, "cuda", None), "empty_cache", None)
    if not callable(empty_cache):
        return False
    empty_cache()
    return True


def require_gguf_f32_type(actual, expected):
    """GGUF quantization types are IntEnums whose str() is the bare integer."""
    if actual is None or int(actual) != int(expected):
        raise ValueError(f"gguf tensor type {actual!r} is not the required {expected!r}")
    return getattr(expected, "name", "F32")


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


def verify_report(report, trace_paths):
    """Authenticate the mounted v3 artefact set against the retained report."""
    thresholds = report.get("thresholds", {})
    if (thresholds.get("max_abs") != STRICT_MAX_ABS
            or thresholds.get("cos_min") != STRICT_COS_MIN
            or thresholds.get("relaxed_for_dtype") is not False):
        raise ValueError("mounted report thresholds differ from the unchanged strict gate")
    inputs = report.get("input_sha256", {})
    if inputs.get("model") != EXPORT_SHA256 or inputs.get("reference") != REFERENCE_SHA256:
        raise ValueError("mounted report input pins differ from the export/reference digests")
    stages = [row for row in report.get("stages", []) if row.get("name") == SEMANTIC]
    if len(stages) != 1:
        raise ValueError("mounted report does not carry exactly one prefill.vision.layer.0 stage")
    stage = stages[0]
    for key, pinned in LAYER0_STAGE_PINS.items():
        actual = stage.get(key)
        if actual is None:
            raise ValueError(f"mounted report stage {key} is missing")
        if key == "shape":
            if list(actual) != pinned:
                raise ValueError(f"mounted report stage shape {actual} != {pinned}")
        elif isinstance(pinned, int):
            if int(actual) != pinned:
                raise ValueError(f"mounted report stage {key} {actual} != {pinned}")
        elif not close(actual, pinned):
            raise ValueError(f"mounted report stage {key} {actual} != {pinned}")
    embedded = [row for row in report.get("stages", []) if row.get("name") == "prefill.vision.embedded"]
    if len(embedded) != 1:
        raise ValueError("mounted report does not carry exactly one embedded stage")
    for key, pinned in EMBEDDED_STAGE_PINS.items():
        if embedded[0].get(key) is None:
            raise ValueError(f"mounted report embedded {key} is missing")
        if not close(embedded[0].get(key), pinned):
            raise ValueError(f"mounted report embedded {key} differs from the pin")
    if stage.get("pass") is not False or stage.get("strict_absolute_pass") is not False:
        raise ValueError("mounted report claims a strict pass for layer 0; this lane does "
                         "not waive the retained failure")
    closure = report.get("source_closure", {})
    if closure.get("staged_runner_sha256") != VISION_PREFILL_RUNNER_SHA256:
        raise ValueError("mounted report was produced by a different vision-prefill runner")
    backend = report.get("backend_evidence", "")
    if BACKEND_TEXT not in backend or "CPU" not in backend:
        raise ValueError(f"mounted report backend evidence differs from the pinned CPU grid: {backend!r}")
    for name, pinned in ((NATIVE_EMBEDDED_TRACE, NATIVE_EMBEDDED_SHA256),
                         (NATIVE_LAYER0_TRACE, NATIVE_LAYER0_SHA256)):
        entry = report.get("native_trace_sha256", {}).get(name)
        if not entry or entry.get("sha256") != pinned or entry.get("bytes") != TRACE_BYTES:
            raise ValueError(f"mounted report native trace entry differs for {name}")
        if not trace_paths[name].is_file():
            raise ValueError(f"mounted native trace missing: {name}")
        if digest(trace_paths[name]) != pinned:
            raise ValueError(f"mounted native trace bytes differ for {name}")
    return {"stage": stage, "embedded_stage": embedded[0]}


def main():
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("this GPU replay must run on Kaggle")
    save("admission")
    free = shutil.disk_usage("/kaggle/working").free
    REPORT["initial_free_bytes"] = free
    if free < 10 * 1024 ** 3:
        raise RuntimeError("insufficient hosted workspace")
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the R1 archive-closure run; CPU fallback forbidden")
    capability = torch.cuda.get_device_capability()
    if capability < (7, 0):
        raise RuntimeError(f"CUDA capability {capability} below 7.0; pinned module requires it")

    # Explicit, recorded precision defaults. TF32 off everywhere: the F32 control
    # must be a true F32 block, and BF16 is excluded from TF32 entirely.
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(False)
    try:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    except (RuntimeError, ValueError):
        pass
    REPORT["device"] = {"name": torch.cuda.get_device_name(0), "capability": list(capability),
                        "cuda": torch.version.cuda}
    device_matches_archive = "T4" in REPORT["device"]["name"]
    REPORT["device"]["matches_retained_t4"] = bool(device_matches_archive)
    if not device_matches_archive:
        REPORT["device"]["note"] = ("the retained archive was produced on a T4; this host is "
                                    "reported as a CPU/GPU/backend ambiguity, not a closure")
    REPORT["torch_flags_after_setup"] = {
        "default_dtype": str(torch.get_default_dtype()),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "grad_enabled": bool(torch.is_grad_enabled()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "threads": torch.get_num_threads(),
    }
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
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionBlock

    REPORT["versions"] = {"transformers": transformers.__version__, "torch": torch.__version__,
                          "numpy": np.__version__,
                          **{name: importlib.metadata.version(name) for name in
                             ("gguf", "safetensors", "huggingface-hub")}}
    for name, expected in (("transformers", TRANSFORMERS), ("gguf", GGUF), ("safetensors", SAFETENSORS)):
        if REPORT["versions"][name] != expected:
            raise ValueError(f"pinned {name} version required: {REPORT['versions'][name]} != {expected}")

    save("source-closure")
    source_file = inspect.getsourcefile(Qwen3_5VisionBlock)
    if not source_file or not Path(source_file).is_file():
        raise RuntimeError("cannot resolve the installed Qwen3_5VisionBlock source file")
    module_file = Path(source_file)
    module_sha = digest(module_file)
    class_source = inspect.getsource(Qwen3_5VisionBlock)
    attention_source = inspect.getsource(modeling.Qwen3_5VisionAttention)
    class_pins = {pin: (pin in class_source) for pin in CLASS_TEXT_PINS}
    attention_pins = {pin: (pin in attention_source) for pin in ATTENTION_TEXT_PINS}
    unmodified = (Qwen3_5VisionBlock is getattr(modeling, SOURCE_CLASS)
                  and Qwen3_5VisionBlock.__module__ == modeling.__name__
                  and not getattr(Qwen3_5VisionBlock, "__crispembed_patched__", False))
    REPORT["original_module"] = {
        "module_file": str(module_file),
        "module_file_sha256": module_sha,
        "module_sha_matches_pin": module_sha == SOURCE_MODULE_SHA256,
        "class": f"{Qwen3_5VisionBlock.__module__}.{Qwen3_5VisionBlock.__name__}",
        "class_source_sha256": digest_bytes(class_source.encode()),
        "class_text_pins": class_pins,
        "attention_source_sha256": digest_bytes(attention_source.encode()),
        "attention_text_pins": attention_pins,
        "unmodified_class": bool(unmodified),
        "init_signature": str(inspect.signature(Qwen3_5VisionBlock.__init__)),
        "forward_signature": str(inspect.signature(Qwen3_5VisionBlock.forward)),
    }
    if module_sha != SOURCE_MODULE_SHA256:
        raise RuntimeError("installed modeling_qwen3_5.py differs from the pinned audited module digest")
    if not unmodified or not all(class_pins.values()) or not all(attention_pins.values()):
        raise RuntimeError("pinned Qwen3_5VisionBlock no longer matches the audited implementation")
    source_ok = True
    REPORT["source_closure"] = {
        "script_version": SCRIPT_VERSION, "base_commit": BASE_COMMIT,
        "runner_sha256": digest(__file__),
        "pinned": {
            "transformers": TRANSFORMERS, "tokenizers": TOKENIZERS, "gguf": GGUF,
            "safetensors": SAFETENSORS, "model_repo": MODEL_REPO, "revision": REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256, "config_sha256": CONFIG_SHA256,
            "export_sha256": EXPORT_SHA256, "reference_sha256": REFERENCE_SHA256,
            "source_module": SOURCE_MODULE, "source_module_sha256": SOURCE_MODULE_SHA256,
            "vision_prefill_runner_sha256": VISION_PREFILL_RUNNER_SHA256,
            "vision_prefill_report_sha256": VISION_PREFILL_REPORT_SHA256,
            "native_embedded_sha256": NATIVE_EMBEDDED_SHA256,
            "native_layer0_sha256": NATIVE_LAYER0_SHA256,
            "tokens": TOKENS, "width": WIDTH, "heads": HEADS, "head_dim": HEAD_DIM,
            "rotary": ROTARY, "layers": LAYERS, "cu_seqlens": CU_SEQLENS, "grid_thw": GRID,
        },
    }
    (OUT / "source-closure.json").write_text(json.dumps(REPORT["source_closure"], indent=2) + "\n")

    save("inputs")
    reference_path = find_unique_input("qianziwen-prefill.gguf", REFERENCE_SHA256)
    export_path = find_unique_input("xiaomi-ocr-0-f32-experimental.gguf", EXPORT_SHA256)
    embedded_path = find_unique_input(NATIVE_EMBEDDED_TRACE, NATIVE_EMBEDDED_SHA256, TRACE_BYTES)
    layer0_path = find_unique_input(NATIVE_LAYER0_TRACE, NATIVE_LAYER0_SHA256, TRACE_BYTES)
    report_path = layer0_path.parent.parent / "report.json"
    if not report_path.is_file():
        raise ValueError(f"mounted vision-prefill report missing next to the native trace: {report_path}")
    report_bytes = report_path.read_bytes()
    mounted_report = json.loads(report_bytes)
    traces = {NATIVE_EMBEDDED_TRACE: embedded_path, NATIVE_LAYER0_TRACE: layer0_path}
    verify_report(mounted_report, traces)
    REPORT["input_paths"] = {"reference": str(reference_path), "export": str(export_path),
                             "native_embedded": str(embedded_path), "native_layer0": str(layer0_path),
                             "vision_prefill_report": str(report_path)}
    REPORT["vision_prefill_report"] = {
        "byte_sha256": digest_bytes(report_bytes),
        "byte_sha256_matches_retained_copy": digest_bytes(report_bytes) == VISION_PREFILL_REPORT_SHA256,
        "staged_runner_sha256": mounted_report["source_closure"]["staged_runner_sha256"],
        "backend_evidence": mounted_report.get("backend_evidence"),
        "layer0_stage_pinned": True,
    }

    save("traces")
    embedded_raw = embedded_path.read_bytes()
    layer0_raw = layer0_path.read_bytes()
    n_values = read_trace(layer0_raw, name=NATIVE_LAYER0_TRACE)
    in_values = read_trace(embedded_raw, name=NATIVE_EMBEDDED_TRACE)
    REPORT["native_traces"] = {
        name: {"bytes": len(raw), "sha256": digest_bytes(raw),
               "absmax": float(np.abs(read_trace(raw)).max())}
        for name, raw in ((NATIVE_LAYER0_TRACE, layer0_raw), (NATIVE_EMBEDDED_TRACE, embedded_raw))
    }
    REPORT["native_traces"][NATIVE_EMBEDDED_TRACE]["fnv1a"] = fnv1a(embedded_raw)
    REPORT["native_traces"][NATIVE_EMBEDDED_TRACE]["fnv1a_matches_report"] = (
        fnv1a(embedded_raw) == NATIVE_EMBEDDED_FNV1A)

    reference_reader = gguf.GGUFReader(str(reference_path))
    a_values = tensor_from_gguf(reference_reader, REFERENCE_LAYER0, (TOKENS, WIDTH))
    ir_values = tensor_from_gguf(reference_reader, REFERENCE_EMBEDDED, (TOKENS, WIDTH))
    cosine = tensor_from_gguf(reference_reader, REFERENCE_COS, (TOKENS, ROTARY))
    sine = tensor_from_gguf(reference_reader, REFERENCE_SIN, (TOKENS, ROTARY))
    rotary_ok(cosine, sine)
    REPORT["reference_activation"] = {
        "layer0": {"name": REFERENCE_LAYER0, "sha256": digest_bytes(a_values.tobytes()),
                   "absmax": float(np.abs(a_values).max()),
                   "bf16_representable_fraction": bf16_representable_fraction(a_values)},
        "embedded": {"name": REFERENCE_EMBEDDED, "sha256": digest_bytes(ir_values.tobytes()),
                     "absmax": float(np.abs(ir_values).max()),
                     "bf16_representable_fraction": bf16_representable_fraction(ir_values)},
        "rotary_cos_sha256": digest_bytes(cosine.tobytes()),
        "rotary_sin_sha256": digest_bytes(sine.tobytes()),
    }
    cu = np.asarray(CU_SEQLENS, dtype=np.int32)
    geometry_ok = (list(cu.tolist()) == CU_SEQLENS and int(cu[-1]) == TOKENS
                   and a_values.shape == (TOKENS, WIDTH) and ir_values.shape == (TOKENS, WIDTH)
                   and n_values.shape == (TOKENS, WIDTH) and in_values.shape == (TOKENS, WIDTH)
                   and cosine.shape == sine.shape == (TOKENS, ROTARY))
    REPORT["geometry_checks"] = {
        "tokens": TOKENS, "width": WIDTH, "heads": HEADS, "head_dim": HEAD_DIM,
        "rotary": ROTARY, "layers": LAYERS, "cu_seqlens": cu.tolist(),
        "cu_seqlens_dtype": str(cu.dtype), "grid_thw": GRID,
        "layer0_shape_token_major": list(a_values.shape),
        "native_shape_token_major": list(n_values.shape),
        "elements": int(a_values.size), "ok": bool(cu_seqlens_ok(cu) and geometry_ok),
    }
    if not geometry_ok or not cu_seqlens_ok(cu):
        raise ValueError("layer-0 geometry does not match the pinned 1984x768 / int32 cu_seqlens")

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
    layout = block_tensor_layout()
    checkpoint_tensors = {}
    checkpoint_dtypes = {}
    with safe_open(str(checkpoint), framework="pt") as handle:
        for name in layout["names"]:
            tensor = handle.get_tensor(layout["checkpoint"][name])
            if tensor.dtype != torch.bfloat16:
                raise ValueError(f"checkpoint tensor {name} is {tensor.dtype}, expected bfloat16")
            checkpoint_dtypes[name] = str(tensor.dtype)
            checkpoint_tensors[name] = np.ascontiguousarray(tensor.float().cpu().numpy())

    save("weights-widening")
    export_reader = gguf.GGUFReader(str(export_path))
    export_tensors = {name: tensor_from_gguf(export_reader, layout["export"][name], shape)
                      for name, shape in BLOCK_TENSORS}
    export_dtypes = {}
    export_types = {}
    for name in layout["names"]:
        found = [tensor for tensor in export_reader.tensors
                 if tensor.name == layout["export"][name]]
        if len(found) != 1:
            raise ValueError(f"export tensor {name}: expected exactly one")
        export_dtypes[name] = str(getattr(found[0], "tensor_type", "unknown"))
        export_types[name] = found[0].tensor_type
    for name in layout["names"]:
        require_gguf_f32_type(export_types[name], gguf.GGMLQuantizationType.F32)
    widening = widening_report(checkpoint_tensors, export_tensors)
    widening["checkpoint_dtypes"] = checkpoint_dtypes
    widening["export_dtypes"] = export_dtypes
    REPORT["weights_widening"] = widening
    if not widening["all_byte_equal"]:
        raise ValueError("exported block tensors are not the exact BF16 widening of the "
                         "checkpoint: " + ", ".join(widening["mismatched"]))

    save("module")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(local))
    vision_config = config.vision_config
    observed = {"hidden_size": int(vision_config.hidden_size),
                "num_heads": int(vision_config.num_heads),
                "intermediate_size": int(vision_config.intermediate_size),
                "depth": int(vision_config.depth),
                "hidden_act": str(vision_config.hidden_act),
                "spatial_merge_size": int(vision_config.spatial_merge_size),
                "patch_size": int(vision_config.patch_size[0] if isinstance(
                    vision_config.patch_size, (list, tuple)) else vision_config.patch_size)}
    REPORT["vision_config"] = observed
    if (observed["hidden_size"], observed["num_heads"], observed["intermediate_size"],
            observed["depth"], observed["spatial_merge_size"]) != (WIDTH, HEADS, INTERMEDIATE, LAYERS, SPATIAL_MERGE):
        raise ValueError("vision config geometry differs from the audited layer-0 prefill")

    def build(dtype, implementation):
        vision_config._attn_implementation = implementation
        block = Qwen3_5VisionBlock(vision_config).to(dtype).eval()
        state = {name: torch.from_numpy(checkpoint_tensors[name]).to(dtype).contiguous()
                 for name in layout["names"]}
        missing, unexpected = block.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"unexpected state keys: missing={list(missing)} unexpected={list(unexpected)}")
        for name in layout["names"]:
            parameter = dict(block.named_parameters())[name]
            if not torch.equal(parameter.detach(), state[name]):
                raise ValueError(f"block parameter {name} did not accept the checkpoint bit-exactly")
        if len(list(block.parameters())) != len(layout["names"]):
            raise ValueError("block carries parameters outside the twelve named tensors")
        return block

    block_f32 = build(torch.float32, "eager")
    block_bf16 = build(torch.bfloat16, "eager")
    if type(block_f32).__name__ != SOURCE_CLASS or "attn_implementation" not in str(
            inspect.signature(Qwen3_5VisionBlock.__init__)):
        raise ValueError("constructed object is not the unmodified pinned block")
    REPORT["module_instance"] = {
        "class": f"{type(block_f32).__module__}.{type(block_f32).__name__}",
        "parameter_count": len(list(block_f32.parameters())),
        "parameters": sorted(name for name, _ in block_f32.named_parameters()),
        "f32_dtype": str(next(block_f32.parameters()).dtype),
        "bf16_dtype": str(next(block_bf16.parameters()).dtype),
        "interface_source": "block.attn.config._attn_implementation (read by the pinned "
                            "attention on every forward)",
        "init_attention_argument": "declared in the pinned __init__ signature",
    }
    interfaces = attention_interfaces(modeling)
    REPORT["attention_interfaces"] = interfaces
    if interfaces.get("eager") == interfaces.get("sdpa"):
        raise ValueError("eager and SDPA resolve to the same interface; the control is meaningless")

    save("execute")
    cuda = torch.device("cuda", 0)
    cpu = torch.device("cpu")
    cos_cpu = torch.from_numpy(cosine).to(cpu)
    sin_cpu = torch.from_numpy(sine).to(cpu)

    def run(block, hidden, device, dtype, implementation):
        block.to(device=device, dtype=dtype)
        block.attn.config._attn_implementation = implementation
        if block.attn.config._attn_implementation != implementation:
            raise ValueError(f"attention implementation was not set to {implementation}")
        inputs = torch.from_numpy(np.asarray(hidden, dtype=np.float32)).to(device=device, dtype=dtype)
        cu_seqlens = torch.from_numpy(cu).to(device)
        embeddings = (cos_cpu.to(device), sin_cpu.to(device))
        with torch.inference_mode():
            output = block(inputs, cu_seqlens=cu_seqlens, position_embeddings=embeddings)
        if tuple(output.shape) != (TOKENS, WIDTH) or not torch.isfinite(output).all():
            raise ValueError(f"block output shape/non-finite failure: {tuple(output.shape)}")
        return np.ascontiguousarray(output.detach().float().cpu().numpy())

    # R1: checkpoint BF16 block, CUDA/T4, SDPA, archive input + archive rotary.
    r1 = run(build(torch.bfloat16, "sdpa"), ir_values, cuda, torch.bfloat16, "sdpa")
    r1_repeat = run(build(torch.bfloat16, "sdpa"), ir_values, cuda, torch.bfloat16, "sdpa")
    repeat_identical = bool(np.array_equal(r1, r1_repeat))
    release_cuda_cache(torch)
    # R2/R3/R4: original module on CPU with the explicitly selected interface.
    r2 = run(build(torch.float32, "eager"), ir_values, cpu, torch.float32, "eager")
    r3 = run(build(torch.float32, "eager"), in_values, cpu, torch.float32, "eager")
    r4 = run(build(torch.bfloat16, "eager"), in_values, cpu, torch.bfloat16, "eager")
    # BF16 device/backend control: identical input, identical dtype, CPU eager vs
    # CUDA SDPA. Captured on its own so it cannot be folded into the strict gate.
    r1_cpu = run(build(torch.bfloat16, "eager"), ir_values, cpu, torch.bfloat16, "eager")
    # Separately captured attention-backend delta: same input, same dtype, CPU.
    r2_sdpa = run(build(torch.float32, "sdpa"), ir_values, cpu, torch.float32, "sdpa")
    r3_sdpa = run(build(torch.float32, "sdpa"), in_values, cpu, torch.float32, "sdpa")
    REPORT["attention_backend"] = {
        "implementations": {"eager": "block.attn.config._attn_implementation='eager'",
                            "sdpa": "block.attn.config._attn_implementation='sdpa'"},
        "resolved_interfaces": interfaces,
        "r2_eager_vs_sdpa": metrics(r2_sdpa, r2),
        "r3_eager_vs_sdpa": metrics(r3_sdpa, r3),
        "note": "same input and dtype on CPU; isolates the attention implementation only",
    }
    REPORT["device_control"] = {
        "bf16_cuda_sdpa_vs_bf16_cpu_eager_same_input": metrics(r1_cpu, r1),
        "bf16_cpu_eager_vs_archive": metrics(r1_cpu, a_values),
        "r1_bf16_cuda_sdpa_vs_r4_bf16_cpu_eager_other_input": metrics(r4, r1),
        "note": "same-input same-dtype control isolates device/backend; the other-input row "
                "also changes the input. Diagnostic only, never a gate",
    }
    REPORT["outputs"] = {}
    values = {"R1": r1, "R1C": r1_cpu, "R2": r2, "R3": r3, "R4": r4, "N": n_values, "A": a_values,
              "IN": in_values, "IR": ir_values}
    for name, array in values.items():
        array = np.ascontiguousarray(array, dtype="<f4")
        path = OUT / (name.lower() + ".layer0.f32")
        array.tofile(str(path))
        REPORT["outputs"][name] = {"path": path.name, "sha256": digest(path),
                                   "bytes": path.stat().st_size,
                                   "bf16_representable_fraction": bf16_representable_fraction(array)}
    for name in ("R1", "R1C", "R2", "R3", "R4", "N", "A"):
        if not np.isfinite(values[name]).all():
            raise ValueError(f"{name} contains non-finite values")

    save("compare")
    decomposition = decompose(n_values, a_values, r2, r3)
    d_report = decomposition["report"]
    terms = decomposition["terms"]
    for name, array in terms.items():
        array = np.ascontiguousarray(array, dtype="<f4")
        path = OUT / (name.lower() + ".term.f32")
        array.tofile(str(path))
        REPORT["outputs"][name] = {"path": path.name, "sha256": digest(path),
                                   "bytes": path.stat().st_size, "term": True}
    comparisons = {
        "r1_vs_archive": metrics(r1, a_values),
        "r1_repeat_vs_first": metrics(r1_repeat, r1),
        "r1_cpu_bf16_vs_archive": metrics(r1_cpu, a_values),
        "r2_vs_archive": metrics(r2, a_values),
        "r3_vs_native": metrics(r3, n_values),
        "r4_vs_native": metrics(r4, n_values),
        "native_vs_archive": metrics(n_values, a_values),
        "negative_permuted_rows": metrics(negative_control(r3), n_values),
        "r2_sdpa_vs_archive": metrics(r2_sdpa, a_values),
        "r3_sdpa_vs_native": metrics(r3_sdpa, n_values),
    }
    for key, metric in comparisons.items():
        metric["strict_pass"] = strict_pass(metric)
    comparison_path = OUT / "comparisons.json"
    comparison_path.write_text(json.dumps({"comparisons": comparisons,
                                           "decomposition": d_report},
                                          indent=2, allow_nan=False) + "\n")
    REPORT["comparisons_sha256"] = digest(comparison_path)
    REPORT["comparisons"] = comparisons
    REPORT["decomposition_report"] = d_report
    REPORT["decomposition_sha256"] = digest_bytes(
        json.dumps(d_report, sort_keys=True, allow_nan=False).encode())

    negative = comparisons["negative_permuted_rows"]
    negative_rejected = negative_control_rejected(negative)
    reproduction = reproduction_checks(comparisons["native_vs_archive"])
    REPORT["report_reproduction"] = reproduction
    strict_failure_reproduced = (
        not strict_pass(comparisons["native_vs_archive"])
        and d_report["cancellation"]["ratio"] is not None)
    try:
        result = verdict(
            source_pins_ok=source_ok and REPORT["original_module"]["module_sha_matches_pin"],
            weights_widening_ok=widening["all_byte_equal"],
            geometry_ok=bool(REPORT["geometry_checks"]["ok"]),
            reference_bf16_representable=REPORT["reference_activation"]["layer0"][
                "bf16_representable_fraction"] == 1.0,
            archive_embedded_bf16_representable=REPORT["reference_activation"]["embedded"][
                "bf16_representable_fraction"] == 1.0,
            repeat_identical=repeat_identical,
            device_matches_archive=device_matches_archive,
            decomposition=d_report,
            r1_vs_archive=comparisons["r1_vs_archive"],
            r2_vs_archive=comparisons["r2_vs_archive"],
            r3_vs_native=comparisons["r3_vs_native"],
            negative_control_rejected=negative_rejected,
            report_reproduction_ok=reproduction["ok"],
            strict_failure_reproduced=strict_failure_reproduced,
            r4_vs_native=comparisons["r4_vs_native"],
        )
    except ValueError as error:
        REPORT["status"] = "failed: verdict could not be assembled from measurements"
        REPORT["error_type"] = "VerdictRejected"
        REPORT["error"] = str(error)
        save("verdict-rejected")
        return 2

    REPORT["conditions"] = result["conditions"]
    REPORT["attribution"] = {
        "comparison_consistency_only": True,
        "confirmed": result["confirmed"],
        "cpu_gpu_backend_ambiguity": result["cpu_gpu_backend_ambiguity"],
        "block_defect_ruled_out": False,
        "b_pass_does_not_prove_no_block_defect": True,
        "sole_causal_label": None,
        "interaction_fraction_of_D": result["interaction_fraction_of_D"],
        "cancellation": result["cancellation"],
        "statement": result["statement"],
        "strict_gate_waived": False,
        "decoded_parity_claimed": False,
    }
    if result["confirmed"]:
        REPORT["status"] = ("completed: pinned original block reproduced on the hosted GPU, "
                            "bitwise archive closure recorded, twelve block tensors verified as "
                            "the exact BF16 widening, and the four-term decomposition satisfied "
                            "its identity. This is comparison consistency only; it does not "
                            "rule out a block defect, prove inheritance, or label a sole cause")
        save("complete")
        return 0
    REPORT["status"] = ("completed with an unconfirmed layer-0 comparison consistency; see "
                        "conditions, residual/ambiguity flags and decomposition; the strict "
                        "gate is unchanged")
    REPORT["error_type"] = "AttributionNotConfirmed"
    REPORT["error"] = json.dumps(result["conditions"], sort_keys=True)
    save("attribution-not-confirmed")
    return 1


def fnv1a(raw):
    value = 0xCBF29CE484222325
    for byte in raw:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


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
        try:
            save(REPORT.get("phase", "startup"))
        except BaseException:
            print("REPORT_WRITE_FAILED", traceback.format_exc()[-2000:], flush=True)
        CODE = 2
    REPORT["exit_code"] = CODE
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        save(REPORT.get("phase", "startup"))
    except BaseException:
        print("REPORT_WRITE_FAILED", traceback.format_exc()[-2000:], flush=True)
    print("LAYER0_REPLAY_RESULT", json.dumps(
        {"status": REPORT["status"],
         "confirmed": REPORT.get("attribution", {}).get("confirmed"),
         "ambiguity": REPORT.get("attribution", {}).get("cpu_gpu_backend_ambiguity"),
         "exit_code": CODE}, sort_keys=True), flush=True)
    raise SystemExit(CODE)
