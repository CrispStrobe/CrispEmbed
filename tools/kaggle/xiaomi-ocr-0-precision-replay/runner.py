#!/usr/bin/env python3
"""Hosted precision attribution of the Xiaomi-OCR-0 first patch stage.

Executes the *actual* pinned original module
(``transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionPatchEmbed``,
transformers 5.8.1) on a GPU with the checkpoint's own BF16 Conv3d weights, and
rebuilds the same stage under explicit precision controls. It answers one
question: is the retained ``prefill.vision.patch_embed`` activation a BF16
Conv3D result (the module casts its input to the weight dtype) or an F32
Conv3D result?

This is attribution, not parity. It does not run a native binary, the rest of
the vision tower, the decoder, OCR text, or quantization. The strict native
gate (max_abs <= 0.002, row cosine >= 0.99999) is recorded unchanged; this
run measures precision effects without assuming whether that gate will pass.

Only the standard library is imported at module scope so the repository test
can import the staged script without torch/numpy on the host.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

SCRIPT_VERSION = "xiaomi-precision-replay-v1"
RUN_ID = "xiaomi-precision-replay-v1"

BASE_COMMIT = "15adb672f71f8470a23db1284dafa102515adae9"
MODEL_REPO = "xiaomi-research/Xiaomi-OCR-0"
REVISION = "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30"
CHECKPOINT_BYTES = 1746937856
CHECKPOINT_SHA256 = "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348"
EXPORT_SHA256 = "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235"
REFERENCE_SHA256 = "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334"
TRANSFORMERS = "5.8.1"
TOKENIZERS = "0.22.2"
GGUF = "0.19.0"
SAFETENSORS = "0.8.0"

PATCHES, WIDTH, CHANNELS = 1984, 768, 1536
TEMPORAL, PATCH = 2, 16
PROJ_WEIGHT = "model.visual.patch_embed.proj.weight"
PROJ_BIAS = "model.visual.patch_embed.proj.bias"
GGUF_WEIGHT = "v.patch_embed.proj.weight"
GGUF_BIAS = "v.patch_embed.proj.bias"

# Unchanged native gate. Never edit these thresholds in this lane.
STRICT_MAX_ABS = 0.002
STRICT_COS_MIN = 0.99999
# Attribution budget for a BF16-stored reference: at least 99% of elements
# bit-identical and RMS <= 1e-4 (BF16 ulp at |v| = 1 is 2^-7 = 0.0078125, so
# this is ~1.3% of one ulp). This is a BF16-regime test, not a relaxation of
# the strict F32 gate above.
BF16_EXACT_MIN = 0.99
BF16_RMS_MAX = 1e-4

OUT = Path("/kaggle/working/xiaomi-precision-replay")
SCRATCH = Path("/kaggle/temp/xiaomi-precision-replay")
REPORT = {
    "script_version": SCRIPT_VERSION,
    "run_id": RUN_ID,
    "scope": "hosted original-module precision attribution of the first patch "
             "embedding stage only; not native parity, not full vision, not decoded text",
    "strict_gate": {"max_abs": STRICT_MAX_ABS, "row_cos_min": STRICT_COS_MIN,
                    "unchanged": True},
    "bf16_attribution_budget": {"exact_fraction_min": BF16_EXACT_MIN, "rms_max": BF16_RMS_MAX,
                                "purpose": "identify the reference compute regime, not a parity pass"},
    "not_tested": ["native CPU/CUDA stage reproduction", "full vision tower", "decoder",
                   "decoded OCR text", "quantization", "speed"],
    "status": "running",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def save(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["phase"] = phase
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=2, allow_nan=False) + "\n")
    print("===", phase, flush=True)


def bf16_round(values):
    """Round F32 to BF16 (round-to-nearest-even) in F32 storage."""
    import numpy as np

    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def bf16_representable_fraction(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    return float(np.mean(bf16_round(values) == values))


def metrics(actual, reference):
    """Pairwise comparison; actual/reference must be equal-shaped 2-D arrays."""
    import numpy as np

    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape or a.ndim != 2:
        raise ValueError(f"comparison shapes differ: {a.shape} vs {r.shape}")
    delta = a - r
    a_norm = np.linalg.norm(a, axis=1)
    r_norm = np.linalg.norm(r, axis=1)
    denominator = a_norm * r_norm
    if (denominator <= 0).any():
        raise ValueError("zero-magnitude row in comparison")
    row_cos = (a * r).sum(axis=1) / denominator
    flat = np.abs(delta).reshape(-1)
    worst_flat = int(np.argmax(flat))
    row, column = divmod(worst_flat, a.shape[1])
    exact = delta == 0
    global_denominator = float(np.linalg.norm(a) * np.linalg.norm(r))
    return {
        "shape": list(a.shape),
        "max_abs": float(np.abs(delta).max()),
        "mean_abs": float(np.abs(delta).mean()),
        "rms": float(np.sqrt(np.square(delta).mean())),
        "row_cos_min": float(row_cos.min()),
        "row_cos_mean": float(row_cos.mean()),
        "row_cos_max": float(row_cos.max()),
        "cos_global": float((a * r).sum() / global_denominator),
        "exact_fraction": float(exact.mean()),
        "mismatch_count": int((~exact).sum()),
        "elements": int(a.size),
        "worst_index": [row, column],
        "worst_abs": float(flat[worst_flat]),
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


def reference_reproduced_exactly(metric):
    return metric["mismatch_count"] == 0


def find_unique_input(name, expected_sha256):
    matches = []
    for path in Path("/kaggle/input").rglob(name):
        if path.is_file() and digest(path) == expected_sha256:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"{name}: expected exactly one input with pinned hash, found {len(matches)}")
    return matches[0]


def find_optional(name, size):
    matches = [path for path in Path("/kaggle/input").rglob(name)
               if path.is_file() and path.stat().st_size == size]
    if len(matches) > 1:
        raise ValueError(f"{name}: ambiguous, found {len(matches)} candidates")
    return matches[0] if matches else None


def tensor_from_gguf(reader, name, shape):
    import numpy as np

    found = [tensor for tensor in reader.tensors if tensor.name == name]
    if len(found) != 1:
        raise ValueError(f"reference tensor {name}: expected exactly one, found {len(found)}")
    values = np.ascontiguousarray(np.asarray(found[0].data, dtype=np.float32).reshape(shape))
    if values.size != int(np.prod(shape)):
        raise ValueError(f"reference tensor {name}: size differs")
    return values


def main():
    save("admission")
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("this GPU replay must run on Kaggle")
    free = shutil.disk_usage("/kaggle/working").free
    REPORT["initial_free_bytes"] = free
    if free < 6 * 1024 ** 3:
        raise RuntimeError("insufficient hosted workspace")
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback forbidden")
    capability = torch.cuda.get_device_capability()
    if capability < (7, 0):
        raise RuntimeError(f"CUDA capability {capability} below 7.0; pinned module requires it")

    # No TF32 anywhere: the F32 control must be a true F32 Conv3D.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    REPORT["torch_flags_after_setup"] = {
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    for name in ("cudnn_allow_tf32", "matmul_allow_tf32"):
        if REPORT["torch_flags_after_setup"][name]:
            raise RuntimeError(f"TF32 still enabled: {name}")
    if REPORT["torch_flags_after_setup"]["float32_matmul_precision"] != "highest":
        raise RuntimeError("float32 matmul precision is not 'highest'")
    REPORT["gpu"] = {
        "name": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_env": {key: value for key, value in os.environ.items() if key.startswith("CUDNN")},
        "cublas_workspace_env": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True)
    REPORT["gpu"]["nvidia_smi"] = smi.stdout.strip()
    (OUT / "nvidia-smi.log").write_text(smi.stdout + smi.stderr)
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
    import safetensors
    import transformers
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchEmbed

    REPORT["versions"] = {"transformers": transformers.__version__, "torch": torch.__version__,
                          "numpy": np.__version__,
                          **{name: importlib.metadata.version(name) for name in
                             ("gguf", "safetensors", "huggingface-hub")}}
    if transformers.__version__ != TRANSFORMERS:
        raise ValueError("pinned transformers version required")

    import inspect
    module_file = Path(inspect.getsourcefile(Qwen3_5VisionPatchEmbed))
    class_source = inspect.getsource(Qwen3_5VisionPatchEmbed)
    REPORT["original_module"] = {
        "class": "transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VisionPatchEmbed",
        "module_file": str(module_file),
        "module_file_sha256": digest(module_file),
        "class_source_sha256": digest_bytes(class_source.encode()),
        "casts_input_to_weight_dtype": "hidden_states.to(dtype=target_dtype)" in class_source,
        "calls_conv3d_proj": "self.proj(" in class_source,
        "class_source": class_source,
    }
    if not (REPORT["original_module"]["casts_input_to_weight_dtype"]
            and REPORT["original_module"]["calls_conv3d_proj"]):
        raise RuntimeError("pinned module no longer matches the audited cast/Conv3d shape")
    save("source-closure")
    REPORT["source_closure"] = {
        "script_version": SCRIPT_VERSION,
        "base_commit": BASE_COMMIT,
        "runner_sha256": digest(__file__),
        "pinned": {"transformers": TRANSFORMERS, "tokenizers": TOKENIZERS, "gguf": GGUF,
                   "safetensors": SAFETENSORS, "model_repo": MODEL_REPO, "revision": REVISION,
                   "checkpoint_sha256": CHECKPOINT_SHA256, "export_sha256": EXPORT_SHA256,
                   "reference_sha256": REFERENCE_SHA256},
    }
    (OUT / "source-closure.json").write_text(json.dumps(REPORT["source_closure"], indent=2) + "\n")

    save("inputs")
    reference_path = find_unique_input("qianziwen-prefill.gguf", REFERENCE_SHA256)
    export_path = find_unique_input("xiaomi-ocr-0-f32-experimental.gguf", EXPORT_SHA256)
    REPORT["input_paths"] = {"reference": str(reference_path), "export": str(export_path)}
    reference_reader = gguf.GGUFReader(str(reference_path))
    pixels = tensor_from_gguf(reference_reader, "input.pixel_values", (PATCHES, CHANNELS))
    saved = tensor_from_gguf(reference_reader, "prefill.vision.patch_embed", (PATCHES, WIDTH))
    if not (np.isfinite(pixels).all() and np.isfinite(saved).all()):
        raise ValueError("frozen reference arrays contain non-finite values")
    REPORT["inputs"] = {
        "pixel_values_sha256": digest_bytes(pixels.tobytes()),
        "saved_activation_sha256": digest_bytes(saved.tobytes()),
        "pixel_absmax": float(np.abs(pixels).max()),
        "saved_absmax": float(np.abs(saved).max()),
        "saved_norm": float(np.linalg.norm(saved.astype(np.float64))),
        "saved_bf16_representable_fraction": bf16_representable_fraction(saved),
    }

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
    REPORT["checkpoint"] = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size,
                            "sha256": CHECKPOINT_SHA256}
    with safe_open(str(checkpoint), framework="pt") as handle:
        weight_bf16 = handle.get_tensor(PROJ_WEIGHT)
        bias_bf16 = handle.get_tensor(PROJ_BIAS)
    if tuple(weight_bf16.shape) != (WIDTH, 3, TEMPORAL, PATCH, PATCH) or bias_bf16.shape != (WIDTH,):
        raise ValueError("checkpoint patch projection shapes differ")
    if weight_bf16.dtype is not torch.bfloat16 or bias_bf16.dtype is not torch.bfloat16:
        raise ValueError("checkpoint patch projection is not BF16")
    REPORT["checkpoint_tensors"] = {
        "weight_source_shape": list(weight_bf16.shape), "weight_dtype": str(weight_bf16.dtype),
        "bias_dtype": str(bias_bf16.dtype),
    }

    save("weights-widening")
    export_reader = gguf.GGUFReader(str(export_path))
    weight_export = tensor_from_gguf(export_reader, GGUF_WEIGHT, (WIDTH, CHANNELS))
    bias_export = tensor_from_gguf(export_reader, GGUF_BIAS, (WIDTH,))
    weight_f32 = weight_bf16.float().reshape(WIDTH, CHANNELS).contiguous()
    bias_f32 = bias_bf16.float().contiguous()
    weight_equal = torch.equal(weight_f32, torch.from_numpy(weight_export))
    bias_equal = torch.equal(bias_f32, torch.from_numpy(bias_export))
    REPORT["weights_widening"] = {
        "weight_bitwise_equal": weight_equal,
        "bias_bitwise_equal": bias_equal,
        "weight_source_f32_sha256": digest_bytes(weight_f32.numpy().tobytes()),
        "weight_export_sha256": digest_bytes(weight_export.tobytes()),
        "bias_source_f32_sha256": digest_bytes(bias_f32.numpy().tobytes()),
        "bias_export_sha256": digest_bytes(bias_export.tobytes()),
        "weight_absmax": float(np.abs(weight_export).max()),
    }
    if not (weight_equal and bias_equal):
        raise ValueError("exported F32 projection is not the exact BF16 widening of the checkpoint")

    save("module")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(local))
    vision_config = config.vision_config
    patch_size = vision_config.patch_size
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    patch_size = int(patch_size)
    in_channels = int(vision_config.in_channels)
    temporal = int(vision_config.temporal_patch_size)
    if (patch_size, temporal, in_channels) != (PATCH, TEMPORAL, 3):
        raise ValueError("vision config geometry differs from the audited 3x2x16x16 projection")
    REPORT["vision_config"] = {"hidden_size": vision_config.hidden_size,
                               "in_channels": in_channels, "patch_size": patch_size,
                               "temporal_patch_size": temporal}
    module = Qwen3_5VisionPatchEmbed(vision_config).to(dtype=torch.bfloat16)
    module.load_state_dict({"proj.weight": weight_bf16, "proj.bias": bias_bf16}, strict=True)
    if not (torch.equal(module.proj.weight, weight_bf16) and torch.equal(module.proj.bias, bias_bf16)):
        raise ValueError("module did not accept the checkpoint BF16 projection bit-exactly")
    module = module.to("cuda").eval()

    captured = {}

    def capture_proj_input(_module, args):
        captured.setdefault("input", args[0].detach())

    handle = module.proj.register_forward_pre_hook(capture_proj_input)
    pixels_cuda = torch.from_numpy(pixels).to("cuda")
    pixels_view = pixels_cuda.view(-1, in_channels, temporal, patch_size, patch_size)
    expected_cast = pixels_view.to(torch.bfloat16)
    try:
        with torch.inference_mode():
            first = module(pixels_cuda)
    finally:
        handle.remove()
    if "input" not in captured:
        raise ValueError("Conv3d received no captured input")
    cast_input = captured["input"]
    if cast_input.dtype is not torch.bfloat16 or tuple(cast_input.shape) != (
            PATCHES, in_channels, temporal, patch_size, patch_size):
        raise ValueError("captured Conv3d input dtype/shape differ from the audited cast")
    cast_equal = torch.equal(cast_input, expected_cast)
    cast_bits = cast_input.view(torch.uint16).cpu().numpy()
    REPORT["cast_input"] = {
        "dtype": str(cast_input.dtype),
        "shape": list(cast_input.shape),
        "sha256_bf16_bits": digest_bytes(cast_bits.tobytes()),
        "equals_explicit_bf16_cast_of_frozen_pixels": bool(cast_equal),
        "quantized_values": int((expected_cast.float() != pixels_view).sum().item()),
        "quantization_max_abs": float((expected_cast.float() - pixels_view).abs().max().item()),
    }
    if not cast_equal:
        raise ValueError("module cast input differs from the explicit BF16 cast of the frozen pixels")
    import torch.nn.functional as F

    weight_f32_cuda = torch.from_numpy(np.ascontiguousarray(weight_export)).to("cuda")
    bias_f32_cuda = torch.from_numpy(np.ascontiguousarray(bias_export)).to("cuda")
    weight_conv = weight_f32_cuda.reshape(WIDTH, in_channels, temporal, patch_size, patch_size)

    def conv32(values):
        out = F.conv3d(values.view(-1, in_channels, temporal, patch_size, patch_size),
                       weight_conv, bias_f32_cuda, stride=(temporal, patch_size, patch_size))
        return out.reshape(-1, WIDTH)

    def cuda_array(tensor):
        return tensor.detach().float().cpu().numpy()

    pixels_bf16 = pixels_cuda.to(torch.bfloat16)
    quantized_pixels_f32 = pixels_bf16.float()
    with torch.inference_mode():
        module_bf16 = first.float()
        module_repeat = module(pixels_cuda).float()
        f0 = conv32(pixels_cuda)
        f1 = conv32(quantized_pixels_f32)
        r0 = torch.from_numpy(bf16_round(cuda_array(f0))).to("cuda")
        r1 = torch.from_numpy(bf16_round(cuda_array(f1))).to("cuda")
        linear_f32 = F.linear(pixels_cuda, weight_f32_cuda, bias_f32_cuda)
        linear_bf16 = F.linear(pixels_bf16, module.proj.weight.view(WIDTH, CHANNELS),
                               module.proj.bias).float()
        flipped = torch.roll(pixels_cuda, shifts=PATCHES // 2, dims=0)
        negative = module(flipped).float()
    for name, tensor in (("module_bf16", module_bf16), ("module_bf16_repeat", module_repeat),
                         ("f32_conv", f0), ("f32_conv_input_cast", f1),
                         ("f32_conv_output_round", r0), ("f32_conv_input_output_round", r1),
                         ("linear_f32", linear_f32), ("linear_bf16", linear_bf16),
                         ("negative_shifted_input", negative)):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite output from {name}")

    outputs = {
        "module_bf16": cuda_array(module_bf16),
        "module_bf16_repeat": cuda_array(module_repeat),
        "f32_conv": cuda_array(f0),
        "f32_conv_input_cast": cuda_array(f1),
        "f32_conv_output_round": cuda_array(r0),
        "f32_conv_input_output_round": cuda_array(r1),
        "linear_f32": cuda_array(linear_f32),
        "linear_bf16": cuda_array(linear_bf16),
        "negative_shifted_input": cuda_array(negative),
    }
    REPORT["output_hashes"] = {name: digest_bytes(np.ascontiguousarray(value, dtype=np.float32).tobytes())
                               for name, value in outputs.items()}
    for name, value in outputs.items():
        np.ascontiguousarray(value, dtype="<f4").tofile(str(OUT / (name + ".f32")))
    REPORT["output_bf16_representable_fraction"] = {
        name: bf16_representable_fraction(value) for name, value in outputs.items()}

    save("compare")
    comparisons = {
        "module_bf16_vs_saved": metrics(outputs["module_bf16"], saved),
        "module_bf16_repeat_vs_first": metrics(outputs["module_bf16_repeat"], outputs["module_bf16"]),
        "f32_conv_vs_saved": metrics(outputs["f32_conv"], saved),
        "f32_conv_input_cast_vs_saved": metrics(outputs["f32_conv_input_cast"], saved),
        "f32_conv_output_round_vs_saved": metrics(outputs["f32_conv_output_round"], saved),
        "f32_conv_input_output_round_vs_saved": metrics(outputs["f32_conv_input_output_round"], saved),
        "module_bf16_vs_rounded_f32": metrics(outputs["module_bf16"], outputs["f32_conv_input_output_round"]),
        "linear_f32_vs_f32_conv": metrics(outputs["linear_f32"], outputs["f32_conv"]),
        "linear_bf16_vs_module_bf16": metrics(outputs["linear_bf16"], outputs["module_bf16"]),
        "negative_shifted_input_vs_saved": metrics(outputs["negative_shifted_input"], saved),
    }
    for metric in comparisons.values():
        metric["strict_pass"] = strict_pass(metric)
        metric["bf16_budget_pass"] = bf16_budget_pass(metric)

    native_cpu_path = find_optional("prefill.vision.patch_embed.f32", PATCHES * WIDTH * 4)
    native_cpu = None
    if native_cpu_path is not None:
        native_cpu = np.fromfile(native_cpu_path, dtype="<f4").reshape(PATCHES, WIDTH)
        if not np.isfinite(native_cpu).all():
            raise ValueError("native CPU dump contains non-finite values")
        comparisons["native_cpu_vs_f32_conv"] = metrics(native_cpu, outputs["f32_conv"])
        comparisons["native_cpu_vs_saved"] = metrics(native_cpu, saved)
        comparisons["native_cpu_vs_module_bf16"] = metrics(native_cpu, outputs["module_bf16"])
        for metric in comparisons.values():
            metric.setdefault("strict_pass", strict_pass(metric))
            metric.setdefault("bf16_budget_pass", bf16_budget_pass(metric))
    REPORT["native_cpu_dump"] = {"path": str(native_cpu_path) if native_cpu_path else None,
                                 "present": native_cpu is not None,
                                 "sha256": digest(native_cpu_path) if native_cpu_path else None,
                                 "authenticated_against_expected_sha256": False,
                                 "scope": "optional diagnostic only; filename and size do not authenticate provenance"}

    key = comparisons["module_bf16_vs_saved"]
    repeat = comparisons["module_bf16_repeat_vs_first"]
    f32_saved = comparisons["f32_conv_vs_saved"]
    negative = comparisons["negative_shifted_input_vs_saved"]
    conditions = {
        "weights_are_exact_bf16_widening": bool(weight_equal and bias_equal),
        "cast_input_is_exact_bf16_cast": bool(cast_equal),
        "saved_activation_fully_bf16_representable":
            REPORT["inputs"]["saved_bf16_representable_fraction"] == 1.0,
        "module_repeat_is_bit_identical": repeat["mismatch_count"] == 0,
        "module_bf16_reproduces_saved_exactly": reference_reproduced_exactly(key),
        "f32_conv_does_not_reproduce_saved": not strict_pass(f32_saved),
        "negative_control_fails_both_gates": (not strict_pass(negative)) and (not bf16_budget_pass(negative)),
    }
    confirmed = all(conditions.values())
    REPORT["conditions"] = conditions
    REPORT["key_comparison"] = "module_bf16_vs_saved"
    REPORT["strict_verdict"] = {
        "gate": {"max_abs": STRICT_MAX_ABS, "row_cos_min": STRICT_COS_MIN},
        "f32_conv_vs_saved_strict_pass": strict_pass(f32_saved),
        "statement": ("The strict gate is unchanged; the F32 Conv3D comparison is reported separately. "
                      "Exact reference reproduction does not establish a unique cause for the native gap."),
        "decoded_parity_claimed": False,
    }
    REPORT["attribution"] = {
        "confirmed": confirmed,
        "reference_compute_regime": ("BF16 Conv3D output stored as F32" if confirmed else "not confirmed"),
        "evidence": {
            "saved_bf16_representable_fraction": REPORT["inputs"]["saved_bf16_representable_fraction"],
            "module_bf16_exact_fraction": key["exact_fraction"],
            "module_bf16_rms": key["rms"],
            "module_bf16_max_abs": key["max_abs"],
            "f32_conv_max_abs": f32_saved["max_abs"],
            "f32_conv_rms": f32_saved["rms"],
        },
    }
    (OUT / "comparisons.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n")
    REPORT["comparisons_sha256"] = digest(OUT / "comparisons.json")

    if confirmed:
        REPORT["status"] = ("completed: saved first-stage activation reproduces exactly from the pinned BF16 "
                            "module; strict gate unchanged, cause not uniquely established, "
                            "and no decoded parity claimed")
        save("complete")
        return 0
    REPORT["status"] = "completed with an unconfirmed attribution; strict verdict unchanged"
    REPORT["error_type"] = "AttributionNotConfirmed"
    REPORT["error"] = json.dumps(conditions, sort_keys=True)
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
    print("PRECISION_REPLAY_RESULT", json.dumps(
        {"status": REPORT["status"], "confirmed": REPORT.get("attribution", {}).get("confirmed"),
         "exit_code": CODE}, sort_keys=True), flush=True)
    raise SystemExit(CODE)
