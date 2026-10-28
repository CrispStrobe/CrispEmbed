#!/usr/bin/env python3
"""Export openbmb/MiniCPM-OCR-1.0 for the experimental native `minicpm_ocr` backend.

Consumer: src/minicpm_ocr.cpp (default-off, CRISPEMBED_MINICPM_OCR=1 plus the
reused qwen35 decoder gate CRISPEMBED_QWEN35_OCR=1). Not parity-qualified; see
docs/minicpm-ocr-inference-spec.md. Every tensor in the pinned checkpoint is
mapped or the export fails; nothing is dropped or defaulted. `vit_merger`
(16x mode only) is kept although the 4x default does not use it.

dtype policy:
  f32   every tensor F32 (parity reference export)
  f16   matrices F16; norms, biases, 1-D vectors, conv1d, A_log, dt_bias F32
  bf16  vision/merger matrices BF16 (lossless vs. the BF16 checkpoint); text
        matrices F16 because the reused qwen35 CPU kernels accept F32/F16 only
Run on a hosted runner, never on a constrained host.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import struct
from pathlib import Path

ARCH = "minicpm_ocr"
FORMAT_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/minicpm-ocr-source.json"
N_TEXT_LAYERS, N_VISION_LAYERS = 24, 27
EOS_IDS = [248044, 248046]

# Audited tensor families (safetensors header at the pinned revision, 779 tensors).
PATTERNS = [
    (r"model\.language_model\.(.+)", "l.{0}"),
    (r"model\.vision_tower\.embeddings\.(.+)", "v.emb.{0}"),
    (r"model\.vision_tower\.encoder\.layers\.(\d+)\.(.+)", "v.blk.{0}.{1}"),
    (r"model\.vision_tower\.post_layernorm\.(.+)", "v.post_ln.{0}"),
    (r"model\.vision_tower\.vit_merger\.(.+)", "v.vmerge.{0}"),
    (r"model\.merger\.mlp\.(\d+)\.(.+)", "m.{0}.{1}"),
]
EXPECTED_SHAPES = {
    "model.language_model.embed_tokens.weight": [248100, 1024],
    "model.vision_tower.embeddings.position_embedding.weight": [4900, 1152],
    "model.vision_tower.embeddings.patch_embedding.weight": [1152, 3, 14, 14],
    "model.merger.mlp.0.linear_1.weight": [4608, 4608],
    "model.merger.mlp.0.linear_2.weight": [1024, 4608],
    "model.merger.mlp.0.pre_norm.weight": [4608],
    "model.merger.mlp.0.pre_norm.bias": [4608],
}


def _qwen35():
    spec = importlib.util.spec_from_file_location("q35conv", ROOT / "models/convert-qwen35-ocr-to-gguf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tensor_name(name):
    for pattern, fmt in PATTERNS:
        m = re.fullmatch(pattern, name)
        if m:
            out = fmt.format(*m.groups())
            if len(out.encode()) >= 64:
                raise ValueError(f"tensor name exceeds ggml limit: {out}")
            return out
    raise ValueError(f"unaudited checkpoint tensor: {name}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def validate(root, contract):
    for name, pin in contract["files"].items():
        p = root / name
        if name in (".gitattributes", "README.md") and not p.exists():
            continue
        if p.stat().st_size != pin["bytes"] or ("sha256" in pin and sha256(p) != pin["sha256"]):
            raise ValueError(f"pinned source file differs: {name}")
    cfg = json.loads((root / "config.json").read_text())
    pre = json.loads((root / "preprocessor_config.json").read_text())
    t, v = cfg["text_config"], cfg["vision_config"]
    checks = {
        "model_type": cfg["model_type"] == "minicpm_ocr",
        "downsample_mode": cfg.get("downsample_mode") == "4x",
        "tied": bool(cfg.get("tie_word_embeddings", t.get("tie_word_embeddings"))),
        "text_type": t["model_type"] == "qwen3_5_text",
        "text_layers": t["num_hidden_layers"] == N_TEXT_LAYERS,
        "text_width": t["hidden_size"] == 1024,
        "vocab": t["vocab_size"] == 248100,
        "vision_layers": v["num_hidden_layers"] == N_VISION_LAYERS,
        "vision_width": v["hidden_size"] == 1152,
        "patch": v["patch_size"] == 14,
        "no_mrope": "mrope_section" not in json.dumps(t),
        "scale_resolution": pre["scale_resolution"] == 448,
        "max_slice_nums": pre["max_slice_nums"] == 9,
        "patch_size": pre.get("patch_size", 14) == 14,
        "mean": list(pre.get("image_mean", [0.5] * 3)) == [0.5] * 3,
        "std": list(pre.get("image_std", [0.5] * 3)) == [0.5] * 3,
    }
    if not all(checks.values()):
        raise ValueError(f"config differs from the audited layout: {[k for k, ok in checks.items() if not ok]}")
    return cfg, pre


def rope_theta(text):
    rope = text.get("rope_parameters") or {}
    return float(rope.get("rope_theta", text.get("rope_theta", 1e7)))


def write_tokenizer(w, tok):
    import gguf
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "qwen35")
    w.add_string(f"{ARCH}.normalizer", tok["normalizer"])
    w.add_string(f"{ARCH}.pretokenizer_pattern", tok["pattern"])
    w.add_array("tokenizer.ggml.tokens", tok["tokens"])
    w.add_array("tokenizer.ggml.merges", tok["merges"])
    for key in ("added", "special"):
        w.add_key_value(f"{ARCH}.{key}_token_ids", tok[key], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    w.add_uint32("tokenizer.ggml.eos_token_id", tok["tokenizer_eos"])
    w.add_uint32("tokenizer.ggml.padding_token_id", tok["pad"])


def target_type(name, src_name, ndim, dtype):
    small = (ndim < 2 or "norm" in name or "A_log" in name or "dt_bias" in name or "conv1d" in name
             or name.endswith(".bias"))
    if dtype == "f32" or small:
        return "f32"
    if dtype == "bf16" and not src_name.startswith("model.language_model."):
        return "bf16"
    return "f16"



class SafetensorsReader:
    """Torch-free safetensors reader; BF16/F16 widen exactly to float32 (same values as torch .float())."""

    def __init__(self, fh):
        n = struct.unpack("<Q", fh.read(8))[0]
        self.header = {k: v for k, v in json.loads(fh.read(n)).items() if k != "__metadata__"}
        self.base, self.fh = 8 + n, fh

    def keys(self):
        return self.header.keys()

    def get_f32(self, name):
        import numpy as np

        info = self.header[name]
        lo, hi = info["data_offsets"]
        self.fh.seek(self.base + lo)
        raw = self.fh.read(hi - lo)
        dt = info["dtype"]
        if dt == "BF16":
            a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F16":
            a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "F32":
            a = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            raise ValueError(f"unsupported safetensors dtype {dt}: {name}")
        return a.reshape(info["shape"])


def convert(root, output, dtype):
    import gguf
    import numpy as np
    contract = json.loads(CONTRACT.read_text())
    cfg, pre = validate(root, contract)
    t, v = cfg["text_config"], cfg["vision_config"]
    q35 = _qwen35()
    tok = q35.read_tokenizer_metadata(root, t["vocab_size"], allow_generation_eos_list=True)
    gen = json.loads((root / "generation_config.json").read_text())
    if gen.get("eos_token_id") != EOS_IDS:
        raise ValueError(f"generation EOS differs: {gen.get('eos_token_id')}")

    output = Path(output)
    pending = output.with_suffix(output.suffix + ".partial")
    w = gguf.GGUFWriter(str(pending), ARCH)
    w.add_string("general.name", contract["repository"])
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.source.huggingface.repository", contract["repository"])
    w.add_string("general.source.huggingface.revision", contract["revision"])
    w.add_string(f"{ARCH}.status", "experimental-native-unqualified")
    w.add_uint32(f"{ARCH}.format_version", FORMAT_VERSION)
    w.add_string(f"{ARCH}.export_dtype", dtype)
    w.add_string(f"{ARCH}.text_matrix_dtype", "f32" if dtype == "f32" else "f16")
    w.add_string(f"{ARCH}.checkpoint_sha256", contract["files"]["model.safetensors"]["sha256"])
    w.add_string(f"{ARCH}.config", (root / "config.json").read_text())
    w.add_string(f"{ARCH}.preprocessor_config", (root / "preprocessor_config.json").read_text())
    if (root / "processor_config.json").exists():
        w.add_string(f"{ARCH}.processor_config", (root / "processor_config.json").read_text())
    w.add_string(f"{ARCH}.downsample_mode", cfg["downsample_mode"])
    w.add_uint32(f"{ARCH}.insert_layer_id", cfg["insert_layer_id"])
    w.add_uint32(f"{ARCH}.image_token_id", cfg["image_token_id"])
    w.add_bool(f"{ARCH}.tie_word_embeddings", True)
    w.add_bool(f"{ARCH}.zero_centered_text_norm", True)
    w.add_key_value(f"{ARCH}.source_eos_token_ids", EOS_IDS, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    # Slicing / preprocessing (minicpm_ocr_tiles.h validates these).
    w.add_uint32(f"{ARCH}.scale_resolution", pre["scale_resolution"])
    w.add_uint32(f"{ARCH}.max_slice_nums", pre["max_slice_nums"])
    w.add_uint32(f"{ARCH}.patch_size", pre.get("patch_size", 14))
    w.add_bool(f"{ARCH}.slice_mode", bool(pre.get("slice_mode", True)))
    w.add_bool(f"{ARCH}.use_image_id", bool(pre.get("use_image_id", True)))
    w.add_string(f"{ARCH}.small_image_upsize", str(pre.get("small_image_upsize", "always")))
    w.add_uint32(f"{ARCH}.merge_factor", 2)
    w.add_uint32(f"{ARCH}.position_grid", 70)
    # Model hyperparameters consumed by the runtime.
    w.add_float32(f"{ARCH}.rope_theta", rope_theta(t))
    w.add_float32(f"{ARCH}.text_rms_eps", float(t.get("rms_norm_eps", 1e-6)))
    w.add_float32(f"{ARCH}.vision_ln_eps", float(v.get("layer_norm_eps", 1e-6)))
    # MiniCPMOcrDownsampleMLP.pre_norm = nn.LayerNorm(4608, eps=1e-6): a literal in
    # transformers 5.21.0 modeling_minicpm_ocr.py:500, not a config field.
    w.add_float32(f"{ARCH}.merger_ln_eps", 1e-6)
    w.add_string("tokenizer.chat_template", (root / "chat_template.jinja").read_text())
    write_tokenizer(w, tok)

    records = []
    with open(root / "model.safetensors", "rb") as fh:
        f = SafetensorsReader(fh)
        for src in sorted(f.keys()):
            a = f.get_f32(src)
            if src in EXPECTED_SHAPES and list(a.shape) != EXPECTED_SHAPES[src]:
                raise ValueError(f"shape differs: {src} {list(a.shape)}")
            name = tensor_name(src)
            if not np.isfinite(a).all():
                raise ValueError(f"non-finite source tensor: {src}")
            kind = target_type(name, src, a.ndim, dtype)
            if kind == "bf16":
                w.add_tensor(name, gguf.quants.quantize(a, gguf.GGMLQuantizationType.BF16),
                             raw_dtype=gguf.GGMLQuantizationType.BF16)
            else:
                arr = np.ascontiguousarray(a, dtype=np.float32 if kind == "f32" else np.float16)
                if not np.isfinite(arr).all():
                    raise ValueError(f"overflowing {kind} export: {src}")
                w.add_tensor(name, arr)
            records.append({"source": src, "name": name, "shape": list(a.shape), "dtype": kind})
    if len(records) != contract["parameters"]["tensors"]:
        raise ValueError(f"tensor count {len(records)} != pinned {contract['parameters']['tensors']}")
    try:
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
    finally:
        w.close()
    pending.replace(output)
    manifest = {"architecture": ARCH, "format_version": FORMAT_VERSION, "dtype": dtype,
                "source": contract["repository"], "revision": contract["revision"],
                "tensors": len(records), "bytes": output.stat().st_size, "records": records}
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def fetch(directory):
    from huggingface_hub import hf_hub_download
    contract = json.loads(CONTRACT.read_text())
    for name in contract["files"]:
        if name in (".gitattributes", "README.md"):
            continue
        hf_hub_download(contract["repository"], name, revision=contract["revision"], local_dir=directory)
    return Path(directory)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_dir", type=Path, help="local snapshot of the pinned revision (see --download)")
    p.add_argument("output", type=Path)
    p.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="f16")
    p.add_argument("--download", action="store_true", help="fetch the pinned files into model_dir first")
    a = p.parse_args()
    root = fetch(a.model_dir) if a.download else a.model_dir
    m = convert(root, a.output, a.dtype)
    print(json.dumps({k: val for k, val in m.items() if k != "records"}, indent=1))


if __name__ == "__main__":
    main()
