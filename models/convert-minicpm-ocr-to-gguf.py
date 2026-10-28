#!/usr/bin/env python3
"""Export openbmb/MiniCPM-OCR-1.0 for the pending native `minicpm_ocr` backend.

SKELETON (2026-10-28): experimental weight format; no native runtime consumes it
yet. See docs/minicpm-ocr-inference-spec.md. Every tensor in the pinned
checkpoint is mapped or the export fails; nothing is dropped or defaulted.
`vit_merger` (16x mode only) is kept although the 4x default does not use it.
Run on a hosted runner, never on a constrained host.
"""

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ARCH = "minicpm_ocr"
ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/minicpm-ocr-source.json"
N_TEXT_LAYERS, N_VISION_LAYERS = 24, 27

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
}


def _qwen35():
    spec = importlib.util.spec_from_file_location("q35conv", ROOT / "models/convert-qwen35-ocr-to-gguf.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
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
        while b := f.read(8 << 20): h.update(b)
    return h.hexdigest()


def validate(root, contract):
    for name, pin in contract["files"].items():
        p = root / name
        if name == ".gitattributes" and not p.exists(): continue
        if p.stat().st_size != pin["bytes"] or ("sha256" in pin and sha256(p) != pin["sha256"]):
            raise ValueError(f"pinned source file differs: {name}")
    cfg = json.loads((root / "config.json").read_text())
    pre = json.loads((root / "preprocessor_config.json").read_text())
    t, v = cfg["text_config"], cfg["vision_config"]
    checks = [cfg["model_type"] == "minicpm_ocr", cfg["downsample_mode"] == "4x", cfg["tie_word_embeddings"],
              t["model_type"] == "qwen3_5_text", t["num_hidden_layers"] == N_TEXT_LAYERS,
              v["num_hidden_layers"] == N_VISION_LAYERS, v["hidden_size"] == 1152, v["patch_size"] == 14,
              "mrope_section" not in json.dumps(t), pre["scale_resolution"] == 448, pre["max_slice_nums"] == 9]
    if not all(checks):
        raise ValueError(f"config differs from the audited layout: {checks}")
    return cfg, pre


def convert(root, output, dtype):
    import gguf
    import numpy as np
    from safetensors import safe_open
    contract = json.loads(CONTRACT.read_text())
    cfg, pre = validate(root, contract)
    q35 = _qwen35()
    tok = q35.read_tokenizer_metadata(root, cfg["text_config"]["vocab_size"], allow_generation_eos_list=True)

    w = gguf.GGUFWriter(str(output), ARCH)
    w.add_string("general.name", contract["repository"])
    w.add_string("general.license", "apache-2.0")
    w.add_string("general.source.huggingface.repository", contract["repository"])
    w.add_string("general.source.huggingface.revision", contract["revision"])
    w.add_string(f"{ARCH}.status", "experimental-export-native-backend-pending")
    w.add_uint32(f"{ARCH}.format_version", 1)
    w.add_string(f"{ARCH}.checkpoint_sha256", contract["files"]["model.safetensors"]["sha256"])
    w.add_string(f"{ARCH}.config", (root / "config.json").read_text())
    w.add_string(f"{ARCH}.preprocessor_config", (root / "preprocessor_config.json").read_text())
    w.add_string(f"{ARCH}.downsample_mode", cfg["downsample_mode"])
    w.add_uint32(f"{ARCH}.insert_layer_id", cfg["insert_layer_id"])
    w.add_uint32(f"{ARCH}.image_token_id", cfg["image_token_id"])
    w.add_bool(f"{ARCH}.tie_word_embeddings", True)
    w.add_bool(f"{ARCH}.zero_centered_text_norm", True)
    w.add_key_value(f"{ARCH}.source_eos_token_ids", [248044, 248046],
                    gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT32)
    w.add_string("tokenizer.chat_template", (root / "chat_template.jinja").read_text())
    q35.write_tokenizer_metadata(w, tok)  # TODO: emits qwen35_ocr.* keys; re-key under ARCH when the runtime lands

    seen = 0
    with safe_open(root / "model.safetensors", framework="pt") as f:
        for name in sorted(f.keys()):
            t = f.get_tensor(name).float()
            if name in EXPECTED_SHAPES and list(t.shape) != EXPECTED_SHAPES[name]:
                raise ValueError(f"shape differs: {name} {list(t.shape)}")
            a = t.numpy()
            # norms, biases, small vectors and delta-net scalars stay F32
            keep32 = dtype == "f32" or a.ndim < 2 or "A_log" in name or "dt_bias" in name or "conv1d" in name
            w.add_tensor(tensor_name(name), a.astype(np.float32 if keep32 else np.float16))
            seen += 1
    if seen != contract["parameters"]["tensors"]:
        raise ValueError(f"tensor count {seen} != pinned {contract['parameters']['tensors']}")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    a = p.parse_args()
    convert(a.model_dir, a.output, a.dtype)


if __name__ == "__main__":
    main()
