#!/usr/bin/env python3
"""Hosted-only conversion/reference audit; no registry, upload or runtime changes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from ultrabert_contract import MODEL, REVISION, expected_tensors, profile, sha256


def main():
    subprocess.run([sys.executable, "tests/test_ultrabert_contract.py", "-v"], cwd=ROOT, check=True)
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import gguf
    import numpy as np

    output = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "ultrabert"
    scratch.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    # Original code is hash-checked before AutoModel is allowed to execute it.
    directory = snapshot_download(MODEL, revision=REVISION,
                                  allow_patterns=list(profile()["inspected_file_sha256"]) + ["model.safetensors"])
    spec = importlib.util.spec_from_file_location("ultrabert_converter", ROOT / "models/convert-ultrabert-to-gguf.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    report = {"source_commit": os.environ["GITHUB_SHA"], "model": MODEL,
              "revision": REVISION, "driver_sha256": sha256(__file__),
              "tool_sha256": {p: sha256(ROOT / p) for p in (
                  "models/convert-ultrabert-to-gguf.py", "tools/ultrabert_contract.py",
                  "tools/dump_ultrabert_reference.py", "tests/test_ultrabert_contract.py")},
              "conversions": [], "native_parity": "not evaluated",
              "status": "started"}
    report_path = output / "ultrabert-reference-audit.json"
    try:
        for dtype in ("f32", "f16"):
            path = scratch / (dtype + ".gguf")
            converted = converter.convert(directory, path, dtype)
            reader = gguf.GGUFReader(str(path))
            actual = {tensor.name: tensor for tensor in reader.tensors}
            expected = expected_tensors()
            metadata = {
                "general.architecture": "ultrabert",
                "general.license": "apache-2.0",
                "general.file_type": 0 if dtype == "f32" else 1,
                "tokenizer.ggml.pre": "gpt2",
                "ultrabert.source.revision": REVISION,
                "ultrabert.source.weight_sha256": converted["source_weights"],
                "ultrabert.attention.causal": False,
                "ultrabert.attention.window_radius": [128, 128, -1] * 8,
                "ultrabert.rope.freq_base_per_layer": [10000., 10000., 2560000.] * 8,
                "ultrabert.pooling": "cls",
                "ultrabert.pooling.normalize_default": False,
                "tokenizer.ggml.bos_token_id": 0,
                "tokenizer.ggml.eos_token_id": 0,
                "tokenizer.ggml.padding_token_id": 0,
                "tokenizer.ggml.separator_token_id": 15,
                "tokenizer.ggml.sep_token_id": 15,
                "tokenizer.ggml.add_bos_token": True,
                "tokenizer.ggml.add_eos_token": False,
                "ultrabert.tokenizer.json": (Path(directory) / "tokenizer.json").read_text(),
            }
            for key, value in metadata.items():
                if reader.fields[key].contents() != value:
                    raise ValueError("GGUF metadata round-trip differs: " + key)
            if set(actual) != {target for target, shape in expected.values()}:
                raise ValueError("Converted GGUF drops/adds tensors")
            with safe_open(str(Path(directory) / "model.safetensors"), framework="np") as weights:
                for source, (target, shape) in expected.items():
                    array = weights.get_tensor(source)
                    precision = np.float32 if dtype == "f32" or len(shape) == 1 else np.float16
                    tensor = actual[target]
                    expected_type = gguf.GGMLQuantizationType.F32 if precision == np.float32 else gguf.GGMLQuantizationType.F16
                    if (list(reversed(tensor.shape.tolist())) != shape
                            or tensor.tensor_type != expected_type
                            or tensor.data.dtype != np.dtype(precision)
                            or tensor.data.tobytes() != array.astype(precision).tobytes()):
                        raise ValueError("GGUF tensor bit/type round-trip differs: " + source)
            converted["all_218_roundtrips_exact"] = True
            converted["metadata_roundtrips_exact"] = True
            report["conversions"].append(converted)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            del actual, reader
        from dump_ultrabert_reference import dump
        reference = dump(directory, output / "reference")
        # Verify archive schemas, all values finite and all recorded magnitudes;
        # the same named F32 tensors feed the future crispembed-diff harness.
        for case in reference["cases"]:
            reader = gguf.GGUFReader(str(output / "reference" / case["archive"]))
            tensors = {tensor.name: tensor for tensor in reader.tensors}
            if set(tensors) != set(case["stages"]):
                raise ValueError("Incomplete reference archive")
            for name, values in case["stages"].items():
                tensor = tensors[name]
                if (list(reversed(tensor.shape.tolist())) != values["shape"]
                        or tensor.tensor_type != gguf.GGMLQuantizationType.F32
                        or not np.isfinite(tensor.data).all()
                        or hashlib.sha256(tensor.data.tobytes()).hexdigest() != values["payload_sha256"]):
                    raise ValueError("Reference round-trip shape/type/payload failure: " + name)
            case["archive_schema_exact"] = True
        (output / "reference/reference.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n")
        report["reference"] = reference
        report["status"] = "completed conversion/reference audit; native parity not evaluated"
    finally:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (output / "summary.md").write_text(
        "UltraBERT original F32 eager source: " + REVISION + "\n\n"
        "Both conversion archives preserve all218 tensors exactly at their declared storage precision. "
        "All24 layers, local/global masks, BOS/padding, raw CLS and optional L2 reference arrays retained. "
        "Original hooked/unhooked outputs are identical. No native runtime/parity, retrieval-quality, "
        "long-context, quantization or publication claim.\n")


if __name__ == "__main__":
    main()
