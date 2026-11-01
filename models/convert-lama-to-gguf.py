#!/usr/bin/env python3
"""Convert a user-supplied Big-LaMa checkpoint losslessly to experimental F32.

The standard-library ZIP reader accepts inert FloatStorage/LongStorage tensor
metadata only; it never imports PyTorch or executes checkpoint pickle globals.
YAML configuration files need PyYAML (JSON configuration is dependency-free).
Checkpoint weights are not downloaded or redistributed by this tool.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import struct
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from lama_reference_contract import (PROFILE, SOURCE_COMMIT, expected_state_shapes,
                                     extract_tensors, load_config_with_sha256, model_metadata,
                                     read_f32_gguf, sha256, write_f32_gguf)


def convert(checkpoint, checkpoint_sha256, config, output):
    checkpoint, config, output = Path(checkpoint), Path(config), Path(output)
    receipt_path = output.with_suffix(".json")
    if output.suffix.lower() != ".gguf":
        raise ValueError("--output must end in .gguf")
    if any(path.exists() or path.is_symlink() for path in (output, receipt_path)):
        raise ValueError("refuse to overwrite an existing model or conversion receipt")
    if not output.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    profile, config_sha256 = load_config_with_sha256(config)
    checkpoint_provenance = {}
    tensors, counters = extract_tensors(checkpoint, checkpoint_sha256, provenance=checkpoint_provenance)
    expected = expected_state_shapes()
    if {name: shape for name, shape, _ in tensors} != expected:
        raise AssertionError("exported tensor schema differs from the supported profile")
    metadata = model_metadata(checkpoint_sha256, config_sha256)
    with tempfile.TemporaryDirectory(prefix=".lama-convert-", dir=output.parent) as temporary:
        stage = Path(temporary)
        model = stage / "model.gguf"
        write_f32_gguf(model, tensors, metadata)
        actual_metadata, actual = read_f32_gguf(model)
        if actual_metadata != metadata or len(actual) != len(tensors):
            raise AssertionError("GGUF readback metadata or tensor count differs")
        rows = []
        for name, shape, raw in tensors:
            if actual[name] != (shape, raw):
                raise AssertionError(f"GGUF readback bytes/shape differ: {name}")
            rows.append({"name": name, "shape": list(shape), "dtype": "float32",
                         "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        receipt = {
            "schema": "crispembed-lama-conversion-v1", "profile": PROFILE,
            "source_commit": SOURCE_COMMIT, "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": checkpoint_provenance["checkpoint_bytes"], "config_sha256": config_sha256,
            "generator_config": profile, "gguf_sha256": sha256(model),
            "gguf_bytes": model.stat().st_size, "tensor_count": len(tensors),
            "all_tensor_bytes_exact": True, "tensors": rows,
            "omitted_inference_inert_counters": counters,
            "checkpoint_metadata": checkpoint_provenance,
            "framework_imported": False, "original_reference_captured": False,
            "native_qualified": False, "representative_quality_qualified": False,
            "checkpoint_license_verified": False, "weights_redistributed": False,
            "converter_sha256": sha256(__file__),
            "contract_sha256": sha256(Path(__file__).resolve().parents[1] / "tools/lama_reference_contract.py"),
        }
        published = []
        try:
            # Exclusive creation, including the receipt, never overwrites files
            # that appeared after the initial preflight.
            with receipt_path.open("x", encoding="utf-8") as stream:
                published.append(receipt_path)
                stream.write(json.dumps(receipt, indent=2) + "\n")
            with output.open("xb") as target, model.open("rb") as source:
                published.append(output)
                shutil.copyfileobj(source, target, 1024 * 1024)
        except BaseException:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            raise
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True,
                        help="independently recorded SHA-256 of the local checkpoint")
    parser.add_argument("--config", type=Path, required=True,
                        help="original Big-LaMa config.yaml or matching JSON config")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = convert(args.checkpoint, args.checkpoint_sha256, args.config, args.output)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError, zipfile.BadZipFile, struct.error) as exc:
        parser.exit(1, f"LaMa conversion failed: {exc}\n")
    print(json.dumps({"model": str(args.output), "receipt": str(args.output.with_suffix('.json')),
                      "gguf_sha256": receipt["gguf_sha256"], "tensor_count": receipt["tensor_count"]}, indent=2))


if __name__ == "__main__":
    main()
