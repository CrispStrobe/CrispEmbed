#!/usr/bin/env python3
"""Build a mixed-precision Tesseract GGUF from compatible F32 and Q8 files.

The base file supplies metadata and the fast quantized tensors.  Matching
tensors from the precision file are copied byte-for-byte when their names
match one of the requested fnmatch patterns.  This keeps the experiment
reproducible and avoids silently re-quantizing a critical recurrent matrix.

Large outputs should be written to the configured external model store, not
committed to the repository.  Example::

  python models/mix-tesseract-gguf.py \
    --base tesseract-frk-q8_0.gguf \
    --precision tesseract-frk-f32.gguf \
    --output tesseract-frk-mixed-lstm3-f32.gguf \
    --pattern 'lstm.3.*'
"""

import argparse
import fnmatch
import hashlib
import os
import sys
from pathlib import Path

try:
    from gguf_merge_core import (
        GGUF_TYPE_ARRAY,
        GGUF_TYPE_INT32,
        GGUF_TYPE_STRING,
        GGUF_TYPE_UINT32,
        read_gguf,
        write_combined_gguf,
    )
except ModuleNotFoundError:  # also support direct import from a test runner
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gguf_merge_core import (
        GGUF_TYPE_ARRAY,
        GGUF_TYPE_INT32,
        GGUF_TYPE_STRING,
        GGUF_TYPE_UINT32,
        read_gguf,
        write_combined_gguf,
    )


def selected_names(names, patterns):
    """Return names matching patterns, preserving GGUF tensor order."""
    return [name for name in names
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_value(value_type, value):
    """Convert the reader's convenient array value to the raw writer form."""
    if value_type != GGUF_TYPE_ARRAY:
        return value
    if not value:
        return (GGUF_TYPE_UINT32, [])
    if isinstance(value[0], str):
        element_type = GGUF_TYPE_STRING
    else:
        # Recoder arrays are signed i32; treating negative entries as u32
        # corrupts otherwise byte-preserving metadata-only repairs.
        element_type = GGUF_TYPE_INT32 if any(v < 0 for v in value) else GGUF_TYPE_UINT32
    return (element_type, value)


def build_mixed(base_path, precision_path, output_path, patterns,
                metadata_only=False):
    base = read_gguf(base_path)
    precision = read_gguf(precision_path)
    if base.metadata.get("general.architecture") != precision.metadata.get(
            "general.architecture"):
        raise ValueError("base and precision files have different architectures")

    base_by_name = {tensor.name: tensor for tensor in base.tensors}
    precision_by_name = {tensor.name: tensor for tensor in precision.tensors}
    selected = selected_names(base_by_name, patterns)
    if not selected and not metadata_only:
        raise ValueError("patterns matched no tensors")

    output_tensors = []
    for base_tensor in base.tensors:
        source = base
        info = base_tensor
        if base_tensor.name in selected:
            precision_tensor = precision_by_name.get(base_tensor.name)
            if precision_tensor is None:
                raise ValueError(f"precision file lacks {base_tensor.name}")
            if (precision_tensor.shape != base_tensor.shape or
                    precision_tensor.dtype != 0):
                raise ValueError(
                    f"{base_tensor.name}: expected matching F32 tensor, "
                    f"got shape={precision_tensor.shape} dtype={precision_tensor.dtype}")
            source = precision
            info = precision_tensor
        output_tensors.append((base_tensor.name, info, source))

    metadata = [(key, value_type, metadata_value(value_type, value))
                for key, value in base.metadata.items()
                for value_type in [base.metadata_types[key]]]
    # Older quantized Tesseract artifacts may predate the serialized seed
    # metadata. Preserve model-contract KVs from the precision source so
    # seeded Convolve padding and input geometry remain reproducible.
    for key, value in precision.metadata.items():
        if key.startswith("tesseract_lstm.") and key not in base.metadata:
            metadata.append((key, precision.metadata_types[key], metadata_value(precision.metadata_types[key], value)))
    metadata.extend([
        ("tesseract_lstm.mixed_precision", GGUF_TYPE_STRING,
         "base=Q8_0;selected=F32"),
        ("tesseract_lstm.mixed_precision_patterns", GGUF_TYPE_ARRAY,
         (GGUF_TYPE_STRING, list(patterns))),
        ("tesseract_lstm.mixed_precision_base_sha256", GGUF_TYPE_STRING,
         sha256(base_path)),
        ("tesseract_lstm.mixed_precision_precision_sha256", GGUF_TYPE_STRING,
         sha256(precision_path)),
    ])
    # The raw writer needs the architecture KV and all original metadata;
    # duplicate keys would make readers disagree about the active value.
    deduped = {}
    for item in metadata:
        deduped[item[0]] = item
    size = write_combined_gguf(output_path, output_tensors,
                               list(deduped.values()))
    print(f"selected ({len(selected)}): {', '.join(selected)}")
    print(f"wrote {output_path}: {size / 1e6:.2f} MB")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", action="append", default=[],
                        help="fnmatch tensor pattern; repeat for multiple patterns")
    parser.add_argument("--metadata-only", action="store_true",
                        help="copy missing Tesseract contract metadata without replacing tensors")
    args = parser.parse_args()
    try:
        build_mixed(args.base, args.precision, args.output, args.pattern,
                    metadata_only=args.metadata_only)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
