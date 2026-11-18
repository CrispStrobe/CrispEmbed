#!/usr/bin/env python3
"""Write a quantized derivative of the admitted de_core_news_sm 3.8.0 F32 GGUF.

Policy spacy-matrices-v1 (mirrored by src/spacy_pipeline.cpp): only the large
matrices (hash-embedding tables, mix/encoder maxout weights, tagger,
morphologizer, lemmatizer, parser reduce/lower) are stored below F32. Biases,
LayerNorm and the per-step parser tensors stay F32. Every row width in this
model (96, 576, 288, 64) is a multiple of 32 but not of 256, so 256-block
K-quants cannot represent any of these matrices; only 32-block types apply.

The source must be the pinned F32 export; the derivative records its SHA-256
and loads only with CRISPEMBED_SPACY_QUANT=1. Writing a file is not quality
qualification: run tests/spacy_ggml_ab.py against the original oracle.

    python3 models/quantize-spacy-gguf.py --input model.gguf --output q8.gguf --type q8_0
    python3 models/quantize-spacy-gguf.py --input model.gguf --output mix.gguf --type q8_0 \
        --override 'embed.*.E=q4_0'
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
from pathlib import Path
import sys

import numpy as np
import gguf
from gguf import quants

F32_SHA256 = "fae2e6ab8d6c611d7a9dddd4b3259a871a1d2f95cacc8259f49c1e4ff31ff6cd"
POLICY = "spacy-matrices-v1"
TYPES = {"f32": gguf.GGMLQuantizationType.F32, "f16": gguf.GGMLQuantizationType.F16,
         "bf16": gguf.GGMLQuantizationType.BF16,
         "q8_0": gguf.GGMLQuantizationType.Q8_0, "q5_0": gguf.GGMLQuantizationType.Q5_0,
         "q4_0": gguf.GGMLQuantizationType.Q4_0}
MATRICES = ([f"embed.{i}.E" for i in range(6)] + ["embed.mix.W"] +
            [f"encoder.{i}.maxout.W" for i in range(4)] +
            ["tagger.W", "morphologizer.W", "lemmatizer.W", "parser.reduce.W", "parser.lower.W"])


def field_value(field):
    kind = field.types[0]
    if kind == gguf.GGUFValueType.STRING:
        return kind, bytes(field.parts[field.data[0]]).decode("utf-8")
    return kind, field.parts[field.data[0]][0].item()


def plan(base: str, overrides: list[str]) -> dict[str, str]:
    types = {name: base for name in MATRICES}
    for rule in overrides:
        pattern, _, kind = rule.partition("=")
        if kind not in TYPES:
            raise SystemExit(f"Unknown type in override {rule!r}")
        matched = [name for name in MATRICES if fnmatch.fnmatchcase(name, pattern)]
        if not matched:
            raise SystemExit(f"Override {rule!r} matches no policy matrix")
        for name in matched:
            types[name] = kind
    return types


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Admitted F32 model.gguf")
    parser.add_argument("--output", type=Path, required=True, help="New GGUF path (must not exist)")
    parser.add_argument("--type", choices=sorted(set(TYPES) - {"f32"}), required=True)
    parser.add_argument("--override", action="append", default=[], help="GLOB=TYPE for policy matrices")
    args = parser.parse_args()
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != F32_SHA256:
        raise SystemExit("Input is not the admitted de_core_news_sm 3.8.0 F32 GGUF")
    if args.output.exists():
        raise SystemExit("Output already exists")
    types = plan(args.type, args.override)
    used = {kind for kind in types.values()}
    precision = args.type if used == {args.type} else "mixed"
    if used == {"f32"}:
        raise SystemExit("Plan quantizes no matrix")

    reader = gguf.GGUFReader(args.input)
    architecture = field_value(reader.fields["general.architecture"])[1]
    writer = gguf.GGUFWriter(args.output, architecture)
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        kind, value = field_value(field)
        if key == "spacy.precision":
            value = precision
        writer.add_key_value(key, value, kind)
    writer.add_string("spacy.quantization.policy", POLICY)
    writer.add_string("spacy.quantization.source_gguf_sha256", F32_SHA256)
    for tensor in reader.tensors:
        data = np.array(tensor.data, dtype=np.float32, copy=True)
        kind = TYPES[types.get(tensor.name, "f32")]
        if kind == gguf.GGMLQuantizationType.F32:
            writer.add_tensor(tensor.name, data)
        else:
            writer.add_tensor(tensor.name, quants.quantize(data, kind), raw_dtype=kind)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{args.output}: precision {precision}, {args.output.stat().st_size} bytes, SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
