#!/usr/bin/env python3
"""Export the de_core_news_sm 3.8.0 NER component as a separate ner.gguf.

Developer tooling only. The accepted model.gguf (tok2vec, tagger,
morphologizer, parser, lemmatizer, attribute_ruler) is unchanged; NER has its
own independent tok2vec (NORM, PREFIX, SUFFIX, SHAPE; width 96, depth 4) and a
TransitionBasedParser (state_type ner, nF=3, hidden 64, maxout 2, upper) whose
18 classes are written in source order. The native runtime loads this file only
through crispembed_spacy_attach_ner behind CRISPEMBED_SPACY_NER.

    python models/convert-spacy-ner-to-gguf.py --wheel de_core_news_sm-3.8.0-py3-none-any.whl \
        --output ner-bundle/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spacy_bundle as base  # noqa: E402

ARCHITECTURE = "spacy_de_core_news_sm_ner"
FORMAT_VERSION = 1
FEATURES = ["NORM", "PREFIX", "SUFFIX", "SHAPE"]
ROWS = [5000, 1000, 2500, 2500]
CLASSES = ["B-LOC", "B-PER", "B-ORG", "B-MISC", "I-LOC", "I-PER", "I-ORG", "I-MISC",
           "L-LOC", "L-PER", "L-ORG", "L-MISC", "U-LOC", "U-PER", "U-ORG", "U-MISC", "U-", "O"]


def expected_tensors() -> dict[str, tuple[int, ...]]:
    result = {f"ner.embed.{i}.E": (rows, 96) for i, rows in enumerate(ROWS)}
    result.update({"ner.embed.mix.W": (96, 3, 384), "ner.embed.mix.b": (96, 3),
                   "ner.embed.norm.G": (96,), "ner.embed.norm.b": (96,)})
    for i in range(4):
        result.update({f"ner.encoder.{i}.maxout.W": (96, 3, 288), f"ner.encoder.{i}.maxout.b": (96, 3),
                       f"ner.encoder.{i}.norm.G": (96,), f"ner.encoder.{i}.norm.b": (96,)})
    result.update({"ner.reduce.W": (64, 96), "ner.reduce.b": (64,),
                   "ner.lower.W": (3, 64, 2, 64), "ner.lower.b": (64, 2), "ner.lower.pad": (1, 3, 64, 2),
                   "ner.upper.W": (len(CLASSES), 64), "ner.upper.b": (len(CLASSES),)})
    return result


def load_with_ner(wheel: Path, destination: Path):
    base.verify_wheel(wheel)
    import spacy
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            name = info.filename
            base.require(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts,
                         "Unsafe archive path")
            if name.startswith(base.MODEL_PREFIX) and not info.is_dir():
                relative = name[len(base.MODEL_PREFIX):]
                base.require(relative and info.file_size < 32 * 1024 * 1024, "Oversized model resource")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
    nlp = spacy.load(destination, exclude=["senter"])
    base.require(nlp.pipe_names == base.COMPONENTS + ["ner"], "Unexpected pipeline order")
    return nlp


def extract(nlp):
    import numpy as np
    ner = nlp.get_pipe("ner")
    model = ner.model
    base.require(model.attrs["has_upper"] is True and not model.attrs["unseen_classes"],
                 "Unsupported NER class mask/upper head")
    base.require([ner.moves.get_class_name(i) for i in range(ner.moves.n_moves)] == CLASSES,
                 "Unexpected NER class inventory/order")
    base.require(ner.cfg.get("beam_width") == 1, "Only greedy NER decoding is supported")
    nodes = list(model.walk())
    tensors, origins = {}, {}

    def put(prefix, index, parameters):
        layer = nodes[index]
        base.require(set(layer.param_names) == set(parameters), f"Unexpected parameters at node {index}")
        for parameter in parameters:
            original = layer.get_param(parameter)
            base.require(original.dtype == np.dtype("float32") and bool(np.isfinite(original).all()),
                         f"Source parameter is not finite F32: {prefix}.{parameter}")
            tensors[f"{prefix}.{parameter}"] = np.array(original, dtype="<f4", order="C", copy=True)
            origins[f"{prefix}.{parameter}"] = {"node_index": index, "node_name": layer.name, "parameter": parameter}

    extract_features = [n for n in nodes if n.name == "extract_features"]
    base.require(len(extract_features) == 1 and
                 [nlp.vocab.strings[c] if isinstance(c, int) else c for c in extract_features[0].attrs["columns"]]
                 == FEATURES, "Unexpected NER feature columns")
    padded = [n for n in nodes if n.name.startswith("with_array(residual")]
    base.require(len(padded) == 1 and padded[0].attrs.get("pad") == 4, "Unexpected NER encoder padding")
    windows = [n for n in nodes if n.name == "expand_window"]
    base.require(len(windows) == 4 and all(n.attrs["window_size"] == 1 for n in windows), "Unexpected NER window")
    hashembeds = [i for i, n in enumerate(nodes) if n.name == "hashembed"]
    base.require(len(hashembeds) == 4, "Unexpected NER hash embedding count")
    for feature, index in enumerate(hashembeds):
        base.require(nodes[index].attrs["seed"] == 8 + feature and nodes[index].attrs["column"] == feature,
                     "Unexpected NER hash embedding seed/column")
        put(f"ner.embed.{feature}", index, ["E"])
    maxouts = [i for i, n in enumerate(nodes) if n.name == "maxout"]
    norms = [i for i, n in enumerate(nodes) if n.name == "layernorm"]
    base.require(len(maxouts) == 5 and len(norms) == 5 and all(n == m + 1 for m, n in zip(maxouts, norms)),
                 "Unexpected NER maxout/layernorm graph")
    put("ner.embed.mix", maxouts[0], ["W", "b"])
    put("ner.embed.norm", norms[0], ["G", "b"])
    for layer in range(4):
        put(f"ner.encoder.{layer}.maxout", maxouts[layer + 1], ["W", "b"])
        put(f"ner.encoder.{layer}.norm", norms[layer + 1], ["G", "b"])
    base.require(nodes[2].name == "precomputable_affine" and nodes[3].name == "linear" and nodes[6].name == "linear",
                 "Unexpected NER parser model graph")
    put("ner.reduce", 6, ["W", "b"])
    put("ner.lower", 2, ["W", "b", "pad"])
    put("ner.upper", 3, ["W", "b"])
    expected = expected_tensors()
    base.require(set(tensors) == set(expected), "Unexpected/missing NER tensor inventory")
    for name, value in tensors.items():
        base.require(tuple(value.shape) == expected[name], f"Unexpected NER tensor shape: {name}")
    return tensors, origins


def convert(wheel: Path, output: Path, dev_environment: bool):
    if not dev_environment:
        base.verify_environment()
    base.require(not output.exists(), "Output already exists; refusing to replace")
    output.parent.mkdir(parents=True, exist_ok=True)
    import gguf
    import numpy as np
    with tempfile.TemporaryDirectory(prefix="spacy-ner-", dir=output.parent) as temporary:
        temp = Path(temporary)
        source, bundle = temp / "source", temp / "bundle"
        source.mkdir()
        bundle.mkdir()
        nlp = load_with_ner(wheel, source)
        tensors, origins = extract(nlp)
        path = bundle / "ner.gguf"
        writer = gguf.GGUFWriter(str(path), ARCHITECTURE)
        writer.add_uint32("spacy.format_version", FORMAT_VERSION)
        writer.add_string("spacy.model", "de_core_news_sm")
        writer.add_string("spacy.model_version", "3.8.0")
        writer.add_string("spacy.source.sha256", base.WHEEL_SHA256)
        writer.add_string("spacy.component", "ner")
        writer.add_string("spacy.feature_order", ",".join(FEATURES))
        writer.add_string("spacy.precision", "f32")
        writer.add_string("spacy.ner.classes", ",".join(CLASSES))
        writer.add_uint32("spacy.width", 96)
        writer.add_uint32("spacy.encoder_depth", 4)
        writer.add_uint32("spacy.encoder_padding", 4)
        writer.add_float32("spacy.layer_norm_epsilon", 1e-8)
        for name in sorted(tensors):
            writer.add_tensor(name, tensors[name])
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        reader = gguf.GGUFReader(str(path))
        base.require(len(reader.tensors) == len(tensors), "GGUF round-trip tensor count mismatch")
        for tensor in reader.tensors:
            original = tensors[tensor.name]
            base.require(tensor.tensor_type == gguf.GGMLQuantizationType.F32 and
                         tuple(tensor.shape) == tuple(reversed(original.shape)) and
                         np.array_equal(tensor.data.reshape(original.shape), original),
                         f"GGUF round-trip mismatch: {tensor.name}")
        del reader
        data = path.read_bytes()
        manifest = {
            "format_version": FORMAT_VERSION, "architecture": ARCHITECTURE, "component": "ner",
            "model": {"name": "de_core_news_sm", "version": "3.8.0", "language": "de"},
            "source": {"file": base.WHEEL_NAME, "sha256": base.WHEEL_SHA256, "size_bytes": base.WHEEL_SIZE},
            "reference_python": platform.python_version(), "dev_environment": dev_environment,
            "classes": CLASSES, "features": FEATURES, "embedding_rows": ROWS, "embedding_seeds": [8, 9, 10, 11],
            "decoding": "greedy BILUO, first maximum among valid classes; sentence starts from the parser",
            "tensors": [{"name": name, "shape": list(tensors[name].shape), "sha256": base.digest(tensors[name].tobytes()),
                         "source": origins[name]} for name in sorted(tensors)],
            "files": {"ner.gguf": {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}},
            "converter_sha256": base.digest(Path(__file__).read_bytes()),
        }
        (bundle / "ner-manifest.json").write_bytes(base.json_bytes(manifest))
        os.rename(bundle, output)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New directory (must not exist)")
    parser.add_argument("--dev-environment", action="store_true",
                        help="Skip the frozen CPython 3.12.14 check (recorded in the manifest)")
    args = parser.parse_args()
    try:
        manifest = convert(args.wheel, args.output, args.dev_environment)
    except Exception as error:
        print(f"spaCy NER conversion failed: {error}", file=sys.stderr)
        return 1
    print(f"Exported {len(manifest['tensors'])} F32 NER tensors; ner.gguf SHA-256 "
          f"{manifest['files']['ner.gguf']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
