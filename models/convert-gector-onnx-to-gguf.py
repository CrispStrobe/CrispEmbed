#!/usr/bin/env python3
"""Convert the authenticated Proofbird German GECToR v2.1 ONNX bundle.

This deliberately is not a general ONNX importer. It accepts one immutable
fine-tuned encoder and its two trained heads, never a replacement BERT model.
The default F32 output is a conversion/runtime control. F16 is a separately
declared precision candidate; decoded quality must be measured independently.

    pip install onnx==1.19.1 gguf==0.19.0 numpy
    python models/convert-gector-onnx-to-gguf.py --bundle proofbird-v2.1 \
        --output gector-de-f32.gguf

No remote code is imported or executed. Tokenization and iterative edits remain
the caller's responsibility; the native interface consumes prepared IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import gguf
import numpy as np
import onnx
from onnx import numpy_helper


REPOSITORY = "proofbird/gector-bert-de"
REVISION = "8e3b50beef7d775cedb7b876009e88bcf2e8d6af"
MODEL_SHA256 = "6d7b43b4bd95c7a00c02c8babc7569d28558df894afe4874d96c202e6ab9a464"
FORMAT_VERSION = 1
# Same immutable bundle as Quds tools/proofing/german_model/artifacts.json.
# The CLI exposes no identity-override flag. Changed bundles need a new profile.
ARTIFACTS = {
    "LICENSE": (11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
    "NOTICE": (444, "52b20b1d66a7eeb6765f1829f08daddca2c36cd2e6c995439a4860e8b71e29e3"),
    "README.md": (13590, "e2e5120e6c25299f962120463936427056fd9b468c714a1bf785f555edabf00f"),
    "quickstart.py": (12746, "f5ce68cc466b5891fc3f13280dc9db3a55bf9016505c2b2d84eb1d2cb07c2b40"),
    "gector-meta.json": (930, "acb7edf298f6ac4ba8b83c628ad512404eadb6c43120c1878d4f3356ebd8b5a5"),
    "config.json": (176821, "c870008059c0f354e1ebc30c816ae5f55fa14a77ffa7da5271310ccf5e24e170"),
    "gector-guards.json": (6088, "57c4824e541d323e9cff4c543bd80cfd93c333d3be64cce83297bf5b8a44627a"),
    "language-rules.json": (1155, "901b350a9c751fc95d51eb5ee3ba7b7f0d11b61c6cc17e67bce2996e17bc5815"),
    "tokenizer_config.json": (370, "4e5738ad97f034b944ce8b7b8c3165f7a6333b524b7b77044f7e9e9b963f4dea"),
    "model.onnx": (452972247, MODEL_SHA256),
    "tokenizer.json": (729060, "45253db94c657c7b89880de2c8872e10cd0561577e9393e1f5f4df27b0165087"),
    "vocab.txt": (239836, "bf186f7fb9cb667d811c08b45ce64d1e8e262277bb053f05d1aef3e9f13039a3"),
    "verb-form-vocab.txt": (121758, "7582e7c2f480d7e49ac18677195cfff4ad1d0c350b0448f1271aacf290205b68"),
}


class ConversionError(ValueError):
    """A source identity, graph contract, or artifact check failed."""


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def verify_bundle(bundle: Path) -> list[dict[str, Any]]:
    records = []
    for name, (size, expected) in ARTIFACTS.items():
        path = bundle / name
        if not path.is_file() or path.stat().st_size != size:
            raise ConversionError(f"Missing or wrong-size pinned artifact: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ConversionError(f"SHA-256 mismatch for pinned artifact: {name}")
        records.append({"file": name, "size_bytes": size, "sha256": actual})
    return records


@dataclass(frozen=True)
class Dimensions:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    vocabulary: int
    positions: int
    types: int
    labels: int
    detection: int

PINNED_DIMENSIONS = Dimensions(768, 3072, 12, 12, 31102, 512, 2, 5000, 2)


def ordered_labels(value: Any, name: str) -> list[str]:
    if not isinstance(value, dict) or not value:
        raise ConversionError(f"Missing label mapping: {name}")
    if set(value) != {str(i) for i in range(len(value))}:
        raise ConversionError(f"Non-contiguous label indices: {name}")
    labels = [value[str(i)] for i in range(len(value))]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ConversionError(f"Invalid label names: {name}")
    if len(set(labels)) != len(labels):
        raise ConversionError(f"Duplicate label names: {name}")
    return labels


@dataclass(frozen=True)
class TensorMapping:
    source_name: str
    name: str
    source_shape: tuple[int, ...]
    transpose: bool
    role: str
    consumer: str


class Graph:
    """Resolve weights through graph use, never numbered initializer order."""

    def __init__(self, model: onnx.ModelProto):
        if model.graph.sparse_initializer:
            raise ConversionError("Sparse ONNX initializers are outside the pinned profile")
        self.model = model
        self.initializers = {}
        for tensor in model.graph.initializer:
            if tensor.name in self.initializers:
                raise ConversionError(f"Duplicate initializer: {tensor.name}")
            if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
                raise ConversionError("External tensor data is outside the authenticated bundle")
            self.initializers[tensor.name] = tensor
        self.producers = {}
        for node in model.graph.node:
            if node.domain not in ("", "ai.onnx"):
                raise ConversionError(f"Unsupported ONNX domain: {node.domain}")
            for output in node.output:
                if output in self.producers or output in self.initializers:
                    raise ConversionError(f"Duplicate tensor producer: {output}")
                self.producers[output] = node
        self.mappings: list[TensorMapping] = []
        self.used: set[str] = set()

    def resolve(self, name: str) -> str:
        seen = set()
        while name in self.producers and self.producers[name].op_type == "Identity":
            if name in seen:
                raise ConversionError("Cycle in Identity nodes")
            seen.add(name)
            node = self.producers[name]
            if len(node.input) != 1:
                raise ConversionError("Malformed Identity node")
            name = node.input[0]
        return name

    def initializer_with_suffix(self, suffix: str) -> str:
        matches = [name for name in self.initializers
                   if name == suffix or name.endswith("." + suffix)]
        if len(matches) != 1:
            raise ConversionError(f"Expected one initializer ending in {suffix}; found {len(matches)}")
        return matches[0]

    def consumers(self, source: str, op_type: str) -> list[onnx.NodeProto]:
        return [node for node in self.model.graph.node if node.op_type == op_type
                and source in {self.resolve(name) for name in node.input}]

    def add(self, source: str, name: str, shape: tuple[int, ...], *,
            transpose: bool = False, role: str, consumer: str) -> None:
        source = self.resolve(source)
        if source not in self.initializers:
            raise ConversionError(f"Weight is not an embedded initializer: {source}")
        if source in self.used:
            raise ConversionError(f"Initializer mapped more than once: {source}")
        tensor = self.initializers[source]
        if tensor.data_type != onnx.TensorProto.FLOAT or tuple(tensor.dims) != shape:
            raise ConversionError(f"Unexpected dtype/shape for {source}: expected F32 {shape}")
        if transpose and len(shape) != 2:
            raise ConversionError(f"Only matrix weights may be transposed: {source}")
        self.used.add(source)
        self.mappings.append(TensorMapping(source, name, shape, transpose, role, consumer))

    def copy(self, suffix: str, name: str, shape: tuple[int, ...], *,
             role: str, op_type: str, input_index: int) -> onnx.NodeProto:
        source = self.initializer_with_suffix(suffix)
        uses = [node for node in self.consumers(source, op_type)
                if len(node.input) > input_index and self.resolve(node.input[input_index]) == source]
        if len(uses) != 1:
            raise ConversionError(f"Expected one {op_type} consumer for {source}")
        node = uses[0]
        self.add(source, name, shape, role=role, consumer=node.name or node.output[0])
        return node

    def linear(self, bias_suffix: str, target: str, width_in: int, width_out: int,
               role: str, output_name: str | None = None) -> onnx.NodeProto:
        bias = self.initializer_with_suffix(bias_suffix)
        uses = self.consumers(bias, "Add")
        if len(uses) != 1 or len(uses[0].input) != 2:
            raise ConversionError(f"Expected one binary Add using trained bias {bias}")
        addition = uses[0]
        others = [self.resolve(name) for name in addition.input if self.resolve(name) != bias]
        if len(others) != 1:
            raise ConversionError(f"Ambiguous bias addition: {bias}")
        matmul = self.producers.get(others[0])
        if matmul is None or matmul.op_type != "MatMul" or len(matmul.input) != 2:
            raise ConversionError(f"Trained bias is not applied directly after MatMul: {bias}")
        if output_name is not None and self.resolve(output_name) not in addition.output:
            raise ConversionError(f"Head {target} does not directly produce {output_name}")
        self.add(matmul.input[1], target + ".weight", (width_in, width_out),
                 transpose=True, role=role, consumer=matmul.name or matmul.output[0])
        self.add(bias, target + ".bias", (width_out,), role="bias",
                 consumer=addition.name or addition.output[0])
        return matmul

    def constant_scalar(self, name: str) -> float | None:
        node = self.producers.get(self.resolve(name))
        if node is None or node.op_type != "Constant":
            return None
        values = [attribute.t for attribute in node.attribute
                  if attribute.name == "value" and attribute.type == onnx.AttributeProto.TENSOR]
        if len(values) != 1:
            return None
        array = numpy_helper.to_array(values[0])
        return float(array.item()) if array.size == 1 else None


def audit_arithmetic(graph: Graph, dimensions: Dimensions,
                     embedding_gathers: dict[str, str], embedding_norm: onnx.NodeProto
                     ) -> dict[str, Any]:
    """Record the actual source expression instead of an equivalent BERT formula."""
    final_add = graph.producers.get(embedding_norm.input[0])
    if final_add is None or final_add.op_type != "Add":
        raise ConversionError("Embedding normalization is not fed by the expected sum")
    position = embedding_gathers["position_embd"]
    if position not in final_add.input:
        raise ConversionError("Embedding addition order differs from word+type, then position")
    remaining = [name for name in final_add.input if name != position]
    first_add = graph.producers.get(remaining[0]) if len(remaining) == 1 else None
    if (first_add is None or first_add.op_type != "Add"
            or set(first_add.input) != {embedding_gathers["token_embd"], embedding_gathers["token_type_embd"]}):
        raise ConversionError("Embedding addition order differs from word+type, then position")
    scales = []
    for node in graph.model.graph.node:
        if node.op_type != "MatMul" or graph.resolve(node.input[1]) in graph.initializers:
            continue
        products = [graph.producers.get(name) for name in node.input]
        if not all(product is not None and product.op_type == "Mul" for product in products):
            continue
        pair = []
        for product in products:
            constants = [graph.constant_scalar(name) for name in product.input]
            constants = [value for value in constants if value is not None]
            if len(constants) != 1:
                raise ConversionError("Attention scaling must use one source scalar per Q/K")
            pair.append(constants[0])
        scales.append(pair)
    expected = float(np.float32((dimensions.hidden // dimensions.heads) ** -0.25))
    if len(scales) != dimensions.layers or any(pair != [expected, expected] for pair in scales):
        raise ConversionError("Attention must scale Q and K separately with the pinned F32 scalar")
    zeroed = 0
    for node in graph.model.graph.node:
        if node.op_type != "Where" or len(node.input) != 3:
            continue
        condition = graph.producers.get(node.input[0])
        if condition is None or condition.op_type != "IsNaN":
            continue
        softmax = graph.producers.get(node.input[2])
        if (softmax is None or softmax.op_type != "Softmax"
                or list(condition.input) != [node.input[2]]
                or graph.constant_scalar(node.input[1]) != 0):
            raise ConversionError("Unsupported post-softmax NaN handling")
        axis = next((attribute.i for attribute in softmax.attribute if attribute.name == "axis"), -1)
        if axis != -1:
            raise ConversionError("Attention softmax must operate on its final axis")
        zeroed += 1
    if zeroed != dimensions.layers:
        raise ConversionError("Missing post-softmax NaN-to-zero controls")
    return {"embedding_addition_order": "(word+type)+position",
            "attention_qk_scale": expected, "attention_scaling": "separate_q_and_k",
            "post_softmax_nan_to_zero": True, "activation": "gelu_erf"}


def build_tensor_plan(model: onnx.ModelProto, dimensions: Dimensions
                      ) -> tuple[list[TensorMapping], dict[str, Any]]:
    graph = Graph(model)
    d = dimensions
    if d.hidden % d.heads:
        raise ConversionError("Hidden width is not divisible by attention heads")
    if {value.name for value in model.graph.input} != {"input_ids", "attention_mask"}:
        raise ConversionError("Expected only input_ids and attention_mask inputs")
    if {value.name for value in model.graph.output} != {"logits_labels", "logits_d"}:
        raise ConversionError("Both named GECToR output heads are required")
    for value in model.graph.input:
        tensor = value.type.tensor_type
        if tensor.elem_type != onnx.TensorProto.INT64 or len(tensor.shape.dim) != 2:
            raise ConversionError(f"Expected rank-2 int64 input: {value.name}")
    for value in model.graph.output:
        tensor = value.type.tensor_type
        width = d.labels if value.name == "logits_labels" else d.detection
        if (tensor.elem_type != onnx.TensorProto.FLOAT or len(tensor.shape.dim) != 3
                or tensor.shape.dim[2].dim_value != width):
            raise ConversionError(f"Wrong shape or dtype for output: {value.name}")
    embedding_gathers = {}
    for suffix, target, rows in (
        ("word_embeddings", "token_embd", d.vocabulary),
        ("position_embeddings", "position_embd", d.positions),
        ("token_type_embeddings", "token_type_embd", d.types),
    ):
        node = graph.copy(f"embeddings.{suffix}.weight", f"{target}.weight",
                          (rows, d.hidden), role="embedding", op_type="Gather", input_index=0)
        axis = next((attribute.i for attribute in node.attribute if attribute.name == "axis"), 0)
        if axis != 0:
            raise ConversionError(f"Embedding Gather has unsupported axis: {node.name}")
        embedding_gathers[target] = node.output[0]
    norms = [("embeddings.LayerNorm", "embd_ln")]
    for layer in range(d.layers):
        prefix = f"encoder.layer.{layer}"
        target = f"enc.{layer}"
        for source, destination in (("query", "q"), ("key", "k"), ("value", "v")):
            graph.linear(f"{prefix}.attention.self.{source}.bias",
                         f"{target}.attn.{destination}", d.hidden, d.hidden, "encoder_matrix")
        graph.linear(f"{prefix}.attention.output.dense.bias", f"{target}.attn.o",
                     d.hidden, d.hidden, "encoder_matrix")
        graph.linear(f"{prefix}.intermediate.dense.bias", f"{target}.ffn.fc1",
                     d.hidden, d.intermediate, "encoder_matrix")
        graph.linear(f"{prefix}.output.dense.bias", f"{target}.ffn.fc2",
                     d.intermediate, d.hidden, "encoder_matrix")
        norms.extend(((f"{prefix}.attention.output.LayerNorm", f"{target}.ln1"),
                      (f"{prefix}.output.LayerNorm", f"{target}.ln2")))
    epsilons = []
    norm_nodes = []
    for suffix, target in norms:
        weight = graph.copy(suffix + ".weight", target + ".weight", (d.hidden,),
                            role="normalization", op_type="LayerNormalization", input_index=1)
        bias = graph.copy(suffix + ".bias", target + ".bias", (d.hidden,),
                          role="normalization", op_type="LayerNormalization", input_index=2)
        if tuple(weight.output) != tuple(bias.output):
            raise ConversionError(f"Mismatched normalization weight/bias consumers: {suffix}")
        norm_nodes.append(weight)
        attrs = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in weight.attribute}
        if attrs.get("axis", -1) != -1 or attrs.get("stash_type", onnx.TensorProto.FLOAT) != onnx.TensorProto.FLOAT:
            raise ConversionError("Unsupported LayerNormalization axis or accumulation precision")
        epsilons.append(float(attrs.get("epsilon", 1e-5)))
    if len(set(epsilons)) != 1 or not math.isfinite(epsilons[0]) or epsilons[0] <= 0:
        raise ConversionError("Inconsistent or invalid LayerNormalization epsilon")
    labels = graph.linear("labels_head.bias", "gector.labels", d.hidden, d.labels, "label_head", "logits_labels")
    detection = graph.linear("detect_head.bias", "gector.detect", d.hidden, d.detection, "detection_head", "logits_d")
    if labels.input[0] != norm_nodes[-1].output[0] or detection.input[0] != norm_nodes[-1].output[0]:
        raise ConversionError("Both heads must read the final encoder output without preprocessing")
    if set(graph.initializers) != graph.used:
        raise ConversionError("Unmapped source initializers: " + ", ".join(sorted(set(graph.initializers) - graph.used)))
    if len(graph.mappings) != 16 * d.layers + 9:
        raise ConversionError("Unexpected converted tensor count")
    operations = Counter(node.op_type for node in model.graph.node)
    if operations["LayerNormalization"] != 2 * d.layers + 1 or operations["Erf"] != d.layers:
        raise ConversionError("Expected post-normalized BERT with exact Erf GELU")
    if operations["MatMul"] != 8 * d.layers + 2:
        raise ConversionError("Unexpected encoder attention/matrix operation count")
    return graph.mappings, {
        **audit_arithmetic(graph, d, embedding_gathers, norm_nodes[0]),
        "source_initializers": len(graph.initializers),
        "converted_tensors": len(graph.mappings),
        "encoder_matrices": 6 * d.layers,
        "layer_norm_epsilon": epsilons[0],
        "operators": dict(sorted(operations.items())),
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
    }


def tensor_array(tensor: onnx.TensorProto, mapping: TensorMapping, dtype: str) -> np.ndarray:
    array = numpy_helper.to_array(tensor)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ConversionError(f"Non-F32 or non-finite source tensor: {mapping.source_name}")
    if mapping.transpose:
        array = array.T
    # Norms, biases, and the two-row type table always preserve source F32.
    reduced = (dtype == "f16" and array.ndim == 2 and mapping.name != "token_type_embd.weight")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(array, dtype="<f2" if reduced else "<f4")
    if not np.isfinite(result).all():
        raise ConversionError(f"Precision conversion overflows: {mapping.name}")
    return result


def metadata(config: dict[str, Any], tokenizer: dict[str, Any], dims: Dimensions,
             audit: dict[str, Any], bundle: Path, dtype: str) -> dict[str, Any]:
    vocab = tokenizer.get("model", {}).get("vocab")
    if tokenizer.get("model", {}).get("type") != "WordPiece" or not isinstance(vocab, dict):
        raise ConversionError("The pinned tokenizer must be WordPiece")
    if len(vocab) != dims.vocabulary or set(vocab.values()) != set(range(dims.vocabulary)):
        raise ConversionError("Tokenizer vocabulary does not match encoder rows")
    tokens = [""] * dims.vocabulary
    for token, index in vocab.items():
        if not isinstance(token, str) or type(index) is not int:
            raise ConversionError("Invalid tokenizer vocabulary entry")
        tokens[index] = token
    if tokens != (bundle / "vocab.txt").read_text(encoding="utf-8").splitlines():
        raise ConversionError("tokenizer.json and vocab.txt disagree")
    labels = ordered_labels(config["id2label"], "id2label")
    detect_labels = ordered_labels(config["d_id2label"], "d_id2label")
    if config.get("keep_label", "$KEEP") not in labels or config.get("incorrect_label", "$INCORRECT") not in detect_labels:
        raise ConversionError("Missing KEEP or INCORRECT label")
    result = {
        "general.name": "Proofbird German GECToR v2.1 (prepared input)",
        "general.license": "apache-2.0",
        "general.file_type": 0 if dtype == "f32" else 1,
        "gector.format_version": FORMAT_VERSION,
        "gector.block_count": dims.layers,
        "gector.embedding_length": dims.hidden,
        "gector.feed_forward_length": dims.intermediate,
        "gector.attention.head_count": dims.heads,
        "gector.context_length": dims.positions,
        "gector.vocab_size": dims.vocabulary,
        "gector.type_vocab_size": dims.types,
        "gector.label_count": dims.labels,
        "gector.detect_count": dims.detection,
        "gector.layer_norm_epsilon": audit["layer_norm_epsilon"],
        "gector.attention.qk_scale": audit["attention_qk_scale"],
        "gector.activation": "gelu_erf",
        "gector.mask_policy": "unpadded_all_ones",
        "gector.position_policy": "zero_based_contiguous",
        "gector.token_type_policy": "all_zero",
        "gector.precision": dtype,
        "gector.source.repository": REPOSITORY,
        "gector.source.revision": REVISION,
        "gector.source.sha256": MODEL_SHA256,
        "gector.source.license": (bundle / "LICENSE").read_text(encoding="utf-8"),
        "gector.source.notice": (bundle / "NOTICE").read_text(encoding="utf-8"),
        "gector.labels": labels,
        "gector.detect_labels": detect_labels,
        "gector.keep_label_id": labels.index(config.get("keep_label", "$KEEP")),
        "gector.incorrect_label_id": detect_labels.index(config.get("incorrect_label", "$INCORRECT")),
        "tokenizer.ggml.model": "bert",
        "tokenizer.ggml.tokens": tokens,
    }
    for name, (_, sha) in ARTIFACTS.items():
        result[f"gector.source.assets.{name}.sha256"] = sha
    for token, key in (("[CLS]", "cls_token_id"), ("[SEP]", "separator_token_id"),
                       ("[UNK]", "unknown_token_id"), ("[PAD]", "padding_token_id")):
        if token not in vocab:
            raise ConversionError(f"Missing tokenizer special token: {token}")
        result[f"tokenizer.ggml.{key}"] = vocab[token]
    return result


def validate_destinations(bundle: Path, output: Path, manifest: Path) -> None:
    destinations = [output, manifest]
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ConversionError("GGUF and manifest destinations must differ")
    sources = [bundle / name for name in ARTIFACTS]
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise ConversionError(f"Refusing to overwrite existing destination: {path.name}")
        if path.resolve() in {source.resolve() for source in sources}:
            raise ConversionError("Output must not replace a source artifact")


def convert(bundle: Path, output: Path, manifest: Path, dtype: str = "f32") -> dict[str, Any]:
    if dtype not in ("f32", "f16"):
        raise ConversionError("Only explicit F32 and F16 conversion recipes are supported")
    validate_destinations(bundle, output, manifest)
    assets = verify_bundle(bundle)
    config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
    # The pinned config is decoding metadata, not a Hugging Face BertConfig.
    # Check every parameter shape against the authenticated ONNX profile below.
    dims = PINNED_DIMENSIONS
    if (len(ordered_labels(config.get("id2label"), "id2label")) != dims.labels
            or len(ordered_labels(config.get("d_id2label"), "d_id2label")) != dims.detection):
        raise ConversionError("Configuration label counts differ from the pinned GECToR profile")
    model = onnx.load(str(bundle / "model.onnx"), load_external_data=False)
    mappings, audit = build_tensor_plan(model, dims)
    meta = metadata(config, json.loads((bundle / "tokenizer.json").read_text(encoding="utf-8")),
                    dims, audit, bundle, dtype)
    tensors = {tensor.name: tensor for tensor in model.graph.initializer}
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".gector-", suffix=".gguf", delete=False) as stream:
            model_temp = Path(stream.name)
        temporary.append(model_temp)
        writer = gguf.GGUFWriter(str(model_temp), arch="gector")
        for key, value in meta.items():
            if isinstance(value, str):
                writer.add_string(key, value)
            elif isinstance(value, list):
                writer.add_array(key, value)
            elif isinstance(value, float):
                writer.add_float32(key, value)
            elif type(value) is int:
                writer.add_uint32(key, value)
            else:
                raise ConversionError(f"Unsupported metadata value: {key}")
        records = []
        for mapping in mappings:
            source = numpy_helper.to_array(tensors[mapping.source_name])
            array = tensor_array(tensors[mapping.source_name], mapping, dtype)
            writer.add_tensor(mapping.name, array)
            records.append({
                "name": mapping.name, "source_name": mapping.source_name,
                "consumer": mapping.consumer, "role": mapping.role,
                "source_shape": list(mapping.source_shape), "source_dtype": "F32",
                "source_sha256": sha256_array(source),
                "transform": "transpose_2d" if mapping.transpose else "copy",
                "shape": list(array.shape), "ggml_shape": list(reversed(array.shape)),
                "type": "F16" if array.dtype.itemsize == 2 else "F32",
                "size_bytes": array.nbytes, "sha256": sha256_array(array),
            })
        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        # Read actual stored types, shapes and bytes back; a recipe name is not proof.
        reader = gguf.GGUFReader(str(model_temp))
        actual = {tensor.name: tensor for tensor in reader.tensors}
        if set(actual) != {record["name"] for record in records}:
            raise ConversionError("GGUF readback changed the tensor inventory")
        for record in records:
            tensor = actual[record["name"]]
            if (tensor.tensor_type.name != record["type"]
                    or list(tensor.shape) != record["ggml_shape"]
                    or sha256_array(tensor.data) != record["sha256"]):
                raise ConversionError(f"GGUF readback failed: {record['name']}")
        # Release memmaps before publication (also necessary on Windows).
        del tensor, actual, reader
        report = {
            "schema": "crispembed.gector.conversion.v1", "architecture": "gector",
            "source": {"repository": REPOSITORY, "revision": REVISION,
                       "model_sha256": MODEL_SHA256, "assets": assets},
            "converter_sha256": sha256_file(Path(__file__)),
            "tools": {name: importlib.metadata.version(name) for name in ("onnx", "numpy", "gguf")},
            "precision": dtype, "graph": audit,
            "output": {"file": output.name, "size_bytes": model_temp.stat().st_size,
                       "sha256": sha256_file(model_temp)},
            "tensor_types": dict(sorted(Counter(record["type"] for record in records).items())),
            "tensors": records, "gguf_readback_verified": True,
            "native_parity_qualified": False, "decoded_quality_qualified": False,
            "notice": "LICENSE and NOTICE are retained verbatim in GGUF metadata; distribute the source notices with model packs.",
        }
        with tempfile.NamedTemporaryFile(dir=manifest.parent, prefix=".gector-", suffix=".json", mode="w",
                                         encoding="utf-8", delete=False) as stream:
            manifest_temp = Path(stream.name)
            temporary.append(manifest_temp)
            json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        # Hard-link publication is atomic and cannot overwrite a destination that
        # appeared after validation. Both temporaries are on their target filesystem.
        os.link(manifest_temp, manifest)
        published.append(manifest)
        os.link(model_temp, output)
        published.append(output)
        return report
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, type=Path, help="Local immutable Proofbird v2.1 bundle")
    parser.add_argument("--output", required=True, type=Path, help="New GGUF artifact; existing files are never overwritten")
    parser.add_argument("--manifest", type=Path, help="New JSON tensor/type census (default: <output>.manifest.json)")
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    args = parser.parse_args()
    manifest = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    try:
        report = convert(args.bundle, args.output, manifest, args.dtype)
    except (ConversionError, OSError, json.JSONDecodeError) as error:
        print(f"Conversion refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": report["output"], "tensor_types": report["tensor_types"],
                      "manifest": manifest.name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
