"""Model-free controls for the authenticated dual-head ONNX converter.

Tiny independent ONNX BERT fixtures exercise actual GGUF write/readback without
downloading a checkpoint. Production's immutable profile has no CLI override.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
gguf = pytest.importorskip("gguf")
from onnx import TensorProto, helper, numpy_helper

CONVERTER = Path(__file__).resolve().parents[1] / "models" / "convert-gector-onnx-to-gguf.py"
spec = importlib.util.spec_from_file_location("gector_onnx_converter", CONVERTER)
converter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = converter
spec.loader.exec_module(converter)


def make_model():
    dimensions = converter.Dimensions(4, 8, 2, 1, 8, 9, 2, 3, 2)
    nodes, initializers = [], []
    rng = np.random.default_rng(391)

    def tensor(name, shape):
        values = rng.normal(0, 0.1, shape).astype(np.float32)
        initializers.append(numpy_helper.from_array(values, name))
        return name

    def node(op, inputs, name, **attrs):
        nodes.append(helper.make_node(op, inputs, [name], name=name, **attrs))
        return name

    def constant(name, value):
        return node("Constant", [], name, value=numpy_helper.from_array(np.asarray(value)))

    def linear(source, bias_suffix, name, n_in, n_out, output=None):
        # Deliberately non-semantic, non-monotonic weight names. The converter
        # must associate a role with its consuming graph, not initializer order.
        weight = tensor(f"unnamed_{379 - len(initializers)}", (n_in, n_out))
        bias = tensor("m.encoder." + bias_suffix, (n_out,))
        product = node("MatMul", [source, weight], name + "/MatMul")
        return node("Add", [bias, product], output or name + "/Add")

    def norm(source, suffix, name):
        weight = tensor("m.encoder." + suffix + ".weight", (dimensions.hidden,))
        bias = tensor("m.encoder." + suffix + ".bias", (dimensions.hidden,))
        return node("LayerNormalization", [source, weight, bias], name, epsilon=1e-12, axis=-1)

    zeros = constant("type_ids", np.zeros((1, 3), dtype=np.int64))
    positions = constant("position_ids", np.array([[0, 1, 2]], dtype=np.int64))
    word = node("Gather", [tensor("m.encoder.embeddings.word_embeddings.weight", (8, 4)), "input_ids"], "word")
    position = node("Gather", [tensor("m.encoder.embeddings.position_embeddings.weight", (9, 4)), positions], "position")
    types = node("Gather", [tensor("m.encoder.embeddings.token_type_embeddings.weight", (2, 4)), zeros], "types")
    value = node("Add", [word, types], "word_plus_type")
    value = node("Add", [value, position], "all_embeddings")
    value = norm(value, "embeddings.LayerNorm", "embedding_norm")
    for layer in range(dimensions.layers):
        prefix = f"encoder.layer.{layer}"
        q = linear(value, f"{prefix}.attention.self.query.bias", f"layer{layer}/query", 4, 4)
        k = linear(value, f"{prefix}.attention.self.key.bias", f"layer{layer}/key", 4, 4)
        v = linear(value, f"{prefix}.attention.self.value.bias", f"layer{layer}/value", 4, 4)
        scale = constant(f"layer{layer}/scale", np.float32(4 ** -0.25))
        q = node("Mul", [q, scale], f"layer{layer}/q_scaled")
        k = node("Transpose", [k], f"layer{layer}/k_transpose", perm=[0, 2, 1])
        k = node("Mul", [k, scale], f"layer{layer}/k_scaled")
        scores = node("MatMul", [q, k], f"layer{layer}/scores")
        soft = node("Softmax", [scores], f"layer{layer}/softmax", axis=-1)
        isnan = node("IsNaN", [soft], f"layer{layer}/nan")
        zero = constant(f"layer{layer}/zero", np.float32(0))
        soft = node("Where", [isnan, zero, soft], f"layer{layer}/nan_zeroed")
        context = node("MatMul", [soft, v], f"layer{layer}/context")
        context = linear(context, f"{prefix}.attention.output.dense.bias", f"layer{layer}/attention_output", 4, 4)
        context = node("Add", [context, value], f"layer{layer}/residual1")
        context = norm(context, f"{prefix}.attention.output.LayerNorm", f"layer{layer}/norm1")
        up = linear(context, f"{prefix}.intermediate.dense.bias", f"layer{layer}/ffn_up", 4, 8)
        root2 = constant(f"layer{layer}/root2", np.float32(2 ** 0.5))
        erf_input = node("Div", [up, root2], f"layer{layer}/erf_input")
        erf = node("Erf", [erf_input], f"layer{layer}/erf")
        one = constant(f"layer{layer}/one", np.float32(1))
        half = constant(f"layer{layer}/half", np.float32(0.5))
        erf = node("Add", [erf, one], f"layer{layer}/erf_plus1")
        gelu = node("Mul", [up, erf], f"layer{layer}/gelu_product")
        gelu = node("Mul", [gelu, half], f"layer{layer}/gelu")
        down = linear(gelu, f"{prefix}.output.dense.bias", f"layer{layer}/ffn_down", 8, 4)
        down = node("Add", [down, context], f"layer{layer}/residual2")
        value = norm(down, f"{prefix}.output.LayerNorm", f"layer{layer}/norm2")
    linear(value, "labels_head.bias", "labels", 4, 3, "logits_labels")
    linear(value, "detect_head.bias", "detection", 4, 2, "logits_d")
    # Reverse the source inventory to catch reliance on initializer order.
    graph = helper.make_graph(nodes, "tiny_gector", [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 3]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 3]),
    ], [
        helper.make_tensor_value_info("logits_labels", TensorProto.FLOAT, [1, 3, 3]),
        helper.make_tensor_value_info("logits_d", TensorProto.FLOAT, [1, 3, 2]),
    ], list(reversed(initializers)))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.checker.check_model(model)
    return model, dimensions


@pytest.fixture
def source():
    return make_model()


def find_tensor(model, suffix):
    return next(tensor for tensor in model.graph.initializer if tensor.name.endswith(suffix))


def test_maps_both_heads_and_every_source_tensor(source):
    model, dimensions = source
    plan, audit = converter.build_tensor_plan(model, dimensions)
    assert len(plan) == 41
    assert len({item.source_name for item in plan}) == len(model.graph.initializer)
    assert audit["encoder_matrices"] == 12
    assert audit["embedding_addition_order"] == "(word+type)+position"
    assert audit["attention_scaling"] == "separate_q_and_k"
    assert audit["post_softmax_nan_to_zero"] is True
    assert {item.name for item in plan} >= {"gector.labels.weight", "gector.labels.bias",
                                           "gector.detect.weight", "gector.detect.bias"}
    assert sum(item.transpose for item in plan) == 14


@pytest.mark.parametrize("target", ["enc.0.attn.q.weight", "enc.0.ffn.fc1.weight",
                                    "enc.1.ffn.fc2.weight", "gector.labels.weight", "gector.detect.weight"])
def test_transposes_square_and_rectangular_matmul_rhs(source, target):
    model, dimensions = source
    plan, _ = converter.build_tensor_plan(model, dimensions)
    mapping = next(item for item in plan if item.name == target)
    original = next(tensor for tensor in model.graph.initializer if tensor.name == mapping.source_name)
    expected = numpy_helper.to_array(original).T.copy()
    actual = converter.tensor_array(original, mapping, "f32")
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert actual.tobytes() == expected.tobytes()


def test_f16_recipe_preserves_norms_biases_and_type_embeddings(source):
    model, dimensions = source
    plan, _ = converter.build_tensor_plan(model, dimensions)
    original = {tensor.name: tensor for tensor in model.graph.initializer}
    for mapping in plan:
        value = converter.tensor_array(original[mapping.source_name], mapping, "f16")
        reduced = len(mapping.source_shape) == 2 and mapping.name != "token_type_embd.weight"
        assert value.dtype == (np.float16 if reduced else np.float32)


@pytest.mark.parametrize("problem", ["missing", "extra", "shape", "type", "external", "duplicate"])
def test_rejects_incomplete_or_unsupported_initializer_inventory(source, problem):
    model, dimensions = source
    tensor = find_tensor(model, "attention.self.query.bias")
    if problem == "missing":
        model.graph.initializer.remove(tensor)
    elif problem == "extra":
        model.graph.initializer.append(numpy_helper.from_array(np.ones(4, dtype=np.float32), "unmapped.weight"))
    elif problem == "shape":
        tensor.dims[0] += 1
    elif problem == "type":
        tensor.data_type = TensorProto.FLOAT16
    elif problem == "external":
        tensor.data_location = TensorProto.EXTERNAL
    elif problem == "duplicate":
        model.graph.initializer.append(copy.deepcopy(tensor))
    with pytest.raises(converter.ConversionError):
        converter.build_tensor_plan(model, dimensions)


@pytest.mark.parametrize("problem", ["head", "shape", "dtype", "extra_input", "head_preprocessing"])
def test_rejects_wrong_public_contract(source, problem):
    model, dimensions = source
    if problem == "head":
        del model.graph.output[-1]
    elif problem == "shape":
        model.graph.output[0].type.tensor_type.shape.dim[2].dim_value += 1
    elif problem == "dtype":
        model.graph.input[0].type.tensor_type.elem_type = TensorProto.INT32
    elif problem == "extra_input":
        model.graph.input.append(helper.make_tensor_value_info("token_type_ids", TensorProto.INT64, [1, 3]))
    elif problem == "head_preprocessing":
        next(node for node in model.graph.node if node.name == "labels/MatMul").input[0] = "embedding_norm"
    with pytest.raises(converter.ConversionError):
        converter.build_tensor_plan(model, dimensions)


@pytest.mark.parametrize("problem", ["norm_epsilon", "norm_axis", "activation", "embedding_order", "scaling", "nan_zero"])
def test_rejects_arithmetic_contract_drift(source, problem):
    model, dimensions = source
    if problem in ("norm_epsilon", "norm_axis"):
        node = next(node for node in model.graph.node if node.op_type == "LayerNormalization")
        attr = next(attr for attr in node.attribute if attr.name == ("epsilon" if problem == "norm_epsilon" else "axis"))
        if problem == "norm_epsilon":
            attr.f = 1e-5
        else:
            attr.i = 1
    elif problem == "activation":
        next(node for node in model.graph.node if node.op_type == "Erf").op_type = "Tanh"
    elif problem == "embedding_order":
        next(node for node in model.graph.node if node.name == "word_plus_type").input[1] = "position"
        next(node for node in model.graph.node if node.name == "all_embeddings").input[1] = "types"
    elif problem == "scaling":
        node = next(node for node in model.graph.node if node.name == "layer0/scale")
        node.attribute[0].t.CopyFrom(numpy_helper.from_array(np.float32(0.5)))
    elif problem == "nan_zero":
        node = next(node for node in model.graph.node if node.name == "layer0/zero")
        node.attribute[0].t.CopyFrom(numpy_helper.from_array(np.float32(1)))
    with pytest.raises(converter.ConversionError):
        converter.build_tensor_plan(model, dimensions)


@pytest.mark.parametrize("value", [np.nan, np.inf, np.float32(1e10)])
def test_rejects_nonfinite_or_overflowing_f16_tensor(source, value):
    model, dimensions = source
    plan, _ = converter.build_tensor_plan(model, dimensions)
    mapping = next(item for item in plan if item.name == "gector.labels.weight")
    tensor = numpy_helper.from_array(np.full(mapping.source_shape, value, dtype=np.float32), mapping.source_name)
    with pytest.raises(converter.ConversionError):
        converter.tensor_array(tensor, mapping, "f16")


@pytest.mark.parametrize("mapping", [{"1": "$KEEP"}, {"0": ""}, {"0": "$KEEP", "1": "$KEEP"}, {}, None])
def test_rejects_invalid_label_maps(mapping):
    with pytest.raises(converter.ConversionError):
        converter.ordered_labels(mapping, "id2label")


@pytest.fixture
def tiny_bundle(tmp_path, monkeypatch, source):
    """Substitute a miniature *test* profile, never a production CLI option."""
    model, dimensions = source
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    onnx.save(model, bundle / "model.onnx")
    config = {"id2label": {"0": "$KEEP", "1": "$DELETE", "2": "$REPLACE_Test"},
              "d_id2label": {"0": "$CORRECT", "1": "$INCORRECT"}}
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "Hallo", "Welt", "##en", "."]
    tokenizer = {"model": {"type": "WordPiece", "vocab": {token: i for i, token in enumerate(tokens)}}}
    (bundle / "config.json").write_text(json.dumps(config))
    (bundle / "tokenizer.json").write_text(json.dumps(tokenizer))
    (bundle / "vocab.txt").write_text("\n".join(tokens) + "\n")
    for name in converter.ARTIFACTS:
        if not (bundle / name).exists():
            (bundle / name).write_text("test-only source " + name)
    artifacts = {path.name: (path.stat().st_size, converter.sha256_file(path)) for path in bundle.iterdir()}
    monkeypatch.setattr(converter, "ARTIFACTS", artifacts)
    monkeypatch.setattr(converter, "PINNED_DIMENSIONS", dimensions)
    monkeypatch.setattr(converter, "MODEL_SHA256", artifacts["model.onnx"][1])
    return bundle


@pytest.mark.parametrize("dtype", ["f32", "f16"])
def test_real_gguf_roundtrip_and_manifest(tiny_bundle, tmp_path, dtype):
    output = tmp_path / f"gector-{dtype}.gguf"
    manifest = tmp_path / f"gector-{dtype}.json"
    report = converter.convert(tiny_bundle, output, manifest, dtype)
    assert report == json.loads(manifest.read_text())
    assert report["gguf_readback_verified"] is True
    assert report["native_parity_qualified"] is False
    assert report["decoded_quality_qualified"] is False
    assert report["output"]["sha256"] == converter.sha256_file(output)
    assert len(report["tensors"]) == 41
    assert report["tensor_types"] == ({"F32": 41} if dtype == "f32" else {"F16": 16, "F32": 25})
    reader = gguf.GGUFReader(str(output))
    assert reader.get_field("general.architecture").contents() == "gector"
    assert reader.get_field("gector.source.sha256").contents() == converter.MODEL_SHA256
    assert reader.get_field("gector.source.notice").contents() == (tiny_bundle / "NOTICE").read_text()
    assert reader.get_field("gector.labels").contents() == ["$KEEP", "$DELETE", "$REPLACE_Test"]
    assert not list(tmp_path.glob(".gector-*"))


@pytest.mark.parametrize("artifact", ["model.onnx", "tokenizer.json", "config.json", "NOTICE", "verb-form-vocab.txt"])
def test_authenticates_whole_bundle_before_creating_output(tiny_bundle, tmp_path, artifact):
    path = tiny_bundle / artifact
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(converter.ConversionError, match="SHA-256 mismatch"):
        converter.convert(tiny_bundle, tmp_path / "out.gguf", tmp_path / "out.json")
    assert not (tmp_path / "out.gguf").exists()
    assert not (tmp_path / "out.json").exists()


def test_refuses_missing_bundle_file(tiny_bundle):
    (tiny_bundle / "LICENSE").unlink()
    with pytest.raises(converter.ConversionError, match="Missing or wrong-size"):
        converter.verify_bundle(tiny_bundle)


@pytest.mark.parametrize("target", ["source", "existing", "same", "symlink", "hardlink"])
def test_output_destinations_never_overwrite_inputs_or_existing_files(tiny_bundle, tmp_path, target):
    output = tmp_path / "output.gguf"
    manifest = tmp_path / "output.json"
    original = (tiny_bundle / "model.onnx").read_bytes()
    if target == "source":
        output = tiny_bundle / "model.onnx"
    elif target == "existing":
        output.write_text("preserve")
    elif target == "same":
        manifest = output
    elif target == "symlink":
        output.symlink_to(tiny_bundle / "model.onnx")
    elif target == "hardlink":
        output.hardlink_to(tiny_bundle / "model.onnx")
    with pytest.raises(converter.ConversionError):
        converter.convert(tiny_bundle, output, manifest)
    assert (tiny_bundle / "model.onnx").read_bytes() == original
    if target == "existing":
        assert output.read_text() == "preserve"


def test_publication_failure_cleans_only_own_partial_artifacts(tiny_bundle, tmp_path, monkeypatch):
    output = tmp_path / "output.gguf"
    manifest = tmp_path / "output.json"
    real_link = converter.os.link

    def fail_model_publication(source, destination):
        if destination == output:
            raise OSError("simulated publication failure")
        return real_link(source, destination)

    monkeypatch.setattr(converter.os, "link", fail_model_publication)
    with pytest.raises(OSError, match="simulated"):
        converter.convert(tiny_bundle, output, manifest)
    assert not output.exists()
    assert not manifest.exists()
    assert not list(tmp_path.glob(".gector-*"))
