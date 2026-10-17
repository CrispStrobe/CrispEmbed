"""Fail-closed contract for the first, immutable UltraBERT retrieval checkpoint.

No torch/gguf imports: shape/source guards run before loading or executing weights.
This profile intentionally does not accept the masked-LM checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "dmis-lab/UltraBERT-Embed-150M"
REVISION = "367d5343b5a0164d57b810cedf5de4b8983d5984"
QUERY = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
MAX_REFERENCE_TOKENS = 512


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def profile():
    audit = json.loads((ROOT / "tests/results/ultrabert/source-audit.json").read_text())
    model = audit["models"][0]
    if model["model"] != MODEL or model["revision"] != REVISION:
        raise ValueError("Unexpected source audit profile")
    return model


def expected_tensors():
    result = {"embed_tokens.weight": ("token_embd.weight", [49152, 768]),
              "norm.weight": ("output_norm.weight", [768])}
    for i in range(24):
        prefix = f"layers.{i}."
        for source, target, shape in (
            ("input_layernorm", "attn_norm", [768]),
            ("post_attention_layernorm", "ffn_norm", [768]),
            ("self_attn.q_proj", "attn_q", [768, 768]),
            ("self_attn.k_proj", "attn_k", [768, 768]),
            ("self_attn.v_proj", "attn_v", [768, 768]),
            ("self_attn.o_proj", "attn_output", [768, 768]),
            ("mlp.gate_proj", "ffn_gate", [1024, 768]),
            ("mlp.up_proj", "ffn_up", [1024, 768]),
            ("mlp.down_proj", "ffn_down", [768, 1024]),
        ):
            result[prefix + source + ".weight"] = (f"blk.{i}.{target}.weight", shape)
    return result


def schema_record():
    return json.loads((ROOT / "tests/results/ultrabert/embed-weight-schema.json").read_text())


def validate_schema(header, payload_bytes):
    expected = expected_tensors()
    actual = {k: v for k, v in header.items() if k != "__metadata__"}
    if set(actual) != set(expected):
        raise ValueError("Unexpected/missing weight tensors")
    intervals = []
    for name, (_, shape) in expected.items():
        tensor = actual[name]
        if tensor.get("shape") != shape or tensor.get("dtype") != "F32":
            raise ValueError("Unexpected tensor shape/dtype: " + name)
        offsets = tensor.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(v) is not int for v in offsets)
                or offsets[0] < 0 or offsets[1] - offsets[0] != 4 * math.prod(shape)):
            raise ValueError("Invalid tensor offsets: " + name)
        intervals.append(tuple(offsets))
    end = 0
    for begin, stop in sorted(intervals):
        if begin != end:
            raise ValueError("Overlapping or noncontiguous weight payloads")
        end = stop
    if end != payload_bytes:
        raise ValueError("Unexpected weight payload length")
    if sum(math.prod(shape) for _, shape in expected.values()) != 151032576:
        raise ValueError("Unexpected parameter total")
    return actual


def validate_weight_file(path):
    path = Path(path)
    record = schema_record()
    if path.stat().st_size != record["size"]:
        raise ValueError("Weight file size differs from pinned LFS metadata")
    with path.open("rb") as stream:
        n = int.from_bytes(stream.read(8), "little")
        if n != record["header_bytes"]:
            raise ValueError("Unexpected safetensors header length")
        raw = stream.read(n)
    if hashlib.sha256(raw).hexdigest() != record["header_sha256"]:
        raise ValueError("Weight header hash differs")
    header = json.loads(raw)
    tensors = validate_schema(header, path.stat().st_size - n - 8)
    if tensors != record["tensors"] or sha256(path) != record["sha256"]:
        raise ValueError("Weight payload/schema differs from pinned checkpoint")
    return record


def validate_sources(directory):
    directory = Path(directory)
    hashes = profile()["inspected_file_sha256"]
    for name, digest in hashes.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024 or sha256(path) != digest:
            raise ValueError("Pinned source hash mismatch: " + name)
    config = json.loads((directory / "config.json").read_text())
    if config != profile()["config"]:
        raise ValueError("Pinned config differs from source audit")
    return hashes


def tokenizer_metadata(tokenizer):
    model = tokenizer["model"]
    vocab = model["vocab"]
    if model["type"] != "BPE" or len(vocab) != 49152 or set(vocab.values()) != set(range(49152)):
        raise ValueError("Unexpected tokenizer vocabulary")
    tokens = [None] * 49152
    types = [1] * 49152
    for token, index in vocab.items():
        tokens[index] = token
    for token in tokenizer["added_tokens"]:
        if tokens[token["id"]] != token["content"]:
            raise ValueError("Added token vocabulary mismatch")
        if token["special"]:
            types[token["id"]] = 3
    if tokens[0] != "<|endoftext|>" or tokens[15] != "<|sep|>" or tokens[16] != "<|mask|>":
        raise ValueError("Unexpected special token IDs")
    merges = model["merges"]
    if len(merges) != 48900:
        raise ValueError("Unexpected merge count")
    merges = [" ".join(m) if isinstance(m, list) else m for m in merges]
    return tokens, types, merges


def reference_cases():
    # Freeze before the port. Boundary IDs are prepared by the original tokenizer;
    # no padding-derived mask may treat the attended BOS (ID zero) as padding.
    return [
        {"id": "empty", "text": ""},
        {"id": "document", "text": "Berlin is the capital of Germany."},
        {"id": "query", "text": "What is the capital of Germany?", "prompt": "query"},
        {"id": "unicode", "text": "Grüße — café\n東京 العربية 👋"},
        {"id": "pair", "text": "A question?", "text_pair": "Its answer."},
        {"id": "window-boundary", "text": "attention " * 260},
    ]
