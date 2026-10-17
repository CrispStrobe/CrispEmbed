#!/usr/bin/env python3
"""Observe the pinned original UltraBERT F32 eager model using forward hooks.

GGUF archives are test inputs for crispembed_diff::Ref; this script does not
reimplement attention, RMSNorm or the encoder. No native parity is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ultrabert_contract import (MAX_REFERENCE_TOKENS, MODEL, QUERY, REVISION,
                                reference_cases, sha256, validate_sources,
                                validate_weight_file)


def metrics(array):
    import numpy as np
    x = array.astype(np.float64)
    return {"shape": list(array.shape), "payload_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            "l2_norm": float(np.linalg.norm(x)),
            "max_abs": float(np.max(np.abs(x))), "rms": float(np.sqrt(np.mean(x*x)))}


def capture(model, encoded):
    import torch
    captured, handles = {}, []

    def save(name, tensor):
        if name in captured:
            raise ValueError("Duplicate reference stage: " + name)
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError("Invalid reference stage: " + name)
        captured[name] = tensor.detach().float().cpu().contiguous().clone()

    def hook(name):
        return lambda module, inputs, output: save(name, output[0] if isinstance(output, tuple) else output)

    handles.append(model.embed_tokens.register_forward_hook(hook("emb_out")))
    for i, layer in enumerate(model.layers):
        prefix = f"layer_{i}."
        def before(module, inputs, kwargs, prefix=prefix):
            save(prefix + "input", inputs[0])
            save(prefix + "attention_mask", kwargs["attention_mask"])
            save(prefix + "position_ids", kwargs["position_ids"])
            save(prefix + "rope_cos", kwargs["position_embeddings"][0])
            save(prefix + "rope_sin", kwargs["position_embeddings"][1])
        handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
        for name, module in (
            ("attn_norm", layer.input_layernorm),
            ("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj),
            ("v", layer.self_attn.v_proj), ("attn_out", layer.self_attn),
            ("ffn_norm", layer.post_attention_layernorm),
            ("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj),
            ("ffn_out", layer.mlp), ("output", layer),
        ):
            handles.append(module.register_forward_hook(hook(prefix + name)))
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(
            lambda module, inputs, prefix=prefix: save(prefix + "post_attn_residual", inputs[0])))
    handles.append(model.norm.register_forward_hook(hook("final_hidden")))
    try:
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != 2 + 24 * 16:
        raise ValueError(f"Incomplete stage capture: {len(captured)}")
    if not torch.equal(output, captured["final_hidden"]):
        raise ValueError("Captured final norm differs from returned hidden states")
    return output, captured


def assert_masks(captured, encoded):
    import torch
    ids, keys = encoded["input_ids"], encoded["attention_mask"].bool()
    n = ids.shape[1]
    positions = torch.arange(n)
    distances = (positions[:, None] - positions[None, :]).abs()
    for i in range(24):
        actual = captured[f"layer_{i}.attention_mask"]
        allowed = keys[:, None, None, :].expand(-1, 1, n, -1)
        if i % 3 != 2:
            allowed = allowed & (distances <= 128)[None, None, :, :]
        if not torch.equal(actual == 0, allowed) or not torch.equal(actual[~allowed], torch.full_like(actual[~allowed], torch.finfo(torch.float32).min)):
            raise ValueError(f"Original mask violates declared contract at layer {i}")
        # The real BOS is zero, yet must remain an attended key. Padding uses the
        # same zero ID with an explicit zero mask, never inference from IDs.
        if not torch.all(actual[:, 0, 0, 0] == 0):
            raise ValueError("BOS was masked as padding")
    boundary_rows = torch.nonzero(keys[:, 129]).flatten() if n > 129 else []
    if len(boundary_rows):
        row = int(boundary_rows[0])
        local = captured["layer_0.attention_mask"]
        global_mask = captured["layer_2.attention_mask"]
        if not (local[row, 0, 0, 127] == 0 and local[row, 0, 0, 128] == 0
                and local[row, 0, 0, 129] < 0 and local[row, 0, 128, 0] == 0
                and local[row, 0, 129, 0] < 0 and global_mask[row, 0, 0, 129] == 0):
            raise ValueError("Local/global 127/128/129 boundary differs")



def check_bos_negative(captured, encoded):
    import torch
    # Captures made under inference_mode are immutable outside that context.
    # Clone outside it and alter only the negative control, never the originals.
    mask = captured["layer_0.attention_mask"].clone()
    negative = {**captured, "layer_0.attention_mask": mask}
    mask[0, 0, 0, 0] = torch.finfo(torch.float32).min
    rejected = False
    try:
        assert_masks(negative, encoded)
    except ValueError:
        rejected = True
    if not rejected:
        raise ValueError("Mask checker accepted a hidden BOS")
    assert_masks(captured, encoded)


def dump(directory, output):
    directory, output = Path(directory), Path(output)
    sources = validate_sources(directory)
    weights = validate_weight_file(directory / "model.safetensors")
    import gguf
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if transformers.__version__ != "5.16.1":
        raise ValueError("Use the separately pinned Transformers 5.16.1 environment")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=True)
    model, loading = AutoModel.from_pretrained(
        directory, local_files_only=True, trust_remote_code=True,
        attn_implementation="eager", dtype=torch.float32, output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Original model weight-loading mismatch")
    model.eval()
    if model.config._attn_implementation != "eager" or len(model.layers) != 24:
        raise ValueError("Unexpected original model backend/layers")
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    cases = reference_cases()
    cases.append({"id": "padded-batch", "text": ["", cases[-1]["text"]]})
    for case in cases:
        text = QUERY + case["text"] if case.get("prompt") == "query" else case["text"]
        encoded = tokenizer(text, text_pair=case.get("text_pair"), padding=True,
                            truncation=False, return_tensors="pt", return_token_type_ids=False)
        # Serialize the actual original tokenization, including the explicit mask.
        if encoded["input_ids"].shape[1] > MAX_REFERENCE_TOKENS:
            raise ValueError("Reference bound exceeded; do not silently truncate")
        if not torch.all(encoded["input_ids"][:, 0] == 0) or not torch.all(encoded["attention_mask"][:, 0] == 1):
            raise ValueError("Original BOS/template mismatch")
        if case["id"] == "empty" and encoded["input_ids"].tolist() != [[0]]:
            raise ValueError("Empty text has extra special tokens")
        if case["id"] == "pair":
            expected = [0] + tokenizer.encode(case["text"], add_special_tokens=False) + [15] + tokenizer.encode(case["text_pair"], add_special_tokens=False)
            if encoded["input_ids"].tolist() != [expected]:
                raise ValueError("Original pair template mismatch")
        if case["id"] == "window-boundary" and encoded["input_ids"].shape[1] <= 258:
            raise ValueError("Boundary fixture does not exercise both sides")
        with torch.inference_mode():
            anchor = model(**encoded).last_hidden_state.detach().clone()
        observed, tensors = capture(model, encoded)
        if not torch.equal(anchor, observed):
            raise ValueError("Reference observation changes ordinary output")
        assert_masks(tensors, encoded)
        check_bos_negative(tensors, encoded)
        tensors["input_ids"] = encoded["input_ids"].float()
        tensors["attention_mask"] = encoded["attention_mask"].float()
        tensors["cls_raw"] = observed[:, 0, :].detach().clone()
        tensors["cls_normalized"] = torch.nn.functional.normalize(tensors["cls_raw"], p=2, dim=-1)
        arrays = {name: value.cpu().float().numpy().copy() for name, value in tensors.items()}
        archive = output / (case["id"] + ".gguf")
        writer = gguf.GGUFWriter(str(archive), arch="ultrabert_ref", use_temp_file=True)
        writer.add_string("ref.model", MODEL)
        writer.add_string("ref.revision", REVISION)
        writer.add_string("ref.case", json.dumps(case, ensure_ascii=False))
        writer.add_string("ref.stage_order", json.dumps(list(arrays)))
        writer.add_uint32("ref.n_layers", 24)
        writer.add_array("ref.input_ids", encoded["input_ids"].flatten().tolist())
        try:
            for name, array in arrays.items():
                writer.add_tensor(name, np.ascontiguousarray(array, dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        reports.append({"case": case, "ids": encoded["input_ids"].tolist(),
                        "attention_mask": encoded["attention_mask"].tolist(),
                        "decoded_input": tokenizer.batch_decode(encoded["input_ids"], skip_special_tokens=False),
                        "neutrality_exact": True, "mask_contract_exact": True,
                        "wrong_bos_mask_rejected": True,
                        "archive": archive.name, "archive_bytes": archive.stat().st_size,
                        "archive_sha256": sha256(archive),
                        "stages": {name: metrics(array) for name, array in arrays.items()}})
        print(f"Captured {case['id']}: {len(arrays)} stages, {encoded['input_ids'].shape}", flush=True)
    return {"model": MODEL, "revision": REVISION, "source_files": sources,
            "source_weights": weights["sha256"], "torch": torch.__version__,
            "transformers": transformers.__version__, "backend": "cpu-f32-eager",
            "reference_token_bound": MAX_REFERENCE_TOKENS, "cases": reports,
            "native_parity": "not evaluated", "retrieval_quality": "not evaluated",
            "pooling_scope": "raw first-token hidden state and optional L2; SentenceTransformer wrapper not yet exercised"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = dump(args.model_dir, args.output_dir)
    (args.output_dir / "reference.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
