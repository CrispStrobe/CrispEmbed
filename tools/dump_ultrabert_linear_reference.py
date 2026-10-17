#!/usr/bin/env python3
"""Additional test-only MLP observations; never replace the frozen encoder arrays."""
from __future__ import annotations

import json
from pathlib import Path

from ultrabert_contract import MODEL, REVISION, profile, sha256, validate_sources, validate_weight_file


def observe_mlp(mlp, inputs, expected):
    """Observe the original modules, requiring exact old anchors before adding stages."""
    import torch
    captures, handles = {}, []

    def save(name, value):
        if name in captures or not torch.isfinite(value).all():
            raise ValueError("Duplicate/nonfinite original MLP stage: " + name)
        captures[name] = value.detach().float().cpu().contiguous().clone()

    for name, module in (("gate", mlp.gate_proj), ("up", mlp.up_proj), ("silu", mlp.act_fn)):
        handles.append(module.register_forward_hook(lambda module, args, result, name=name: save(name, result)))
    handles.append(mlp.down_proj.register_forward_pre_hook(lambda module, args: save("product", args[0])))
    try:
        with torch.inference_mode():
            observed = mlp(inputs)
    finally:
        for handle in handles:
            handle.remove()
    with torch.inference_mode():
        ordinary = mlp(inputs)
    if not torch.equal(observed, ordinary):
        raise ValueError("MLP observation changed ordinary output")
    save("ffn_out", observed)
    for name in ("gate", "up", "ffn_out"):
        actual, frozen = captures[name], expected[name]
        if actual.shape != frozen.shape or not torch.equal(actual.view(torch.int32), frozen.view(torch.int32)):
            error = float((actual - frozen.detach()).abs().max()) if actual.shape == frozen.shape else None
            raise ValueError("Frozen MLP anchor drift: " + name + "; max_abs=" + str(error))
    return captures



def observe_local_operators(mlp, gate, up):
    """Additional same-input operator targets, never frozen full-model targets."""
    import torch
    with torch.inference_mode():
        silu = mlp.act_fn(gate)
        # This is the product expression in the inspected original MLP forward.
        # Its result is supplied identically to original down_proj and native down.
        product = silu * up
        down = mlp.down_proj(product)
    return {name: value.detach().float().cpu().contiguous().clone()
            for name, value in (("silu", silu), ("product", product), ("local_down", down))}


class FrozenForwardDrift(ValueError):
    def __init__(self, drifts):
        self.drifts = drifts
        super().__init__("Full original forward anchor drift: " + drifts[0]["stage"] +
                         "; max_abs=" + str(drifts[0]["max_abs_error"]))


def check_full_anchors(captured, records, tensors):
    import hashlib
    from dump_ultrabert_reference import metrics
    drifts = []
    for name, value in captured.items():
        array = value.numpy()
        expected = records[name]
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        if list(array.shape) != expected["shape"] or digest != expected["payload_sha256"]:
            original = tensors[name].data.reshape(expected["shape"])
            drifts.append({"stage": name,
                           "max_abs_error": float(abs(array - original).max()) if array.shape == original.shape else None,
                           "actual": metrics(array), "original": expected})
    if drifts:
        raise FrozenForwardDrift(drifts)


def observe_full(model, encoded, record, tensors):
    """Attach two additional hooks to the unchanged original full forward."""
    import torch
    from dump_ultrabert_reference import capture
    extra, handles = {}, []
    def save(name, value):
        if name in extra or not torch.isfinite(value).all():
            raise ValueError("Invalid extra full-forward observation")
        extra[name] = value.detach().float().cpu().contiguous().clone()
    mlp = model.layers[14].mlp
    handles.append(mlp.act_fn.register_forward_hook(lambda module, args, output: save("silu", output)))
    handles.append(mlp.down_proj.register_forward_pre_hook(lambda module, args: save("product", args[0])))
    try:
        observed, captured = capture(model, encoded)
    finally:
        for handle in handles:
            handle.remove()
    ordinary, neutral = capture(model, encoded)
    if (not torch.equal(observed, ordinary) or set(captured) != set(neutral)
            or any(not torch.equal(value.view(torch.int32), neutral[name].view(torch.int32))
                   for name, value in captured.items())):
        raise ValueError("Additional full-forward hooks changed an original stage")
    if len(captured) != 386 or set(extra) != {"silu", "product"}:
        raise ValueError("Incomplete full original forward observation")
    check_full_anchors(captured, record["stages"], tensors)
    return {**extra, **{name: captured["layer_14." + name] for name in ("gate", "up", "ffn_out")}}


def dump_linear(current, reference_directory, output, full_forward=False, local_operators=False):
    """Run only original layer14 MLP on saved inputs, with full checkpoint guards."""
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(MODEL, revision=REVISION,
                         allow_patterns=list(profile()["inspected_file_sha256"]) + ["model.safetensors"]))
    sources = validate_sources(directory)
    weights = validate_weight_file(directory / "model.safetensors")
    import gguf
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel
    from dump_ultrabert_reference import metrics

    if transformers.__version__ != "5.16.1" or torch.__version__.split('+')[0] != "2.14.0":
        raise ValueError("Use the frozen original Torch/Transformers environment")
    torch.set_num_threads(2)
    model, loading = AutoModel.from_pretrained(directory, local_files_only=True, trust_remote_code=True,
                       attn_implementation="eager", dtype=torch.float32, output_loading_info=True)
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Original operator model loading mismatch")
    model.eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"source_files": sources, "source_weights": weights["sha256"],
              "torch": torch.__version__, "transformers": transformers.__version__,
              "dumper_sha256": sha256(__file__), "cases": [],
              "scope": "additional original-input MLP observations; frozen encoder anchors unchanged",
              "mode": ("same-input local original operators; not frozen full-model targets" if local_operators else
                       "original full forward with all386 anchor checks" if full_forward else "isolated original MLP")}
    target = output / "linear-reference.json"
    for case in current["reference"]["cases"]:
        source = Path(reference_directory) / case["archive"]
        if sha256(source) != case["archive_sha256"]:
            raise ValueError("Frozen original archive changed before operator observation")
        reader = gguf.GGUFReader(str(source))
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        def original(name):
            stage = "layer_14." + name
            array = tensors[stage].data.copy().reshape(case["stages"][stage]["shape"])
            return torch.from_numpy(array)
        entry = {"case": case["case"], "original_archive_sha256": case["archive_sha256"],
                 "status": "observing; exact old anchors required"}
        report["cases"].append(entry)
        target.write_text(json.dumps(report, indent=2) + "\n")
        try:
            if local_operators:
                values = {name: original(name) for name in ("gate", "up", "ffn_out")}
                values.update(observe_local_operators(model.layers[14].mlp, values["gate"], values["up"]))
                entry["frozen_gate_up_output_retained"] = True
                entry["target_scope"] = "local_down is an additional same-product operator target; ffn_out remains frozen"
                entry["local_down_vs_frozen_ffn_out_max_abs"] = float((values["local_down"] - values["ffn_out"]).abs().max())
            elif full_forward:
                encoded = {name: torch.from_numpy(tensors[name].data.copy().reshape(case["stages"][name]["shape"])).long()
                           for name in ("input_ids", "attention_mask")}
                values = observe_full(model, encoded, case, tensors)
                entry["full_forward_anchors_exact"] = True
                entry["full_forward_anchor_count"] = 386
            else:
                values = observe_mlp(model.layers[14].mlp, original("ffn_norm"),
                                     {name: original(name) for name in ("gate", "up", "ffn_out")})
        except ValueError as error:
            entry.update(status="failed; frozen anchors preserved", error=str(error),
                         anchor_drifts=getattr(error, "drifts", None))
            target.write_text(json.dumps(report, indent=2) + "\n")
            raise
        archive = output / (case["case"]["id"] + ".gguf")
        writer = gguf.GGUFWriter(str(archive), arch="ultrabert_linear_ref", use_temp_file=True)
        writer.add_string("ref.model", MODEL)
        writer.add_string("ref.linear_scope", "local-original-operators" if local_operators else "frozen-model-anchors")
        writer.add_string("ref.revision", REVISION)
        writer.add_string("ref.case", json.dumps(case["case"], ensure_ascii=False))
        try:
            for name, tensor in values.items():
                writer.add_tensor("layer_14." + name, np.ascontiguousarray(tensor.numpy(), dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        written = gguf.GGUFReader(str(archive))
        actual = {tensor.name: tensor for tensor in written.tensors}
        if set(actual) != {"layer_14." + name for name in values}:
            raise ValueError("Additional operator archive stage coverage differs")
        for name, tensor in values.items():
            stored = actual["layer_14." + name]
            if (stored.tensor_type != gguf.GGMLQuantizationType.F32
                    or list(reversed(stored.shape.tolist())) != list(tensor.shape)
                    or stored.data.tobytes() != tensor.numpy().tobytes()):
                raise ValueError("Additional original operator archive bytes differ")
        entry.update(status=("frozen gate/up/output retained; additional local targets" if local_operators else
                             "exact old anchors; neutral additional observations"), anchor_exact=True,
                     archive_payloads_exact=True,
                     archive=archive.name, archive_bytes=archive.stat().st_size, archive_sha256=sha256(archive),
                     stages={"layer_14." + name: metrics(tensor.numpy()) for name, tensor in values.items()})
        target.write_text(json.dumps(report, indent=2) + "\n")
    return report
