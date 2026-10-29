#!/usr/bin/env python3
"""NEW authored20 CUDA-only original Josev reference; no historical recapture.

Adapted from frozen tools/dump_josev2b_reference.py atb2edf8c8; same
official math/loader, environment, observer logic and raw-array finite policy.

Uses the pinned publisher module unchanged. No native model outputs are accepted.
All requests/actual tokens/spans are frozen before loading backbone/head weights.
See docs/josev2b.md for the review barrier and unexecuted resource assumptions.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import importlib.util
import inspect
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from josev2b_task_quality_v1 import (CONTRACT, ROOT, environment_versions, panel, protocol, source_guard, declaration, expected_stages,
                            require_cuda_capability, sha256, verify_snapshot)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_loaded_head(head, checkpoint: Path, weights: dict) -> None:
    """Read-only official-loader identity proof, before the first forward.

    Checkpoint BF16/F16/F32 values are promoted by torch to F32; no calibration,
    parameter mutation, silent key filtering or intentional transform is accepted.
    """
    import torch
    from safetensors import safe_open
    state=head.state_dict()
    weights.update(head_checkpoint_sha256=sha256(checkpoint),head_identity_exact=False,head_tensors={})
    with safe_open(str(checkpoint),framework="pt",device="cpu") as source:
        keys=set(source.keys())
        if keys!=set(state):
            weights["head_key_difference"]={"checkpoint_only":sorted(keys-set(state)),"loaded_only":sorted(set(state)-keys)}
            raise ValueError("official loaded head key set differs from published checkpoint")
        for name in sorted(keys):
            original=source.get_tensor(name)
            actual=state[name].detach()
            entry={"shape":list(original.shape),"loaded_shape":list(actual.shape),
                "checkpoint_dtype":str(original.dtype).removeprefix("torch."),
                "loaded_dtype":str(actual.dtype).removeprefix("torch.")}
            weights["head_tensors"][name]=entry
            if tuple(original.shape)!=tuple(actual.shape) or not original.is_floating_point() or actual.dtype!=torch.float32:
                raise ValueError("official loaded head shape/F32 dtype differs: "+name)
            promoted=original.to(dtype=torch.float32).contiguous()
            loaded=actual.to(device="cpu").contiguous()
            if not bool(torch.isfinite(original).all()) or not bool(torch.isfinite(promoted).all()) or not bool(torch.isfinite(loaded).all()):
                raise ValueError("nonfinite published/promoted/loaded head tensor: "+name)
            expected_bytes=promoted.numpy().astype("<f4",copy=False).tobytes()
            actual_bytes=loaded.numpy().astype("<f4",copy=False).tobytes()
            entry.update(checkpoint_promoted_f32_sha256=hashlib.sha256(expected_bytes).hexdigest(),
                         loaded_f32_sha256=hashlib.sha256(actual_bytes).hexdigest())
            if expected_bytes!=actual_bytes:
                raise ValueError("official loaded head differs from published F32 promotion: "+name)
    weights.update(head_identity_exact=True,head_tensor_count=len(state),
                   head_state_elements=sum(value.numel() for value in state.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "capture"])
    parser.add_argument("--reviewed-input-freeze-sha256")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviewed-contract-sha256", required=True)
    args = parser.parse_args()
    if args.reviewed_contract_sha256 != sha256(CONTRACT):
        raise ValueError("explicit reviewed contract digest required")
    cfg = protocol()
    owned = panel()
    if args.phase == "prepare":
        if args.out.exists():
            raise ValueError("output directory must be new; preserve prior attempts")
        args.out.mkdir(parents=True)
    elif (not args.out.is_dir() or not args.reviewed_input_freeze_sha256 or
          sha256(args.out / "input-freeze.json") != args.reviewed_input_freeze_sha256 or
          (args.out / "reference.json").exists()):
        raise ValueError("exact prior new20 tokenizer freeze and fresh capture required")
    started = time.monotonic()
    report = {"schema": cfg["schema"], "status": "rejected", "conclusive": False, "native_outputs": False,
              "contract_sha256": sha256(CONTRACT), "panel_sha256": cfg["panel"]["sha256"],
              "model_revision": cfg["revision"], "cases": []}
    manifest = []
    report["arrays"] = manifest
    try:
        report["source_closure"] = source_guard()
        versions = environment_versions()
        if any(importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")):
            raise ValueError("optional unpinned attention/conv kernels must be absent")
        # No framework is imported by --help or source-only contract controls.
        import numpy as np
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
        from transformers import AutoProcessor
        if sha256(Path(inspect.getfile(qwen))) != cfg["transformers_source"]["sha256"]:
            raise ValueError("installed original Qwen implementation differs")
        if not torch.cuda.is_available():
            raise ValueError("CUDA required; CPU fallback prohibited")
        require_cuda_capability(torch.cuda.get_device_capability())
        free, total = torch.cuda.mem_get_info()
        if free < cfg["scope"]["min_free_cuda_gib"] * 2**30 or total < cfg["scope"]["min_total_cuda_gib"] * 2**30:
            raise ValueError("GPU memory preflight failed")
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.mha.set_fastpath_enabled(False)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.set_num_threads(2)
        report["environment"] = {"packages": versions, "python": platform.python_version(),
            "torch_cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()), "free_bytes": free, "total_bytes": total,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            # Package/version inventory excludes potentially private direct-URL origins.
            "installed_packages": sorted({(dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions() if dist.metadata["Name"]})}
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before process startup")
        report["snapshot"] = verify_snapshot(args.snapshot, source_only=args.phase == "prepare")
        notices = args.out / "source-notices"
        notices.mkdir(exist_ok=True)
        for name in ("LICENSE", "README.md", "joint_schema_model.py", "config.json", "joint_head_config.json",
                     "unsloth_decision_config.json", "tokenizer_config.json", "processor_config.json"):
            shutil.copyfile(args.snapshot / name, notices / name)
        report["source"] = {str(path.relative_to(ROOT)): sha256(path) for path in (
            CONTRACT, ROOT / cfg["panel"]["path"], Path(__file__), ROOT / "tools/josev2b_contract.py",
            ROOT / "tools/josev2b_environment.py", ROOT / "tools/josev2b_task_quality_v1.py", ROOT / cfg["environment_lock"]["path"],
            ROOT / cfg["environment_lock"]["requirements_path"])}
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        spec = importlib.util.spec_from_file_location("josev_original_pinned", args.snapshot / "joint_schema_model.py")
        source = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = source
        spec.loader.exec_module(source)
        processor = AutoProcessor.from_pretrained(args.snapshot, local_files_only=True)
        encoded, unclipped_lengths = [], []
        for case in owned["cases"]:
            record = source.encode_record(processor.tokenizer, case["request"], max_length=512, processor=processor)
            unclipped = source.encode_record(processor.tokenizer, case["request"], max_length=16384, processor=processor)
            unclipped_lengths.append(len(unclipped.input_ids))
            if record.media or len(record.input_ids) > 512:
                raise ValueError("text-only length contract failed")
            for question in record.questions:
                for start, end in (question.question_span, *question.option_spans):
                    if not 0 <= start < end <= len(record.input_ids):
                        raise ValueError("empty/out-of-range semantic span")
            encoded.append(record)
            if len(unclipped.input_ids) != len(record.input_ids):
                raise ValueError("authored factual state would be clipped; reject before weights/forward")
        if sum(unclipped_lengths) > declaration()["original_scope"]["maximum_total_tokens"]:
            raise ValueError("predeclared total token/raw-array resource bound exceeded before weights")
        # Real tokenizer, original serializer, exact spans; this file precedes weights/forward.
        freeze = {"panel_sha256": cfg["panel"]["sha256"], "max_length": 512,
            "records": [{"id": case["id"], "request": case["request"], "encoded": dataclasses.asdict(record),
                         "unclipped_tokens_under_16384": full_length,
                         "attention_mask": [1] * len(record.input_ids)}
                        for case, record, full_length in zip(owned["cases"], encoded, unclipped_lengths)]}
        if args.phase == "prepare":
            write_json(args.out / "input-freeze.json", freeze)
            report.update(status="input-freeze-completed", conclusive=False, original_forward_executed=False,
                          model_weights_loaded=False, cases=freeze["records"])
            report["input_freeze_sha256"] = sha256(args.out / "input-freeze.json")
            return
        # Frozen before any weight payload download by the new GPU recipe.
        if json.loads(json.dumps(freeze)) != json.loads((args.out / "input-freeze.json").read_text()):
            raise ValueError("actual second tokenizer pass differs from frozen20 IDs/masks/spans/order")
        report["input_freeze_sha256"] = sha256(args.out / "input-freeze.json")
        # Real publisher loader: no hand-rebuilt head and no extra calibration.
        model, loaded_processor = source.load_release_model(args.snapshot, device="cuda", dtype=torch.float32,
            local_files_only=True, attn_implementation="eager")
        report["weights"]={}
        verify_loaded_head(model.head,args.snapshot/"joint_head.safetensors",report["weights"])
        if loaded_processor.tokenizer.get_vocab() != processor.tokenizer.get_vocab():
            raise ValueError("loader tokenizer changed")
        language = model.language_model.model.language_model
        embedding = model.language_model.get_output_embeddings().weight
        report["weights"].update({"embedding_tied": embedding.data_ptr() == language.embed_tokens.weight.data_ptr(),
            "head_parameters": sum(p.numel() for p in model.head.parameters()),
            "head_scalars": {name: float(getattr(model.head, name).detach().cpu()) for name in (
                "prior_logit_scale", "joint_logit_scale", "residual_gate")}})
        directory = args.out / "arrays"
        directory.mkdir()
        array_bytes, sequence = 0, 0
        current = ""

        def save(label, value, allow_infinity=False):
            nonlocal array_bytes, sequence
            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().contiguous().numpy()
                if np.isnan(arr).any() or np.isposinf(arr).any() or (not allow_infinity and not np.isfinite(arr).all()):
                    raise ValueError(f"nonfinite activation: {label}")
                array_bytes += arr.nbytes
                if array_bytes > cfg["scope"]["max_array_bytes"]:
                    raise ValueError("bounded raw-array budget exceeded")
                path = directory / f"{sequence:05d}.npy"
                sequence += 1
                np.save(path, arr, allow_pickle=False)
                manifest.append({"case": current, "stage": label, "file": str(path.relative_to(args.out)),
                    "dtype": str(arr.dtype), "shape": list(arr.shape), "bytes": path.stat().st_size,
                    "sha256": sha256(path), "negative_infinity": int(np.isneginf(arr).sum()) if allow_infinity else 0})
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    save(f"{label}/{index}", item, allow_infinity)
            elif isinstance(value, dict):
                for key, item in value.items():
                    save(f"{label}/{key}", item, allow_infinity)

        def answer(case, record, logits):
            return {"model": case["request"]["model"], "answers": {
                question.question_id: source.systemone_answer(case["request"]["questions"][question.question_id],
                    dict(zip(question.option_ids, row.float().softmax(-1).tolist())))
                for question, row in zip(record.questions, logits)},
                "usage": {"input_tokens": len(record.input_ids), "output_tokens": 0}}

        # Keep the same math/fastpath policy for all ordinary and observed passes.
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for case, record in zip(owned["cases"], encoded):
                if time.monotonic() - started > cfg["scope"]["max_wall_seconds"]:
                    raise ValueError("reference wall-time bound exceeded")
                current = case["id"]
                batch = source.collate_records([record], processor.tokenizer.pad_token_id, torch.device("cuda"))
                if not torch.equal(batch["input_ids"].cpu(), torch.tensor([record.input_ids])) or not torch.equal(
                    batch["attention_mask"].cpu(), torch.ones((1, len(record.input_ids)), dtype=torch.long)):
                    raise ValueError("actual collator differs from frozen single-record IDs/mask")
                official = source.systemone(model, processor, case["request"], max_length=512)
                ordinary = model(batch)[0]
                if official != answer(case, record, ordinary):
                    raise ValueError("official SystemOne response differs from captured-logit conversion")
                hooks = []
                names = ["hidden_norm", "memory_projection", "question_projection", "option_question_projection",
                    "global_projection", "option_context_projection", "option_lexical_projection", "type_embedding",
                    "option_summary_norm", "field_norm", "option_norm", "residual_scorer"]
                modules = [(f"head/{name}", getattr(model.head, name)) for name in names]
                modules += [(f"head/evidence/{i}", layer) for i, layer in enumerate(model.head.evidence_layers)]
                modules += [(f"head/field/{i}", layer) for i, layer in enumerate(model.head.layers)]
                modules += [("backbone/embedding", language.embed_tokens), ("backbone/final_norm", language.norm),
                            ("backbone/rotary", language.rotary_emb)]
                modules += [(f"backbone/layer/{i}", layer) for i, layer in enumerate(language.layers)]
                for label, module in modules:
                    hooks.append(module.register_forward_hook(lambda m, a, out, label=label: save(label, out)))
                # Capture actual inputs for span means, lexical lookup and residual features.
                for name in ("hidden_norm", "question_projection", "option_lexical_projection", "residual_scorer"):
                    hooks.append(getattr(model.head, name).register_forward_pre_hook(
                        lambda m, a, name=name: save(f"head/{name}/input", a)))

                input_observations = {}

                def inputs(label, module, args_, kwargs):
                    bound = inspect.signature(module.forward).bind_partial(*args_, **kwargs).arguments
                    for key in ("position_ids", "attention_mask", "causal_mask"):
                        if key in bound:
                            value = bound[key]
                            input_observations[f"{label}/{key}"] = None if value is None else list(value.shape)
                            if value is not None:
                                save(f"{label}/{key}", value, allow_infinity=("mask" in key))

                hooks.append(language.rotary_emb.register_forward_pre_hook(
                    lambda m, a, kw: inputs("backbone/rotary", m, a, kw), with_kwargs=True))
                for index in (0, 3):
                    hooks.append(language.layers[index].register_forward_pre_hook(
                        lambda m, a, kw, index=index: inputs(f"backbone/layer/{index}/input", m, a, kw), with_kwargs=True))
                try:
                    observed = model(batch)[0]
                finally:
                    for hook in hooks:
                        hook.remove()
                repeated = model(batch)[0]
                if input_observations.get("backbone/rotary/position_ids") != [3, 1, len(record.input_ids)]:
                    raise ValueError("actual rotary position capture missing or wrong shape")
                if "backbone/layer/0/input/attention_mask" not in input_observations or input_observations[
                    "backbone/layer/0/input/attention_mask"] is not None:
                    raise ValueError("expected actual unpadded linear mask=None")
                if input_observations.get("backbone/layer/3/input/attention_mask") != [1, 1, len(record.input_ids), len(record.input_ids)]:
                    raise ValueError("actual eager full-attention mask capture missing or wrong shape")
                if len(ordinary) != len(record.questions) or any(
                    not torch.equal(a, b) or not torch.equal(a, c) for a, b, c in zip(ordinary, observed, repeated)):
                    raise ValueError("observer/repeat neutrality failed")
                fields = []
                for question, logits in zip(record.questions, ordinary):
                    save(f"logits/{question.question_id}", logits)
                    probabilities = logits.float().softmax(-1)
                    save(f"probabilities/{question.question_id}", probabilities)
                    fields.append({"id": question.question_id, "option_ids": question.option_ids,
                        "logits": logits.float().tolist(), "probabilities": probabilities.tolist()})
                report["cases"].append({"id": current, "tokens": len(record.input_ids), "fields": fields,
                    "actual_position_mask_inputs": input_observations,
                    "official_response": official, "observer_bit_exact": True, "repeat_bit_exact": True})
        from collections import Counter
        template = declaration()["original_stage_template"]
        for case in owned["cases"]:
            expected = expected_stages(case)
            actual = Counter(row["stage"] for row in manifest if row["case"] == case["id"])
            if actual != expected or sum(actual.values()) != template["per_case_arrays"]:
                raise ValueError("mandatory original20 matched/input stage coverage differs: " + case["id"])
        if len(manifest) != template["total_arrays"] or len(report["cases"]) != 20:
            raise ValueError("exact20case/1480array coverage required")
        from josev2b_task_quality_v1 import evaluate
        report["independent_quality"] = evaluate({row["id"]: row["official_response"] for row in report["cases"]})
        report.update(original_forward_executed=True, model_weights_loaded=True,
            native_outputs=False, old_four_recaptured=False, publication_authorized=False)
        report.update(status="completed", conclusive=True, arrays=manifest, array_payload_bytes=array_bytes,
            elapsed_seconds=time.monotonic() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            limits="New20 authored text requests/10translated scenario pairs only; no native or broad representative/media/long-context qualification")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.out / ("preflight.json" if args.phase == "prepare" else "reference.json"), report)
        if args.phase == "capture":
            write_json(args.out / "raw-manifest.json", manifest)


if __name__ == "__main__":
    main()
