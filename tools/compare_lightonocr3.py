#!/usr/bin/env python3
"""Compare bounded LightOnOCR-3 original/native traces; strict failures persist.

Owner acceptance is the minimum per-position final-vocabulary cosine, strictly
above the precision bar. Every required stage and decisive logit must exist and
be finite. Decoded, EOS, grounding, cap and strict-stage differences are reported
separately; this tool never publishes or infers a benchmark score.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
from lightonocr3_output import parse_grounding
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from lightonocr3_profile import (MANIFEST, MODEL_ID, REVISION, PROFILE, VOCAB_SIZE,
                                STOP_IDS, REFERENCE_PACKAGES, sha256, trace_shapes, validate_generation_ids)

BARS = {"f32": 0.99, "f16": 0.98, "q8_0": 0.97, "q4_k": 0.94}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite JSON constant")
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def valid_seconds(value):
    return type(value) in (float, int) and 0 <= value <= 24 * 3600 and math.isfinite(value)


def validate_reference(meta):
    require(isinstance(meta, dict), "reference must be an object")
    required = {"schema_version": 2, "model_id": MODEL_ID, "revision": REVISION,
                "source_manifest_sha256": sha256(MANIFEST), "dtype": "float32", "device": "cpu",
                "attention": "eager", "do_sample": False, "use_cache": True,
                "stop_ids": list(STOP_IDS), "forced_eos_token_ids": []}
    for key, expected in required.items():
        require(key in meta and type(meta[key]) is type(expected) and meta[key] == expected,
                "reference provenance/generation mismatch: " + key)
    versions = meta.get("versions", {})
    require(isinstance(versions, dict) and all(versions.get(k) == v for k, v in REFERENCE_PACKAGES.items()),
            "reference package versions differ from the frozen recipe")
    validate_generation_ids(meta["token_ids"], meta["max_new_tokens"])
    require(type(meta.get("hit_cap")) is bool and meta["hit_cap"] == (meta["token_ids"][-1] not in STOP_IDS),
            "reference cap flag contradicts generated IDs")
    require(valid_seconds(meta.get("seconds")), "invalid reference elapsed time")
    for key in ("image_sha256", "decoded_sha256", "forced_ids_sha256"):
        require(valid_sha(meta.get(key)), "missing reference digest: " + key)
    shapes, prompt = trace_shapes(meta["image_size"], meta["mode"], len(meta["token_ids"]))
    require(isinstance(meta.get("arrays"), dict) and set(meta["arrays"]) == set(shapes),
            "reference does not contain all model stages/full-vocabulary logits")
    return shapes, prompt


def read_reference_array(directory, name, descriptor, shape):
    path = directory / f"{name}.npy"
    require(isinstance(descriptor, dict) and descriptor.get("shape") == list(shape),
            "reference descriptor shape differs from the model contract")
    require(valid_sha(descriptor.get("sha256")) and sha256(path) == descriptor["sha256"],
            "frozen reference digest mismatch")
    array = np.load(path, allow_pickle=False)
    require(array.dtype == np.dtype("float32") and array.shape == shape and np.isfinite(array).all(),
            "reference tensor shape/dtype/nonfinite mismatch")
    return array


def native_receipt(directory, meta, precision, arm):
    receipt, execution = load_json(directory / "trace.json"), load_json(directory / "execution.json")
    require(isinstance(receipt, dict) and isinstance(execution, dict), "native receipts must be objects")
    cap = len(meta["token_ids"]) if arm == "forced" else meta["max_new_tokens"]
    for key, expected in {"schema_version": 2, "profile": PROFILE, "mode": meta["mode"],
                          "image_sha256": meta["image_sha256"], "max_new_tokens": cap,
                          "observed": arm != "plain",
                          "forced_ids": meta["token_ids"] if arm == "forced" else []}.items():
        require(key in receipt and type(receipt[key]) is type(expected) and receipt[key] == expected,
                "native input/forcing mismatch: " + key)
    validate_generation_ids(receipt["generated_ids"], cap)
    argmax = receipt.get("argmax_ids")
    require(isinstance(argmax, list) and len(argmax) == len(receipt["generated_ids"])
            and all(type(v) is int and 0 <= v < VOCAB_SIZE for v in argmax), "invalid native argmax IDs")
    require(receipt["generated_ids"] == (meta["token_ids"] if arm == "forced" else argmax),
            "native did not apply the declared greedy/forced prefix")
    require(valid_seconds(receipt.get("seconds")), "invalid native elapsed time")
    require(type(receipt.get("model_bytes")) is int and receipt["model_bytes"] > 0, "missing native model size")
    require(type(execution.get("schema_version")) is int and execution["schema_version"] == 1 and execution.get("build_mode") == "portable"
            and execution.get("precision") == precision and execution.get("arm") == arm,
            "native execution precision/arm/build mismatch")
    require(isinstance(execution.get("git_commit"), str)
            and re.fullmatch(r"[0-9a-f]{40}", execution["git_commit"]), "native source revision missing")
    for key in ("model_sha256", "binary_sha256", "trace_sha256"):
        require(valid_sha(execution.get(key)), "native execution digest missing: " + key)
    require(execution.get("model_bytes") == receipt["model_bytes"], "native model size differs from inspected GGUF")
    require(execution["trace_sha256"] == sha256(directory / "trace.json"), "native trace receipt digest mismatch")
    process = execution.get("process", {})
    require(isinstance(process, dict) and type(process.get("exit")) is int and process["exit"] == 0 and process.get("timed_out") is False,
            "native process did not succeed")
    return receipt, execution


def statistics(reference, actual):
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    if r.shape != a.shape or not r.size or not np.isfinite(r).all() or not np.isfinite(a).all():
        raise ValueError("empty, mismatched or nonfinite trace")
    diff = np.abs(a - r)
    rr, aa = float(np.dot(r, r)), float(np.dot(a, a))
    cosine = float(np.dot(r, a) / np.sqrt(rr * aa)) if rr and aa else (1.0 if np.array_equal(r, a) else 0.0)
    require(all(math.isfinite(v) for v in (rr, aa, cosine)), "nonfinite trace statistics")
    return {"cosine": cosine, "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "rms_error": float(np.sqrt(np.mean(diff * diff))), "reference_rms": float(np.sqrt(rr / r.size)),
            "native_rms": float(np.sqrt(aa / a.size)), "argmax_reference": int(r.argmax()),
            "argmax_native": int(a.argmax()), "count": int(r.size),
            "strict_pass": cosine >= 0.99999 and float(diff.max()) <= 0.002}


def compare(reference, native, autonomous, plain, precision):
    reference, native, autonomous, plain = map(Path, (reference, native, autonomous, plain))
    require(precision in BARS, "unsupported precision")
    stages, errors, decode, meta = {}, [], None, None
    failures = (OSError, ValueError, KeyError, TypeError, OverflowError)
    try:
        meta = load_json(reference / "reference.json")
        shapes, prompt = validate_reference(meta)
        require(sha256(reference / "decoded.txt") == meta["decoded_sha256"], "reference decoded digest mismatch")
        require(sha256(reference / "forced-ids.txt") == meta["forced_ids_sha256"], "reference forced-ID digest mismatch")
        require([int(v) for v in (reference / "forced-ids.txt").read_text().split()] == meta["token_ids"],
                "reference forced IDs differ from generation")
        ids = read_reference_array(reference, "generated_ids", meta["arrays"]["generated_ids"], shapes["generated_ids"])
        require(np.array_equal(ids, meta["token_ids"]), "reference generated IDs differ from observed array")
    except failures as exc:
        errors.append({"stage": "reference_contract", "reason": str(exc)})
        shapes, prompt = {}, []
    receipts = {}
    if shapes:
        for arm, directory in (("forced", native), ("autonomous", autonomous), ("plain", plain)):
            try:
                receipts[arm] = native_receipt(directory, meta, precision, arm)
            except failures as exc:
                errors.append({"stage": arm + "_receipt", "reason": str(exc)})
    for name, shape in sorted(shapes.items()):
        if name == "generated_ids":
            continue
        try:
            r = read_reference_array(reference, name, meta["arrays"][name], shape)
            path = native / f"{name}.f32"
            require(path.stat().st_size == r.size * 4, "native tensor byte count differs from the model contract")
            a = np.fromfile(path, dtype="<f4")
            stages[name] = statistics(r, a)
            if name == "prompt_ids":
                require(np.array_equal(r, prompt) and np.array_equal(r, a), "prompt IDs differ from the pinned input")
            if name == "pixel_values":
                stages[name]["exact"] = bool(np.array_equal(r.reshape(-1), a))
            if name.startswith("logits_step"):
                index = int(name.removeprefix("logits_step"))
                require(int(r.argmax()) == meta["token_ids"][index], "original ID is not the observed greedy argmax")
                if "forced" in receipts:
                    require(int(a.argmax()) == receipts["forced"][0]["argmax_ids"][index],
                            "native argmax receipt differs from the observed logits")
        except failures as exc:
            errors.append({"stage": name, "reason": str(exc)})
    decisive = [stages.get(f"logits_step{i}") for i in range(len(meta["token_ids"]))] if shapes else []
    min_cosine = min((r["cosine"] for r in decisive if r), default=None)
    complete = not errors and all(decisive) and bool(decisive)
    try:
        require(len(receipts) == 3, "all native execution receipts required")
        values = [row[1] for row in receipts.values()]
        for key in ("model_sha256", "model_bytes", "binary_sha256", "git_commit"):
            require(all(v[key] == values[0][key] for v in values), "native arms have different " + key)
        observed, unobserved = receipts["autonomous"][0], receipts["plain"][0]
        ids = observed["generated_ids"]
        text = (autonomous / "decoded.txt").read_text(encoding="utf-8")
        ref_text = (reference / "decoded.txt").read_text(encoding="utf-8")
        plain_text = (plain / "decoded.txt").read_text(encoding="utf-8")
        decode = {"ids_exact": ids == meta["token_ids"], "text_exact": text == ref_text,
                  "observer_text_exact": text == plain_text, "observer_ids_exact": ids == unobserved["generated_ids"],
                  "reference_ids": meta["token_ids"], "native_ids": ids,
                  "reference_text": ref_text, "native_text": text,
                  "reference_hit_cap": meta["hit_cap"], "native_hit_cap": ids[-1] not in STOP_IDS,
                  "reference_chars": len(ref_text), "native_chars": len(text),
                  "reference_seconds": meta["seconds"], "native_seconds": observed["seconds"],
                  "grounding": parse_grounding(text) if meta["mode"] == "grounding" else None}
        require(decode["observer_text_exact"] and decode["observer_ids_exact"], "observer changes native text or IDs")
    except failures as exc:
        errors.append({"stage": "autonomous_output", "reason": str(exc)})
        complete = False
    intermediates = {k: v for k, v in stages.items() if not k.startswith("logits_step")}
    worst = min(intermediates, key=lambda k: intermediates[k]["cosine"], default=None)
    complete = complete and not errors
    return {"schema_version": 2, "scope": "bounded greedy parity; not document quality or benchmark reproduction",
            "precision": precision, "owner_bar_strictly_greater_than": BARS[precision],
            "minimum_final_logit_cosine": min_cosine, "complete_trace": complete,
            "owner_bar_pass": bool(complete and min_cosine is not None and min_cosine > BARS[precision]),
            "errors": errors, "worst_intermediate_stage": worst,
            "strict_failures": [k for k, v in stages.items() if not v["strict_pass"]],
            "stages": stages, "decoded": decode}


def main():
    p = argparse.ArgumentParser(__doc__)
    for arg in ("reference", "native", "autonomous", "plain", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    p.add_argument("--precision", choices=list(BARS), required=True)
    a = p.parse_args()
    report = compare(a.reference, a.native, a.autonomous, a.plain, a.precision)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["owner_bar_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
