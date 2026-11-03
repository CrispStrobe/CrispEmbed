#!/usr/bin/env python3
"""Compare bounded LightOnOCR-3 original/native traces; strict failures persist.

Owner acceptance is the minimum per-position final-vocabulary cosine, strictly
above the precision bar. Every required stage and decisive logit must exist and
be finite. Decoded, EOS, grounding, cap and strict-stage differences are reported
separately; this tool never publishes or infers a benchmark score.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from lightonocr3_output import parse_grounding

BARS = {"f32": 0.99, "f16": 0.98, "q8_0": 0.97, "q4_k": 0.94}


def statistics(reference, actual):
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    if r.shape != a.shape or not r.size or not np.isfinite(r).all() or not np.isfinite(a).all():
        raise ValueError("empty, mismatched or nonfinite trace")
    diff = np.abs(a - r)
    rr, aa = float(np.dot(r, r)), float(np.dot(a, a))
    cosine = float(np.dot(r, a) / np.sqrt(rr * aa)) if rr and aa else (1.0 if np.array_equal(r, a) else 0.0)
    return {"cosine": cosine, "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "rms_error": float(np.sqrt(np.mean(diff * diff))), "reference_rms": float(np.sqrt(rr / r.size)),
            "native_rms": float(np.sqrt(aa / a.size)), "argmax_reference": int(r.argmax()),
            "argmax_native": int(a.argmax()), "count": int(r.size),
            "strict_pass": cosine >= 0.99999 and float(diff.max()) <= 0.002}


def compare(reference, native, autonomous, plain, precision):
    reference, native, autonomous, plain = map(Path, (reference, native, autonomous, plain))
    meta = json.loads((reference / "reference.json").read_text())
    expected = set(meta["arrays"]) - {"generated_ids"}
    required = {"pixel_values", "prompt_ids", "patch_conv", "vis_ln_pre", "proj_out", "decoder_embed", "llm_final_norm"}
    required |= {f"vis_layer_{i}" for i in range(24)} | {f"llm_layer_{i}" for i in range(28)}
    required |= {f"logits_step{i}" for i in range(len(meta["token_ids"]))}
    stages, errors = {}, []
    if expected != required:
        errors.append({"stage": "reference_coverage", "reason": "reference does not contain all model stages/logits"})
    for name in sorted(expected):
        try:
            rp = reference / f"{name}.npy"
            if hashlib.sha256(rp.read_bytes()).hexdigest() != meta["arrays"][name]["sha256"]:
                raise ValueError("frozen reference digest mismatch")
            r = np.load(rp, allow_pickle=False)
            a = np.fromfile(native / f"{name}.f32", dtype="<f4")
            stages[name] = statistics(r, a)
            if name == "prompt_ids" and not np.array_equal(r.reshape(-1), a):
                errors.append({"stage": name, "reason": "prompt IDs differ; decisive logit prefixes are not aligned"})
        except (OSError, ValueError) as exc:
            errors.append({"stage": name, "reason": str(exc)})
    decisive = [stages.get(f"logits_step{i}") for i in range(len(meta["token_ids"]))]
    min_cosine = min((r["cosine"] for r in decisive if r), default=None)
    complete = not errors and all(decisive) and bool(decisive)
    try:
        ids = np.fromfile(autonomous / "generated_ids.f32", dtype="<f4").astype(np.int64).tolist()
        text = (autonomous / "decoded.txt").read_text()
        ref_text = (reference / "decoded.txt").read_text()
        plain_text = (plain / "decoded.txt").read_text()
        decode = {"ids_exact": ids == meta["token_ids"], "text_exact": text == ref_text,
                  "observer_text_exact": text == plain_text, "reference_ids": meta["token_ids"], "native_ids": ids,
                  "reference_text": ref_text, "native_text": text,
                  "reference_hit_cap": meta["hit_cap"], "native_hit_cap": not ids or ids[-1] not in meta["stop_ids"],
                  "reference_chars": len(ref_text), "native_chars": len(text),
                  "reference_seconds": meta["seconds"], "native_seconds": float((autonomous / "seconds.txt").read_text()),
                  "grounding": parse_grounding(text) if meta["mode"] == "grounding" else None}
        complete = complete and decode["observer_text_exact"]
    except (OSError, ValueError) as exc:
        errors.append({"stage": "autonomous_output", "reason": str(exc)})
        decode = None
        complete = False
    intermediates = {k: v for k, v in stages.items() if not k.startswith("logits_step")}
    worst = min(intermediates, key=lambda k: intermediates[k]["cosine"], default=None)
    return {"schema_version": 1, "scope": "bounded greedy parity; not document quality or benchmark reproduction",
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
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["owner_bar_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
