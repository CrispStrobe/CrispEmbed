#!/usr/bin/env python3
"""Hosted MinerU 2605 native measurement against ORIGINAL Python final logits.

The prefill stage probe and freely generated output are retained diagnostics.
The precision decision uses every generated-position raw logit row, conditioned
on the original token prefix. Missing positions or changed sources fail closed.
This tool creates evidence only: no registry, alias, default or publication edit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import mineru2605_validate as protocol

ROOT = Path(__file__).resolve().parents[1]
BARS = {"f32": .99, "f16": .98, "q8_0": .97, "q4_k": .94}
REQUIRED_STAGES = ("input_patches", "vis_patch_embed", *(f"vis_layer_{i}" for i in range(32)),
                   "vis_merger_output", "input_ids", "mrope_positions", "llm_embed",
                   *(f"llm_layer_{i}" for i in range(24)), "llm_final_norm", "llm_logits_last")


def require(value, message):
    if not value:
        raise ValueError(message)


def candidate_contract(parsed, precision):
    """Bind the precision label to actual checkpoint identity and stored tensors."""
    require(precision in BARS, "unknown precision tier")
    manifest = json.loads((ROOT / "models/mineru25-pro-2605-source.json").read_text())
    metadata = parsed["metadata"]
    expected = {"general.architecture": "qwen2vl", "qwen2vl.model_family": "mineru2.5-pro-2605",
                "qwen2vl.source.repository": manifest["repository"], "qwen2vl.source.revision": manifest["revision"],
                "qwen2vl.source.expected_revision": manifest["revision"],
                "qwen2vl.blueprint.revision": "mineru-vl-utils==2.0.7",
                "qwen2vl.blueprint.artifact_sha256": manifest["blueprint"]["wheel_sha256"]}
    require(all(metadata.get(key, {}).get("value") == value for key, value in expected.items()),
            "candidate GGUF is not the exact 2605 source profile")
    hashes = json.loads(metadata.get("qwen2vl.source.file_sha256", {}).get("value", "null"))
    require(hashes == {name: row["sha256"] for name, row in manifest["files"].items()},
            "candidate checkpoint-file provenance differs")
    spec = importlib.util.spec_from_file_location("mineru2605_candidate_converter", ROOT / "models/convert-mineru25-pro-to-gguf.py")
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    expected_shapes = {row.target: list(reversed((1280, 1176) if row.flatten_patch else row.shape))
                       for row in converter.tensor_specs()}
    tensors = parsed["tensors"]
    require(set(tensors) == set(expected_shapes) and
            all(row["stored_shape"] == expected_shapes[name] for name, row in tensors.items()),
            "candidate tensor inventory/shape differs")
    counts = Counter(row["type"] for row in tensors.values())
    allowed, required = {"f32": ({0}, 0), "f16": ({0, 1}, 1), "q8_0": ({0, 1, 8}, 8),
                         "q4_k": ({0, 1, 8, 12}, 12)}[precision]
    require(set(counts) <= allowed and required in counts, "declared precision differs from actual tensor storage")
    if precision == "q4_k":
        # Explicit aligned recipe v1: the original plain request produced
        # 144 Q4_0 fallbacks, which remain rejected. Do not broaden the parser.
        require(counts == {0: 384, 8: 273, 12: 24}, "aligned Q4_K/Q8_0 recipe counts differ")
        for name, row in tensors.items():
            expected_type = 0
            if re.fullmatch(r"l\.blk\.\d+\.ffn_down\.weight", name):
                expected_type = 12
            elif (name == "l.embed_tokens.weight"
                  or re.fullmatch(r"v\.blk\.\d+\.(attn_qkv|attn_proj|ffn_fc1|ffn_fc2)\.weight", name)
                  or re.fullmatch(r"l\.blk\.\d+\.(attn_[qkvo]|ffn_gate|ffn_up)\.weight", name)):
                expected_type = 8
            require(row["type"] == expected_type, "aligned Q4_K role differs: " + name)
    return {"source": expected, "source_file_sha256": hashes, "tensor_count": len(tensors),
            "stored_type_counts": dict(counts), "precision": precision, "file_bytes": parsed["bytes"],
            "recipe": "aligned-q4k-q8-v1" if precision == "q4_k" else precision}


def validate_reference(reference):
    manifest = json.loads((ROOT / "models/mineru25-pro-2605-source.json").read_text())
    require(reference.get("model") == manifest["repository"] and
            reference.get("model_revision") == manifest["revision"] and
            reference.get("blueprint_revision") == "mineru-vl-utils==2.0.7",
            "reference source identity differs from the frozen 2605 protocol")
    require(reference.get("source_sha256") == {n: x["sha256"] for n, x in manifest["files"].items()},
            "reference source byte hashes are incomplete or changed")
    require(reference.get("engine") == "original-python" and
            reference.get("packages", {}).get("transformers") == "5.10.1" and
            reference.get("vision_weights_gguf") is None,
            "reference must be the untouched original Transformers 5.10.1 model")
    require(reference.get("dtype") == "f32" and reference.get("device") == "cpu",
            "this qualification needs the original Python F32 CPU reference")
    ids = reference.get("token_ids", [])
    rows = reference.get("final_logits") or []
    require(bool(ids) and len(rows) == len(ids), "reference final-logit coverage is incomplete")
    require(all(type(i) is int and 0 <= i < 151936 for i in ids), "invalid reference token IDs")
    for index, row in enumerate(rows):
        expected_name = "c0_prefill_logits_last.f32" if index == 0 else f"c0_decode{index}_logits.f32"
        require(row.get("position") == index and row.get("file") == expected_name and
                row.get("elements") == 151936 and isinstance(row.get("sha256"), str) and
                len(row["sha256"]) == 64, "malformed reference logit position")


def precision_result(reference, forced, rows, stage_report, precision, build_mode):
    """Pure receipt decision, shared by the CLI and small negative controls."""
    validate_reference(reference)
    require(precision in BARS, "unknown precision tier")
    require(build_mode == "portable", "precision qualification requires a portable CPU build")
    input_ids = forced.get("input_ids", forced.get("prompt_ids"))
    same_prefix = input_ids == reference.get("input_ids") and forced.get("token_ids") == reference["token_ids"]
    expected_positions = len(reference["token_ids"])
    complete = len(rows) == expected_positions and all(
        row.get("position") == index and row.get("finite") is True and row.get("elements") == 151936 and
        type(row.get("cosine")) in (float, int) and math.isfinite(row["cosine"])
        for index, row in enumerate(rows))
    stages = stage_report.get("stages", [])
    require(isinstance(stages, list) and len(stages) == len(REQUIRED_STAGES) and
            all(isinstance(row, dict) and isinstance(row.get("name"), str) for row in stages) and
            {row["name"] for row in stages} == set(REQUIRED_STAGES),
            "intermediate stage diagnostics must cover every required name exactly once")
    observed = [row for row in stages if row.get("observed") is True and row.get("finite") is True and
                type(row.get("visits")) is int and row["visits"] == 1 and
                type(row.get("elements")) is int and row["elements"] > 0 and
                type(row.get("cos_min")) in (float, int) and math.isfinite(row["cos_min"])]
    require(len(observed) == len(REQUIRED_STAGES), "intermediate stage diagnostics are incomplete")
    worst_stage = min(observed, key=lambda row: row["cos_min"])
    minimum = min((row["cosine"] for row in rows), default=None) if complete else None
    return {
        "schema": "crispembed.mineru2605.precision.v1", "precision": precision,
        "build_mode": build_mode, "bar_exclusive": BARS[precision],
        "metric": "minimum raw final-logit cosine over EVERY original generated position, teacher-forced on original prefixes",
        "reference_engine": "original-python", "same_prefix": same_prefix,
        "final_output_positions": len(rows), "final_output_positions_total": expected_positions,
        "final_output_complete": complete, "final_output_min_cosine": minimum,
        "precision_pass": bool(same_prefix and complete and minimum > BARS[precision]),
        "positions": rows, "worst_stage": worst_stage,
        "strict_stage_pass": stage_report.get("pass", False),
        "magnitude_stage_pass": stage_report.get("pass_v2", False),
        "first_failed_stage": stage_report.get("first_failed_stage"),
        "strict_stages": stages,
        "reference_eos": reference.get("eos"), "reference_capped": reference.get("capped"),
        "full_document_qualified": False, "benchmark_qualified": False,
        "qualification_scope": "this selected request only; aggregate every frozen case separately",
    }


def clean_native_environment():
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("CRISPEMBED_QWEN2VL_FORCE_") or key.startswith("CRISPEMBED_QWEN2VL_STAGE_DUMP"):
            env.pop(key)
    for key in ("CRISPEMBED_QWEN2VL_REF", "CRISPEMBED_NO_KV_CACHE", "CRISPEMBED_BPE_LEGACY_WHITESPACE"):
        env.pop(key, None)
    env.update(CRISPEMBED_MINERU25_PRO="1", CRISPEMBED_MINERU25_PRO_2605="1",
               QWEN2VL_OCR_FORCE_CPU="1")
    return env


def portable_cache(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, rest = line.split(":", 1)
            values[key] = rest.split("=", 1)[1]
    require(values.get("GGML_NATIVE") == "OFF" and values.get("CRISPEMBED_NATIVE") == "OFF",
            "CMakeCache must show GGML_NATIVE=OFF and CRISPEMBED_NATIVE=OFF")


def execute(command, env, stem):
    result = subprocess.run([str(x) for x in command], env=env, capture_output=True, text=True)
    Path(str(stem) + ".stdout.json").write_text(result.stdout)
    Path(str(stem) + ".stderr.log").write_text(result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-stages", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--diff", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--precision", choices=list(BARS), required=True)
    parser.add_argument("--cmake-cache", type=Path, required=True)
    parser.add_argument("--native-logits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    portable_cache(args.cmake_cache)
    from gguf_file_type_canonicalize import parse as parse_gguf
    candidate = candidate_contract(parse_gguf(args.gguf), args.precision)
    protocol.write_json(args.out / "candidate.json", candidate)
    ref = json.loads(args.reference.read_text())
    validate_reference(ref)
    require(protocol.sha256(args.image) == ref["image_sha256"], "candidate image differs from reference")
    require(protocol.sha256(args.reference_stages) == ref["stage_reference_sha256"], "reference stages changed")
    env = clean_native_environment()
    stage = execute([args.diff, args.gguf, args.reference_stages, ref["threads"]], env, args.out / "stages")
    require(stage.returncode in (0, 1), "native stage runner failed; see retained stderr")
    stage_report = json.loads(stage.stdout)
    cmd = [args.native, "--model", args.gguf, "--image", args.image, "--task", ref["task"],
           "--max-tokens", ref["max_tokens"], "--threads", ref["threads"], "--repeat"]
    free = execute(cmd + [len(ref["timing"]["runs_ms"])], env, args.out / "free")
    require(free.returncode == 0, "native free decode failed; see retained stderr")
    native = json.loads(free.stdout)
    native["input_ids"] = native.get("input_ids", native.get("prompt_ids"))
    native["machine_fingerprint"] = protocol.machine_fingerprint()
    decoded = protocol.comparison(ref, native, 1.0)
    protocol.write_json(args.out / "decoded.json", decoded)
    require(not args.native_logits.exists() or not any(args.native_logits.iterdir()), "native logit directory must be empty")
    args.native_logits.mkdir(parents=True, exist_ok=True)
    ids_file = args.out / "original-token-ids.txt"
    ids_file.write_text(" ".join(map(str, ref["token_ids"])) + "\n")
    forced_env = {**env, "CRISPEMBED_QWEN2VL_FORCE_IDS": str(ids_file.resolve()),
                  "CRISPEMBED_QWEN2VL_STAGE_DUMP": str(args.native_logits.resolve()),
                  "CRISPEMBED_QWEN2VL_STAGE_DUMP_STEPS": str(len(ref["token_ids"]))}
    forced_run = execute(cmd + [1], forced_env, args.out / "forced")
    require(forced_run.returncode == 0, "native forced decode failed; see retained stderr")
    forced = json.loads(forced_run.stdout)
    import numpy as np
    from qwen2vl_stage_cosine import logit_stats
    rows = []
    for capture in ref["final_logits"]:
        rpath = args.reference_logits / capture["file"]
        cpath = args.native_logits / capture["file"]
        require(protocol.sha256(rpath) == capture["sha256"], "original logit bytes changed")
        require(rpath.stat().st_size == 151936 * 4 and cpath.stat().st_size == 151936 * 4,
                "missing, truncated or malformed final logits")
        r, c = np.fromfile(rpath, dtype="<f4"), np.fromfile(cpath, dtype="<f4")
        require(bool(np.isfinite(r).all() and np.isfinite(c).all()), "nonfinite final logits")
        rows.append({"position": capture["position"], **logit_stats(r, c),
                     "reference_sha256": capture["sha256"], "native_sha256": protocol.sha256(cpath)})
    report = precision_result(ref, forced, rows, stage_report, args.precision, "portable")
    report.update(reference_sha256=protocol.sha256(args.reference), gguf_sha256=protocol.sha256(args.gguf),
                  image_sha256=ref["image_sha256"], native_executable_sha256=protocol.sha256(args.native),
                  diff_executable_sha256=protocol.sha256(args.diff), cmake_cache_sha256=protocol.sha256(args.cmake_cache),
                  packages=protocol.packages(), threads=ref["threads"], decoded=decoded,
                  candidate=candidate,
                  model=protocol.MODEL_ID, revision=protocol.MODEL_REVISION,
                  blueprint=protocol.BLUEPRINT_REVISION)
    protocol.write_json(args.out / "precision.json", report)
    print(json.dumps({key: report[key] for key in ("precision", "precision_pass", "final_output_complete", "final_output_min_cosine", "strict_stage_pass")}))
    return 0 if report["precision_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
