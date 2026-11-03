#!/usr/bin/env python3
"""Hosted bounded qualification; dispatch lightonocr3_1b.sh for the full setup.

GGUFs/cache stay in --scratch. Raw reference/native arrays, immutable identities,
process resource receipts and failures stay in --output for artifact retention.
No publication, registry alias or benchmark claim follows from this runner.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "models"), str(ROOT / "tools")]
from lightonocr3_profile import source_directory, sha256, MODEL_ID, REVISION, PROFILE, MANIFEST
from ci_process import run as execute
from gguf_file_type_canonicalize import parse as parse_gguf


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def model_census(path, precision):
    """Check actual storage before assigning a precision bar; do not trust a filename."""
    parsed = parse_gguf(path)
    metadata = parsed["metadata"]
    for name, expected in {"general.architecture": "lightonocr", "lightonocr.profile": PROFILE,
                           "lightonocr.profile_version": 1, "lightonocr.source.model": MODEL_ID,
                           "lightonocr.source.revision": REVISION}.items():
        if metadata.get(name, {}).get("value") != expected:
            raise ValueError("GGUF profile/source mismatch: " + name)
    counts = Counter(row["type"] for row in parsed["tensors"].values())
    allowed, required = {"f32": ({0}, 0), "f16": ({0, 1}, 1),
                         "q8_0": ({0, 1, 8}, 8), "q4_k": ({0, 1, 8, 12}, 12)}[precision]
    if not set(counts) <= allowed or required not in counts:
        raise ValueError("Declared precision does not match actual GGUF tensor storage")
    return {"sha256": sha256(path), "bytes": parsed["bytes"], "precision": precision,
            "tensor_type_counts": dict(counts), "tensors": parsed["tensors"],
            "general_file_type": metadata.get("general.file_type", {}).get("value")}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    default_scratch = Path(os.environ["HEAVY_SCRATCH"]) / "lightonocr3-1b" if os.environ.get("HEAVY_SCRATCH") else None
    p.add_argument("--scratch", type=Path, default=default_scratch)
    p.add_argument("--model-dir", help="Exact verified snapshot; otherwise download the pinned checkpoint")
    p.add_argument("--formats", default="f32,f16")
    p.add_argument("--model-format", action="append", default=[], metavar="FORMAT=GGUF")
    a = p.parse_args()
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("CRISPEMBED_BUILD_MODE") != "portable"
            or not sys.platform.startswith("linux")):
        p.error("run on hosted Linux with CRISPEMBED_BUILD_MODE=portable")
    if a.scratch is None:
        p.error("--scratch or HEAVY_SCRATCH is required for large model files")
    a.output, a.build, a.scratch = (v.resolve() for v in (a.output, a.build, a.scratch))
    if a.output.is_relative_to(a.scratch) or a.scratch.is_relative_to(a.output):
        p.error("artifact output and model scratch must be separate directory trees")
    if a.output.exists() and any(a.output.iterdir()):
        p.error("output must be empty; previous evidence cannot be overwritten")
    a.output.mkdir(parents=True, exist_ok=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    formats = a.formats.split(",")
    if not formats or len(set(formats)) != len(formats) or any(f not in ("f32", "f16", "q8_0", "q4_k") for f in formats):
        p.error("select unique supported precisions")
    models = {}
    for item in a.model_format:
        if "=" not in item:
            p.error("model-format must be FORMAT=GGUF")
        precision, filename = item.split("=", 1)
        if precision not in formats or precision in models or not filename:
            p.error("model-format must uniquely name a selected precision")
        models[precision] = Path(filename).resolve()
        if models[precision].is_relative_to(a.output):
            p.error("large GGUFs must not be placed in the uploaded evidence directory")
    if any(f in ("q8_0", "q4_k") and f not in models for f in formats):
        p.error("quantized arms need explicit --model-format; no quantization recipe is assumed")
    recipe_path = ROOT / "tests/fixtures/lightonocr3/qualification.json"
    recipe = json.loads(recipe_path.read_text())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    summary = {"schema_version": 2, "scope": recipe["scope"], "build_mode": "portable", "git_commit": commit,
               "recipe_sha256": sha256(recipe_path), "source_manifest_sha256": sha256(MANIFEST), "cases": [],
               "commands": [], "models": {}, "completed": False, "pass": False, "published": False,
               "policy": "minimum per-position full-vocabulary cosine; decoded and strict failures retained"}
    def save():
        write_json(a.output / "summary.json", summary)
    save()
    try:
        cache_path = a.build / "CMakeCache.txt"
        cache = cache_path.read_text()
        for name in ("GGML_NATIVE", "CRISPEMBED_NATIVE"):
            if re.search(r"^" + name + r":BOOL=OFF$", cache, re.MULTILINE) is None:
                raise ValueError(name + " must be OFF in the actual CMakeCache")
        binary = a.build / "lightonocr3-trace"
        binary_sha = sha256(binary)
        build_receipt = json.loads((a.build / "lightonocr3-build.json").read_text())
        if (build_receipt.get("git_commit") != commit or build_receipt.get("binary_sha256") != binary_sha
                or build_receipt.get("cmake_cache_sha256") != sha256(cache_path)):
            raise ValueError("Build receipt differs from the actual source/binary/CMake cache; use the shell entrypoint")
        shutil.copyfile(cache_path, a.output / "CMakeCache.txt")
        write_json(a.output / "build.json", build_receipt)
        memory = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                          if line.startswith("MemAvailable:"))) * 1024
        scratch_free, output_free = shutil.disk_usage(a.scratch).free, shutil.disk_usage(a.output).free
        same_disk = a.scratch.stat().st_dev == a.output.stat().st_dev
        preflight = {"memory_available_bytes": memory, "scratch_free_bytes": scratch_free,
                     "output_free_bytes": output_free, "same_filesystem": same_disk, "load_average": os.getloadavg()}
        write_json(a.output / "preflight.json", preflight)
        if memory < 12 * 2**30 or scratch_free < 12 * 2**30 or output_free < 8 * 2**30 or same_disk and output_free < 20 * 2**30:
            raise ValueError("Need 12 GiB available RAM, 12 GiB model scratch and 8 GiB evidence space")
        (a.output / "requirements-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        source = source_directory(a.model_dir or MODEL_ID, cache_dir=a.scratch / "hf" / "hub")
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("CRISPEMBED_LIGHTON", "LOCR_")) and key != "CRISPEMBED_MAX_PIXELS"}
        env.update(CRISPEMBED_LIGHTONOCR3="1", CRISPEMBED_LIGHTONOCR_FORCE_CPU="1", OMP_NUM_THREADS="4",
                   MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
        def run(command, log, extra=None):
            result = execute(command, log, cwd=ROOT, env=env | (extra or {}), timeout=3600)
            result.update(argv=list(map(str, command)), log=str(log.relative_to(a.output)), log_sha256=sha256(log))
            summary["commands"].append(result)
            save()
            return result
        model_dir = a.scratch / "models"
        model_dir.mkdir(exist_ok=True)
        for precision in formats:
            if precision not in models:
                models[precision] = model_dir / f"candidate-{precision}.gguf"
                if models[precision].exists():
                    raise ValueError("Scratch candidate already exists; use a fresh lane or explicit --model-format")
                result = run([sys.executable, "models/convert-lightonocr-to-gguf.py", "--model", source,
                              "--profile", PROFILE, "--dtype", precision, "--output", models[precision]],
                             a.output / f"convert-{precision}.log")
                if result["exit"] or result["timed_out"]:
                    raise RuntimeError(precision + " conversion failed; receipt/log retained")
            census = model_census(models[precision], precision)
            write_json(a.output / f"model-{precision}.json", census)
            summary["models"][precision] = {k: v for k, v in census.items() if k != "tensors"}
            save()
        from compare_lightonocr3 import compare, validate_reference, load_json
        failed = False
        for case in recipe["cases"]:
            image = ROOT / case["image"]
            if sha256(image) != case["sha256"]:
                raise ValueError("Frozen image mismatch: " + case["id"])
            for mode in case["modes"]:
                directory = a.output / f"{case['id']}-{mode}"
                directory.mkdir()
                ref = directory / "reference"
                result = run([sys.executable, "tools/dump_lightonocr3_reference.py", "--model-dir", source,
                              "--image", image, "--output", ref, "--mode", mode,
                              "--max-new-tokens", recipe["max_new_tokens"]], directory / "original.log")
                try:
                    if result["exit"] or result["timed_out"]:
                        raise ValueError("original reference process failed")
                    meta = load_json(ref / "reference.json")
                    validate_reference(meta)
                    if (meta["image_sha256"] != case["sha256"] or meta["mode"] != mode
                            or meta["max_new_tokens"] != recipe["max_new_tokens"]):
                        raise ValueError("original reference differs from the frozen case")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    summary["cases"].append({"case": case["id"], "mode": mode, "pass": False, "error": str(error)})
                    failed = True
                    save()
                    continue
                for precision, model in models.items():
                    dirs = {arm: directory / f"{precision}-{arm}" for arm in ("forced", "autonomous", "plain")}
                    for arm, output in dirs.items():
                        forced = ref / "forced-ids.txt" if arm == "forced" else "-"
                        cap = len(meta["token_ids"]) if arm == "forced" else recipe["max_new_tokens"]
                        process = run([binary, model, image, output, cap, forced, "plain" if arm == "plain" else "observe"],
                                      directory / f"{precision}-{arm}.log",
                                      {"CRISPEMBED_LIGHTONOCR3_GROUNDING": "1" if mode == "grounding" else "0"})
                        output.mkdir(exist_ok=True)
                        write_json(output / "execution.json", {"schema_version": 1, "build_mode": "portable",
                                   "git_commit": commit, "precision": precision, "arm": arm, "process": process,
                                   "model_sha256": summary["models"][precision]["sha256"],
                                   "model_bytes": summary["models"][precision]["bytes"], "binary_sha256": binary_sha,
                                   "trace_sha256": sha256(output / "trace.json") if (output / "trace.json").is_file() else None})
                    report = compare(ref, dirs["forced"], dirs["autonomous"], dirs["plain"], precision)
                    write_json(directory / f"{precision}-comparison.json", report)
                    summary["cases"].append({"case": case["id"], "mode": mode, "precision": precision,
                                             "pass": report["owner_bar_pass"],
                                             "minimum_final_logit_cosine": report["minimum_final_logit_cosine"],
                                             "decoded": report["decoded"], "strict_failures": report["strict_failures"],
                                             "errors": report["errors"]})
                    failed |= not report["owner_bar_pass"]
                    save()
        if sha256(binary) != binary_sha or any(sha256(path) != summary["models"][name]["sha256"] for name, path in models.items()):
            raise ValueError("Native binary/model changed during the lane")
        summary.update(completed=True, **{"pass": not failed})
        return 1 if failed else 0
    except Exception as error:
        summary["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    raise SystemExit(main())
