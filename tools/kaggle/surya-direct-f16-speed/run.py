#!/usr/bin/env python3
"""Private hosted Surya 2.5 F16 gate-off/on decoded and timing A/B.

Requires the already-built SPEED_BUILD/test-surya25-run and the sibling
quant runner in this source tree. No build, install, registry write or publish.
The reported timing is one-host, one synthetic crop, not a promotion decision.
"""
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QUANT_PATH = ROOT / "tools/kaggle/surya-ocr-25-quant/run.py"
RECEIPT = ROOT / "tests/results/surya25/native-ab-local.json"
NATIVE = Path(os.environ.get("SPEED_BUILD", "/kaggle/temp/build-speed")) / "test-surya25-run"
WORKING = Path("/kaggle/working")
FROZEN_RECEIPT_SHA256 = "f4a42803a045cf448e783d31306b17ffcc3def88d34385561e4fc0e3233ddc49"
F16_SHA256 = "5f968d8321e8656e17a5b648725911ba83feaa948addd5eaa3768551e46c5358"
SOURCE_PINS = {
    "tools/kaggle/surya-ocr-25-quant/run.py": "a9fbf935c80f9c145e845e638da7bbf4f0b7761ec779f17d34a719ea3c1a5eea",
    "src/qwen35_linear.cpp": "38e342810caff1aca42dfdae1bff67ed4fddc60049fcfbd68d7f01f076275af3",
    "models/qwen35_surya25_profile.py": "08cef7838c5d8af49cf8d50bc7e1e3104e5c15eceead818fec28e2c49065d972",
    "models/convert-qwen35-ocr-to-gguf.py": "b85629c77e37296d6355514fe7e8badc4b16f205395157ce03793710fe67d3f3",
    "tests/surya25_run.cpp": "135b8ce4475e5b5985cddf7711e6c4cd91594b7c57b482e95533d6e137015c10",
}
THREADS = 4
BENCH_CASE = "invoice"  # 24 frozen output tokens; a nontrivial single text line.
ARMS = ("off", "direct")
REPORT = {
    "script_version": "surya-direct-f16-speed-v1",
    "phase": "startup", "status": "running", "cases": {}, "pairs": [],
    "scope": "Private CPU diagnostic on six synthetic crops; no registry, publication, default or semantic promotion.",
    "limits": [
        "Frozen input hashes authenticate PNG bytes, not a separately frozen raw-pixel digest.",
        "The frozen HF receipt has no per-token confidences; confidence differences are direct versus off only.",
        "The direct flag is requested; this CLI does not expose a per-operation dispatch counter.",
        "The prebuilt binary's digest is recorded, but its linkage to the checked source is not independently attested by this runner.",
        "Timing is a three-pair single-crop diagnostic, not a general speed or quality claim.",
    ],
}
REPORT_PATH = None


def save(phase):
    REPORT["phase"] = phase
    if REPORT_PATH is not None:
        temporary = REPORT_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(REPORT, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(REPORT_PATH)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_output(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def confidence_delta(reference, arm):
    # Scores are only comparable while the token prefix is identical.
    left, right = reference["token_ids"], arm["token_ids"]
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    diffs = [arm["confidences"][i] - reference["confidences"][i] for i in range(count)]
    return {"aligned_identical_prefix_tokens": count,
            "signed_deltas_arm_minus_reference": diffs,
            "max_absolute_delta": max(map(abs, diffs), default=None),
            "mean_absolute_delta": sum(map(abs, diffs)) / count if count else None,
            "mean_confidence_delta": arm["mean_confidence"] - reference["mean_confidence"]}


def compare(quant, reference, arm):
    if not isinstance(reference.get("text"), str) or not isinstance(arm.get("text"), str):
        raise ValueError("missing text in comparison")
    if not isinstance(reference.get("token_ids"), list) or not isinstance(arm.get("token_ids"), list):
        raise ValueError("missing token ids in comparison")
    result = quant.compare_arm(reference, arm)
    result["eos_required_and_matched"] = (
        result["stop_exact"] and result["stop"] == "EOS"
        and bool(reference["token_ids"]) and reference["token_ids"][-1] == 2
        and bool(arm["token_ids"]) and arm["token_ids"][-1] == 2)
    return result


def valid_comparison(value):
    return all(value.get(k) is True for k in ("ids_exact", "text_exact", "stop_exact", "eos_required_and_matched"))


def native(quant, model, image, arm, profile=False):
    # No inherited CRISPEMBED flags, HF credentials, profiler or debug gates.
    env = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LANG", "LC_ALL")
           if key in os.environ}
    env.update({"CRISPEMBED_QWEN35_OCR": "1", "CRISPEMBED_SURYA25_FRONTEND": "1",
                "CRISPEMBED_SURYA25_INFERENCE": "1", "CRISPEMBED_QWEN35_DIRECT_F16": "1" if arm == "direct" else "0",
                "CRISPEMBED_QWEN35_PROFILE": "1" if profile else "0"})
    command = [str(NATIVE), str(model), str(image), str(quant.CAP), str(THREADS)]
    start = time.perf_counter()
    run = subprocess.run(command, capture_output=True, text=True, env=env)
    elapsed = time.perf_counter() - start
    result = {"arm": arm, "process_wall_seconds": elapsed, "returncode": run.returncode,
              "stderr_tail": run.stderr[-6000:]}
    if profile:
        profile_lines = [line for line in run.stdout.splitlines() if line.startswith(("PROFILE ", "PROFILE_MATRIX ", "PROFILE_COMPLETE "))]
        result["profile"] = {"lines": profile_lines, "complete": any(
            line.startswith("PROFILE_COMPLETE active=1 dropped=0 ") for line in profile_lines),
            "matrix_calls": sum(int(part.split("=", 1)[1]) for line in profile_lines
                                if line.startswith("PROFILE_MATRIX ")
                                for part in line.split() if part.startswith("calls="))}
    lines = [row for row in run.stdout.splitlines() if row.startswith("SURYA25_RESULT ")]
    if run.returncode != 0 or len(lines) != 1:
        result.update(error="native process failed or returned an invalid result count", stdout_tail=run.stdout[-6000:])
        return result
    try:
        data = json.loads(lines[0].split(" ", 1)[1])
        data["text"] = bytes.fromhex(data.pop("raw_text_utf8_hex")).decode("utf-8", "strict")
        data["stop"] = {1: "EOS", 2: "cap", 3: "context_limit"}.get(data["stop_kind"], "unknown")
        ids, conf = data["token_ids"], data["confidences"]
        if (not isinstance(ids, list) or not ids or not isinstance(conf, list) or len(ids) != len(conf)
                or any(not isinstance(x, int) or isinstance(x, bool) for x in ids)
                or any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in conf)
                or not math.isfinite(data["mean_confidence"])
                or not math.isfinite(data["init_seconds"]) or data["init_seconds"] < 0
                or not math.isfinite(data["wall_seconds"]) or data["wall_seconds"] <= 0):
            raise ValueError("invalid native ids, confidence or timing")
        result.update(data)
    except (KeyError, TypeError, ValueError, UnicodeError) as error:
        result.update(error=f"invalid native result: {error}", stdout_tail=run.stdout[-6000:])
    return result


def main():
    global REPORT_PATH
    if not WORKING.is_dir() or not Path("/kaggle/temp").is_dir():
        raise RuntimeError("requires hosted /kaggle/working and /kaggle/temp")
    run_dir = Path(tempfile.mkdtemp(prefix="surya-direct-f16-speed-", dir=WORKING))
    REPORT_PATH = run_dir / "report.json"
    REPORT["report_path"] = str(REPORT_PATH)
    scratch = Path(tempfile.mkdtemp(prefix="surya-direct-f16-speed-", dir="/kaggle/temp"))
    REPORT["scratch"] = str(scratch)
    os.environ["TMPDIR"] = str(scratch)
    tempfile.tempdir = str(scratch)
    REPORT["free_bytes"] = {"scratch": shutil.disk_usage(scratch).free,
                            "working": shutil.disk_usage(WORKING).free}
    save("admission")

    import hashlib
    def digest(path):
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                sha.update(block)
        return sha.hexdigest()

    REPORT["source"] = {"git_head": git_output("rev-parse", "HEAD"),
                        "baseline_reference": "9498e67e3558a9a0b0ec20f93c40698b2434204e",
                        "runner_sha256": digest(Path(__file__)), "files": {}}
    for rel, expected in SOURCE_PINS.items():
        actual = digest(ROOT / rel)
        REPORT["source"]["files"][rel] = {"sha256": actual, "expected_sha256": expected}
        save("source-pins")
        if actual != expected:
            raise ValueError(f"source pin mismatch: {rel}")
    actual_receipt = digest(RECEIPT)
    REPORT["frozen_receipt"] = {"path": str(RECEIPT.relative_to(ROOT)), "sha256": actual_receipt,
                                "expected_sha256": FROZEN_RECEIPT_SHA256}
    if actual_receipt != FROZEN_RECEIPT_SHA256:
        raise ValueError("frozen receipt digest mismatch")
    if not NATIVE.is_file():
        raise RuntimeError(f"native binary missing: {NATIVE}")
    REPORT["binary"] = {"path": str(NATIVE), "sha256": digest(NATIVE)}
    REPORT["host"] = {"platform": platform.platform(), "machine": platform.machine(),
                      "logical_cpus": os.cpu_count(), "threads": THREADS,
                      "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}
    REPORT["versions"] = {"python": sys.version.split()[0]}
    for package in ("numpy", "gguf", "safetensors", "huggingface-hub", "pillow", "tokenizers", "transformers", "torch"):
        try:
            REPORT["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            REPORT["versions"][package] = None
    quant = load_module("surya25_quant_helpers", QUANT_PATH)
    REPORT["fonts"] = {str(path): digest(path) for path in (quant.FONT_REGULAR, quant.FONT_BOLD)}
    frozen = json.loads(RECEIPT.read_text())
    expected_cases = [row[0] for row in quant.CASE_TABLE]
    if (len(expected_cases) != 6 or len(set(expected_cases)) != 6
            or set(frozen["cases"]) != set(expected_cases) or frozen["revision"] != quant.REVISION):
        raise ValueError("frozen case set or revision mismatch")
    REPORT["model"] = {"repo": quant.MODEL, "revision": quant.REVISION, "profile": quant.PROFILE,
                       "checkpoint_expected_sha256": quant.CHECKPOINT_SHA256,
                       "checkpoint_expected_bytes": quant.CHECKPOINT_BYTES, "cap": quant.CAP,
                       "reference_cap": frozen["cap"], "reference": frozen["reference"]}
    save("source-pins")

    # Isolate the HF cache and disable implicit credentials; token=False on every download.
    os.environ["HF_HOME"] = str(scratch / "hf-home")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    sys.path.insert(0, str(ROOT / "models"))
    import qwen35_surya25_profile as profile
    from huggingface_hub import snapshot_download
    if (profile.MODEL, profile.REVISION, profile.CHECKPOINT_SHA256, profile.CHECKPOINT_BYTES) != (
            quant.MODEL, quant.REVISION, quant.CHECKPOINT_SHA256, quant.CHECKPOINT_BYTES):
        raise ValueError("profile and helper model pins differ")
    REPORT["metadata_expected_sha256"] = profile.FILE_SHA256
    save("download-metadata")
    root = Path(snapshot_download(quant.MODEL, revision=quant.REVISION, token=False,
                                  allow_patterns=list(profile.METADATA_FILES), local_dir=str(scratch / "checkpoint")))
    profile.verify_metadata(str(root))
    REPORT["metadata_verified"] = {name: digest(root / name) for name in profile.METADATA_FILES}
    save("download-checkpoint")
    root = Path(snapshot_download(quant.MODEL, revision=quant.REVISION, token=False,
                                  allow_patterns=list(profile.METADATA_FILES) + ["model.safetensors"], local_dir=str(root)))
    profile.verify_metadata(str(root))
    checkpoint = root / "model.safetensors"
    REPORT["checkpoint"] = {"bytes": checkpoint.stat().st_size, "sha256": digest(checkpoint),
                            "anonymous_token_false": True}
    save("download-checkpoint")
    if (REPORT["checkpoint"]["bytes"] != quant.CHECKPOINT_BYTES
            or REPORT["checkpoint"]["sha256"] != quant.CHECKPOINT_SHA256):
        raise ValueError("checkpoint pin mismatch")
    save("convert")
    converter = load_module("convert_qwen35", ROOT / "models/convert-qwen35-ocr-to-gguf.py")
    f16 = scratch / "surya25-f16.gguf"
    manifest = converter.convert(root, f16, quant.MODEL, quant.REVISION, "f16", (), profile=quant.PROFILE)
    REPORT["convert"] = {"sha256": digest(f16), "expected_sha256": F16_SHA256,
                         "bytes": f16.stat().st_size, "manifest_bytes": manifest["bytes"],
                         "export_tensor_count": manifest["export_tensor_count"],
                         "checkpoint_tensor_count": manifest["checkpoint_tensor_count"]}
    save("convert")
    if REPORT["convert"]["sha256"] != F16_SHA256:
        raise ValueError("F16 digest differs from frozen hosted conversion")

    paths = {}
    for name, width, height, lines in quant.CASE_TABLE:
        save("render-" + name)
        original = frozen["cases"][name]
        if original["size"] != [width, height] or original["lines"] != lines:
            raise ValueError(f"render definition differs from frozen receipt: {name}")
        path, image = quant.render(name, width, height, lines, scratch)
        png_hash = digest(path)
        paths[name] = path
        REPORT["cases"][name] = {"frozen_original": original["original"],
                                 "input_png_sha256": png_hash, "frozen_input_png_sha256": original["input_sha256"],
                                 "input_rgb_bytes_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                                 "png_matches_frozen": png_hash == original["input_sha256"], "arms": {}}
        save("render")
    if not all(c["png_matches_frozen"] for c in REPORT["cases"].values()):
        raise ValueError("rendered PNG hash mismatch; no decoded or timing comparison is valid")

    for name in expected_cases:
        case = REPORT["cases"][name]
        for arm in ARMS:
            save("decoded-" + name + "-" + arm)
            result = native(quant, f16, paths[name], arm)
            case["arms"][arm] = result
            save("decoded-" + name + "-" + arm)
            if "error" not in result:
                result["vs_frozen_hf"] = compare(quant, case["frozen_original"], result)
            save("decoded-" + name + "-" + arm)
        off, direct = case["arms"]["off"], case["arms"]["direct"]
        if "error" not in off and "error" not in direct:
            for ref, other in ((off, direct), (direct, off)):
                other["vs_other_arm"] = compare(quant, ref, other)
                other["confidence_vs_other_arm"] = confidence_delta(ref, other)
        save("decoded-" + name)
    quality_ok = all(
        all("error" not in c["arms"].get(arm, {})
            and valid_comparison(c["arms"].get(arm, {}).get("vs_frozen_hf", {}))
            and valid_comparison(c["arms"].get(arm, {}).get("vs_other_arm", {})) for arm in ARMS)
        for c in REPORT["cases"].values())
    REPORT["decoded_gate_passed"] = quality_ok
    save("decoded-gate")
    if not quality_ok:
        REPORT["status"] = "decoded mismatch or native failure; timing not qualified"
        return 1

    REPORT["profiled_controls"] = {}
    for arm in ARMS:
        save("profile-" + arm)
        control = native(quant, f16, paths["short"], arm, profile=True)
        REPORT["profiled_controls"][arm] = control
        if "error" not in control:
            control["vs_frozen_hf"] = compare(quant, REPORT["cases"]["short"]["frozen_original"], control)
        save("profile-" + arm)
    controls = REPORT["profiled_controls"]
    REPORT["direct_dispatch_observed"] = all(
        "error" not in controls[a] and valid_comparison(controls[a].get("vs_frozen_hf", {}))
        and controls[a].get("profile", {}).get("complete") is True for a in ARMS
    ) and controls["direct"]["profile"]["matrix_calls"] < controls["off"]["profile"]["matrix_calls"]
    if not REPORT["direct_dispatch_observed"]:
        REPORT["status"] = "direct dispatch not confirmed by profile; timing not qualified"
        save("profile-incomplete")
        return 1

    REPORT["timing_method"] = {"case": BENCH_CASE, "pairs_requested": 3,
        "order": [["off", "direct"], ["direct", "off"], ["off", "direct"]],
        "fresh_process_per_arm": True, "profile_enabled": False, "threads": THREADS,
        "recognition_ratio_definition": "off wall_seconds / direct wall_seconds; >1 favors direct",
        "process_ratio_definition": "off process_wall_seconds / direct process_wall_seconds; >1 favors direct",
        "warmup": "six-case decoded runs precede timing; no dedicated timing warmup; filesystem cache not controlled"}
    for index, order in enumerate(REPORT["timing_method"]["order"], 1):
        pair = {"index": index, "order": order, "arms": {}}
        REPORT["pairs"].append(pair)
        for arm in order:
            save(f"timing-pair-{index}-{arm}")
            result = native(quant, f16, paths[BENCH_CASE], arm)
            pair["arms"][arm] = result
            save(f"timing-pair-{index}-{arm}")
            if "error" not in result:
                result["vs_frozen_hf"] = compare(quant, REPORT["cases"][BENCH_CASE]["frozen_original"], result)
                result["vs_quality_arm"] = compare(quant, REPORT["cases"][BENCH_CASE]["arms"][arm], result)
            save(f"timing-pair-{index}-{arm}")
        off, direct = pair["arms"]["off"], pair["arms"]["direct"]
        if all("error" not in r and valid_comparison(r.get("vs_frozen_hf", {}))
               and valid_comparison(r.get("vs_quality_arm", {})) for r in (off, direct)):
            pair["direct_vs_off"] = compare(quant, off, direct)
            pair["confidence_direct_vs_off"] = confidence_delta(off, direct)
            if valid_comparison(pair["direct_vs_off"]):
                pair["recognition_off_over_direct"] = off["wall_seconds"] / direct["wall_seconds"]
                pair["process_off_over_direct"] = off["process_wall_seconds"] / direct["process_wall_seconds"]
        save(f"timing-pair-{index}")
    ratios = [pair["recognition_off_over_direct"] for pair in REPORT["pairs"] if "recognition_off_over_direct" in pair]
    process_ratios = [pair["process_off_over_direct"] for pair in REPORT["pairs"] if "process_off_over_direct" in pair]
    REPORT["timing_summary"] = {"valid_pair_count": len(ratios), "requested_pair_count": 3,
        "recognition_off_over_direct_ratios": ratios, "recognition_median": statistics.median(ratios) if ratios else None,
        "process_off_over_direct_ratios": process_ratios, "process_median": statistics.median(process_ratios) if process_ratios else None}
    if len(ratios) != 3:
        REPORT["status"] = "incomplete: one or more timing pairs failed decoded gates"
        save("timing-incomplete")
        return 1
    REPORT["status"] = "complete: decoded gates and all three diagnostic timing pairs passed"
    save("complete")
    return 0


if __name__ == "__main__":
    REPORT["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        code = main()
    except BaseException as error:
        REPORT.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc()[-10000:])
        code = 2
    REPORT["exit_code"] = code
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        save(REPORT["phase"])
    except BaseException:
        print("REPORT_WRITE_FAILED", traceback.format_exc(), file=sys.stderr, flush=True)
        code = 2
    print("SURYA_DIRECT_F16_SPEED_RESULT", json.dumps({"status": REPORT["status"],
          "phase": REPORT["phase"], "report": str(REPORT_PATH) if REPORT_PATH else None,
          "exit_code": code}), flush=True)
    raise SystemExit(code)
