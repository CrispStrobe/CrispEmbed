#!/usr/bin/env python3
"""Hosted test-only ggml parity prototype, using immutable original reference code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from ultrabert_contract import sha256

FROZEN_REFERENCE_SOURCE = "0573f1847db68914052a98341a22fceeb1358518"


def parse_log(log):
    stages, exact, totals = {}, {}, None
    for line in log.splitlines():
        fields = line.split("\t")
        if fields[0] == "STAGE":
            if len(fields) != 9 or fields[1] in stages:
                raise ValueError("Invalid/duplicate native stage")
            values = [float(v) for v in fields[2:8]]
            if any(math.isfinite(v) and v < 0 for v in values[1:]):
                raise ValueError("Negative error/norm in native log")
            passed = all(math.isfinite(v) for v in values) and values[0] >= 0.99999 and values[1] <= 0.002
            if int(fields[8]) != int(passed):
                raise ValueError("Native stage verdict mismatch")
            stages[fields[1]] = dict(zip(("cos_min", "max_abs", "mean_abs", "rms", "native_norm", "reference_norm"),
                                        [v if math.isfinite(v) else None for v in values]))
            stages[fields[1]]["pass"] = passed
        elif fields[0] == "EXACT":
            if len(fields) != 3 or fields[1] in exact or fields[2] not in ("0", "1"):
                raise ValueError("Invalid/duplicate exact control")
            exact[fields[1]] = fields[2] == "1"
        elif fields[0] == "TOTAL":
            if len(fields) != 4 or totals is not None:
                raise ValueError("Invalid/duplicate native total")
            totals = [int(v) for v in fields[1:]]
    failures = sum(not row["pass"] for row in stages.values()) + sum(not v for v in exact.values())
    if len(stages) != 292 or len(exact) != 50 or totals != [292, 50, failures]:
        raise ValueError("Incomplete native comparisons")
    return dict(stages=stages, exact=exact, failures=failures)



def parse_operator_log(log):
    expected = {
        "original_embedding/layer_0.attn_norm", "original_residual/layer_14.ffn_norm",
        "original_ffn_norm/layer_14.gate", "original_ffn_norm/layer_14.up",
        "original_ffn_norm/layer_14.ffn_out", "original_gate_up/layer_14.ffn_out",
    }
    return _parse_replay_log(log, "OPERATOR", expected)


def parse_linear_log(log):
    return _parse_replay_log(log, "LINEAR", {
        "original_gate/layer_14.silu", "original_gate_up/layer_14.product",
        "original_product/layer_14.ffn_out",
    })


def _parse_replay_log(log, tag, expected):
    result = {}
    for line in log.splitlines():
        fields = line.split("\t")
        if fields[0] != tag:
            continue
        if len(fields) != 10:
            raise ValueError("Malformed operator record")
        key = fields[1] + "/" + fields[2]
        if key not in expected or key in result:
            raise ValueError("Unexpected/duplicate operator record")
        values = [float(v) for v in fields[3:9]]
        if any(math.isfinite(v) and v < 0 for v in values[1:]):
            raise ValueError("Negative operator error/norm")
        passed = all(math.isfinite(v) for v in values) and values[0] >= 0.99999 and values[1] <= 0.002
        if fields[9] not in ("0", "1") or int(fields[9]) != int(passed):
            raise ValueError("Operator verdict mismatch")
        result[key] = dict(zip(("cos_min", "max_abs", "mean_abs", "rms", "native_norm", "reference_norm"),
                               [v if math.isfinite(v) else None for v in values]))
        result[key]["pass"] = passed
    if set(result) != expected:
        raise ValueError("Incomplete operator audit")
    return result


def verify_frozen_reference(baseline, current):
    if (baseline.get("source_commit") != FROZEN_REFERENCE_SOURCE
            or baseline.get("status") != "completed conversion/reference audit; native parity not evaluated"
            or not baseline.get("reference") or not baseline.get("conversions")):
        raise ValueError("Incomplete/wrong-source frozen reference report")
    if (baseline["reference"] != current.get("reference")
            or baseline["conversions"] != current.get("conversions")):
        raise ValueError("Frozen original arrays/conversion records drifted")



def read_frozen_manifest(baseline, directory):
    manifest = directory / "ultrabert-reference-audit.json"
    expected_raw_sha = baseline.get("audit_provenance", {}).get("raw_report_sha256")
    if not expected_raw_sha or sha256(manifest) != expected_raw_sha:
        raise ValueError("Original artifact report bytes differ from reviewed producer")
    current = json.loads(manifest.read_text())
    verify_frozen_reference(baseline, current)
    return current


def prepare_frozen_artifacts(baseline, directory, scratch):
    """Consume original tensor files without rerunning floating-point Python."""
    current = read_frozen_manifest(baseline, directory)
    import importlib.util
    import gguf
    import numpy as np
    from huggingface_hub import snapshot_download
    from ultrabert_contract import MODEL, REVISION, profile

    reference_dir = directory / "reference"
    for case in current["reference"]["cases"]:
        name = case["archive"]
        if Path(name).name != name or name != case["case"]["id"] + ".gguf":
            raise ValueError("Unexpected original archive name")
        path = reference_dir / name
        if path.stat().st_size != case["archive_bytes"] or sha256(path) != case["archive_sha256"]:
            raise ValueError("Frozen original archive bytes differ: " + name)
        reader = gguf.GGUFReader(str(path))
        actual = {tensor.name: tensor for tensor in reader.tensors}
        if set(actual) != set(case["stages"]):
            raise ValueError("Frozen original archive stage set differs")
        for stage, record in case["stages"].items():
            tensor = actual[stage]
            if (tensor.tensor_type != gguf.GGMLQuantizationType.F32
                    or list(reversed(tensor.shape.tolist())) != record["shape"]
                    or not np.isfinite(tensor.data).all()
                    or hashlib.sha256(tensor.data.tobytes()).hexdigest() != record["payload_sha256"]):
                raise ValueError("Frozen original tensor differs: " + stage)
        del actual, reader
    model_dir = snapshot_download(MODEL, revision=REVISION,
                                 allow_patterns=list(profile()["inspected_file_sha256"]) + ["model.safetensors"])
    spec = importlib.util.spec_from_file_location("ultrabert_converter", ROOT / "models/convert-ultrabert-to-gguf.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    for frozen in current["conversions"]:
        dtype = frozen["dtype"]
        if dtype not in ("f32", "f16"):
            raise ValueError("Unexpected frozen precision")
        converted = converter.convert(model_dir, scratch / (dtype + ".gguf"), dtype)
        if any(value != frozen.get(key) for key, value in converted.items()):
            raise ValueError("Frozen conversion bytes/records differ: " + dtype)
    return current, reference_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--linear-forward-audit", action="store_true", help="Capture extra operators during full original forward")
    parser.add_argument("--linear-audit", action="store_true", help="Observe original SiLU/product and replay isolated down")
    parser.add_argument("--operator-audit", action="store_true", help="Separate original-input RMSNorm/MLP/GLU-down replay")
    parser.add_argument("--frozen-report", type=Path,
                        help="Reviewed complete original report at the immutable pre-port source")
    parser.add_argument("--frozen-artifacts", type=Path,
                        help="Downloaded original artifact; verify every byte instead of rerunning Python")
    args = parser.parse_args()
    if args.linear_forward_audit and not args.linear_audit:
        raise ValueError("Full forward observations require the separate linear audit")
    if args.linear_audit and (not args.operator_audit or args.build_only):
        raise ValueError("Linear audit requires the full separate operator audit")
    if not args.build_only and (args.frozen_report is None or not args.frozen_report.is_file()):
        raise ValueError("Review and freeze the completed original report before native model comparisons")
    output = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "ultrabert"
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "tests/test_ultrabert_contract.py", "-v"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "tests/test_ultrabert_native_log.py", "-v"], cwd=ROOT, check=True)
    # Actions checkout is shallow: obtain the immutable original before comparing
    # bytes. This cannot silently accept a changed dumper as a new blueprint.
    subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", FROZEN_REFERENCE_SOURCE], cwd=ROOT, check=True)
    # Freeze the original and conversion code before the first native output.
    for name in ("tools/ultrabert_contract.py", "models/convert-ultrabert-to-gguf.py",
                 "tools/dump_ultrabert_reference.py", "tools/ci-heavy/ultrabert_reference.py",
                 "tests/test_ultrabert_inference_capture.py",
                 "tests/results/ultrabert/source-audit.json", "tests/results/ultrabert/embed-weight-schema.json"):
        old = subprocess.check_output(["git", "show", FROZEN_REFERENCE_SOURCE + ":" + name], cwd=ROOT)
        if hashlib.sha256(old).hexdigest() != sha256(ROOT / name):
            raise ValueError("Original reference/conversion source drift: " + name)
    build = scratch / "native-build"
    subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF",
                    "-DCRISPEMBED_BUILD_SHARED=OFF"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "test-ultrabert-diff", "-j2"], check=True)
    binary = next(p for p in (build / "test-ultrabert-diff", build / "bin/test-ultrabert-diff") if p.is_file())
    env = dict(os.environ, CRISPEMBED_ULTRABERT="0")
    control = subprocess.run([str(binary), "absent.gguf", "absent-ref.gguf"], env=env, capture_output=True, text=True)
    if control.returncode != 2:
        raise ValueError("Literal-zero opt-in control failed")
    default_env = dict(os.environ)
    default_env.pop("CRISPEMBED_ULTRABERT", None)
    default_control = subprocess.run([str(binary), "absent.gguf", "absent-ref.gguf"],
                                     env=default_env, capture_output=True, text=True)
    if default_control.returncode != 2:
        raise ValueError("Default-off opt-in control failed")
    report = {"source_commit": os.environ["GITHUB_SHA"], "frozen_reference_source": FROZEN_REFERENCE_SOURCE,
              "native_cpp_sha256": sha256(ROOT / "tests/test_ultrabert_diff.cpp"),
              "driver_sha256": sha256(__file__), "build_passed": True,
              "literal_zero_gate_off": True, "default_gate_off": True, "arms": [],
              "scope": "test-only CPU graph over original prepared IDs; not tokenizer/public API or task quality",
              "status": "build-only; model parity not evaluated"}
    target = output / "ultrabert-native.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    if args.build_only:
        return
    baseline = json.loads(args.frozen_report.read_text())
    if args.frozen_artifacts is not None:
        current, reference_directory = prepare_frozen_artifacts(baseline, args.frozen_artifacts, scratch)
        report["reference_mode"] = "byte-exact original tensor archives; no Python inference replay"
    else:
        import ultrabert_reference
        ultrabert_reference.main()
        current = json.loads((output / "ultrabert-reference-audit.json").read_text())
        verify_frozen_reference(baseline, current)
        reference_directory = output / "reference"
        report["reference_mode"] = "exact Python inference and conversion replay"
    report["frozen_report_sha256"] = sha256(args.frozen_report)
    report["original_and_conversion_replay_exact"] = True
    reference = current["reference"]
    if args.linear_audit:
        subprocess.run([sys.executable, "tests/test_ultrabert_linear_capture.py", "-v"], cwd=ROOT, check=True)
        from dump_ultrabert_linear_reference import dump_linear
        report["linear_reference"] = dump_linear(current, reference_directory, output / "linear-reference",
                                                       full_forward=args.linear_forward_audit)
        target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    env["CRISPEMBED_ULTRABERT"] = "1"
    for dtype in ("f32", "f16"):
        for case in reference["cases"]:
            proc = subprocess.run([str(binary), str(scratch / (dtype + ".gguf")),
                                   str(reference_directory / case["archive"])],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log = output / (dtype + "-" + case["case"]["id"] + ".log")
            log.write_text(proc.stdout)
            if proc.returncode not in (0, 1):
                raise ValueError("Native structural/compute error; see " + log.name)
            audit = parse_log(proc.stdout)
            if proc.returncode != int(audit["failures"] > 0):
                raise ValueError("Native exit code disagrees with retained failures")
            ordered = ["input_ids", "attention_mask"] + [name for name in case["stages"]
                       if (name in audit["stages"] or name in audit["exact"])
                       and name not in ("input_ids", "attention_mask")]
            if len(ordered) != 342:
                raise ValueError("Compared stages differ from original reference")
            # Original capture order, never alphabetical map order, defines the
            # earliest difference. Input IDs here are supplied, not retokenized.
            first = next((name for name in ordered if not (audit["stages"][name]["pass"]
                         if name in audit["stages"] else audit["exact"][name])), None)
            audit.update(dtype=dtype, case=case["case"]["id"], first_failed_stage=first,
                         reference_archive_sha256=case["archive_sha256"], log=log.name,
                         log_sha256=sha256(log), compared_order=ordered,
                         uncaptured_native_stages=[name for name in case["stages"] if name not in ordered])
            if args.operator_audit:
                command = [str(binary), str(scratch / (dtype + ".gguf")),
                           str(reference_directory / case["archive"]), "--operator-audit"]
                if args.linear_audit:
                    command.append(str(output / "linear-reference" / case["archive"]))
                probe = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
                probe_log = output / (dtype + "-" + case["case"]["id"] + "-operators.log")
                probe_log.write_text(probe.stdout)
                if probe.returncode != proc.returncode or parse_log(probe.stdout) != parse_log(proc.stdout):
                    raise ValueError("Operator observation changed ordinary native metrics/exit")
                audit["operators"] = parse_operator_log(probe.stdout)
                if args.linear_audit:
                    audit["linear"] = parse_linear_log(probe.stdout)
                audit["operator_primary_metrics_exact"] = True
                audit["operator_log"] = probe_log.name
                audit["operator_log_sha256"] = sha256(probe_log)
            report["arms"].append(audit)
            target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["status"] = "completed prepared-ID native comparisons; all failures retained"
    report["failures"] = sum(arm["failures"] for arm in report["arms"])
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["operator_failures"] = sum(not row["pass"] for arm in report["arms"]
                                      for row in arm.get("operators", {}).values())
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["linear_failures"] = sum(not row["pass"] for arm in report["arms"]
                                    for row in arm.get("linear", {}).values())
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["failures"] or report["operator_failures"] or report["linear_failures"]:
        raise AssertionError("UltraBERT strict native comparisons failed; see retained report/logs")


if __name__ == "__main__":
    main()
