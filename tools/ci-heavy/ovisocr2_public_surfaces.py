"""Hosted, model-backed OvisOCR2 public C/Python/CLI/HTTP/Rust/Dart audit.

No GGUF is fetched by this script. The caller supplies the preserved model-audit
artifact after --preflight verifies a reviewed positive F16 receipt and run.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "tests/results/qwen35-ocr/ovisocr2-model-audit.json"
LIGHTON_RECEIPT = ROOT / "tests/results/qwen35-ocr/receipt.json"
GATES = ("CRISPEMBED_QWEN35_OCR", "CRISPEMBED_OVISOCR2_FRONTEND",
         "CRISPEMBED_OVISOCR2_INFERENCE")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_receipt(data):
    """Pure stdlib gate; does not import native or model packages."""
    if not isinstance(data, dict) or not isinstance(data.get("independent_review"), dict):
        raise ValueError("Reviewed Ovis model-audit receipt is absent")
    review = data["independent_review"]
    producer = data.get("producer", data)
    if review.get("f16_positive") is not True:
        raise ValueError("Independent F16-positive model review is required")
    if (producer.get("status") not in ("completed diagnostic audit", "completed with failures",
                                       "failed hosted model audit")
            or not isinstance(producer.get("source_commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", producer["source_commit"])):
        raise ValueError("Model producer status/source differs")
    commands = producer.get("commands", {})
    required = ("f16-delta", "f16-vision", "f16-decoder-prepared", "f16-decoder-native-vision",
                "f16-decoder-native-positions", "f16-decoder-native-frontend", "f16-context")
    if any(commands.get(name, {}).get("exit_code") != 0 or
           commands[name].get("passed") is not True or
           commands[name].get("comparison_coverage") is not True for name in required):
        raise ValueError("Reviewed F16 native model-backed comparisons are incomplete")
    if not re.fullmatch(r"[0-9a-f]{64}", commands["f16-context"].get("sha256", "")):
        raise ValueError("F16 low-level/context log identity is missing")
    export = producer.get("exports", {}).get("f16", {})
    readable = producer.get("readable", {})
    original = producer.get("original_capture", {})
    expected = original.get("decoded")
    generated = original.get("generated_ids")
    generation = original.get("generation_result", {})
    cache_error = original.get("cached_vs_full_prefix_max_abs")
    control = original.get("observational_control", {})
    if (not re.fullmatch(r"[0-9a-f]{64}", export.get("model_sha256", ""))
            or export.get("tensor_count") != 473
            or not re.fullmatch(r"[0-9a-f]{64}", readable.get("png_sha256", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", readable.get("raw_rgb_sha256", ""))
            or not isinstance(expected, list) or len(expected) != 1
            or not isinstance(expected[0], str) or not expected[0]
            or not isinstance(generated, list) or len(generated) != 1
            or not isinstance(generated[0], list) or len(generated[0]) > 12
            or generation.get("max_new_tokens") != 12
            or generation.get("eos_ids") != [248044, 248046]
            or generation.get("generated_token_count") != len(generated[0])
            or control.get("prefill_logits_exact") is not True
            or control.get("greedy_ids_exact") is not True
            or original.get("cached_vs_full_prefix_same_argmax") is not True
            or not isinstance(cache_error, (int, float)) or not math.isfinite(cache_error)
            or cache_error > .002):
        raise ValueError("Reviewed F16 model/image or bounded original text is missing")
    return {"source_commit": producer["source_commit"], "producer_status": producer["status"],
            "producer_failures": producer.get("failures", []),
            "model_sha256": export["model_sha256"],
            "image_sha256": readable["png_sha256"], "rgb_sha256": readable["raw_rgb_sha256"],
            "expected_text": expected[0], "original_generation": generation,
            "original_generated_ids": generated[0],
            "original_cache_max_abs": cache_error,
            "f16_context_log_sha256": commands["f16-context"].get("sha256")}


def service_gate(run_id, head, call=subprocess.check_output):
    if not re.fullmatch(r"[1-9][0-9]*", str(run_id)):
        raise ValueError("An explicit model-audit run ID is required")
    result = json.loads(call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/runs/{run_id}"]))
    if (result.get("status") != "completed" or result.get("conclusion") not in ("success", "failure")
            or result.get("head_sha") != head
            or result.get("path") != ".github/workflows/ovisocr2-metadata.yml"):
        raise ValueError("Model-audit service run/head/workflow differs")
    return {key: result[key] for key in ("id", "status", "conclusion", "head_sha", "path")}


def require_artifacts(folder, pins):
    folder = Path(folder)
    matches = {}
    for key, name, digest in (("model", "ovisocr2-f16.gguf", pins["model_sha256"]),
                              ("image", "owned-readable.png", pins["image_sha256"])):
        candidates = list(folder.rglob(name))
        if len(candidates) != 1 or sha(candidates[0]) != digest:
            raise ValueError(f"Preserved model-audit artifact {name} missing or SHA differs")
        matches[key] = candidates[0]
    return matches


def require_lighton_artifact(path):
    evidence = json.loads(LIGHTON_RECEIPT.read_text())
    expected = evidence.get("weights", {})
    if (evidence.get("model") != "lightonai/LightOnOCR-3-0.8B"
            or not re.fullmatch(r"[0-9a-f]{64}", expected.get("sha256", ""))
            or not Path(path).is_file() or Path(path).stat().st_size != expected.get("bytes")
            or sha(path) != expected["sha256"]):
        raise ValueError("Real LightOn GGUF differs from its qualified public receipt")
    return {"sha256": expected["sha256"], "bytes": expected["bytes"]}


def metadata_complete_truncation(model, target):
    """Copy only a verified GGUF header plus 64 tensor bytes for a load-fail control."""
    from gguf import GGUFReader
    reader = GGUFReader(str(model))
    if (len(reader.tensors) != 473
            or reader.fields["qwen35_ocr.format_version"].contents() != 2
            or reader.fields["qwen35_ocr.profile"].contents() != "ovisocr2"):
        raise ValueError("Truncation source is not the qualified format-2 Ovis checkpoint")
    offset = int(reader.data_offset)
    del reader
    cut = offset + 64
    if offset < 24 or cut >= Path(model).stat().st_size:
        raise ValueError("GGUF metadata/data boundary is invalid")
    with Path(model).open("rb") as source, Path(target).open("xb") as output:
        remaining = cut
        while remaining:
            block = source.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Qualified GGUF ended before the metadata boundary")
            output.write(block)
            remaining -= len(block)
    return {"sha256": sha(target), "bytes": cut, "metadata_bytes": offset,
            "retained_tensor_bytes": 64, "tensor_count": 473,
            "source_sha256": sha(model)}


def validate_stage_cache_probe(command, log_text, expected_text, truncated_path):
    """Require actual two-stage C API traces, not final fallback text alone."""
    rows = [line.removeprefix("RESULT stage-cache ") for line in log_text.splitlines()
            if line.startswith("RESULT stage-cache ")]
    if len(rows) != 1:
        raise ValueError("Ovis two-stage result record missing or ambiguous")
    value = json.loads(rows[0])
    metrics = value.get("metrics", [])
    if (command.get("exit_code") != 0 or len(metrics) != 2
            or value.get("stages_tried") != 2
            or [row.get("index") for row in metrics] != [1, 2]
            or [row.get("engine") for row in metrics] != ["ovisocr2", "ovisocr2"]
            or metrics[0].get("text_chars") != len(expected_text.encode("utf-8"))
            or metrics[0].get("accepted") is not False
            or metrics[1].get("text_chars") != 0
            or metrics[1].get("accepted") is not False
            or "truncated/corrupt GGUF" not in log_text
            or str(truncated_path) not in log_text
            or "ocr_orchestrator: qwen35/ovis load failed" not in log_text):
        raise ValueError("Second Ovis stage reused first model or truncated weight load was not proven")
    return {"stages_tried": 2, "metrics": metrics,
            "fallback_text": value.get("fallback_text"),
            "metadata_opened_second": True, "second_weight_load_failed": True}


def run_cmd(label, args, out, *, env=None, cwd=ROOT, timeout=1800):
    log = out / (label + ".log")
    with log.open("w") as stream:
        try:
            result = subprocess.run([str(a) for a in args], cwd=cwd, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            stream.write(f"Driver error: {type(exc).__name__}: {exc}\n")
            code = -1
    content = log.read_text(errors="replace")
    return {"exit_code": code, "log": log.name, "log_sha256": sha(log),
            "first_failure": next((line for line in content.splitlines()
                                   if re.search(r"\b(?:FAIL|ERROR|Error:|Traceback)\b", line)), None),
            "metric_lines": [line for line in content.splitlines()
                             if line.startswith(("PASS", "FAIL", "ABI", "RESULT", "DECODED"))],
            "bytes": log.stat().st_size, "passed": code == 0 and log.stat().st_size > 0}


def env_with_gates(enabled=True):
    env = dict(os.environ)
    for gate in GATES:
        env[gate] = "1" if enabled else "0"
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    return env


def image_data(path, pins, scratch):
    import numpy as np
    from PIL import Image
    rgb = np.asarray(Image.open(path).convert("RGB"))
    raw = scratch / "owned-readable.rgb"
    raw.write_bytes(rgb.tobytes())
    if sha(raw) != pins["rgb_sha256"] or rgb.shape != (448, 448, 3):
        raise ValueError("Frozen readable raw RGB identity/shape differs")
    return rgb, raw


def abi_arm(lib, model, rgb, expected):
    import numpy as np
    dll = ctypes.CDLL(str(lib))
    dll.crispembed_ocr_model_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    dll.crispembed_ocr_model_init.restype = ctypes.c_void_p
    dll.crispembed_ocr_model_free.argtypes = [ctypes.c_void_p]
    dll.crispembed_ocr_model_set_max_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int]
    dll.crispembed_ocr_model_get_max_tokens.argtypes = [ctypes.c_void_p]
    dll.crispembed_ocr_model_get_max_tokens.restype = ctypes.c_int
    dll.crispembed_ocr_model_set_prompt.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    dll.crispembed_ocr_model_set_prompt.restype = ctypes.c_int
    dll.crispembed_ocr_model_recognize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                    ctypes.POINTER(ctypes.c_int)]
    dll.crispembed_ocr_model_recognize.restype = ctypes.c_void_p
    dll.crispembed_ocr_model_confidences.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    dll.crispembed_ocr_model_confidences.restype = ctypes.POINTER(ctypes.c_float)
    dll.crispembed_ocr_model_mean_confidence.argtypes = [ctypes.c_void_p]
    dll.crispembed_ocr_model_mean_confidence.restype = ctypes.c_float
    ctx = dll.crispembed_ocr_model_init(str(model).encode(), 2)
    if not ctx:
        raise ValueError("C ABI model initialization failed")
    try:
        dll.crispembed_ocr_model_set_max_tokens(ctx, 12)
        assert dll.crispembed_ocr_model_get_max_tokens(ctx) == 12
        assert dll.crispembed_ocr_model_set_prompt(ctx, b"different instruction") == 0
        assert dll.crispembed_ocr_model_set_prompt(ctx, None) == 1
        length = ctypes.c_int()
        ptr = dll.crispembed_ocr_model_recognize(ctx, rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                                                  rgb.shape[1], rgb.shape[0], 3, ctypes.byref(length))
        if not ptr or ctypes.string_at(ptr, length.value).decode("utf-8") != expected:
            raise ValueError("C ABI decoded text differs from reviewed readable output")
        n = ctypes.c_int()
        p = dll.crispembed_ocr_model_confidences(ctx, ctypes.byref(n))
        scores = list(np.ctypeslib.as_array(p, shape=(n.value,))) if p and n.value else []
        mean = float(dll.crispembed_ocr_model_mean_confidence(ctx))
        if not (0 < len(scores) <= 12 and all(math.isfinite(x) and 0 <= x <= 1 for x in scores)
                and math.isfinite(mean) and abs(mean - sum(scores)/len(scores)) <= 1e-5):
            raise ValueError("C ABI selected-token confidences invalid")
        return {"text_exact": True, "text_bytes": length.value, "confidence_count": n.value,
                "mean_confidence": mean}
    finally:
        dll.crispembed_ocr_model_free(ctx)


def python_arm(repo, lib, model, image, rgb, expected):
    sys.path.insert(0, str(repo / "python"))
    from crispembed import CrispOcrModel, CrispOcrOrchestrator
    recognizer = CrispOcrModel(str(model), n_threads=2, lib_path=str(lib))
    try:
        recognizer.set_max_tokens(12)
        assert recognizer.get_max_tokens() == 12
        assert recognizer.set_prompt("different instruction") is False
        assert recognizer.set_prompt(None) is True
        text = recognizer.recognize(rgb)
        scores = recognizer.confidences()
        mean = recognizer.mean_confidence
        if (text != expected or not 0 < len(scores) <= 12
                or not math.isfinite(mean) or abs(mean - sum(scores)/len(scores)) > 1e-5):
            raise ValueError("Python unified output/confidence differs")
    finally:
        recognizer.__del__()
    pipe = CrispOcrOrchestrator(stages=[{"engine": 21, "model_a": str(model),
                                        "vlm_max_tokens": 12}], router=False, n_threads=2,
                                    lib_path=str(lib))
    try:
        result = pipe.run(str(image))
        if result["text"] != expected or result["n_regions"] != 1:
            raise ValueError("Python stage 21 output differs")
    finally:
        pipe.__del__()
    return {"text_exact": True, "confidence_count": len(scores), "mean_confidence": mean,
            "stage_text_exact": True, "stage_n_regions": result["n_regions"]}


def cli_http_arm(build, model, lighton, image, expected, out, report, save, enabled):
    env = env_with_gates(enabled)
    cli = build / "crispembed"
    for label, args, key in (
        ("cli-unified", ["-m", model, "--ocr", image], "text"),
        ("cli-stage21", ["--ocr-pipeline", image, "--ocr-engine", "ovisocr2",
                         "--ocr-rec", model], "full_text")):
        result = run_cmd(label, [cli, *args, "--offline", "--ocr-max-tokens", "12", "--json", "-t", "2"],
                         out, env=env)
        report["arms"][label] = result
        if result["passed"]:
            lines = [line for line in (out/result["log"]).read_text().splitlines() if line.startswith("{")]
            try:
                answer = json.loads(lines[-1])
                result["decoded"] = answer[key]
                if label == "cli-stage21":
                    result["mean_confidence"] = answer.get("mean_confidence")
                    result["n_regions"] = answer.get("n_regions")
                result["passed"] = answer[key] == expected
                if label == "cli-stage21":
                    result["passed"] &= answer.get("n_regions") == 1
            except (IndexError, KeyError, ValueError) as exc:
                result["passed"] = False
                result["parse_error"] = str(exc)
        save()


    for label, engine, extra, use_model in (
        ("cli-stage21-custom-prompt", "ovisocr2", ["--ocr-prompt", "different instruction"], model),
        ("cli-stage21-lighton-profile", "ovisocr2", [], lighton),
        ("cli-stage20-ovis-profile", "qwen35-ocr", [], model)):
        result = run_cmd(label, [cli, "--ocr-pipeline", image, "--ocr-engine", engine,
                                 "--ocr-rec", use_model, "--ocr-max-tokens", "12", *extra], out, env=env)
        rejection = (out/result["log"]).read_text(errors="replace").lower()
        hint = "profile" if "profile" in label else "ovisocr2"
        result["passed"] = result["exit_code"] > 0 and hint in rejection
        report["arms"][label] = result
        save()
    for gate in GATES:
        disabled = env_with_gates()
        disabled[gate] = "0"
        label = "cli-stage21-gate0-" + gate.lower()
        result = run_cmd(label, [cli, "--ocr-pipeline", image, "--ocr-engine", "ovisocr2",
                                 "--ocr-rec", model, "--ocr-max-tokens", "12"], out, env=disabled)
        rejection = (out/result["log"]).read_text(errors="replace").lower()
        result["passed"] = result["exit_code"] > 0 and "ovisocr2" in rejection
        report["arms"][label] = result
        save()
    server = build/"crispembed-server"
    for label, engine, use_model, extra, disabled_gate in (
        ("http-start-custom-prompt", "ovisocr2", model,
         ["--ocr-prompt", "different instruction"], None),
        ("http-start-lighton-profile", "ovisocr2", lighton, [], None),
        ("http-start-ovis-on-stage20", "qwen35-ocr", model, [], None),
        *(("http-start-gate0-" + gate.lower(), "ovisocr2", model, [], gate) for gate in GATES)):
        startup_env = env_with_gates()
        if disabled_gate: startup_env[disabled_gate] = "0"
        result = run_cmd(label, [server, "--offline", "--ocr-pipeline", "--ocr-engine", engine,
                                 "--ocr-rec", use_model, "--ocr-max-tokens", "12", *extra,
                                 "--host", "127.0.0.1", "--port", "0"], out,
                         env=startup_env, timeout=15)
        rejection = (out/result["log"]).read_text(errors="replace").lower()
        hint = "profile" if ("profile" in label or "stage20" in label) else "ovisocr2"
        result["passed"] = result["exit_code"] > 0 and hint in rejection
        report["arms"][label] = result
        save()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    path = out / "http-stage21.log"
    with path.open("w") as log:
        proc = subprocess.Popen([str(build/"crispembed-server"), "--offline", "--ocr-pipeline",
                                 "--ocr-engine", "ovisocr2", "--ocr-rec", str(model),
                                 "--ocr-max-tokens", "12", "--host", "127.0.0.1",
                                 "--port", str(port), "--image-root", str(image.parent)],
                                env=env, stdout=log, stderr=subprocess.STDOUT)
        result = {"passed": False, "log": path.name}
        try:
            deadline = time.monotonic() + 90
            while True:
                if proc.poll() is not None:
                    raise RuntimeError("HTTP service exited before ready")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                        break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("HTTP service readiness timeout")
                    time.sleep(.25)
            request = urllib.request.Request(f"http://127.0.0.1:{port}/ocr/pipeline",
                                             json.dumps({"image": str(image)}).encode(),
                                             {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                answer = json.load(response)
            result.update(decoded=answer.get("text"), n_regions=answer.get("n_regions"),
                          mean_confidence=answer.get("mean_confidence"),
                          passed=answer.get("text") == expected and answer.get("n_regions") == 1
                          and isinstance(answer.get("mean_confidence"), (int, float)))
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    result.update(log_sha256=sha(path), bytes=path.stat().st_size)
    report["arms"]["http-stage21"] = result
    save()


def bindings_arm(repo, build, model, raw, image, expected, out, scratch, report, save):
    env = env_with_gates()
    env["CRISPEMBED_SYS_LIB_DIR"] = str(build)
    env["CARGO_BUILD_JOBS"] = "2"
    env["LD_LIBRARY_PATH"] = str(build) + ":" + str(build/"ggml/src") + ":" + env.get("LD_LIBRARY_PATH", "")
    rust = scratch / "rust-ovis-public"
    (rust/"src").mkdir(parents=True, exist_ok=True)
    (rust/"Cargo.toml").write_text('[package]\nname="ovisocr2-probe"\nversion="0.0.0"\nedition="2021"\n[dependencies]\ncrispembed={path='+json.dumps(str(repo/'crispembed'))+'}\n')
    shutil.copyfile(repo/"tests/ovisocr2_public_bindings.rs", rust/"src/main.rs")
    report["arms"]["rust-build"] = run_cmd("rust-build", ["cargo", "build", "--manifest-path", rust/"Cargo.toml"], out, env=env)
    save()
    if report["arms"]["rust-build"]["passed"]:
        report["arms"]["rust-run"] = run_cmd("rust-run", [rust/"target/debug/ovisocr2-probe",
            model, raw, "448", "448", image, expected], out, env=env)
        report["arms"]["rust-run"]["passed"] &= len([s for s in report["arms"]["rust-run"]["metric_lines"]
            if s.startswith("PASS Rust")]) == 2
        save()
    dart = scratch/"dart-ovis-public"
    (dart/"lib").mkdir(parents=True, exist_ok=True)
    for name in ("crispembed.dart", "crispembed_bindings.dart"):
        shutil.copyfile(repo/"flutter/crispembed/lib/src"/name, dart/"lib"/name)
    (dart/"pubspec.yaml").write_text("name: ovisocr2_probe\nenvironment:\n  sdk: '>=3.0.0 <4.0.0'\ndependencies:\n  ffi: ^2.0.0\n")
    shutil.copyfile(repo/"tests/ovisocr2_public_bindings.dart", dart/"main.dart")
    for label, args in (("dart-pub", ["dart", "pub", "get"]),
                        ("dart-analyze", ["dart", "analyze", "--no-fatal-warnings"]),
                        ("dart-run", ["dart", "run", dart/"main.dart", model, raw, "448", "448", image,
                                      build/"libcrispembed.so", expected])):
        if label != "dart-pub" and not report["arms"]["dart-pub"]["passed"]:
            break
        report["arms"][label] = run_cmd(label, args, out, env=env, cwd=dart)
        if label == "dart-run":
            report["arms"][label]["passed"] &= len([s for s in report["arms"][label]["metric_lines"]
                if s.startswith("PASS Dart")]) == 2
        save()


def compare_confidences(arms):
    """Public API exposes scores, while IDs and stops belong to low-level receipt."""
    baseline = arms.get("c-abi", {})
    if not baseline.get("passed"):
        raise ValueError("C ABI confidence baseline absent")
    count, mean = baseline["confidence_count"], baseline["mean_confidence"]
    python = arms.get("python", {})
    if (not python.get("passed") or python.get("confidence_count") != count
            or abs(python.get("mean_confidence", -1) - mean) > 1e-5):
        raise ValueError("Python confidence count/mean differs from C ABI")
    for name, tolerance in (("cli-stage21", .0006), ("http-stage21", .00006)):
        arm = arms.get(name, {})
        if (not arm.get("passed") or not isinstance(arm.get("mean_confidence"), (int, float))
                or abs(arm["mean_confidence"] - mean) > tolerance):
            raise ValueError(name + " mean confidence differs from C ABI")
    for name, tag in (("rust-run", "Rust"), ("dart-run", "Dart")):
        arm = arms.get(name, {})
        lines = [line for line in arm.get("metric_lines", []) if line.startswith("PASS " + tag + " unified")]
        match = re.search(r"confidence_count=(\d+) mean=([0-9.eE+-]+)", lines[0]) if len(lines) == 1 else None
        if (not arm.get("passed") or not match or int(match.group(1)) != count
                or abs(float(match.group(2)) - mean) > 1e-5):
            raise ValueError(name + " confidence count/mean differs from C ABI")
    return {"confidence_count": count, "c_abi_mean": mean,
            "max_abs_public_mean_error": max(abs(arms[name]["mean_confidence"] - mean)
                                         for name in ("python", "cli-stage21", "http-stage21"))}


def compare_abi(native_text, dart_lines, python_layout):
    """Bind the copied Dart stage FFI to the actual compiled C++ header layout."""
    pattern = (r"ABI stage_size=(\d+) cleanup_size=(\d+) params_size=(\d+)"
               r"(?: vlm_max_tokens_offset=(\d+) vlm_prompt_offset=(\d+))?")
    native_rows = re.findall(pattern, native_text)
    dart_rows = [re.fullmatch(pattern, line) for line in dart_lines if line.startswith("ABI ")]
    if len(native_rows) != 1 or len(dart_rows) != 1 or dart_rows[0] is None:
        raise ValueError("Native or Dart ABI declaration absent or ambiguous")
    names = ("stage_size", "cleanup_size", "params_size",
             "vlm_max_tokens_offset", "vlm_prompt_offset")
    native = dict(zip(names, map(int, native_rows[0])))
    if not all(native_rows[0]):
        raise ValueError("Native ABI offsets are missing")
    dart = dict(zip(names[:3], map(int, dart_rows[0].groups()[:3])))
    if python_layout != native or dart != {key: native[key] for key in dart}:
        raise ValueError("Python/Dart OCR stage ABI differs from compiled C++")
    return {"native": native, "python": python_layout, "dart": dart}


def python_abi_layout():
    sys.path.insert(0, str(ROOT/"python"))
    from crispembed._binding import _CrispOcrStage, _ScanCleanupParams, _CrispOcrPipelineParams
    return {"stage_size": ctypes.sizeof(_CrispOcrStage),
            "cleanup_size": ctypes.sizeof(_ScanCleanupParams),
            "params_size": ctypes.sizeof(_CrispOcrPipelineParams),
            "vlm_max_tokens_offset": _CrispOcrStage.vlm_max_tokens.offset,
            "vlm_prompt_offset": _CrispOcrStage.vlm_prompt.offset}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--lighton-model", type=Path, help="real format-1 GGUF for wrong-profile check")
    args = parser.parse_args()
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"])/"ovis-public"
    out.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "phase": "preflight", "arms": {}, "failures": [],
              "source_commit": os.environ.get("GITHUB_SHA"),
              "source_sha256": {name: sha(ROOT/name) for name in (
                  "tools/ci-heavy/ovisocr2_public_surfaces.py",
                  "tools/ci-heavy/ovisocr2_stage_cache_probe.py",
                  "tests/ovisocr2_public_bindings.rs", "tests/ovisocr2_public_bindings.dart")},
              "public_api_limit": "Public OCR surfaces expose text/confidences; generated IDs and stop IDs are qualified in the independent low-level/context model audit.",
              "generic_prompt_semantics": {
                  "single_model_cli": "Fixed-task Ovis prompt override is ignored with an explicit warning; the default task still runs.",
                  "single_model_http": "A custom fixed-task prompt returns prompt_applied:false; the default task still runs.",
                  "explicit_ovis_stage": "CLI/HTTP startup rejects a nonempty custom prompt; binding setters return false without mutation."}}
    report_path = out/"ovisocr2-public-surfaces.json"
    def save(): report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    save()
    try:
        if not RECEIPT.is_file():
            raise ValueError("Reviewed F16-positive receipt absent; stop before imports/weights")
        pins = require_receipt(json.loads(RECEIPT.read_text()))
        report["model_prerequisite"] = {**pins, "receipt_sha256": sha(RECEIPT),
            "service_run": service_gate(args.model_run_id, pins["source_commit"])}
        report["phase"] = "preflight-complete"
        save()
        if args.preflight:
            report["status"] = "preflight passed; no model or framework loaded"
            save()
            return
        if not args.artifacts:
            raise ValueError("Preserved model-audit artifact directory required")
        scratch.mkdir(parents=True, exist_ok=True)
        artifacts = require_artifacts(args.artifacts, pins)
        if not args.lighton_model or not args.lighton_model.is_file():
            raise ValueError("Real LightOn format-1 GGUF required for model-backed wrong-profile check")
        report["artifacts"] = {key: {"sha256": sha(path), "bytes": path.stat().st_size}
                               for key, path in artifacts.items()}
        report["artifacts"]["lighton"] = require_lighton_artifact(args.lighton_model)
        rgb, raw = image_data(artifacts["image"], pins, scratch)
        report["phase"] = "portable-build"
        save()
        build = scratch/"build"
        configure = ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                     "-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_METAL=OFF",
                     "-DGGML_BLAS=OFF", "-DCRISPEMBED_BUILD_SHARED=ON"]
        for label, command in (("cmake-configure", configure),
            ("cpu-baseline-check", [sys.executable, ROOT/"scripts/check-cpu-baseline.py", build]),
            ("public-target-build", ["cmake", "--build", build, "--config", "Release", "--target",
                                      "crispembed-shared", "crispembed-cli", "crispembed-server",
                                      "test-qwen35-public", "-j2"])):
            report["arms"][label] = run_cmd(label, command, out, timeout=2400)
            save()
            if not report["arms"][label]["passed"]:
                raise ValueError(label + " failed")
        lib = build/"libcrispembed.so"
        report["binary_sha256"] = {name: sha(path) for name, path in
            (("shared", lib), ("cli", build/"crispembed"), ("server", build/"crispembed-server"))}
        report["phase"] = "public-model-backed-arms"
        save()
        # The existing Qwen public checker prints the authoritative C++ ABI
        # before loading references. Deliberately missing test inputs make it
        # exit 2 after that print, without reading any model weights.
        abi_probe = run_cmd("native-abi-layout", [build/"test-qwen35-public",
            "__missing_abi_model__", "__missing_abi_image__", "__missing_abi_reference__",
            "__unused__", "__missing_abi_prompted_reference__"], out, timeout=30)
        abi_probe["passed"] = abi_probe["exit_code"] == 2 and len([
            line for line in abi_probe["metric_lines"] if line.startswith("ABI ")]) == 1
        report["arms"]["native-abi-layout"] = abi_probe
        save()
        if not abi_probe["passed"]:
            raise ValueError("Existing native public ABI checker did not return a bounded layout record")
        env = env_with_gates()
        os.environ.update({gate: "1" for gate in GATES})
        truncated = scratch/"ovisocr2-metadata-complete-truncated.gguf"
        stage_cache = None
        try:
            report["stage_cache_truncation"] = metadata_complete_truncation(artifacts["model"], truncated)
            stage_cache = run_cmd("python-stage21-two-model-cache", [
                sys.executable, ROOT/"tools/ci-heavy/ovisocr2_stage_cache_probe.py",
                "--repo", ROOT, "--lib", lib, "--first-model", artifacts["model"],
                "--truncated-second", truncated, "--image", artifacts["image"]],
                out, env=env, timeout=1200)
            stage_cache.update(validate_stage_cache_probe(
                stage_cache, (out/stage_cache["log"]).read_text(errors="replace"),
                pins["expected_text"], truncated))
            stage_cache["passed"] = True
            report["arms"]["python-stage21-two-model-cache"] = stage_cache
        except Exception as exc:
            report["arms"]["python-stage21-two-model-cache"] = {
                **(stage_cache or {}), "passed": False,
                "error": f"{type(exc).__name__}: {exc}"}
        save()
        for label, arm in (("c-abi", lambda: abi_arm(lib, artifacts["model"], rgb, pins["expected_text"])),
                           ("python", lambda: python_arm(ROOT, lib, artifacts["model"], artifacts["image"],
                                                         rgb, pins["expected_text"]))):
            try: report["arms"][label] = {"passed": True, **arm()}
            except Exception as exc: report["arms"][label] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            save()
        cli_http_arm(build, artifacts["model"], args.lighton_model, artifacts["image"],
                     pins["expected_text"], out, report, save, True)
        bindings_arm(ROOT, build, artifacts["model"], raw, artifacts["image"],
                     pins["expected_text"], out, scratch, report, save)
        try:
            report["abi_layout"] = {"passed": True, **compare_abi(
                (out/"native-abi-layout.log").read_text(),
                report["arms"].get("dart-run", {}).get("metric_lines", []),
                python_abi_layout())}
        except Exception as exc:
            report["abi_layout"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        save()
        try:
            report["public_confidence_comparison"] = {"passed": True,
                **compare_confidences(report["arms"])}
        except Exception as exc:
            report["public_confidence_comparison"] = {"passed": False,
                "error": f"{type(exc).__name__}: {exc}"}
        save()
        report["failures"] = [name for name, arm in report["arms"].items() if not arm["passed"]]
        if not report["public_confidence_comparison"]["passed"]:
            report["failures"].append("public-confidence-comparison")
        if not report["abi_layout"]["passed"]:
            report["failures"].append("public-abi-layout")
        report["status"] = "completed public audit" if not report["failures"] else "completed with failures"
        save()
        if report["failures"]: raise ValueError("Public model-backed arms failed")
    except BaseException as exc:
        report["status"] = "failed hosted public audit"
        report["error"] = f"{type(exc).__name__}: {exc}"
        if not report["failures"]: report["failures"].append(report["phase"])
        save()
        raise


if __name__ == "__main__":
    main()
