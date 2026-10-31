#!/usr/bin/env python3
"""Hosted, test-only composed native CPU vision audit against the frozen
qianziwen prefill GGUF.

Scope: qwen35_ocr::vision_cpu (patch embed, position, embedded, rotary cos/sin,
all twelve blocks, merger.norm/up/activation/merger) compared with
crispembed_diff at a literal max_abs 0.002 / cos_min 0.99999. The decoder,
decoded text, image frontend and quantized artifacts are not tested. The native
vision path and hosted session are CPU-only; this is not a GPU measurement.
"""
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zlib

BUNDLE = "__CRISPEMBED_SOURCE_BUNDLE__"
OUT = Path("/kaggle/working/xiaomi-vision-prefill")
ROOT = Path("/kaggle/temp/xiaomi-vision-prefill/source")
MODEL_SHA = "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235"
REF_SHA = "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334"
BASE = "15adb672f71f8470a23db1284dafa102515adae9"
GGML = "a06fc0bb46b73cc10011fc90ae4054aee8438aeb"
MODEL_NAME = "xiaomi-ocr-0-f32-experimental.gguf"
REF_NAME = "qianziwen-prefill.gguf"
ALLOWED = {
    "audit/CMakeLists.txt", "tests/test_xiaomi_vision_prefill.cpp",
    "src/qwen35_vision.cpp", "src/qwen35_vision.h", "src/qwen35_image_bounds.h",
}
MAX_ABS, COS_MIN = 0.002, 0.99999
ROTARY = 64
EXPECTED = (["patch_embed", "position", "embedded", "rotary_cos", "rotary_sin"] +
            [f"layer.{i}" for i in range(12)] +
            ["merger.norm", "merger.up", "merger.activation", "merger"])
STAGE_SHAPE = {}
for _name in EXPECTED:
    if _name in ("rotary_cos", "rotary_sin"):
        STAGE_SHAPE[_name] = [ROTARY, 1984]
    elif _name in ("merger.up", "merger.activation"):
        STAGE_SHAPE[_name] = [3072, 496]
    elif _name == "merger":
        STAGE_SHAPE[_name] = [1024, 496]
    else:
        STAGE_SHAPE[_name] = [768, 1984]
REPORT = {
    "scope": "test-only native CPU composed vision prefill; no decoder, decoded text, "
             "image frontend, quantized parity or end-to-end OCR claim",
    "thresholds": {"max_abs": MAX_ABS, "cos_min": COS_MIN, "relaxed_for_dtype": False},
    "reference_note": "frozen archive stores the original BF16 vision activations widened to F32; "
                      "the gate is unchanged by that dtype difference",
    "backend_note": "qwen35_ocr::vision_cpu and this hosted session are CPU-only",
    "status": "running",
}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["phase"] = phase
    REPORT["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=2, allow_nan=False) + "\n")
    print("===", phase, flush=True)


def unpack(root):
    if BUNDLE == "__CRISPEMBED_SOURCE_BUNDLE__":
        raise ValueError("source bundle was not staged")
    data = json.loads(zlib.decompress(base64.b64decode(BUNDLE, validate=True)))
    if data["base_commit"] != BASE or data["ggml_commit"] != GGML or set(data["files"]) != ALLOWED:
        raise ValueError("source closure differs")
    for name, entry in data["files"].items():
        content = base64.b64decode(entry["content"], validate=True)
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("source digest differs: " + name)
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return {"base_commit": BASE, "ggml_commit": GGML,
            "runner_template_sha256": data["runner_template_sha256"],
            "files_sha256": {name: entry["sha256"] for name, entry in data["files"].items()}}


def find_input(name, expected):
    matches = [path for path in Path("/kaggle/input").rglob(name) if digest(path) == expected]
    if len(matches) != 1:
        raise ValueError(f"{name}: expected exactly one input with the pinned hash, found {len(matches)}")
    return matches[0]


def command(label, args, cwd=None, env=None, allowed=(0,)):
    print("RUN", label, flush=True)
    result = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    print(result.stdout[-20000:], flush=True)
    if result.returncode not in allowed:
        raise RuntimeError(f"{label} failed with exit {result.returncode}")
    return result


def parse_stages(text):
    rows = []
    for line in text.splitlines():
        if line.startswith("STAGE "):
            rows.append(json.loads(line[len("STAGE "):]))
    return rows


def main():
    save("admission")
    if not Path("/kaggle/working").is_dir() or shutil.disk_usage("/kaggle/working").free < 3 * 1024**3:
        raise RuntimeError("insufficient hosted workspace")
    REPORT["device"] = "CPU"
    model = find_input(MODEL_NAME, MODEL_SHA)
    reference = find_input(REF_NAME, REF_SHA)
    REPORT["input_sha256"] = {"model": MODEL_SHA, "reference": REF_SHA}
    REPORT["input_bytes"] = {"model": model.stat().st_size, "reference": reference.stat().st_size}
    save("checkout")
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    command("clone", ["git", "clone", "--no-checkout", "https://github.com/CrispStrobe/CrispEmbed.git", str(ROOT)])
    command("checkout", ["git", "checkout", "--detach", BASE], cwd=ROOT)
    command("submodule", ["git", "submodule", "update", "--init", "--recursive"], cwd=ROOT)
    for name, cwd, expected in (("repo", ROOT, BASE), ("ggml", ROOT / "ggml", GGML)):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
        if actual != expected:
            raise ValueError(name + " commit mismatch")
    if not (ROOT / "ggml/CMakeLists.txt").is_file():
        raise RuntimeError("ggml submodule has no CMakeLists.txt at the pinned commit")
    closure = unpack(ROOT)
    closure["staged_runner_sha256"] = digest(__file__)
    REPORT["source_closure"] = closure
    (OUT / "source-closure.json").write_text(json.dumps(closure, indent=2) + "\n")
    # The standalone audit target lives under audit/ while its sources live
    # at the checkout root. Symlink only these known directories.
    for name in ("src", "tests", "ggml"):
        (ROOT / "audit" / name).symlink_to(ROOT / name, target_is_directory=True)
    save("configure")
    build = ROOT / "build-vision-prefill"
    command("configure", ["cmake", "-S", str(ROOT / "audit"), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF",
            "-DGGML_VULKAN=OFF", "-DGGML_BLAS=OFF", "-DGGML_OPENMP=OFF"])
    save("build")
    command("build", ["cmake", "--build", str(build), "--target", "test-xiaomi-vision-prefill",
            "--parallel", "4"])
    binary = build / "test-xiaomi-vision-prefill"
    if not binary.is_file():
        raise RuntimeError("vision prefill binary missing after build")
    REPORT["binary_sha256"] = digest(binary)
    save("execute")
    dump = OUT / "native-trace"
    dump.mkdir()
    env = dict(os.environ, CRISPEMBED_DIFF_DUMP_DIR=str(dump))
    # 0 = all stages within the strict gate, 1 = a numerical mismatch. Both are
    # legitimate observations; only 2 (setup/control) aborts the audit.
    run = command("vision-prefill", [str(binary), str(model), str(reference)], env=env, allowed=(0, 1))
    stages = parse_stages(run.stdout)
    names = [row["name"] for row in stages]
    expected_names = ["prefill.vision." + name for name in EXPECTED]
    if names != expected_names:
        raise ValueError(f"stage set/order differs: {names}")
    control = "NEGATIVE_CONTROL rejected=1" in run.stdout
    backend = next((line for line in run.stdout.splitlines() if line.startswith("AUDIT_BACKEND ")), "")
    if not control:
        raise ValueError("negative control was not rejected")
    if "CPU" not in backend:
        raise ValueError("backend evidence missing or not CPU: " + backend)
    results, failures = [], []
    for row in stages:
        short = row["name"].removeprefix("prefill.vision.")
        if row["shape"] != STAGE_SHAPE[short]:
            raise ValueError(f"shape mismatch at {row['name']}: {row['shape']}")
        metrics = [row[key] for key in ("max_abs", "mean_abs", "rms", "cos_min", "cos_mean",
                                        "cos_global", "native_norm", "reference_norm")]
        if not all(isinstance(value, (int, float)) and value == value and abs(value) != float("inf")
                   for value in metrics):
            raise ValueError("non-finite metric at " + row["name"])
        strict = row["max_abs"] <= MAX_ABS and row["cos_min"] >= COS_MIN
        if strict != row["pass"]:
            raise ValueError("native verdict disagrees with the strict gate at " + row["name"])
        row["strict_absolute_pass"] = strict
        row["max_abs_budget"] = MAX_ABS
        row["cos_min_floor"] = COS_MIN
        results.append(row)
        if not strict:
            failures.append(row["name"])
    REPORT["native_exit"] = run.returncode
    REPORT["negative_control_rejected"] = control
    REPORT["backend_evidence"] = backend
    REPORT["stages"] = results
    REPORT["stages_total"] = len(results)
    REPORT["stages_passed"] = len(results) - len(failures)
    REPORT["strict_failures"] = failures
    REPORT["first_failing_stage"] = failures[0] if failures else None
    trace = {}
    for row in results:
        path = dump / (row["name"] + ".f32")
        expected_bytes = 4 * row["n"]
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError("native trace missing or truncated: " + path.name)
        trace[path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    REPORT["native_trace_sha256"] = trace
    if run.returncode != (0 if not failures else 1):
        raise ValueError("native exit code disagrees with parsed verdicts")
    REPORT["status"] = ("completed: native CPU composed vision compared stage by stage against the frozen "
                        "prefill; decoder, decoded text, image frontend and quantized parity not tested")
    save("complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        REPORT["status"] = "failed"
        REPORT["error_type"] = type(error).__name__
        REPORT["error"] = str(error)[:1000]
        REPORT["traceback"] = traceback.format_exc()[-5000:]
        save(REPORT.get("phase", "startup"))
        raise
