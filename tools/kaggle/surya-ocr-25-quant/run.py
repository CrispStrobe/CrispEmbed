#!/usr/bin/env python3
"""Hosted Surya OCR 2.5 convert + quantize lane (Kaggle CPU).

The branch already shows that native F16 reproduces the original model's token
ids, text and EOS on the owned crops. This lane answers the next question: what
does the project's own quantizer do to that exact output?

Runs, all on the same pixels in one kernel so no cross-host image drift can
contaminate the comparison:

  f16    convert-qwen35-ocr-to-gguf.py --profile surya25 --dtype f16
  q8_0   crispembed-quantize F16 -> q8_0
  q4_k   crispembed-quantize F16 -> q4_k

For each arm: a stored-type census (no norm/bias may be quantized, and the
quantized-matrix count is recorded), the decoded token ids, text and stop, and
the comparison against the same-kernel F16 arm and against the frozen original
receipt. The F16 arm is the gate: it must still reproduce the frozen original
ids and text, otherwise the host renders different pixels and no quantized
comparison in this run is meaningful. Quantized arms are measured, never gated
and never promoted here.

Exit codes: 0 every arm completed and the F16 anchor reproduced the frozen
original; 1 the F16 anchor did not reproduce (pixels or host drift);
2 setup, pin or policy failure.

Only the standard library is imported at module scope so the repository test can
import this file without gguf, torch or model bytes.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

SCRIPT_VERSION = "surya25-kaggle-quant-v1"

MODEL = "datalab-to/surya-ocr-2.5"
REVISION = "00c843e9cae45611dca0a6dc2c9bbbf2dadaaed3"
CHECKPOINT_BYTES = 1378772208
CHECKPOINT_SHA256 = "6940baca2d16a20b15b7302cac7253246c78c95469920fcdb064963f48b5cb17"
PROFILE = "surya25"
CAP = 256

ARMS = ("f16", "q8_0", "q4_k")
ANCHOR_ARM = "f16"

# Package `scale_to_fit` bounds: at least 1792x28 area, at most 3072x2048.
MIN_AREA = 1792 * 28
MAX_AREA = 3072 * 2048
# Native frontend bound for this model today (patch 16, no widening).
MAX_PATCHES = 1024
PATCH = 16

# Identical to tools/ci-heavy/surya25_native.py so the same pixels are rendered.
# (name, width, height, lines) — every side a multiple of the 28-px grid.
CASE_TABLE = (
    ("hello", 560, 112, ["Hello, world."]),
    ("invoice", 896, 84, ["Invoice No. 2026-10-31: 1,234.56 EUR"]),
    ("german", 728, 112, ["Grüße aus München, Straße 5"]),
    ("paragraph", 672, 252, ["The quick brown fox jumps", "over the lazy dog. 42%", "of tests pass (n=17)."]),
    ("bold", 504, 140, ["Total: $99.95"]),
    ("short", 448, 112, ["OCR test 7"]),
)

FROZEN_RECEIPT = Path("tests/results/surya25/native-ab-local.json")
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"

OUT = Path("/kaggle/working/surya-ocr-25-quant")
REPORT = {"script_version": SCRIPT_VERSION, "model": MODEL, "revision": REVISION,
          "cap": CAP, "arms": list(ARMS), "anchor_arm": ANCHOR_ARM}

PROTECTED_SUBSTRINGS = ("norm", "bias", "embd", "position_embd", "conv")


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def save(phase):
    if not OUT.is_dir():
        return
    REPORT["phase"] = phase
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False) + "\n")


def census_violations(census):
    """Tensor names that must stay F32/F16 but were stored quantized."""
    bad = []
    for name, stored in sorted(census.items()):
        if stored in ("F32", "F16"):
            continue
        lowered = name.lower()
        if any(part in lowered for part in PROTECTED_SUBSTRINGS):
            bad.append(name)
    return bad


def census_summary(census):
    by_type = {}
    for stored in census.values():
        by_type[stored] = by_type.get(stored, 0) + 1
    quantized = sum(count for stored, count in by_type.items() if stored not in ("F32", "F16"))
    return {"tensors": len(census), "by_type": by_type, "quantized_matrices": quantized}


def compare_arm(reference, arm):
    """Token-level comparison of one arm against a reference arm."""
    ref_ids = list(reference["token_ids"])
    ids = list(arm["token_ids"])
    first = next((index for index, (left, right) in enumerate(zip(ids, ref_ids)) if left != right),
                 None if len(ids) == len(ref_ids) else min(len(ids), len(ref_ids)))
    return {
        "ids_exact": ids == ref_ids,
        "text_exact": arm.get("text") == reference.get("text"),
        "stop_exact": arm.get("stop") == reference.get("stop"),
        "first_divergent_step": first,
        "token_count": len(ids),
        "reference_token_count": len(ref_ids),
        "stop": arm.get("stop"),
    }


def read_types(path):
    """Stored GGUF type per tensor, read with the gguf library inside the run."""
    import gguf

    reader = gguf.GGUFReader(str(path))
    return {tensor.name: tensor.tensor_type.name for tensor in reader.tensors}


def quantize(model, output, kind, extra):
    command = [str(QUANTIZE), str(model), str(output), kind] + list(extra)
    print("+", " ".join(command), flush=True)
    run = subprocess.run(command, capture_output=True, text=True)
    (OUT / f"quantize-{kind}.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(f"crispembed-quantize {kind} failed with exit {run.returncode}: "
                           f"{run.stderr[-1500:]}")
    return output


def native(model, image, prompt=None):
    env = dict(os.environ, CRISPEMBED_QWEN35_OCR="1", CRISPEMBED_SURYA25_FRONTEND="1",
               CRISPEMBED_SURYA25_INFERENCE="1")
    command = [str(NATIVE), str(model), str(image), str(CAP), "4"]
    if prompt:
        command.append(prompt)
    run = subprocess.run(command, capture_output=True, text=True, env=env)
    line = next((row for row in run.stdout.splitlines() if row.startswith("SURYA25_RESULT ")), None)
    if run.returncode or not line:
        return {"error": run.returncode, "stderr": run.stderr[-1500:]}
    data = json.loads(line.split(" ", 1)[1])
    data["text"] = bytes.fromhex(data.pop("raw_text_utf8_hex")).decode("utf-8", "replace")
    data["stop"] = "EOS" if data.get("stop_kind") == "eos" else (data.get("stop_kind") or "unknown")
    return data


def render(name, width, height, lines, scratch):
    from PIL import Image, ImageDraw, ImageFont

    font_path = FONT_BOLD if name == "bold" else FONT_REGULAR
    font = ImageFont.truetype(str(font_path), 30)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = (height - 38 * len(lines)) // 2
    for line in lines:
        draw.text((16, y), line, fill=(0, 0, 0), font=font)
        y += 38
    path = scratch / f"{name}.png"
    image.save(path)
    return path, image


def main():
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("this lane must run on the hosted CPU worker")
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = Path("/kaggle/temp/surya-ocr-25-quant")
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage("/kaggle/working").free
    REPORT["initial_free_bytes"] = free
    if free < 8 * 1024 ** 3:
        raise RuntimeError("insufficient hosted workspace")
    save("admission")

    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT / "models"))
    import importlib.metadata

    import qwen35_surya25_profile as profile

    REPORT["versions"] = {"python": sys.version.split()[0]}
    for name in ("numpy", "gguf", "safetensors", "huggingface-hub", "pillow"):
        try:
            REPORT["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            REPORT["versions"][name] = "missing"
    for path, label in ((QUANTIZE, "crispembed-quantize"), (NATIVE, "test-surya25-run")):
        if not Path(path).is_file():
            raise RuntimeError(f"{label} was not built: {path}")
        REPORT.setdefault("binaries", {})[label] = digest(path)
    save("binaries")

    save("checkpoint")
    from huggingface_hub import snapshot_download

    patterns = list(profile.METADATA_FILES) + ["model.safetensors"]
    root = Path(snapshot_download(MODEL, revision=REVISION, token=False,
                                  allow_patterns=patterns, local_dir=str(scratch / "checkpoint")))
    profile.verify_metadata(str(root))
    checkpoint = root / "model.safetensors"
    if checkpoint.stat().st_size != CHECKPOINT_BYTES or digest(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("downloaded checkpoint differs from the pinned revision digest")
    REPORT["checkpoint"] = {"bytes": checkpoint.stat().st_size, "sha256": CHECKPOINT_SHA256,
                            "anonymous": True, "token": False}
    save("checkpoint")

    converter = ROOT / "models/convert-qwen35-ocr-to-gguf.py"
    spec = __import__("importlib.util", fromlist=["util"]).spec_from_file_location("convert_qwen35", converter)
    module = __import__("importlib.util", fromlist=["util"]).module_from_spec(spec)
    spec.loader.exec_module(module)
    f16 = scratch / "surya25-f16.gguf"
    manifest = module.convert(root, f16, MODEL, REVISION, "f16", (), profile=PROFILE)
    REPORT["convert"] = {"bytes": manifest["bytes"],
                         "export_tensor_count": manifest["export_tensor_count"],
                         "checkpoint_tensor_count": manifest["checkpoint_tensor_count"],
                         "gguf_sha256": digest(f16)}
    print("CONVERTED", json.dumps(REPORT["convert"]), flush=True)
    save("convert")

    models = {ANCHOR_ARM: f16}
    for kind in ("q8_0", "q4_k"):
        models[kind] = quantize(f16, scratch / f"surya25-{kind}.gguf", kind, [])
    save("quantize")

    for arm, path in models.items():
        census = read_types(path)
        violations = census_violations(census)
        REPORT.setdefault("census", {})[arm] = {
            **census_summary(census),
            "violations": violations,
            "bytes": path.stat().st_size,
            "gguf_sha256": digest(path),
        }
        if violations:
            raise RuntimeError(f"{arm}: protected tensors were quantized: {', '.join(violations[:8])}")
        print("CENSUS", arm, json.dumps(REPORT["census"][arm]), flush=True)
    save("census")

    frozen = json.loads((ROOT / FROZEN_RECEIPT).read_text())
    REPORT["frozen_receipt"] = {"path": str(FROZEN_RECEIPT), "sha256": digest(ROOT / FROZEN_RECEIPT)}

    results = {}
    for name, width, height, lines in CASE_TABLE:
        path, _ = render(name, width, height, lines, scratch)
        reference = frozen["cases"][name]["original"]
        case = {"input_sha256": digest(path),
                "frozen_input_sha256": frozen["cases"][name].get("input_sha256")}
        case["pixels_match_frozen"] = case["input_sha256"] == case["frozen_input_sha256"]
        for arm, model in models.items():
            result = native(model, path)
            case[arm] = result
            if "token_ids" in result:
                case[arm]["vs_frozen_original"] = compare_arm(reference, result)
            print("NATIVE", arm, name, json.dumps(
                {"text": result.get("text"), "error": result.get("error"),
                 "vs_frozen_original": result.get("vs_frozen_original")}, ensure_ascii=False), flush=True)
        anchor = case.get(ANCHOR_ARM, {})
        if "token_ids" in anchor:
            for arm in ARMS:
                if arm != ANCHOR_ARM and "token_ids" in case.get(arm, {}):
                    case[arm]["vs_anchor"] = compare_arm(anchor, case[arm])
        results[name] = case
    REPORT["cases"] = results
    save("compare")

    anchor_rows = [case[ANCHOR_ARM].get("vs_frozen_original", {}) for case in results.values()]
    anchor_ok = all(row.get("ids_exact") and row.get("text_exact") for row in anchor_rows)
    REPORT["anchor"] = {
        "arm": ANCHOR_ARM,
        "cases": len(anchor_rows),
        "ids_exact_cases": sum(1 for row in anchor_rows if row.get("ids_exact")),
        "text_exact_cases": sum(1 for row in anchor_rows if row.get("text_exact")),
        "pixels_match_frozen_cases": sum(1 for case in results.values() if case["pixels_match_frozen"]),
        "reproduced_frozen_original": bool(anchor_ok),
    }
    REPORT["summary"] = {
        arm: {
            "cases": len(results),
            "ids_exact_vs_anchor": sum(1 for case in results.values()
                                       if case.get(arm, {}).get("vs_anchor", {}).get("ids_exact")),
            "ids_exact_vs_frozen_original": sum(
                1 for case in results.values()
                if case.get(arm, {}).get("vs_frozen_original", {}).get("ids_exact")),
            "errors": sum(1 for case in results.values() if "error" in case.get(arm, {})),
        }
        for arm in ARMS
    }
    REPORT["claims"] = {"decoded_quality_claimed": False, "speed_claimed": False,
                        "registry_claimed": False, "publication_claimed": False,
                        "f16_anchor_is_the_only_gate": True}
    if anchor_ok:
        REPORT["status"] = ("completed: F16 anchor reproduced the frozen original ids and text, and "
                            "every quantized arm ran; quantized arms are measured, not qualified")
        save("complete")
        return 0
    REPORT["status"] = ("completed without an anchor: the F16 arm did not reproduce the frozen "
                        "original on this host, so the quantized comparison is not meaningful")
    REPORT["error_type"] = "AnchorNotReproduced"
    save("anchor-missing")
    return 1


QUANTIZE = Path("/kaggle/temp/build/crispembed-quantize")
NATIVE = Path("/kaggle/temp/build/test-surya25-run")

if __name__ == "__main__":
    START = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    REPORT["started_utc"] = START
    try:
        CODE = main()
    except BaseException as error:
        REPORT["status"] = "failed"
        REPORT["error_type"] = type(error).__name__
        REPORT["error"] = str(error)[:1500]
        REPORT["traceback"] = traceback.format_exc()[-6000:]
        try:
            save(REPORT.get("phase", "startup"))
        except BaseException:
            print("REPORT_WRITE_FAILED", traceback.format_exc()[-2000:], flush=True)
        CODE = 2
    REPORT["exit_code"] = CODE
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        save(REPORT.get("phase", "startup"))
    except BaseException:
        print("REPORT_WRITE_FAILED", traceback.format_exc()[-2000:], flush=True)
    print("SURYA25_QUANT_RESULT", json.dumps(
        {"status": REPORT["status"], "anchor": REPORT.get("anchor"), "summary": REPORT.get("summary"),
         "exit_code": CODE}, ensure_ascii=False), flush=True)
    raise SystemExit(CODE)
