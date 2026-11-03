#!/usr/bin/env python3
"""Separate current-source F16/Q8 replay; historical quality gates stay frozen.

One immutable document per hosted job. Reconstructs the exact old model bytes,
captures the original before native execution, and keeps every strict/text
failure alongside the current final-output precision policy. No publication.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/ci-heavy")]
import lighton08_precision_review as review
import ovisocr2_lighton_fixture as fixture
import qwen35_quality as quality
import qwen35_quality_quant as quant
from qwen35_decoded_panel import validate
from qwen35_page_parity import parse
from qwen35_receipt import add_confidences, teacher_quality

PROTOCOL = ROOT / "tests/fixtures/lighton08-portable/protocol.json"
FROZEN_FILES = (
    'tests/results/qwen35-ocr/quality.json',
    'tests/results/qwen35-ocr/page-parity.json',
    'tests/results/qwen35-ocr/receipt.json',
    'tests/results/ocr-quality-prepared.json',
    'tests/results/qwen35-ocr/ovisocr2-quality-prepared-preservation.json',
    'tests/regression/ocr-quality/panel.json',
    'tools/ocr_quality.py',
    'tools/dump_qwen35_ocr_reference.py',
    'tools/dump_qwen35_ocr_page_reference.py',
    'tools/qwen35_decode_reference.py',
    'tools/ci-heavy/ovisocr2_lighton_converter_frozen.py',
    'tools/ci-heavy/ovisocr2_lighton_fixture.py',
    'tools/ci-heavy/qwen35_quality.py',
    'tools/ci-heavy/qwen35_quality_quant.py',
    'tools/ci-heavy/ovisocr3_quality_run.py',
    'tests/test_qwen35_quant.cpp',
    'tests/qwen35_parity.h',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(case_id):
    protocol = json.loads(PROTOCOL.read_text())
    require(protocol["schema"] == "crispembed.lighton08.current-portable.v1" and
            protocol["model"] == fixture.MODEL and protocol["revision"] == fixture.REVISION and
            protocol["case_order"] == list(review.CASES) and case_id in review.CASES and
            protocol["precision_bars_exclusive"] == {"f16": .98, "q8": .97} and
            protocol["max_new_tokens"] == 1024 and protocol["prompt"] == "" and
            protocol["gguf_sha256"] == fixture.GGUF_SHA and
            protocol["historical_verdicts_changed"] is False and protocol["publication_qualified"] is False, "changed portable replay protocol")
    require(set(protocol["frozen_files_sha256"]) == set(FROZEN_FILES), "frozen source coverage differs")
    for name, digest in protocol["frozen_files_sha256"].items():
        require(fixture.sha(ROOT / name) == digest, "changed frozen input/producer: " + name)
    recipes = json.loads((ROOT / "tests/results/qwen35-ocr/page-parity.json").read_text())
    recipe = next(a for a in recipes["arms"] if a["name"] == "q8-vision-f32")
    require(recipe["overrides"] == ["v.*=f32"] and recipe["gguf_sha256"] == protocol["q8_sha256"],
            "frozen Q8 recipe changed")
    fixture.require_receipt()
    quality.require_reviewed_preparation(ROOT)
    prior = json.loads((ROOT / "tests/results/qwen35-ocr/quality.json").read_text())
    old = next(c for c in prior["cases"] if c["id"] == case_id)
    require(old["specification"]["prompt"] == "" and old["specification"]["max_new_tokens"] == 1024,
            "changed frozen task")
    return protocol, prior, old


def portable_cache(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if ":" in line and "=" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    require(values.get("GGML_NATIVE") == values.get("CRISPEMBED_NATIVE") == "OFF",
            "both portable CPU flags must be OFF")
    require(values.get("GGML_CUDA") == values.get("GGML_METAL") == "OFF", "CPU backend required")
    return {name: values[name] for name in ("GGML_NATIVE", "CRISPEMBED_NATIVE", "GGML_CUDA", "GGML_METAL")}


def native_environment(prior):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CRISPEMBED_")}
    env.update(prior["native_environment"])
    env.update(CRISPEMBED_QWEN35_OCR="1", CRISPEMBED_QWEN35_QUANT="1", CRISPEMBED_BUILD_MODE="portable",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    return env


def stop_boundary(ids, eos_ids):
    require(isinstance(ids, list) and bool(ids) and isinstance(eos_ids, list) and bool(eos_ids) and
            not any(token in eos_ids for token in ids[:-1]), "tokens after EOS cannot be complete generation")


def storage_capacity(scratch, output):
    scratch, output = Path(scratch), Path(output)
    scratch_free, output_free = shutil.disk_usage(scratch).free, shutil.disk_usage(output).free
    shared = scratch.stat().st_dev == output.stat().st_dev
    require(scratch_free >= 15 * 1024**3 and output_free >= 8 * 1024**3 and
            (not shared or min(scratch_free, output_free) >= 23 * 1024**3),
            "need 15 GiB scratch plus 8 GiB retained-array capacity (combined on a shared filesystem)")
    return {"scratch_free_bytes": scratch_free, "output_free_bytes": output_free,
            "same_filesystem": shared, "scratch_required_bytes": 15 * 1024**3,
            "output_required_bytes": 8 * 1024**3}


def summarize(case_id, arms, reference):
    require([arm.get("profile") for arm in arms] == ["f16", "q8"], "both ordered precision arms are required")
    stop_boundary(reference["generated_ids"], reference["eos_ids"])
    for arm in arms:
        stop_boundary(arm["decode"]["native_ids"], reference["eos_ids"])
    rows = [review.case_review(case_id, arm, reference, arm["profile"]) for arm in arms]
    return {"complete_case_comparison": True, "profiles": rows,
            "all_selected_precision_bars_met": all(row["exceeds_bar_on_recorded_build"] for row in rows),
            "all_native_documents_complete": all(row["complete_document"] for row in rows),
            "all_three_case_panel_complete": False, "historical_verdicts_changed": False,
            "current_model_qualified": False, "publication_qualified": False}


def run(case_id):
    protocol, prior, old = preflight(case_id)
    require(os.environ.get("GITHUB_ACTIONS") == "true", "model execution is hosted-only")
    require(os.environ.get("CRISPEMBED_BUILD_MODE", "portable") == "portable", "portable build required")
    out = Path(os.environ["HEAVY_OUT"]) / ("lighton08-portable-" + case_id)
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / ("lighton08-portable-" + case_id)
    require(not out.exists(), "output directory must be new")
    out.mkdir(parents=True); scratch.mkdir(parents=True, exist_ok=True)
    report = {"schema": protocol["schema"], "protocol_sha256": fixture.sha(PROTOCOL), "case_id": case_id,
              "model": fixture.MODEL, "revision": fixture.REVISION, "phase": "preflight", "arms": [],
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "historical_verdicts_changed": False, "current_model_qualified": False,
              "publication_qualified": False, "upstream_benchmark_score": None}
    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    def command(args, name, env=None, timeout=3600):
        with (out / (name + ".log")).open("w") as stream:
            subprocess.run(list(map(str, args)), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           timeout=timeout, check=True)
    save()
    try:
        report["storage_capacity"] = storage_capacity(scratch, out)
        report["packages"] = fixture.check_versions()
        command([sys.executable, "tools/qwen35_source_pins.py", "--check"], "source-pin-guard")
        command(["git", "diff", "--exit-code", "HEAD"], "clean-source")
        report["source_amendments_sha256"] = fixture.sha(ROOT / "tests/results/qwen35-ocr/source-pin-amendments.json")
        report["historical_receipt_sha256"] = fixture.sha(ROOT / "tests/results/qwen35-ocr/quality.json")
        report["runtime_sources"] = {str(p.relative_to(ROOT)): fixture.sha(p) for p in
            sorted((ROOT / "src").glob("qwen35*")) if p.is_file() and p.suffix in (".cpp", ".h")}
        report["hub_source"] = fixture.require_hub_info(fixture.model_api(fixture.MODEL, fixture.REVISION),
            fixture.MODEL, fixture.REVISION, fixture.SOURCE_FILES)
        report["hub_base_license"] = fixture.require_hub_info(fixture.model_api(fixture.BASE, fixture.BASE_REVISION),
            fixture.BASE, fixture.BASE_REVISION, {"LICENSE": fixture.BASE_LICENSE_FILE})
        save()
        from huggingface_hub import hf_hub_download, snapshot_download
        from PIL import Image, ImageDraw
        import gguf
        import torch
        torch.set_num_threads(4)
        local = Path(snapshot_download(fixture.MODEL, revision=fixture.REVISION, allow_patterns=list(fixture.SOURCE_FILES)))
        report["checkpoint_files"] = {name: fixture.verify_file(local / name, spec) for name, spec in fixture.SOURCE_FILES.items()}
        license_path = Path(hf_hub_download(fixture.BASE, "LICENSE", revision=fixture.BASE_REVISION))
        fixture.verify_file(license_path, fixture.BASE_LICENSE_FILE)
        shutil.copyfile(license_path, out / "BASE_LICENSE")
        converter = fixture.frozen_converter(); receipt = fixture.require_receipt()
        baseline = scratch / fixture.GGUF_NAME
        manifest = converter.convert(local, baseline, fixture.MODEL, fixture.REVISION, dtype="f16",
                                     keep_f32=fixture.receipt_vision_recipe(receipt, converter))
        report["weights"] = fixture.verify_export(baseline, manifest, receipt)
        (out / "export-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Reuse the independently verified immutable PNG transport. No re-render.
        prepared = json.loads((ROOT / "tests/results/qwen35-ocr/ovisocr2-quality-prepared-preservation.json").read_text())
        archive = scratch / "prepared.zip"
        with urllib.request.urlopen(prepared["url"], timeout=90) as response:
            payload = response.read(prepared["bytes"] + 1)
        require(len(payload) == prepared["bytes"], "prepared archive size differs")
        archive.write_bytes(payload)
        source = load_module("lighton08_immutable_inputs", ROOT / "tools/ci-heavy/ovisocr3_quality_run.py")
        _, members = source.verify_prepared_archive(archive, prepared,
            json.loads((ROOT / "tests/results/ocr-quality-prepared.json").read_text()))
        image_path = out / old["prepared"]["file"]
        image_path.write_bytes(members[old["prepared"]["file"]])
        image = Image.open(image_path).convert("RGB")
        require(list(image.size) == old["prepared"]["size"] and
                hashlib.sha256(image.tobytes()).hexdigest() == old["prepared"]["rgb_sha256"], "frozen image differs")
        report["case"] = {key: old[key] for key in ("id", "specification", "prepared", "gold")}
        # Original full/compact anchor and document capture precede native runs.
        report["phase"] = "original"; save()
        anchor = Image.new("RGB", (256, 256), "white")
        ImageDraw.Draw(anchor).text((30, 80), "Hello OCR\n12345", fill="black", font_size=24)
        anchor_path = out / "anchor.png"; anchor.save(anchor_path)
        full = load_module("lighton08_original_full", ROOT / "tools/dump_qwen35_ocr_reference.py")
        full.dump_reference(local, anchor_path, out / "anchor-full.npz")
        gc.collect()
        capture = load_module("lighton08_original_page", ROOT / "tools/dump_qwen35_ocr_page_reference.py")
        report["compact_anchor"] = capture.dump_reference(local, anchor_path, out / "anchor-compact",
            max_new_tokens=12, anchor=out / "anchor-full.npz")
        gc.collect()
        original_dir = out / "original"
        reference = capture.dump_reference(local, image_path, original_dir, max_new_tokens=1024, prompt="",
                                           long_document=True, prefill_controls=True, vision_probes=True)
        report["reference"] = reference
        stop_boundary(reference["generated_ids"], reference["eos_ids"])
        report["frozen_original_exact"] = quant.frozen_original(reference, old["reference"])
        report["reference_quality"] = quality.score(old["gold"], reference["decoded"],
            complete=quality.reference_complete(reference, 1024))
        require(report["reference_quality"] == old["reference_quality"], "frozen original quality differs")
        report["all_original_outputs_frozen_before_native"] = True; save(); gc.collect()
        build = scratch / "build"
        command(["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
                 "-DCRISPEMBED_NATIVE=OFF", "-DGGML_METAL=OFF", "-DGGML_CUDA=OFF", "-DGGML_BLAS=OFF"], "cmake")
        command(["cmake", "--build", build, "-j2", "--target", "test-qwen35-quant", "crispembed-quantize"], "build")
        report["build"] = portable_cache(build / "CMakeCache.txt")
        shutil.copyfile(build / "CMakeCache.txt", out / "CMakeCache.txt")
        report["native_binaries"] = {name: fixture.sha(build / name) for name in ("test-qwen35-quant", "crispembed-quantize")}
        command([build / "test-qwen35-quant", "--request-controls"], "native-request-controls")
        recipe = next(a for a in json.loads((ROOT / "tests/results/qwen35-ocr/page-parity.json").read_text())["arms"]
                      if a["name"] == "q8-vision-f32")
        candidate = scratch / "q8-vision-f32.gguf"
        command([build / "crispembed-quantize", baseline, candidate, "q8_0", "--tensor-type", "v.*=f32"], "quantize-q8")
        require(candidate.stat().st_size == recipe["bytes"] and fixture.sha(candidate) == recipe["gguf_sha256"],
                "frozen Q8 recipe bytes differ")
        def tensor_schema(path):
            reader = gguf.GGUFReader(str(path))
            return {t.name: {"shape": t.shape.tolist(), "type": t.tensor_type.name, "bytes": t.data.nbytes,
                    "sha256": hashlib.sha256(t.data.tobytes()).hexdigest()} for t in reader.tensors}
        report["q8_protected_tensors_exact"] = quant.tensor_audit(tensor_schema(baseline), tensor_schema(candidate),
                                                                recipe["overrides"], "q8_0")
        report["q8_sha256"] = recipe["gguf_sha256"]
        report["harness_gate_scope"] = "QWEN35_QUANT=1 during both computations; literal 0 only for the frozen negative admission control"
        env = native_environment(prior); report["native_environment"] = {k: v for k, v in env.items() if k.startswith("CRISPEMBED_")}
        names = [m["stage"] for m in old["prefill"]["metrics"]]
        report["phase"] = "native"; save()
        for profile, model in (("f16", baseline), ("q8", candidate)):
            arm = {"profile": profile}; report["arms"].append(arm)
            arm_env = dict(env)  # The frozen harness toggles quant=0 only for its admission control.
            request = [build / "test-qwen35-quant", model, image_path, original_dir / "official-delta.gguf", "", "1024"]
            command([*request, "--prefill-only"], profile + "-prefill", env=arm_env)
            arm["prefill"] = quality.strict_prefill((out / (profile + "-prefill.log")).read_text(), names)
            command([*request, "--full-quality"], profile + "-decode", env=arm_env, timeout=7200)
            log = (out / (profile + "-decode.log")).read_text()
            arm["decode"] = add_confidences(log, validate(parse(log, reference), reference, 1024), reference)
            stop_boundary(arm["decode"]["native_ids"], reference["eos_ids"])
            arm["teacher_quality"] = teacher_quality(log, len(reference["generated_ids"]))
            require(arm["decode"]["metrics"][:77] == arm["prefill"]["metrics"], "independent native prefill differs")
            arm["native_quality"] = quality.score(old["gold"], arm["decode"]["native_text"],
                complete=arm["decode"]["native_ids"][-1] in reference["eos_ids"])
            save()
        report["summary"] = summarize(case_id, report["arms"], reference)
        report["phase"] = "complete"; save()
        return 0 if report["summary"]["all_selected_precision_bars_met"] else 1
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id", choices=list(review.CASES))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        protocol, _, _ = preflight(args.case_id)
        print(json.dumps({"preflight": "passed", "protocol": protocol["schema"], "model_execution": False}))
        return 0
    return run(args.case_id)


if __name__ == "__main__":
    raise SystemExit(main())
