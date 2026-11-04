#!/usr/bin/env python3
"""Retained-reference A/B: no original model forward, new gold or budget change."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "models"), str(ROOT / "tools"), str(ROOT / "tools/ci-heavy")]
import extract_ocr_nov4_evidence as archive
from ci_process import run as execute

PROTOCOL = ROOT / "docs/lighton3-pixtral-epsilon-ab-v1.json"
GATE = "CRISPEMBED_LIGHTONOCR3_PIXTRAL_EPS"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return archive.sha256(Path(path).read_bytes()) if Path(path).stat().st_size < 2**20 else file_sha(path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def trace_control(old, new):
    # Source/binary/timing receipts are deliberately separate; all actual tensor
    # payloads, decoded bytes and non-timing trace fields must reproduce.
    a = archive.read_json((old / "trace.json").read_bytes())
    b = archive.read_json((new / "trace.json").read_bytes())
    require({k: v for k, v in a.items() if k != "seconds"} ==
            {k: v for k, v in b.items() if k != "seconds"}, "gate0 trace/IDs changed")
    require((old / "decoded.txt").read_bytes() == (new / "decoded.txt").read_bytes(), "gate0 decoded bytes changed")
    paths = sorted(p.name for p in old.glob("*.f32"))
    require(paths == sorted(p.name for p in new.glob("*.f32")), "gate0 tensor inventory changed")
    rows = []
    for name in paths:
        expected, actual = old / name, new / name
        require(expected.stat().st_size == actual.stat().st_size and file_sha(expected) == file_sha(actual),
                "gate0 tensor bytes changed: " + name)
        rows.append({"name": name, "bytes": actual.stat().st_size, "sha256": file_sha(actual)})
    return {"tensor_files": len(rows), "all_bytes_exact": True, "tensors": rows}


def source_guard(protocol):
    require(protocol["schema"] == "crispembed.lighton3.pixtral-epsilon-ab.v1"
            and protocol["baseline_source"] == "4476be6eb21ecacc3c001ed4e02928d0a98be0ea"
            and len(protocol["cases"]) == 6 and "form-grounding" in protocol["cases"], "frozen A/B protocol differs")
    require(os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_REPOSITORY") == archive.REPOSITORY,
            "hosted repository runner required")
    require(os.environ.get("EPS_SELECTED_SCRIPT") == "tools/ci-heavy/lighton3_pixtral_eps_ab.py"
            and os.environ.get("EPS_SELECTED_RUNNER") == "ubuntu-latest"
            and os.environ.get("EPS_SELECTED_MODE") == "portable"
            and os.environ.get("EPS_SELECTED_PIP", "") == "", "closed A/B dispatch differs")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(source == os.environ["GITHUB_SHA"] and not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], text=True), "source must be clean and exact")
    for name, expected in protocol["owned_sha256"].items():
        require(sha(ROOT / name) == expected, "owned source changed: " + name)
    for name, expected in protocol["protected_git_objects"].items():
        require(subprocess.check_output(["git", "rev-parse", "HEAD:" + name], text=True).strip() == expected,
                "protected model/reference source changed: " + name)
    changed = subprocess.check_output(["git", "diff", "--name-only", protocol["baseline_source"], "HEAD", "--", "src"], text=True).splitlines()
    require(set(changed) == {"src/lightonocr.cpp", "src/lightonocr3_profile.h"}, "unreviewed runtime delta")
    return source


def download(target, destination, token):
    with urllib.request.build_opener(archive.NoRedirect).open(archive.archive_location(target, token), timeout=60) as stream:
        return archive.copy_verified(stream, destination, target["bytes"], target["sha256"], time.monotonic() + 20 * 60)


def first_proof(args, protocol, scratch, token):
    require(args.first_run and args.first_artifact and args.first_sha, "full phase requires reviewed first proof")
    run = archive.fetch_metadata("/actions/runs/" + str(args.first_run), token)
    item = archive.fetch_metadata("/actions/artifacts/" + str(args.first_artifact), token)
    require(run["conclusion"] == "success" and run["status"] == "completed" and run["event"] == "workflow_dispatch"
            and run["path"] == ".github/workflows/heavy-cpu.yml" and run["repository"]["full_name"] == archive.REPOSITORY,
            "first proof producer differs")
    require(item["name"] == "lighton3-eps-review-" + str(args.first_run) and not item["expired"]
            and item["digest"] == "sha256:" + args.first_sha and item["workflow_run"]["id"] == args.first_run
            and item["workflow_run"]["head_sha"] == run["head_sha"] and item["size_in_bytes"] <= 8 * 2**20,
            "first proof artifact differs")
    target = {"artifact_id": args.first_artifact, "bytes": item["size_in_bytes"], "sha256": args.first_sha}
    path = scratch / "first.zip"; download(target, path, token)
    with zipfile.ZipFile(path) as z:
        archive.safe_members(z)
        data = z.read("summary.json")
    first = archive.read_json(data)
    require(first["phase"] == "first" and first["completed"] and first["pass"]
            and first["protocol_sha256"] == sha(PROTOCOL) and first["original_forward_calls"] == 0,
            "first proof is not a positive unchanged-protocol A/B")
    require(len(first["rows"]) == 4, "first proof coverage differs")
    return first


def extract_inputs(path, destination, cases):
    require(not destination.exists(), "retained input root must be fresh")
    destination.mkdir()
    selected_bytes, rows = 0, []
    with zipfile.ZipFile(path) as z:
        archive.safe_members(z)
        for info in z.infolist():
            p = Path(info.filename)
            parts = p.parts
            if (info.is_dir() or len(parts) < 3 or parts[0] != "lightonocr3-1b" or parts[1] not in cases
                    or p.suffix not in (".npy", ".f32", ".json", ".txt")):
                continue
            selected_bytes += info.file_size
            require(info.file_size <= 512 * 2**20 and selected_bytes <= 24 * 2**30, "retained member budget exceeded")
            output = destination / info.filename; output.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, output.open("xb") as stream:
                shutil.copyfileobj(source, stream, 2**20)
            require(output.stat().st_size == info.file_size, "retained member truncation")
            rows.append({"path": info.filename, "bytes": info.file_size, "sha256": file_sha(output)})
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--phase", choices=("first", "full"), default="first")
    parser.add_argument("--first-run", type=int); parser.add_argument("--first-artifact", type=int)
    parser.add_argument("--first-sha")
    args = parser.parse_args()
    protocol = archive.read_json(PROTOCOL.read_bytes()); source = source_guard(protocol)
    out, scratch = Path(os.environ["EPS_OUT"]), Path(os.environ["EPS_SCRATCH"])
    require(not out.exists() and not scratch.exists(), "fresh output/scratch required")
    out.mkdir(parents=True); scratch.mkdir(parents=True)
    review, raw = out / "review", out / "raw"; review.mkdir(); raw.mkdir()
    report = {"schema": protocol["schema"], "phase": args.phase, "source": source,
              "protocol_sha256": sha(PROTOCOL), "original_forward_calls": 0, "rows": [], "commands": [],
              "default_changed": False, "publication_qualified": False, "completed": False, "pass": False,
              "native_request_processes": 0}
    def save(): write(review / "summary.json", report)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CRISPEMBED_LIGHTON", "LOCR_"))
           and k not in ("CRISPEMBED_MAX_PIXELS", "GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "HF_TOKEN_2",
                         "KAGGLE_API_TOKEN", "KAGGLE_ACCOUNT")}
    env.update(CRISPEMBED_BUILD_MODE="portable", CRISPEMBED_LIGHTONOCR3="1", CRISPEMBED_LIGHTONOCR_FORCE_CPU="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               HF_HOME=str(scratch / "hf"), PIP_NO_CACHE_DIR="1")
    def command(name, argv, extra=None, timeout=3600):
        log = review / (name + ".log")
        result = execute(argv, log, cwd=ROOT, env=env | (extra or {}), timeout=timeout)
        report["commands"].append({"name": name, **result, "log": log.name}); save()
        require(not result["timed_out"] and result["exit"] == 0, "A/B command failed: " + name)
        return result
    save()
    try:
        memory = int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:"))) * 1024
        require(memory >= 12 * 2**30 and shutil.disk_usage(scratch).free >= 40 * 2**30,
                "need 12GiB RAM and 40GiB combined scratch/output before build or transfers")
        # Small operator controls precede archive/model acquisition. The ordinary
        # production library still excludes the observer compile definition.
        build = scratch / "build"
        command("configure", ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
                              "-DCRISPEMBED_NATIVE=OFF", "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF",
                              "-DGGML_BLAS=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"])
        command("build", ["cmake", "--build", build, "--parallel", "4", "--target", "test-lightonocr3-contract", "lightonocr3-trace"])
        command("norm-controls", [build / "test-lightonocr3-contract"])
        token = os.environ["GH_TOKEN"]
        imported = first_proof(args, protocol, scratch, token) if args.phase == "full" else None
        if imported:
            report["rows"] = imported["rows"]
            report["imported_first_run"] = args.first_run
        target = protocol["original_artifact"]
        run = archive.fetch_metadata("/actions/runs/" + str(target["run_id"]), token)
        item = archive.fetch_metadata("/actions/artifacts/" + str(target["artifact_id"]), token)
        require(run["head_sha"] == protocol["baseline_source"] and run["status"] == "completed" and run["conclusion"] == "success"
                and run["event"] == "workflow_dispatch" and run["repository"]["full_name"] == archive.REPOSITORY
                and item["workflow_run"]["id"] == target["run_id"] and item["workflow_run"]["head_sha"] == run["head_sha"]
                and item["name"] == "heavy-" + str(target["run_id"]) and not item["expired"]
                and item["size_in_bytes"] == target["bytes"] and item["digest"] == "sha256:" + target["sha256"], "original artifact identity differs")
        zip_path = scratch / "original.zip"; report["verified_archive"] = download(target, zip_path, token); save()
        cases = ["form-grounding"] if args.phase == "first" else protocol["cases"]
        retained = scratch / "retained"
        members = extract_inputs(zip_path, retained, cases)
        write(review / "retained-members.json", members)
        command("requirements", [sys.executable, "-m", "pip", "install", "--report", review / "resolver.json",
                                 "-r", ROOT / "models/requirements-lightonocr3-reference.txt"])
        command("dependency-check", [sys.executable, "-m", "pip", "check"])
        command("source-check", [sys.executable, "-c", "import sys;sys.path.insert(0,'tools');from dump_lightonocr3_reference import verify_environment;print(verify_environment())"])
        from compare_lightonocr3 import compare, validate_reference
        from lightonocr3_profile import source_directory, MODEL_ID
        models = {}
        checkpoint = source_directory(MODEL_ID, cache_dir=scratch / "hf/hub")
        for precision, expected in protocol["models"].items():
            model = scratch / (precision + ".gguf")
            command("convert-" + precision, [sys.executable, "models/convert-lightonocr-to-gguf.py", "--model", checkpoint,
                                           "--profile", "lightonocr3-1b", "--dtype", precision, "--output", model])
            require(model.stat().st_size == expected["bytes"] and file_sha(model) == expected["sha256"], "frozen GGUF changed")
            models[precision] = model
        binary = build / "lightonocr3-trace"; binary_sha = file_sha(binary)
        # No original/reference generator is called. Copy the immutable original
        # arrays into the new raw artifact so its evidence survives source expiry.
        for case in cases:
            old = retained / "lightonocr3-1b" / case
            ref = old / "reference"; meta = archive.read_json((ref / "reference.json").read_bytes()); validate_reference(meta)
            shutil.copytree(ref, raw / "references" / case)
            if imported and case == "form-grounding": continue
            fixture = next(x for x in protocol["fixture_cases"] if case == x["id"] + "-" + meta["mode"])
            image = ROOT / fixture["image"]; require(file_sha(image) == fixture["sha256"] == meta["image_sha256"], "image source changed")
            for precision, model in models.items():
                for gate in ("0", "1"):
                    base = raw / case / precision / ("gate" + gate)
                    dirs = {arm: base / arm for arm in ("forced", "autonomous", "plain")}
                    extra = {GATE: gate, "CRISPEMBED_LIGHTONOCR3_GROUNDING": "1" if meta["mode"] == "grounding" else "0"}
                    controls = {}
                    for arm, output in dirs.items():
                        cap = len(meta["token_ids"]) if arm == "forced" else 128
                        forced = ref / "forced-ids.txt" if arm == "forced" else "-"
                        report["native_request_processes"] += 1
                        result = command(case + "-" + precision + "-g" + gate + "-" + arm,
                                         [binary, model, image, output, str(cap), forced, "plain" if arm == "plain" else "observe"], extra)
                        write(output / "execution.json", {"schema_version": 1, "build_mode": "portable", "git_commit": source,
                              "precision": precision, "arm": arm, "process": result, "model_sha256": protocol["models"][precision]["sha256"],
                              "model_bytes": model.stat().st_size, "binary_sha256": binary_sha, "trace_sha256": file_sha(output / "trace.json"), "gate": gate})
                        if gate == "0": controls[arm] = trace_control(old / (precision + "-" + arm), output)
                    result = compare(ref, dirs["forced"], dirs["autonomous"], dirs["plain"], precision)
                    write(review / (case + "-" + precision + "-gate" + gate + ".json"), result)
                    row = {"case": case, "precision": precision, "gate": gate, "comparison": result,
                           "gate0_byte_controls": controls}
                    if gate == "1":
                        old_row = next(x for x in report["rows"] if x["case"] == case and x["precision"] == precision and x["gate"] == "0")
                        row["first_norm_improved"] = (result["stages"]["vis_ln_pre"]["max_abs"] <
                                                     old_row["comparison"]["stages"]["vis_ln_pre"]["max_abs"])
                    report["rows"].append(row); save()
        require(file_sha(binary) == binary_sha, "native binary changed")
        new = [x for x in report["rows"] if x["gate"] == "1"]
        require(len(report["rows"]) == (4 if args.phase == "first" else 24), "A/B coverage incomplete")
        report.update(completed=True, **{"pass": all(x["first_norm_improved"] and x["comparison"]["owner_bar_pass"] and x["comparison"]["decoded"]["ids_exact"]
                      and x["comparison"]["decoded"]["text_exact"] for x in new)})
        save()
        return 0 if report["pass"] else 1
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}; save(); raise


if __name__ == "__main__":
    raise SystemExit(main())
