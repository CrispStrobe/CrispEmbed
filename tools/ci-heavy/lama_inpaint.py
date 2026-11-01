#!/usr/bin/env python3
"""Manual hosted CPU build/original-first Big-LaMa acceptance orchestration.

Build-only never downloads checkpoints. The explicit original-native phase
downloads the pinned official archive, captures the original BEFORE converting
weights, then replays fixed numerical gates and preserves partial failures.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from lama_contract import load_contract, sha256, verify_source
from lama_reference_contract import CASES, THRESHOLDS


def binary_path(build, target):
    executable = "crispembed" if target == "crispembed-cli" else target
    found = [path for path in Path(build).rglob(executable) if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"expected one built executable {executable}, found {len(found)}")
    return found[0]


def assert_observer_symbols(symbols, testing):
    present = [token in symbols for token in ("lama::run_reference(", "lama::set_observer(")]
    if (testing and not all(present)) or (not testing and any(present)):
        raise AssertionError("production/test observer symbol isolation differs")


def terminate_tree(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The group can still contain compiler children even if its parent exited.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build-only", "original-native"), default="build-only")
    parser.add_argument("--cases", choices=("subtitle_bar", "full-panel"), default="subtitle_bar")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--deadline-minutes", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists() or args.scratch.exists() or not 5 <= args.deadline_minutes <= 170:
        parser.error("output/scratch must be new directories and deadline must be in[5,170]")
    args.output.mkdir(parents=True)
    args.scratch.mkdir(parents=True)
    deadline = time.monotonic() + args.deadline_minutes * 60
    report = {"schema": 1, "phase": args.phase, "status": "running", "native_qualified": False,
              "thresholds": THRESHOLDS, "commands": [], "cases": []}

    def save():
        (args.output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def run(label, command, required=True):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("acceptance deadline reached")
        command = [str(value) for value in command]
        print(label + ": " + " ".join(command), flush=True)
        log = args.output / (label + ".log")
        start = time.monotonic()
        record = {"label": label, "command": command, "status": "running", "returncode": None, "log": log.name}
        report["commands"].append(record)
        save()
        process = None
        try:
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=os.name == "posix")
                record["pid"] = process.pid
                save()
                try:
                    record["returncode"] = process.wait(timeout=remaining)
                    record["status"] = "completed"
                except BaseException as error:
                    terminate_tree(process)
                    record["returncode"] = process.returncode
                    record["status"] = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
                    raise
        except BaseException as error:
            record["error"] = str(error)
            if process is None:
                record["status"] = "failed-to-start"
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            if log.exists():
                record["sha256"] = sha256(log)
            save()
        if required and record["returncode"]:
            raise RuntimeError(f"{label} failed ({record['returncode']}); see {log.name}")
        return record

    build = args.scratch / "build"

    def binary(name):
        return binary_path(build, name)

    try:
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        report["source_contract_sha256"] = sha256(ROOT / "docs/lama-source-contract.json")
        run("python-contracts", [sys.executable, "tests/test_lama_contract.py", "-v"])
        run("hosted-controls", [sys.executable, "tests/test_lama_hosted.py", "-v"])
        run("configure", ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                          "-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_BLAS=OFF",
                          "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF",
                          "-DGGML_OPENMP=OFF", "-DGGML_LLAMAFILE=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"])
        run("build", ["cmake", "--build", build, "--target", "crispembed-cli", "test-lama-ops", "test-lama-diff", "-j2"])
        run("native-operators", [binary("test-lama-ops")])
        run("native-reference-contracts", [binary("test-lama-diff"), "--contracts"])
        artifacts = args.output / "build-artifacts"
        artifacts.mkdir()
        for name in ("crispembed-cli", "test-lama-ops", "test-lama-diff"):
            executable = binary(name)
            shutil.copy2(executable, artifacts / executable.name)
        for name in ("crispembed-lama", "crispembed-lama-test"):
            archive = list(build.rglob("lib" + name + ".a"))
            if len(archive) != 1:
                raise RuntimeError("missing/ambiguous native archive " + name)
            symbols = subprocess.check_output(["nm", "-g", "--defined-only", "-C", archive[0]], text=True)
            assert_observer_symbols(symbols, name.endswith("-test"))
            (args.output / (name + ".symbols.txt")).write_text(symbols, encoding="utf-8")
        report["build"] = "passed"
        report["model_free_controls"] = "passed"
        save()
        if args.phase == "original-native":
            contract = load_contract()
            source = args.scratch / "original-source"
            run("source-init", ["git", "init", source])
            run("source-remote", ["git", "-C", source, "remote", "add", "origin", contract["source"]["repository"]])
            run("source-fetch", ["git", "-C", source, "fetch", "--depth", "1", "origin", contract["source"]["commit"]])
            run("source-checkout", ["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"])
            verify_source(source, contract)
            archive_path = args.scratch / "big-lama.zip"
            checkpoint = contract["checkpoint"]
            with urllib.request.urlopen(checkpoint["url"], timeout=60) as response, archive_path.open("xb") as stream:
                count = 0
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("checkpoint download exceeded acceptance deadline")
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > checkpoint["bytes"]:
                        raise ValueError("official archive exceeds pinned byte count")
                    stream.write(block)
            if count != checkpoint["bytes"] or sha256(archive_path) != checkpoint["sha256"]:
                raise ValueError("downloaded official archive identity differs")
            report["checkpoint_archive_sha256"] = checkpoint["sha256"]
            selected = list(CASES) if args.cases == "full-panel" else [args.cases]
            reference = args.output / "original"
            run("capture-original", [sys.executable, "tools/dump_lama_reference.py", "--source-dir", source,
                                     "--checkpoint-archive", archive_path, "--output", reference, "--cases", *selected])
            model = args.scratch / "big-lama-f32.gguf"
            run("convert-after-original", [sys.executable, "models/convert-lama-to-gguf.py", "--archive", archive_path,
                                            "--source-dir", source, "--output", model])
            receipt_path = model.with_suffix(".gguf.json")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            shutil.copy2(receipt_path, args.output / "conversion.json")
            original = json.loads((reference / "reference.json").read_text(encoding="utf-8"))
            if [row["id"] for row in original["cases"]] != selected:
                raise AssertionError("original fixture coverage differs")
            report["gguf_sha256"] = receipt["gguf_sha256"]
            from PIL import Image
            for row in original["cases"]:
                name = row["id"]
                target = args.output / ("native-" + name)
                result = run("replay-" + name, [binary("test-lama-diff"), "--model", model,
                             "--model-sha256", receipt["gguf_sha256"], "--reference", reference / name / "reference.gguf",
                             "--reference-sha256", row["reference_sha256"], "--output", target], required=False)
                report["cases"].append({"id": name, "passed": result["returncode"] == 0,
                                        "reference_sha256": row["reference_sha256"], "log": result["log"]})
                for image_name in ("native", "difference-x16"):
                    pixels = target / (image_name + ".rgb8")
                    if pixels.exists() and pixels.stat().st_size == row["width"] * row["height"] * 3:
                        Image.frombytes("RGB", (row["width"], row["height"]), pixels.read_bytes()).save(target / (image_name + ".png"))
                save()
            report["numerical_acceptance"] = "passed_selected_cases" if all(row["passed"] for row in report["cases"]) else "failed"
            if report["numerical_acceptance"] == "failed":
                raise AssertionError("native/original comparison failed; thresholds were not changed")
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save()
        raise
    save()
    print(json.dumps({key: report[key] for key in ("status", "phase", "native_qualified")}, indent=2))


if __name__ == "__main__":
    main()
