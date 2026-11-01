#!/usr/bin/env python3
"""Manual hosted CPU/CUDA/native-Metal graph controls and optional MI-GAN timing.

No model is downloaded. Components do not qualify model quality. The optional
MI-GAN benchmark consumes only an explicitly supplied, SHA-pinned local model;
the separate original-first replay workflows remain the model acceptance gate.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("test-inpaint-graph", "test-inpaint-graph-spectral",
           "test-latent-inpaint-graph-unet", "test-migan-graph")
BUILD_TARGETS = TARGETS + ("test-lama-diff", "test-latent-inpaint", "test-migan-diff",
                          "test-migan512-diff", "test-inpaint-benchmark", "crispembed-cli")
CPU_GATES = ("CRISPEMBED_INPAINT_GEMM_LAYOUT", "CRISPEMBED_INPAINT_REFLECT_GATHER",
             "CRISPEMBED_INPAINT_CPU_FFT", "CRISPEMBED_INPAINT_CPU_GLA",
             "CRISPEMBED_MOEBIUS_RELATIVE_REASSOC", "CRISPEMBED_MOEBIUS_LOCAL_REASSOC",
             "CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM",
             "CRISPEMBED_MIGAN_CPU_KERNELS", "CRISPEMBED_INPAINT_CPU_ATTENTION_TILES",
             "CRISPEMBED_INPAINT_CPU_DEPTHWISE")
# Hashing is an independent opt-in; resetting it must not expand the existing
# CPU component preset.
RESET_GATES = CPU_GATES + ("CRISPEMBED_INPAINT_SHA256_ACCEL",)
OBSERVER_SYMBOLS = {
    "crispembed-inpaint-graph": ("inpaint_graph::Builder::set_observer(",),
    "crispembed-latent-inpaint": ("latent_inpaint::run_reference(", "latent_inpaint::set_observer(",
                                  "latent_inpaint::Engine::set_observer("),
    "crispembed-lama": ("lama::run_reference(", "lama::set_observer("),
    "crispembed-migan": ("migan::test::run_observed(", "migan::test::observer_retention_regression(",
                          "migan::test::graph_backend_controls(", "migan::test::get_observed_execution_stats("),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def separate_output_trees(output, scratch):
    output, scratch = Path(output).resolve(), Path(scratch).resolve()
    if output == scratch or output in scratch.parents or scratch in output.parents:
        raise ValueError("output and scratch must be separate directory trees")


def validate_ggml_identity(root, declared_root, commit, gitlink):
    if Path(declared_root).resolve() != (Path(root) / "ggml").resolve():
        raise ValueError("ggml directory is not its own initialized Git repository")
    match = re.fullmatch(r"160000 commit ([0-9a-f]{40}|[0-9a-f]{64})\tggml", gitlink)
    if match is None or match.group(1) != commit:
        raise ValueError("ggml HEAD differs from the reviewed parent commit's gitlink")


def binary_path(build, name):
    name = "crispembed" if name == "crispembed-cli" else name
    found = [p for suffix in ("", ".exe") for p in Path(build).rglob(name + suffix) if p.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one built {name}, found {len(found)}")
    return found[0]


def archive_path(build, name):
    found = [p for p in Path(build).rglob("lib" + name + ".a") if p.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one built archive {name}, found {len(found)}")
    return found[0]


def assert_observer_symbols(symbols, family, testing):
    expected = OBSERVER_SYMBOLS[family]
    present = [token in symbols for token in expected]
    if testing and not all(present):
        raise RuntimeError(f"test archive {family} is missing its reference hooks")
    if not testing and any(token in symbols for tokens in OBSERVER_SYMBOLS.values() for token in tokens):
        raise RuntimeError(f"production archive/executable {family} contains a test observer hook")


def configure_command(build, backend):
    if backend not in ("cpu", "cuda", "metal"):
        raise ValueError("explicit cpu, cuda or metal backend is required")
    result = ["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
              "-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_BLAS=OFF",
              "-DGGML_OPENMP=OFF", "-DGGML_LLAMAFILE=ON", "-DGGML_VULKAN=OFF",
              "-DGGML_BACKEND_DL=OFF", "-DBUILD_SHARED_LIBS=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF",
              "-DGGML_CUDA=" + ("ON" if backend == "cuda" else "OFF"),
              "-DGGML_METAL=" + ("ON" if backend == "metal" else "OFF")]
    if backend == "cuda":
        result += ["-DCMAKE_CUDA_ARCHITECTURES=native", "-DGGML_CUDA_GRAPHS=ON"]
    if backend == "metal":
        result += ["-DCMAKE_OSX_ARCHITECTURES=arm64", "-DGGML_METAL_EMBED_LIBRARY=ON"]
    return result


def reset_gate_environment(environment):
    result = dict(environment)
    result.update({name: "0" for name in RESET_GATES})
    return result


def component_profiles(backend):
    """Reset inherited candidates and exercise each CPU alternative explicitly."""
    if backend not in ("cpu", "cuda", "metal"):
        raise ValueError("component profiles require an explicit backend")
    baseline = reset_gate_environment({})
    profiles = [("portable", TARGETS, baseline)]
    if backend == "cpu":
        optimized = dict(baseline)
        optimized.update({name: "1" for name in CPU_GATES if name != "CRISPEMBED_INPAINT_CPU_ATTENTION_TILES"})
        profiles.append(("cpu-optimized", TARGETS, optimized))
        profiles.append(("cpu-attention-attribution", ("test-inpaint-graph",),
                         dict(baseline, CRISPEMBED_INPAINT_CPU_ATTENTION_TILES="1")))
    return profiles


def backend_matches(requested, selected):
    selected = selected.lower()
    if requested == "cpu":
        return selected == "cpu"
    if requested == "cuda":
        return re.fullmatch(r"cuda[0-9]*", selected) is not None
    if requested == "metal":
        return re.fullmatch(r"(?:metal|mtl)[0-9]*", selected) is not None
    return False


def device_evidence(text, requested, full_model=False):
    """Fail closed when a successful command does not identify its real device."""
    records, names = [], set()
    for line in text.splitlines():
        payload = line[line.find("{"):] if "{" in line else ""
        if payload and (line.startswith("MIGAN_GRAPH_") or line.startswith('{"suite":')):
            def reject_constant(value):
                raise ValueError("nonfinite native JSON metric: " + value)
            row = json.loads(payload, parse_constant=reject_constant)
            records.append(row)
            kind = row.get("kind")
            if not full_model or kind == "production" or "selected_backend" in row:
                name = row.get("selected_backend", row.get("backend"))
                if name:
                    names.add(name)
        if not full_model:
            names.update(re.findall(r"(?<!\w)backend=([A-Za-z0-9_-]+)", line))
    if not names or any(not backend_matches(requested, name) for name in names):
        raise RuntimeError(f"requested {requested}, native selected devices {sorted(names)}; no fallback is accepted")
    if any(row.get("passed") is False or row.get("failures", 0) for row in records):
        raise RuntimeError("native receipt reports failed component or model checks")
    if full_model:
        benchmark = [row for row in records if "warm_reuse_verified" in row]
        if len(benchmark) != 1 or not all(benchmark[0].get(key) is True for key in (
                "passed", "warm_reuse_verified", "repeat_bit_exact", "observer_isolated", "observer_bit_exact")):
            raise RuntimeError("full-model receipt lacks successful reuse/parity/observer evidence")
    return {"selected_devices": sorted(names), "native_records": records}


def migan_kernel_evidence(evidence, enabled):
    """The MI test parses its own CLI flag, so environment alone is not proof."""
    rows = evidence.get("native_records", [])
    actual = [row for row in rows if row.get("kind") == "synthetic-controls"]
    if len(actual) != 1 or actual[0].get("cpu_kernels") is not enabled:
        raise RuntimeError("MI-GAN component receipt did not execute the requested CPU kernel profile")


def terminate_tree(process):
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The parent may have exited while compiler grandchildren remain.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
        if process.poll() is None:
            process.kill()
    process.wait(timeout=15)


class Runbook:
    def __init__(self, output, deadline, report, environment=None):
        self.output, self.deadline, self.report = Path(output), deadline, report
        self.environment = environment

    def save(self):
        target = self.output / "graph-acceptance.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(target)

    def run(self, label, command, *, required=True, timeout=900, cwd=ROOT, stdin_path=None, environment=None):
        command = [str(value) for value in command]
        log = self.output / (label + ".log")
        record = {"label": label, "command": command, "status": "starting", "returncode": None, "log": log.name}
        effective_environment = self.environment if environment is None else environment
        if effective_environment is not None:
            record["cpu_candidate_gates"] = {name: effective_environment.get(name, "0") for name in RESET_GATES}
        if stdin_path is not None:
            record["stdin"] = str(Path(stdin_path).name)
        self.report["commands"].append(record)
        self.save()
        remaining = min(timeout, self.deadline - time.monotonic())
        started, process = time.monotonic(), None
        print(label + ": " + " ".join(command), flush=True)
        try:
            if remaining <= 0:
                record["status"] = "deadline-exceeded-before-start"
                raise TimeoutError("graph acceptance deadline reached")
            with ExitStack() as stack:
                stream = stack.enter_context(log.open("w", encoding="utf-8"))
                source = stack.enter_context(Path(stdin_path).open("rb")) if stdin_path is not None else None
                process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=source, env=effective_environment, start_new_session=os.name == "posix")
                record.update({"pid": process.pid, "status": "running"})
                self.save()
                try:
                    record["returncode"] = process.wait(timeout=remaining)
                    record["status"] = "completed"
                except BaseException as error:
                    record["status"] = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "interrupted"
                    terminate_tree(process)
                    record["returncode"] = process.returncode
                    raise
            if required and record["returncode"]:
                raise RuntimeError(f"{label} failed ({record['returncode']}); see {log.name}")
        except BaseException as error:
            record["error"] = str(error)
            if process is None and record["status"] != "deadline-exceeded-before-start":
                record["status"] = "failed-to-start"
            raise
        finally:
            record["seconds"] = time.monotonic() - started
            if log.exists():
                record["sha256"] = sha256(log)
            self.save()
        return record

    def text(self, record):
        return (self.output / record["log"]).read_text(encoding="utf-8", errors="replace")


def audit_observers(runner, build):
    report = runner.report
    report["observer_audit"] = {"status": "running", "archives": [], "production_cli_clean": False}
    directory = runner.output / "archives"
    directory.mkdir()

    def symbols(label, path):
        options = ["-gU"] if platform.system() == "Darwin" else ["-g", "--defined-only"]
        raw = runner.run(label + "-nm", ["nm", *options, path], timeout=60)
        demangled = runner.run(label + "-demangled", ["c++filt"], timeout=60,
                               stdin_path=runner.output / raw["log"])
        return runner.text(demangled), demangled["log"]

    for family in OBSERVER_SYMBOLS:
        for testing in (False, True):
            name = family + ("-test" if testing else "")
            archive = archive_path(build, name)
            text, log = symbols(name, archive)
            assert_observer_symbols(text, family, testing)
            destination = directory / archive.name
            shutil.copy2(archive, destination)
            artifact = {"path": str(destination.relative_to(runner.output)), "sha256": sha256(destination),
                        "bytes": destination.stat().st_size}
            report["artifacts"].append(artifact)
            report["observer_audit"]["archives"].append({"name": name, "testing": testing, "log": log, **artifact})
            runner.save()
    for target, label in (("crispembed-cli", "production_cli"), ("test-inpaint-benchmark", "production_benchmark")):
        text, log = symbols(label, binary_path(build, target))
        assert_observer_symbols(text, "crispembed-inpaint-graph", False)
        report["observer_audit"].update({label + "_clean": True, label + "_symbols": log})
    report["observer_audit"]["status"] = "passed"
    runner.save()


def model_identity(path, expected_sha, checkpoint_sha):
    if not all(re.fullmatch("[0-9a-f]{64}", value or "") for value in (expected_sha, checkpoint_sha)):
        raise ValueError("reviewed lowercase GGUF and checkpoint SHA-256 pins are required")
    path = Path(path)
    if not path.is_file() or not 32 <= path.stat().st_size <= 512 * 1024 * 1024:
        raise ValueError("MI-GAN local GGUF must be a regular file in [32 B,512 MiB]")
    actual = sha256(path)
    if actual != expected_sha:
        raise ValueError("local MI-GAN GGUF SHA-256 differs from the supplied independent pin")
    return {"gguf_sha256": actual, "gguf_bytes": path.stat().st_size, "checkpoint_sha256": checkpoint_sha}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda", "metal"), required=True)
    parser.add_argument("--gpu-admitted", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--deadline-minutes", type=int, default=50)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--migan-model", type=Path)
    parser.add_argument("--migan-model-sha256")
    parser.add_argument("--migan-checkpoint-sha256")
    parser.add_argument("--migan-size", type=int, choices=(256, 512), default=256)
    args = parser.parse_args(argv)
    args.output = args.output.expanduser().resolve()
    args.scratch = args.scratch.expanduser().resolve()
    if args.migan_model:
        args.migan_model = args.migan_model.expanduser().resolve()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        parser.error("heavy graph orchestration is restricted to the manually dispatched hosted workflow")
    if args.backend != "cpu" and not args.gpu_admitted:
        parser.error("native GPU execution requires the workflow's explicit runner admission")
    if args.backend == "metal" and (platform.system() != "Darwin" or platform.machine().lower() not in ("arm64", "aarch64")):
        parser.error("Metal requires a provisioned native Apple Silicon runner")
    if args.backend == "cuda" and platform.system() != "Linux":
        parser.error("this CUDA admission lane requires the provisioned Linux GPU runner")
    try:
        separate_output_trees(args.output, args.scratch)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists() or args.scratch.exists():
        parser.error("output and scratch must be distinct new directories")
    if not (5 <= args.deadline_minutes <= 170 and 1 <= args.threads <= 8 and 1 <= args.parallel <= 4 and 1 <= args.warm_runs <= 100):
        parser.error("deadline/threads/parallel/warm-runs exceed their bounded ranges")
    pins = (bool(args.migan_model), bool(args.migan_model_sha256), bool(args.migan_checkpoint_sha256))
    if any(pins) and not all(pins):
        parser.error("a local MI-GAN model requires both independent SHA-256 pins")
    if args.migan_model and os.environ.get("RUNNER_ENVIRONMENT") != "self-hosted":
        parser.error("full-model paths are admitted only on the provisioned self-hosted runner")
    args.output.mkdir(parents=True)
    args.scratch.mkdir(parents=True)
    report = {"schema": 1, "status": "running", "requested_backend": args.backend,
              "precision_profile": "fast-diagnostic" if args.fast and args.backend != "cpu" else "strict-f32",
              "model_free_only": not bool(args.migan_model), "full_model_qualified": False,
              "original_reference_replayed": False, "model_free_controls": "pending", "commands": [],
              "components": [], "artifacts": [], "platform": {"system": platform.system(), "machine": platform.machine()},
              "full_model_benchmark": {"status": "pending" if args.migan_model else "not_requested"}}
    env = reset_gate_environment(os.environ)
    env.update({"OMP_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": str(args.threads),
                "MKL_NUM_THREADS": str(args.threads), "CRISPEMBED_INPAINT_GRAPH": "1",
                "CRISPEMBED_INPAINT_FAST": "1" if args.fast else "0", "CRISPEMBED_INPAINT_PROFILE": "1",
                "CRISPEMBED_INPAINT_BACKEND": args.backend})
    # Pinned ggml treats even the string "0" as disabling CUDA capture.
    env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    runner = Runbook(args.output, time.monotonic() + args.deadline_minutes * 60, report, env)
    runner.save()
    previous_signals = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f"hosted orchestration interrupted by signal {signum}")

    for name in (signal.SIGTERM, signal.SIGINT):
        previous_signals[name] = signal.signal(name, interrupted)
    build = args.scratch / "build"
    try:
        report["commit"] = runner.text(runner.run("source-head", ["git", "rev-parse", "HEAD"], timeout=30)).strip()
        report["ggml_commit"] = runner.text(runner.run("ggml-head", ["git", "-C", "ggml", "rev-parse", "HEAD"], timeout=30)).strip()
        ggml_root = runner.text(runner.run("ggml-root", ["git", "-C", "ggml", "rev-parse", "--show-toplevel"], timeout=30)).strip()
        report["ggml_gitlink"] = runner.text(runner.run("ggml-gitlink", ["git", "ls-tree", "HEAD", "ggml"], timeout=30)).strip()
        validate_ggml_identity(ROOT, ggml_root, report["ggml_commit"], report["ggml_gitlink"])
        report["source_dirty"] = bool(runner.text(runner.run("source-status", ["git", "status", "--porcelain"], timeout=30)).strip())
        report["ggml_dirty"] = bool(runner.text(runner.run("ggml-status", ["git", "-C", "ggml", "status", "--porcelain"], timeout=30)).strip())
        runner.run("source-patch", ["git", "diff", "--binary", "HEAD"], timeout=60)
        runner.run("ggml-patch", ["git", "-C", "ggml", "diff", "--binary", "HEAD"], timeout=60)
        runner.run("source-archive", ["git", "archive", "--format=tar.gz", "--output", args.output / "source-head.tar.gz", "HEAD"], timeout=120)
        runner.run("ggml-archive", ["git", "-C", "ggml", "archive", "--format=tar.gz", "--output", (args.output / "ggml-head.tar.gz").resolve(), "HEAD"], timeout=120)
        if report["source_dirty"] or report["ggml_dirty"]:
            raise RuntimeError("hosted graph build requires a clean reviewed checkout and ggml submodule")
        if os.environ.get("GITHUB_SHA") and report["commit"] != os.environ["GITHUB_SHA"]:
            raise RuntimeError("checked-out source HEAD differs from dispatched GitHub SHA")
        runner.run("cmake-version", ["cmake", "--version"], timeout=30)
        if args.backend == "cuda":
            runner.run("cuda-device", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], timeout=60)
            runner.run("cuda-toolkit", ["nvcc", "--version"], timeout=30)
        elif args.backend == "metal":
            runner.run("metal-toolchain", ["xcrun", "-f", "metal"], timeout=30)
            runner.run("macos-version", ["sw_vers"], timeout=30)
            runner.run("apple-gpu", ["system_profiler", "SPDisplaysDataType", "-json"], timeout=120)
        else:
            runner.run("cpu-information", ["lscpu"], required=False, timeout=30)
        runner.run("hosted-driver-controls", [sys.executable, "tests/test_inpaint_graph_hosted.py", "-v"], timeout=60)
        runner.run("configure", configure_command(build, args.backend), timeout=300)
        runner.run("build", ["cmake", "--build", build, "--config", "Release", "--parallel", args.parallel,
                             "--target", *BUILD_TARGETS], timeout=args.deadline_minutes * 60)
        after_head = runner.text(runner.run("source-head-after-build", ["git", "rev-parse", "HEAD"], timeout=30)).strip()
        after_ggml = runner.text(runner.run("ggml-head-after-build", ["git", "-C", "ggml", "rev-parse", "HEAD"], timeout=30)).strip()
        after_status = runner.text(runner.run("source-status-after-build", ["git", "status", "--porcelain"], timeout=30)).strip()
        after_ggml_status = runner.text(runner.run("ggml-status-after-build", ["git", "-C", "ggml", "status", "--porcelain"], timeout=30)).strip()
        if after_head != report["commit"] or after_ggml != report["ggml_commit"] or after_status or after_ggml_status:
            raise RuntimeError("reviewed source changed during the build; no clean-source receipt can be issued")
        artifacts = args.output / "executables"
        artifacts.mkdir()
        binaries = []
        for target in BUILD_TARGETS:
            binary = binary_path(build, target)
            destination = artifacts / binary.name
            shutil.copy2(binary, destination)
            identity = {"path": str(destination.relative_to(args.output)), "sha256": sha256(destination), "bytes": destination.stat().st_size}
            report["artifacts"].append(identity)
            binaries.append({"name": destination.stem if destination.suffix == ".exe" else destination.name, **identity})
        build_receipt = {"schema": "crispembed-inpaint-graph-build-v1", "native_commit": report["commit"],
                         "ggml_commit": report["ggml_commit"], "source_tree_clean": True, "binaries": binaries,
                         "requested_backend": args.backend,
                         "producer": {"github_sha": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID")}}
        receipt_path = args.output / "build-receipt.json"
        receipt_path.write_text(json.dumps(build_receipt, indent=2) + "\n", encoding="utf-8")
        report["build_receipt"] = {"path": receipt_path.name, "sha256": sha256(receipt_path)}
        cache = build / "CMakeCache.txt"
        if cache.is_file():
            shutil.copy2(cache, args.output / "CMakeCache.txt")
        runner.save()
        audit_observers(runner, build)
        for profile, targets, gates in component_profiles(args.backend):
            component_env = dict(env, **gates)
            for target in targets:
                command = [binary_path(build, target)]
                if target == "test-migan-graph":
                    command += ["--contracts", "--warm-runs", args.warm_runs]
                    if gates["CRISPEMBED_MIGAN_CPU_KERNELS"] == "1":
                        command += ["--cpu-kernels"]
                    if args.fast:
                        command += ["--fast"]
                command += ["--backend", args.backend, "--threads", args.threads]
                record = runner.run(profile + "-" + target, command, timeout=180, environment=component_env)
                evidence = device_evidence(runner.text(record), args.backend)
                if target == "test-migan-graph":
                    migan_kernel_evidence(evidence, gates["CRISPEMBED_MIGAN_CPU_KERNELS"] == "1")
                report["components"].append({"name": target, "profile": profile, "cpu_candidate_gates": gates,
                                             "status": "passed", "log": record["log"], **evidence})
                runner.save()
        report["model_free_controls"] = "passed"
        if args.migan_model:
            report["full_model_benchmark"].update(model_identity(args.migan_model, args.migan_model_sha256, args.migan_checkpoint_sha256))
            # The model consumed by both runs is an owned, verified snapshot;
            # later edits to the runner's provisioned source cannot alter it.
            pinned_model = args.scratch / "migan-pinned.gguf"
            shutil.copy2(args.migan_model, pinned_model)
            model_identity(pinned_model, args.migan_model_sha256, args.migan_checkpoint_sha256)
            runner.save()
            command = [binary_path(build, "test-migan-graph"), "--model", pinned_model,
                       "--checkpoint-sha256", args.migan_checkpoint_sha256, "--size", args.migan_size,
                       "--backend", args.backend, "--threads", args.threads, "--warm-runs", args.warm_runs]
            if args.fast:
                command += ["--fast"]
            record = runner.run("migan-full-model-benchmark", command, timeout=1800)
            report["full_model_benchmark"].update({"status": "passed", "log": record["log"],
                                                  **device_evidence(runner.text(record), args.backend, full_model=True)})
        for name in ("source-head.tar.gz", "ggml-head.tar.gz"):
            path = args.output / name
            report["artifacts"].append({"path": name, "sha256": sha256(path), "bytes": path.stat().st_size})
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed"
        report["error"] = str(error)
        raise
    finally:
        runner.save()
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)
    print(json.dumps({key: report[key] for key in ("status", "requested_backend", "model_free_only", "full_model_qualified")}, indent=2))


if __name__ == "__main__":
    main()
