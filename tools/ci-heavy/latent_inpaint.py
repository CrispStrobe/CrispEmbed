#!/usr/bin/env python3
"""Manual hosted build or pinned original-first Moebius/PixelHacker replay.

No model access occurs in build-only mode. The heavy mode requires an explicitly
provisioned CUDA worker and a separate existing original-framework interpreter;
it does not mutate that interpreter or substitute a CPU implementation for FLA.
Full weights stay in the worker scratch directory. Reports, original tensors,
native differences, notices and build artifacts go in the evidence directory.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import signal
import shutil
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from latent_inpaint_contract import (CONTRACT, CASES, fixture, load_contract, model_record,
                                    sha256, verify_checkpoint, verify_source)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def text_command(arguments):
    return subprocess.check_output([str(arg) for arg in arguments], cwd=ROOT, text=True).strip()


def resource_snapshot(path):
    result = {"platform": platform.platform(), "python": platform.python_version(),
              "cpu_count": os.cpu_count(), "disk": dict(zip(("total", "used", "free"), shutil.disk_usage(path)))}
    if hasattr(os, "getloadavg"):
        result["load_average"] = list(os.getloadavg())
    if Path("/proc/meminfo").is_file():
        result["memory_kib"] = {key: int(value.split()[0]) for key, value in
                                (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    return result


def contained_file(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in relative.split("/")) or "\\" in relative or "\0" in relative:
        raise ValueError("unsafe path in frozen source contract")
    destination = root.joinpath(*path.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("source path escaped task directory")
    return destination


def observer_symbols(symbols, test_archive):
    present = ["latent_inpaint::run_reference(" in symbols, "latent_inpaint::set_observer(" in symbols,
               "latent_inpaint::Engine::set_observer(" in symbols]
    if (test_archive and not all(present)) or (not test_archive and any(present)):
        raise ValueError("test observer symbol isolation differs")


class Run:
    def __init__(self, args):
        self.args = args
        self.deadline = time.monotonic() + args.deadline_minutes * 60
        self.output, self.scratch = args.output.resolve(), args.scratch.resolve()
        for directory in (self.output, self.scratch):
            if directory.exists() or directory.is_symlink():
                raise ValueError("choose new output and scratch directories")
        if self.output == self.scratch or self.output.is_relative_to(self.scratch) or self.scratch.is_relative_to(self.output):
            raise ValueError("evidence and scratch must be separate directory trees")
        self.output.mkdir(parents=True)
        self.scratch.mkdir(parents=True)
        self.report = {"schema": "crispembed-latent-inpaint-hosted-v1", "phase": args.phase,
                       "model": args.model, "complete": False, "passed": False,
                       "source_contract_sha256": sha256(CONTRACT), "commands": [], "downloads": [],
                       "build": "not-run", "original_reference": "not-run", "native_replay": "not-run",
                       "real_cli_empty_mask": "not-run", "native_qualified": False,
                       "representative_quality_qualified": False, "errors": [],
                       "resources_before": resource_snapshot(self.scratch)}
        self.save()

    def save(self):
        write_json(self.output / "report.json", self.report)

    def remaining(self, maximum=None):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("hosted run deadline reached; partial evidence retained")
        return remaining if maximum is None else min(maximum, remaining)

    def command(self, name, arguments, *, env=None, maximum=None, expected=0):
        arguments = [str(arg) for arg in arguments]
        record = {"name": name, "argv": arguments, "state": "running", "log": name + ".log"}
        self.report["commands"].append(record)
        self.save()
        start = time.monotonic()
        child_env = dict(os.environ)
        # A prior job's experimental flags must not silently activate a path in
        # model-free controls. Heavy native calls set their exact gate below.
        child_env.pop("CRISPEMBED_MOEBIUS", None)
        child_env.pop("CRISPEMBED_PIXELHACKER", None)
        child_env.update({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                          "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"})
        if env:
            child_env.update(env)
        print("Running " + name, flush=True)
        try:
            with (self.output / record["log"]).open("wb") as log:
                with subprocess.Popen(arguments, cwd=ROOT, env=child_env, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=os.name == "posix") as child:
                    try:
                        returncode = child.wait(timeout=self.remaining(maximum))
                    except BaseException:
                        # Terminate the entire task-owned process group: killing
                        # only cmake would leave compiler grandchildren running.
                        if os.name == "posix":
                            try:
                                os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        else:
                            child.kill()
                        child.wait(timeout=30)
                        raise
            record["returncode"] = returncode
            record["state"] = "passed" if returncode == expected else "failed"
            if returncode != expected:
                raise RuntimeError(f"{name} returned {returncode}; see {record['log']}")
        except subprocess.TimeoutExpired:
            record["state"] = "timed-out"
            raise
        except BaseException:
            if record["state"] == "running":
                record["state"] = "failed"
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            self.save()
        return self.output / record["log"]

    def download(self, url, destination, expected_bytes, expected_hash):
        if urllib.parse.urlsplit(url).scheme != "https" or expected_bytes < 0 or expected_bytes > 4_500_000_000:
            raise ValueError("invalid pinned HTTPS download contract")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError("download destination already exists")
        record = {"url": url, "bytes": expected_bytes, "sha256": expected_hash,
                  "state": "running", "downloaded_bytes": 0}
        self.report["downloads"].append(record)
        self.save()
        temporary = destination.with_suffix(destination.suffix + ".partial")
        digest, count = hashlib.sha256(), 0
        request = urllib.request.Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "CrispEmbed-latent-inpaint-reference/1"})
        try:
            with urllib.request.urlopen(request, timeout=min(60, self.remaining())) as response, temporary.open("xb") as output:
                if response.status != 200 or urllib.parse.urlsplit(response.url).scheme != "https":
                    raise ValueError("pinned download returned non-HTTPS or partial content")
                advertised = response.headers.get("Content-Length")
                if advertised and int(advertised) != expected_bytes:
                    raise ValueError("pinned download Content-Length differs")
                while True:
                    self.remaining()
                    block = response.read(min(1024 * 1024, expected_bytes - count + 1))
                    if not block:
                        break
                    count += len(block)
                    if count > expected_bytes:
                        raise ValueError("pinned download exceeded exact byte count")
                    digest.update(block)
                    output.write(block)
            if count != expected_bytes or digest.hexdigest() != expected_hash:
                raise ValueError("pinned download byte count/SHA-256 differs")
            temporary.replace(destination)
            record["state"] = "verified"
        except BaseException:
            record["state"] = "failed"
            raise
        finally:
            record["downloaded_bytes"] = count
            self.save()

    def source(self, name, contract):
        record = contract["source"][name]
        destination = self.output / "original-sources" / name
        destination.mkdir(parents=True)
        repository = urllib.parse.urlsplit(record["repository"])
        if repository.scheme != "https" or repository.netloc != "github.com" or repository.path not in ("/hustvl/Moebius", "/hustvl/PixelHacker"):
            raise ValueError("unexpected upstream source repository")
        for relative, expected in record["files"].items():
            path = contained_file(destination, relative)
            url = "https://raw.githubusercontent.com" + repository.path + "/" + record["commit"] + "/" + urllib.parse.quote(relative, safe="/")
            self.download(url, path, expected["bytes"], expected["sha256"])
        verify_source(destination, name, contract)
        return destination

    def provenance(self):
        head = text_command(["git", "rev-parse", "HEAD"])
        github_sha = os.environ.get("GITHUB_SHA")
        if github_sha and github_sha != head:
            raise ValueError("checked-out source differs from the dispatched commit")
        if text_command(["git", "status", "--porcelain", "--untracked-files=no"]):
            raise ValueError("tracked native source is dirty")
        tree = text_command(["git", "ls-tree", "HEAD", "ggml"]).split()
        actual_ggml = text_command(["git", "-C", ROOT / "ggml", "rev-parse", "HEAD"])
        if len(tree) != 4 or tree[0] != "160000" or tree[2] != actual_ggml or text_command(["git", "-C", ROOT / "ggml", "status", "--porcelain", "--untracked-files=no"]):
            raise ValueError("ggml submodule does not match the committed pin")
        paths = ["CMakeLists.txt", ".github/workflows/latent-inpaint.yml", "tools/ci-heavy/latent_inpaint.py",
                 "tools/dump_latent_inpaint_reference.py", "tools/latent_inpaint_contract.py",
                 "models/convert-latent-inpaint-to-gguf.py", "tests/test_latent_inpaint.cpp",
                 "src/latent_inpaint.cpp", "src/latent_inpaint_ops.cpp", "src/latent_inpaint_unet.cpp",
                 "src/latent_inpaint_vae.cpp", "src/latent_inpaint_schema.h", "examples/cli/inpaint_cli.cpp"]
        self.report["producer"] = {"source_commit": head, "github_sha": github_sha,
                                   "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                                   "ggml_commit": actual_ggml, "source_sha256": {path: sha256(ROOT / path) for path in paths}}
        self.save()

    def binary(self, name):
        filename = "crispembed" if name == "crispembed-cli" else name
        paths = [self.scratch / "build" / filename, self.scratch / "build" / "bin" / filename]
        found = [path for path in paths if path.is_file()]
        if len(found) != 1:
            raise ValueError("expected exactly one built binary: " + name)
        return found[0]

    def build(self):
        self.report["build"] = "running"
        self.save()
        self.command("python-contracts", [sys.executable, "tests/test_latent_inpaint_contract.py", "-v"], maximum=180)
        self.command("driver-contracts", [sys.executable, "tests/test_latent_inpaint_hosted.py", "-v"], maximum=180)
        self.command("configure", ["cmake", "-S", ROOT, "-B", self.scratch / "build", "-DCMAKE_BUILD_TYPE=Release",
                                   "-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_BLAS=OFF",
                                   "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF",
                                   "-DGGML_OPENMP=OFF", "-DGGML_LLAMAFILE=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"], maximum=600)
        targets = ["crispembed-latent-inpaint", "crispembed-cli", "test-latent-inpaint",
                   "test-latent-inpaint-attention", "test-latent-inpaint-ops", "test-latent-inpaint-pipeline", "test-inpaint"]
        self.command("build", ["cmake", "--build", self.scratch / "build", "--target", *targets, "-j2"], maximum=2400)
        self.command("native-contracts", [self.binary("test-latent-inpaint"), "--contracts"], maximum=120)
        for name in ("test-latent-inpaint-attention", "test-latent-inpaint-ops", "test-latent-inpaint-pipeline", "test-inpaint"):
            self.command(name, [self.binary(name)], maximum=180)
        cli_help = self.command("real-cli-help", [self.binary("crispembed-cli"), "--help"], maximum=60).read_text()
        if not all(token in cli_help for token in ("moebius", "pixelhacker", "--inpaint-vae", "--inpaint-model-sha256")):
            raise ValueError("real CLI does not advertise the implemented latent profiles")
        artifacts = self.output / "build-artifacts"
        artifacts.mkdir()
        for name in ("crispembed-cli", "test-latent-inpaint", "test-latent-inpaint-attention", "test-latent-inpaint-ops", "test-latent-inpaint-pipeline"):
            binary = self.binary(name)
            shutil.copy2(binary, artifacts / binary.name)
        for target in ("crispembed-latent-inpaint", "crispembed-latent-inpaint-test"):
            found = list((self.scratch / "build").rglob("lib" + target + ".a"))
            if len(found) != 1:
                raise ValueError("expected one native archive: " + target)
            symbols = text_command(["nm", "-g", "--defined-only", "-C", found[0]])
            (self.output / (target + ".symbols.txt")).write_text(symbols + "\n")
            observer_symbols(symbols, target.endswith("-test"))
            shutil.copy2(found[0], artifacts / found[0].name)
        self.report["build"] = "passed"
        self.report["production_observer_symbols_absent"] = True
        self.report["model_free_controls"] = "passed"
        self.save()

    def original_preflight(self, source):
        python = self.args.reference_python
        if not python or not Path(python).is_absolute() or not Path(python).is_file():
            raise ValueError("heavy phase requires an absolute path to the model's preprovisioned reference Python")
        preflight = r'''
import importlib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd() / "tools"))
from dump_latent_inpaint_reference import check_dependencies
from latent_inpaint_contract import load_contract
model, source, destination = sys.argv[1:]
versions = check_dependencies(model, load_contract())
import torch
if not torch.cuda.is_available():
    raise RuntimeError("actual original FLA reference requires the provisioned CUDA worker")
sys.path.insert(0, source)
modules = ("removal.v1_2.removal_model", "removal.v1_2.pipeline") if model == "moebius" else ("utils", "pipeline")
for module in modules:
    importlib.import_module(module)
properties = torch.cuda.get_device_properties(0)
free_bytes, total_bytes = torch.cuda.mem_get_info(0)
record = {"versions": versions, "cuda": torch.version.cuda, "device": properties.name,
          "compute_capability": [properties.major, properties.minor],
          "gpu_free_bytes": free_bytes, "gpu_total_bytes": total_bytes,
          "torch_config": torch.__config__.show(), "model_modules_imported": list(modules),
          "full_weights_downloaded": False, "original_inference_performed": False}
pathlib.Path(destination).write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
'''
        self.command("original-dependency-preflight", [python, "-c", preflight, self.args.model, source, self.output / "original-environment.json"], maximum=300,
                     env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})

    def convert(self, model, checkpoint, source):
        destination = self.scratch / "converted" / (model + "-f32.gguf")
        destination.parent.mkdir(exist_ok=True)
        self.command("convert-" + model, [sys.executable, "models/convert-latent-inpaint-to-gguf.py",
                                         "--model", model, "--checkpoint", checkpoint, "--source-dir", source,
                                         "--output", destination], maximum=900)
        receipt_path = destination.with_suffix(destination.suffix + ".json")
        receipt = json.loads(receipt_path.read_text())
        if receipt["gguf_sha256"] != sha256(destination) or receipt["source_contract_sha256"] != sha256(CONTRACT):
            raise ValueError("new GGUF receipt identity differs")
        evidence = self.output / "conversions"
        evidence.mkdir(exist_ok=True)
        shutil.copy2(receipt_path, evidence / receipt_path.name)
        notices = destination.with_suffix(destination.suffix + ".licenses")
        shutil.copytree(notices, evidence / notices.name)
        return destination, receipt["gguf_sha256"]

    def real_cli_control(self, denoiser, denoiser_hash, vae, vae_hash):
        directory = self.output / "real-cli"
        directory.mkdir()
        rgb, mask = fixture("empty")
        png(directory / "input.png", rgb, channels=3)
        png(directory / "mask.png", mask, channels=1)
        self.report["real_cli_empty_mask"] = "running"
        self.save()
        self.command("real-cli-empty-mask", [self.binary("crispembed-cli"), "--inpaint-engine", self.args.model,
                    "--inpaint-model", denoiser, "--inpaint-vae", vae,
                    "--inpaint-model-sha256", denoiser_hash, "--inpaint-vae-sha256", vae_hash,
                    "--inpaint", directory / "input.png", "--mask", directory / "mask.png",
                    "--inpaint-output", directory / "output.png", "--threads", "2"],
                    env={"CRISPEMBED_" + self.args.model.upper(): "1"}, maximum=600)
        check = r'''
import pathlib, sys
from PIL import Image
sys.path.insert(0, str(pathlib.Path.cwd() / "tools"))
from latent_inpaint_contract import fixture
with Image.open(sys.argv[1]) as actual:
    if actual.size != (512, 512) or actual.mode != "RGB" or actual.tobytes() != fixture("empty")[0]:
        raise AssertionError("real CLI empty-mask output changed known RGB pixels")
print("Real CLI decoded-input / exact-known-pixel control passed; no diffusion quality claim.")
'''
        self.command("real-cli-pixels", [self.args.reference_python, "-c", check, directory / "output.png"], maximum=60)
        self.report["real_cli_empty_mask"] = "passed"
        self.save()

    def heavy(self):
        if platform.system() != "Linux":
            raise ValueError("heavy reference lane requires the explicitly provisioned Linux CUDA worker")
        resources = resource_snapshot(self.scratch)
        self.report["resources_before_models"] = resources
        self.save()
        if resources["disk"]["free"] < 25 * 1024**3:
            raise ValueError("heavy lane requires 25GiB free for exact checkpoints, conversion and retained evidence")
        if resources.get("memory_kib", {}).get("MemAvailable", 0) < 12 * 1024**2:
            raise ValueError("heavy lane requires at least 12GiB available host RAM before original model loading")
        contract = load_contract()
        sources = {self.args.model: self.source(self.args.model, contract)}
        if "pixelhacker" not in sources:
            sources["pixelhacker"] = self.source("pixelhacker", contract)
        self.original_preflight(sources[self.args.model])
        checkpoints = {}
        for model in (self.args.model, "sdxl_vae"):
            record = model_record(model, contract)
            checkpoint = self.scratch / "checkpoints" / (model + ".bin")
            self.download(record["checkpoint"]["url"], checkpoint, record["checkpoint"]["bytes"], record["checkpoint"]["sha256"])
            verify_checkpoint(checkpoint, record)
            checkpoints[model] = checkpoint
        # Actual unaltered original execution comes before conversion/replay.
        # A failed import, CUDA operator or neutrality check cannot become a
        # native pass by swapping in a locally reimplemented reference.
        self.report["original_reference"] = "running"
        self.save()
        original = self.output / "original"
        selected = CASES if self.args.cases == "full-panel" else ("subtitle_bar",)
        self.command("capture-original", [self.args.reference_python, "tools/dump_latent_inpaint_reference.py",
                    "--model", self.args.model, "--source-dir", sources[self.args.model],
                    "--checkpoint", checkpoints[self.args.model], "--vae-checkpoint", checkpoints["sdxl_vae"],
                    "--output", original, "--device", "cuda:0", "--threads", "2", "--cases", *selected],
                    env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        original_report = json.loads((original / "report.json").read_text())
        if not original_report["complete"] or [case["id"] for case in original_report["cases"]] != list(selected):
            raise ValueError("actual original capture is incomplete")
        self.report["original_reference"] = "passed"
        self.report["original_report_sha256"] = sha256(original / "report.json")
        self.save()
        denoiser, denoiser_hash = self.convert(self.args.model, checkpoints[self.args.model], sources[self.args.model])
        vae, vae_hash = self.convert("sdxl_vae", checkpoints["sdxl_vae"], sources["pixelhacker"])
        self.report["native_replay"] = "running"
        self.save()
        native = self.output / "native"
        self.command("native-parity", [self.binary("test-latent-inpaint"), "--model", self.args.model,
                    "--denoiser", denoiser, "--vae", vae, "--model-sha256", denoiser_hash,
                    "--vae-sha256", vae_hash, "--reference-dir", original, "--out", native, "--threads", "2"],
                    env={"CRISPEMBED_" + self.args.model.upper(): "1"})
        native_report = json.loads((native / "report.json").read_text())
        if not native_report["complete"] or not native_report["synthetic_parity_pass"]:
            raise ValueError("native replay did not produce a complete passing report")
        self.report["native_replay"] = "passed"
        self.report["native_report_sha256"] = sha256(native / "report.json")
        self.report["full_synthetic_panel_pass"] = native_report["full_synthetic_panel_pass"]
        self.real_cli_control(denoiser, denoiser_hash, vae, vae_hash)

    def execute(self):
        try:
            self.provenance()
            self.build()
            if self.args.phase == "original-native":
                self.heavy()
            else:
                self.report["original_reference"] = "skipped: build-only has no model access"
                self.report["native_replay"] = "skipped: build-only has no model access"
                self.report["real_cli_empty_mask"] = "skipped: models unavailable in build-only"
            self.report["complete"] = True
            self.report["passed"] = True
            return 0
        except BaseException as error:
            self.report["errors"].append(type(error).__name__ + ": " + str(error))
            for stage in ("build", "original_reference", "native_replay", "real_cli_empty_mask"):
                if self.report[stage] == "running":
                    self.report[stage] = "failed or timed-out; inspect command log and partial evidence"
            raise
        finally:
            self.report["resources_after"] = resource_snapshot(self.scratch)
            self.save()


def png(path, raw, *, channels):
    if channels not in (1, 3) or len(raw) != 512 * 512 * channels:
        raise ValueError("PNG control dimensions differ")
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
    stride = 512 * channels
    rows = b"".join(b"\0" + raw[y * stride:(y + 1) * stride] for y in range(512))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 512, 512, 8, 2 if channels == 3 else 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("build-only", "original-native"), default="build-only")
    parser.add_argument("--model", choices=("moebius", "pixelhacker"), default="moebius")
    parser.add_argument("--cases", choices=("subtitle_bar", "full-panel"), default="subtitle_bar")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--reference-python", default="")
    parser.add_argument("--deadline-minutes", type=int, default=165)
    args = parser.parse_args(argv)
    if not 5 <= args.deadline_minutes <= 165:
        parser.error("deadline must be 5..165 minutes, leaving time for artifact upload")
    try:
        return Run(args).execute()
    except BaseException as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
