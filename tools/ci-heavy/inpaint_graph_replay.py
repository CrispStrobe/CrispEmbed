#!/usr/bin/env python3
"""Hosted replay of prepared original captures on native CPU/CUDA/Metal graphs.

This driver does not build, download, convert weights, import a framework, or
regenerate references. It invokes the existing full-model comparators with
their unchanged numerical budgets. --compare-eager also runs the CPU eager
implementation against the same original bundle. A mixed profile receives
exactly the same full-model acceptance checks as F32.

The model receipt is the converter's adjacent MODEL.gguf.json sidecar. The
reference manifest is reference.json for LaMa and report.json for latent
models; its caller-supplied hash pins the chosen original capture bundle.
Use --build-receipt with the graph build workflow's build-receipt.json to bind
the executable bytes to its producer revisions. Without it, numerical replay
is permitted but build_provenance_qualified remains false.

Only run this driver on an authorized hosted worker. Its unit controls mock
every native invocation and do not require model files or a GPU.
"""

import argparse
from datetime import datetime, timezone
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
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
CASE_ID = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
MAX_JSON_BYTES = 16 * 1024 * 1024
GATES = ("CRISPEMBED_LAMA", "CRISPEMBED_MOEBIUS", "CRISPEMBED_PIXELHACKER",
         "CRISPEMBED_MIGAN", "CRISPEMBED_MIGAN512")
CPU_GATES = ("CRISPEMBED_INPAINT_GEMM_LAYOUT", "CRISPEMBED_INPAINT_REFLECT_GATHER",
             "CRISPEMBED_INPAINT_CPU_FFT", "CRISPEMBED_INPAINT_CPU_GLA",
             "CRISPEMBED_MOEBIUS_RELATIVE_REASSOC", "CRISPEMBED_MIGAN_CPU_KERNELS",
             "CRISPEMBED_INPAINT_CPU_ATTENTION_TILES", "CRISPEMBED_MOEBIUS_LOCAL_REASSOC",
             "CRISPEMBED_MOEBIUS_LOCAL_PAD_GEMM")
CPU_ENGINE_GATES = {
    "lama": CPU_GATES[:3],
    "moebius": (CPU_GATES[0], CPU_GATES[4]),
    "pixelhacker": (CPU_GATES[0], CPU_GATES[3]),
}
PROFILE = re.compile(
    rb"^inpaint_graph label=(?P<label>.*?) backend=(?P<backend>\S+) "
    rb"precision=(?P<precision>\S+) nodes=(?P<nodes>\d+) "
    rb"allocation_bytes=(?P<allocation_bytes>\d+) runs=(?P<runs>\d+) "
    rb"upload_bytes=(?P<upload_bytes>\d+) download_bytes=(?P<download_bytes>\d+) "
    rb"compute_ms=(?P<compute_ms>[0-9]+(?:\.[0-9]+)?)\s*$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value, label):
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(label + " must be a lowercase SHA-256")
    return value


def regular_file(path, label, maximum=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(label + " must be a regular file, not a symlink")
    if maximum is not None and path.stat().st_size > maximum:
        raise ValueError(label + " exceeds the bounded byte count")
    return path.resolve()


def unique_pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_nonfinite(value):
    raise ValueError("nonfinite JSON number: " + value)


def read_json(path):
    path = regular_file(path, "JSON evidence", MAX_JSON_BYTES)
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs,
                       parse_constant=reject_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("JSON evidence root must be an object")
    return value


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def overlaps(first, second):
    return first == second or first in second.parents or second in first.parents


def terminate_tree(process):
    """Stop the task-owned process group, including descendants of an exited parent."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def profile_records(path):
    records = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.startswith(b"inpaint_graph "):
                continue
            match = PROFILE.fullmatch(line)
            if match is None:
                raise ValueError("malformed native graph execution profile")
            row = {}
            for name, raw in match.groupdict().items():
                row[name] = (raw.decode("utf-8", errors="strict") if name in ("label", "backend", "precision")
                             else float(raw) if name == "compute_ms" else int(raw))
            records.append(row)
    return records


def native_marker(path, prefix):
    records = []
    with path.open("rb") as stream:
        for line in stream:
            if line.startswith(prefix):
                if len(line) > MAX_JSON_BYTES:
                    raise ValueError("native verdict line exceeds the bounded byte count")
                row = json.loads(line[len(prefix):].decode("utf-8"), object_pairs_hook=unique_pairs,
                                 parse_constant=reject_nonfinite)
                if not isinstance(row, dict):
                    raise ValueError("native verdict must be a JSON object")
                records.append(row)
    if len(records) != 1:
        raise ValueError("expected exactly one native final verdict")
    return records[0]


def backend_matches(requested, actual):
    actual = actual.lower()
    if requested == "cpu":
        return actual == "cpu"
    if requested == "cuda":
        return re.fullmatch(r"cuda[0-9]*", actual) is not None
    if requested == "metal":
        return re.fullmatch(r"(?:metal|mtl)[0-9]*", actual) is not None
    return False


class Run:
    def __init__(self, args):
        self.args = args
        self.deadline = time.monotonic() + args.deadline_minutes * 60
        requested = (args.output, args.scratch)
        if any(path.exists() or path.is_symlink() for path in requested):
            raise ValueError("choose new output and scratch directories")
        self.output, self.scratch = (path.resolve() for path in requested)
        if overlaps(self.output, self.scratch):
            raise ValueError("output and scratch must be separate directory trees")
        protected = [args.reference_dir, args.binary, args.model, args.model_receipt,
                     args.vae, args.vae_receipt, args.build_receipt]
        for path in (Path(value).resolve() for value in protected if value is not None):
            if overlaps(path, self.output) or overlaps(path, self.scratch):
                raise ValueError("output/scratch overlap a protected input")
        self.output.mkdir(parents=True)
        self.scratch.mkdir(parents=True)
        (self.output / "inputs").mkdir()
        (self.output / "logs").mkdir()
        self.report = {
            "schema": "crispembed-inpaint-graph-replay-v1", "engine": args.engine,
            "requested_backend": args.backend, "requested_precision": args.precision,
            "cpu_optimized": args.cpu_optimized,
            "compare_eager": args.compare_eager, "status": "running", "complete": False,
            "passed": False, "graph_parity_pass": False, "eager_parity_pass": None,
            "build_provenance_qualified": False, "native_qualified": False,
            "representative_quality_qualified": False, "performance_qualified": False,
            "qualification_note": "Existing original/native comparator budgets are unchanged. Passing selected "
                                  "capture cases does not establish representative image quality, performance, "
                                  "or release qualification.",
            "started_utc": datetime.now(timezone.utc).isoformat(), "deadline_minutes": args.deadline_minutes,
            "platform": platform.platform(), "python": platform.python_version(),
            "scratch": str(self.scratch), "commands": [], "errors": [],
            "network_access_performed": False, "reference_capture_performed": False,
            "model_conversion_performed": False, "framework_imported": False,
        }
        self.cases = []
        self.save()

    def save(self):
        write_json(self.output / "report.json", self.report)

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("hosted replay deadline reached; partial evidence retained")
        return remaining

    def git(self, directory, *arguments):
        try:
            return subprocess.check_output(["git", "-C", str(directory), *arguments],
                                           stderr=subprocess.PIPE, timeout=min(30, self.remaining())).decode("utf-8").strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    def provenance(self):
        head = self.git(ROOT, "rev-parse", "HEAD")
        ggml_head = self.git(ROOT / "ggml", "rev-parse", "HEAD")
        # An uninitialized submodule/archive directory can resolve to its parent
        # repository. Only a ggml repository rooted here supplies a ggml HEAD.
        ggml_root = self.git(ROOT / "ggml", "rev-parse", "--show-toplevel")
        if ggml_root is None or Path(ggml_root).resolve() != (ROOT / "ggml").resolve():
            ggml_head = None
        gitlink = self.git(ROOT, "ls-tree", "HEAD", "ggml")
        status = self.git(ROOT, "status", "--porcelain=v1", "--untracked-files=all", "--",
                          "CMakeLists.txt", "cmake", "src", "include", "ggml", "tests", "tools", "models",
                          "docs/lama-source-contract.json", "docs/latent-inpaint-source-contract.json")
        expected = os.environ.get("GITHUB_SHA") or None
        if expected and (COMMIT.fullmatch(expected) is None or head is not None and expected != head):
            raise ValueError("GITHUB_SHA differs from the native checkout HEAD")
        source = {"native_commit": head, "ggml_commit": ggml_head, "ggml_gitlink": gitlink,
                  "expected_github_sha": expected, "source_tree_clean": status == "" if status is not None else None,
                  "archive_without_git": head is None, "binary_revision_assumed_from_checkout": False}
        if status is not None:
            file = self.output / "native-source-status.txt"
            file.write_text(status + ("\n" if status else ""), encoding="utf-8")
            source["source_status_file"] = file.name
            source["source_status_sha256"] = sha256(file)
        # Capture the actual comparator and driver source identities as well as
        # the revision; a local executable is not presumed to come from HEAD.
        comparator = ROOT / "tests" / ("test_lama_diff.cpp" if self.args.engine == "lama" else "test_latent_inpaint.cpp")
        source["comparator_source"] = {"path": str(comparator.relative_to(ROOT)), "sha256": sha256(comparator)}
        source["driver_sha256"] = sha256(__file__)
        self.report["source"] = source
        self.save()

    def preserve_json(self, path, name, expected_sha=None):
        regular_file(path, name, MAX_JSON_BYTES)
        observed = sha256(path)
        if expected_sha is not None and observed != digest(expected_sha, name + " hash"):
            raise ValueError(name + " SHA-256 differs from the caller's pin")
        value = read_json(path)
        destination = self.output / "inputs" / name
        shutil.copyfile(path, destination)
        return value, {"path": str(Path(path).resolve()), "sha256": observed, "bytes": path.stat().st_size,
                       "preserved_file": str(destination.relative_to(self.output))}

    def conversion(self, model, receipt_path, expected_sha, name):
        model = regular_file(model, name + " GGUF")
        digest(expected_sha, name + " GGUF hash")
        sidecar = model.with_suffix(model.suffix + ".json")
        if Path(receipt_path).resolve() != sidecar:
            raise ValueError("native comparators require the adjacent " + name + " GGUF .json receipt")
        receipt, evidence = self.preserve_json(receipt_path, name + "-conversion.json")
        if (receipt.get("gguf_sha256") != expected_sha or type(receipt.get("gguf_bytes")) is not int
                or receipt["gguf_bytes"] != model.stat().st_size):
            raise ValueError(name + " receipt does not bind the supplied GGUF digest/byte count")
        evidence.update({"gguf_path": str(model), "expected_gguf_sha256": expected_sha,
                         "gguf_bytes": model.stat().st_size,
                         "gguf_payload_validation": "delegated to the unchanged native comparator/loader"})
        return evidence

    def build_receipt(self):
        if self.args.build_receipt is None:
            self.report["build_provenance_note"] = "No build receipt supplied; binary-to-source revision is not attested."
            return
        receipt, evidence = self.preserve_json(self.args.build_receipt, "build-receipt.json")
        if (receipt.get("schema") != "crispembed-inpaint-graph-build-v1"
                or receipt.get("source_tree_clean") is not True
                or COMMIT.fullmatch(str(receipt.get("native_commit", ""))) is None
                or COMMIT.fullmatch(str(receipt.get("ggml_commit", ""))) is None):
            raise ValueError("build receipt source identity or clean-source assertion differs")
        producer = receipt.get("producer")
        if not isinstance(producer, dict) or producer.get("github_sha") != receipt["native_commit"] or not str(producer.get("github_run_id", "")).isdigit():
            raise ValueError("build receipt requires matching producer SHA and workflow run ID")
        source = self.report["source"]
        for key in ("native_commit", "ggml_commit"):
            if source[key] is not None and source[key] != receipt[key]:
                raise ValueError("build receipt " + key + " differs from the replay checkout")
        if source["expected_github_sha"] is not None and source["expected_github_sha"] != receipt["native_commit"]:
            raise ValueError("build receipt differs from GITHUB_SHA")
        if source["source_tree_clean"] is False:
            raise ValueError("build receipt cannot attest a modified replay source tree")
        rows = receipt.get("binaries")
        if not isinstance(rows, list):
            raise ValueError("build receipt executable census is missing")
        selected = [row for row in rows if isinstance(row, dict) and row.get("name") == self.binary_name]
        if len(selected) != 1 or selected[0].get("sha256") != self.report["binary"]["sha256"] or type(selected[0].get("bytes")) is not int or selected[0]["bytes"] != self.report["binary"]["bytes"]:
            raise ValueError("build receipt does not bind the exact selected executable")
        evidence["declared_native_commit"] = receipt["native_commit"]
        evidence["declared_ggml_commit"] = receipt["ggml_commit"]
        evidence["producer"] = producer
        evidence["binary"] = selected[0]
        self.report["build_receipt"] = evidence
        self.report["build_provenance_qualified"] = True

    def prepare(self):
        args = self.args
        if args.precision == "mixed" and args.backend == "cpu":
            raise ValueError("mixed precision requires CUDA or Metal; CPU executes F32")
        if args.cpu_optimized and (args.backend != "cpu" or args.precision != "f32"):
            raise ValueError("CPU optimizations require the explicit CPU/F32 profile")
        if args.engine == "lama" and (args.vae is not None or args.vae_sha256 is not None or args.vae_receipt is not None):
            raise ValueError("LaMa has no VAE")
        if args.engine != "lama" and any(value is None for value in (args.vae, args.vae_sha256, args.vae_receipt)):
            raise ValueError("latent replay requires --vae, --vae-sha256 and its adjacent receipt")
        self.binary = regular_file(args.binary, "native comparator")
        self.binary_name = "test-lama-diff" if args.engine == "lama" else "test-latent-inpaint"
        if self.binary.name not in (self.binary_name, self.binary_name + ".exe"):
            raise ValueError("selected engine requires the actual " + self.binary_name + " executable")
        if os.name == "posix" and not os.access(self.binary, os.X_OK):
            raise ValueError("native comparator is not executable")
        self.report["binary"] = {"path": str(self.binary), "name": self.binary_name,
                                 "sha256": sha256(self.binary), "bytes": self.binary.stat().st_size}
        self.report["model"] = self.conversion(args.model, args.model_receipt, args.model_sha256, "model")
        if args.engine != "lama":
            self.report["vae"] = self.conversion(args.vae, args.vae_receipt, args.vae_sha256, "vae")
        reference = args.reference_dir
        if reference.is_symlink() or not reference.is_dir():
            raise ValueError("reference directory must be a real directory")
        self.reference = reference.resolve()
        filename = "reference.json" if args.engine == "lama" else "report.json"
        manifest, evidence = self.preserve_json(self.reference / filename, "original-" + filename,
                                               args.reference_manifest_sha256)
        evidence["validation"] = "source/checkpoint/fixture/stage/raw-file validation delegated to existing comparator"
        self.report["original_reference"] = evidence
        rows = manifest.get("cases")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
            raise ValueError("original manifest must select a nonempty bounded case list")
        for row in rows:
            name = row.get("id") if isinstance(row, dict) else None
            if not isinstance(name, str) or CASE_ID.fullmatch(name) is None or name in [item["id"] for item in self.cases]:
                raise ValueError("original manifest case IDs must be unique safe names")
            case = {"id": name}
            if args.engine == "lama":
                case["reference_sha256"] = digest(row.get("reference_sha256"), "LaMa case reference hash")
                directory = self.reference / name
                if directory.is_symlink() or not directory.is_dir():
                    raise ValueError("LaMa case must be a real child directory")
                regular_file(directory / "reference.gguf", "LaMa case reference")
            self.cases.append(case)
        self.report["selected_cases"] = self.cases
        self.provenance()
        self.build_receipt()
        self.save()

    def environment(self, arm, temporary):
        env = dict(os.environ)
        for key in tuple(env):
            if key.startswith("CRISPEMBED_INPAINT_"):
                del env[key]
        controlled = {name: "0" for name in GATES + CPU_GATES}
        controlled["CRISPEMBED_" + self.args.engine.upper()] = "1"
        if arm == "graph" and self.args.cpu_optimized:
            controlled.update({name: "1" for name in CPU_ENGINE_GATES[self.args.engine]})
        controlled.update({"CRISPEMBED_INPAINT_GRAPH": "1" if arm == "graph" else "0",
                           "CRISPEMBED_INPAINT_BACKEND": self.args.backend if arm == "graph" else "cpu",
                           "CRISPEMBED_INPAINT_FAST": "1" if arm == "graph" and self.args.precision == "mixed" else "0",
                           "CRISPEMBED_INPAINT_PROFILE": "1",
                           "OMP_NUM_THREADS": str(self.args.threads), "OPENBLAS_NUM_THREADS": str(self.args.threads),
                           "MKL_NUM_THREADS": str(self.args.threads), "TMPDIR": str(temporary),
                           "TMP": str(temporary), "TEMP": str(temporary)})
        env.update(controlled)
        return env, controlled

    def argv(self, case, output):
        args = self.args
        if args.engine == "lama":
            return [str(self.binary), "--model", str(args.model.resolve()), "--model-sha256", args.model_sha256,
                    "--reference", str(self.reference / case["id"] / "reference.gguf"),
                    "--reference-sha256", case["reference_sha256"], "--output", str(output),
                    "--threads", str(args.threads)]
        return [str(self.binary), "--model", args.engine, "--denoiser", str(args.model.resolve()),
                "--vae", str(args.vae.resolve()), "--model-sha256", args.model_sha256,
                "--vae-sha256", args.vae_sha256, "--reference-dir", str(self.reference),
                "--out", str(output), "--threads", str(args.threads)]

    def verify_input_pins(self):
        # Recheck the small routing/receipt files around each process. The
        # comparators verify every large model/reference payload themselves.
        # An edited bundle manifest must not silently select a different panel
        # between the eager and graph arms.
        try:
            binary = regular_file(self.binary, "native comparator")
            intact = (binary.stat().st_size == self.report["binary"]["bytes"]
                      and sha256(binary) == self.report["binary"]["sha256"])
        except (OSError, ValueError):
            intact = False
        if not intact:
            self.report["build_provenance_qualified"] = False
            raise ValueError("native executable changed after its exact build identity was checked")
        for name in ("original_reference", "model", "vae", "build_receipt"):
            if name not in self.report:
                continue
            evidence = self.report[name]
            path = regular_file(evidence["path"], name + " pinned evidence", MAX_JSON_BYTES)
            if sha256(path) != evidence["sha256"]:
                if name == "build_receipt":
                    self.report["build_provenance_qualified"] = False
                raise ValueError(name + " evidence changed after its supplied pin was checked")

    def verdict(self, record, output, stdout, stderr):
        self.verify_input_pins()
        if self.args.engine == "lama":
            final = native_marker(stdout, b"LAMA_FINAL ")
            if final.get("case") != record["case"]:
                raise ValueError("native LaMa case identity differs from the selected manifest row")
            comparator_passed = final.get("passed") is True
        else:
            final = native_marker(stdout, b"LATENT_INPAINT_PARITY ")
            path = output / "report.json"
            native = read_json(path)
            if (native.get("schema") != "crispembed-latent-inpaint-native-v1" or native.get("model") != self.args.engine
                    or Path(str(final.get("report", ""))).resolve() != path.resolve()
                    or native.get("original_report_sha256") != self.args.reference_manifest_sha256):
                raise ValueError("native latent report identity differs")
            record["native_report"] = {"file": str(path.relative_to(self.output)), "sha256": sha256(path),
                                        "complete": native.get("complete"),
                                        "synthetic_parity_pass": native.get("synthetic_parity_pass"),
                                        "full_synthetic_panel_pass": native.get("full_synthetic_panel_pass")}
            comparator_passed = (final.get("passed") is True and native.get("complete") is True
                                 and native.get("synthetic_parity_pass") is True and native.get("errors") == [])
        record["native_final"] = final
        profiles = profile_records(stderr)
        record["graph_profiles"] = profiles
        if record["arm"] == "graph":
            wanted_precision = "mixed-f16" if self.args.precision == "mixed" else "f32"
            if not profiles or not any(row["runs"] > 0 for row in profiles):
                raise ValueError("native comparator produced no executed graph backend/precision evidence")
            if not all(backend_matches(self.args.backend, row["backend"]) and row["precision"] == wanted_precision for row in profiles):
                raise ValueError("actual native graph backend/precision differs from the explicit request")
        elif any(row["runs"] > 0 for row in profiles):
            raise ValueError("eager baseline unexpectedly executed an inpainting graph")
        record["backend_precision_verified"] = True
        record["native_comparator_passed"] = comparator_passed
        record["passed"] = record["returncode"] == 0 and comparator_passed
        if not record["passed"]:
            record["errors"].append("native comparator rejected original/reference parity; budgets were unchanged")

    def command(self, arm, case):
        name = case["id"] if self.args.engine == "lama" else "bundle"
        label = arm + "-" + name
        output = self.output / "native" / arm / name
        stdout, stderr = (self.output / "logs" / (label + suffix) for suffix in (".stdout.log", ".stderr.log"))
        temporary = self.scratch / label
        temporary.mkdir()
        env, controlled = self.environment(arm, temporary)
        argv = self.argv(case, output)
        record = {"arm": arm, "case": name, "argv": argv, "environment": controlled, "state": "running",
                  "returncode": None, "passed": False, "backend_precision_verified": False,
                  "native_output": str(output.relative_to(self.output)), "errors": []}
        self.report["commands"].append(record)
        self.save()
        start = time.monotonic()
        process = None
        try:
            with stdout.open("xb") as out, stderr.open("xb") as err:
                remaining = self.remaining()
                self.verify_input_pins()
                print("Running " + label, flush=True)
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=out, stderr=err,
                                           start_new_session=os.name == "posix")
                record["pid"] = process.pid
                self.save()
                try:
                    record["returncode"] = process.wait(timeout=remaining)
                except BaseException:
                    terminate_tree(process)
                    record["returncode"] = process.returncode
                    raise
            record["state"] = "completed"
            try:
                self.verdict(record, output, stdout, stderr)
            except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as error:
                record["errors"].append(str(error))
            return record
        except (TimeoutError, subprocess.TimeoutExpired) as error:
            record["state"] = "timed-out"
            record["errors"].append(str(error))
            raise
        except BaseException as error:
            record["state"] = "failed-to-start" if process is None else "interrupted"
            record["errors"].append(str(error))
            raise
        finally:
            record["seconds"] = time.monotonic() - start
            for key, file in (("stdout", stdout), ("stderr", stderr)):
                if file.is_file():
                    record[key] = {"file": str(file.relative_to(self.output)), "sha256": sha256(file),
                                   "bytes": file.stat().st_size}
            self.save()

    def execute(self):
        result = 1
        try:
            self.prepare()
            arms = ("eager", "graph") if self.args.compare_eager else ("graph",)
            cases = self.cases if self.args.engine == "lama" else [{"id": "bundle"}]
            self.report["planned_commands"] = len(arms) * len(cases)
            self.save()
            for arm in arms:
                for case in cases:
                    self.command(arm, case)
            self.report["complete"] = True
            for arm in arms:
                rows = [row for row in self.report["commands"] if row["arm"] == arm]
                self.report[arm + "_parity_pass"] = bool(rows) and all(row["passed"] for row in rows)
            self.report["passed"] = self.report["graph_parity_pass"] and (
                not self.args.compare_eager or self.report["eager_parity_pass"])
            self.report["status"] = "passed" if self.report["passed"] else "failed"
            result = 0 if self.report["passed"] else 1
        except (TimeoutError, subprocess.TimeoutExpired) as error:
            self.report["status"] = "timed-out"
            self.report["errors"].append(str(error))
            result = 124
        except (KeyboardInterrupt, InterruptedError) as error:
            self.report["status"] = "interrupted"
            self.report["errors"].append(str(error) or "interrupted")
            result = 130
        except Exception as error:
            self.report["status"] = "failed"
            self.report["errors"].append(str(error))
            result = 1
        finally:
            self.report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            self.save()
        print(json.dumps({key: self.report[key] for key in ("status", "complete", "graph_parity_pass",
                         "eager_parity_pass", "build_provenance_qualified", "native_qualified")}), flush=True)
        return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=("lama", "moebius", "pixelhacker"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "metal"), required=True)
    parser.add_argument("--precision", choices=("f32", "mixed"), default="f32")
    parser.add_argument("--cpu-optimized", action="store_true",
                        help="explicitly test the gated model-specific CPU optimizations against the same reference")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-receipt", type=Path, help="defaults to MODEL.gguf.json")
    parser.add_argument("--vae", type=Path)
    parser.add_argument("--vae-sha256")
    parser.add_argument("--vae-receipt", type=Path, help="defaults to VAE.gguf.json")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest-sha256", required=True,
                        help="SHA-256 of original reference.json (LaMa) or report.json (latent)")
    parser.add_argument("--compare-eager", action="store_true", help="also run eager CPU/F32 against the same capture")
    parser.add_argument("--threads", type=int, default=2, help="1..8 CPU threads, including the LaMa comparator")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--deadline-minutes", type=int, default=120)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 8 or not 1 <= args.deadline_minutes <= 1440:
        parser.error("threads must be in[1,8] and deadline-minutes in[1,1440]")
    if args.cpu_optimized and (args.backend != "cpu" or args.precision != "f32"):
        parser.error("--cpu-optimized requires --backend cpu --precision f32")
    if args.model_receipt is None:
        args.model_receipt = args.model.with_suffix(args.model.suffix + ".json")
    if args.vae is not None and args.vae_receipt is None:
        args.vae_receipt = args.vae.with_suffix(args.vae.suffix + ".json")
    return args


def main(argv=None):
    args = arguments(argv)
    try:
        run = Run(args)
    except (OSError, ValueError) as error:
        print("Cannot create replay evidence: " + str(error), file=sys.stderr)
        return 2
    def interrupted(signum, _frame):
        raise InterruptedError("hosted replay received signal " + str(signum))
    previous = {kind: signal.signal(kind, interrupted) for kind in (signal.SIGTERM, signal.SIGINT)}
    try:
        return run.execute()
    finally:
        for kind, handler in previous.items():
            signal.signal(kind, handler)


if __name__ == "__main__":
    sys.exit(main())
