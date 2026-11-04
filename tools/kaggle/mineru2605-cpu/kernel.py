#!/usr/bin/env python3
"""Genuine CPU-provider adapter for the unchanged frozen MinerU qualification job.

The private submitter may prefix MINERU_CASE_ID with one protocol case. No
credentials, operator identity, fake GitHub environment or GPU allocation is used.
All model/array/build work stays on this hosted worker. The numerical source,
precision recipes, cap, threads, repeats, original-first order and gates are fixed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

SOURCE = "1bb0ddb1d9ac12ded51eb5bd0edd89be0a8259e6"
CASE_ID = globals().get("MINERU_CASE_ID", "formula")
CASES = ("receipt", "formula", "japanese", "arabic", "table", "german", "layout-page", "layout-rot90")
UV_URL = "https://files.pythonhosted.org/packages/8f/35/cb47d2d07a383c07b0e5043c6fe5555f0fd79683c6d7f9760222987c8be9/uv-0.8.17-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
UV_SHA = "b6d30d02fb65193309fc12a20f9e1a9fab67f469d3e487a254ca1145fd06788f"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def validate_provider(case_id, env, *, working, inputs, machine, system, cpus, nvidia):
    require(case_id in CASES, "exactly one frozen case required")
    require(working and inputs and env.get("KAGGLE_KERNEL_RUN_TYPE") in ("Batch", "Interactive"),
            "genuine Kaggle worker required")
    require(env.get("GITHUB_ACTIONS") is None and env.get("GITHUB_SHA") is None, "GitHub impersonation forbidden")
    require(machine == "x86_64" and system.startswith("linux") and type(cpus) is int and cpus >= 4,
            "declared Linux x86 CPU profile required")
    require(not nvidia, "this CPU task must not occupy a GPU worker")


def main():
    validate_provider(CASE_ID, os.environ, working=Path("/kaggle/working").is_dir(),
                      inputs=Path("/kaggle/input").is_dir(), machine=platform.machine(),
                      system=sys.platform, cpus=os.cpu_count(), nvidia=Path("/dev/nvidia0").exists())
    out = Path("/kaggle/working") / ("mineru2605-" + CASE_ID)
    out.mkdir(exist_ok=False)
    scratch = Path(tempfile.mkdtemp(prefix="crisp-mineru2605-" + CASE_ID + "-"))
    report = {"schema": "crispembed.mineru2605.kaggle-cpu.v1", "algorithm_source": SOURCE,
              "case": CASE_ID, "provider": "kaggle-cpu", "completed": False, "commands": [],
              "originals_regenerated_to_bypass_a_guard": False, "publication_qualified": False,
              "external_exact_version_get_status_every_output_page_required": True}
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CRISPEMBED_", "PIP_"))
           and k not in ("GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "HF_TOKEN_2", "PYTHONPATH", "PYTHONHOME")}
    env.update(CRISPEMBED_BUILD_MODE="portable", HEAVY_OUT=str(out / "panel"),
               HEAVY_SCRATCH=str(scratch / "model"), HF_HOME=str(scratch / "hf"),
               PIP_NO_CACHE_DIR="1", PIP_CONFIG_FILE="/dev/null", PIP_USER="0",
               PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    (out / "panel").mkdir(); (scratch / "model").mkdir()
    repo = scratch / "source"
    accounting = None

    def save():
        temporary = out / "provider.partial"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(out / "provider.json")

    def command(name, args, *, cwd=None, timeout=600, allowed=(0,)):
        log = out / (name + ".log")
        if accounting is None:
            began = time.monotonic()
            with log.open("wb") as stream:
                result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=timeout)
            status = {"exit": result.returncode, "timed_out": False, "seconds": time.monotonic() - began}
        else:
            status = accounting(args, log, cwd=cwd, env=env, timeout=timeout)
        report["commands"].append({"name": name, **status, "log": log.name, "sha256": sha(log)})
        save()
        require(not status["timed_out"] and status["exit"] in allowed, "provider command failed: " + name)
        return status

    save()
    try:
        command("clone", ["git", "clone", "--filter=blob:none", "--no-checkout", "-q",
                          "https://github.com/CrispStrobe/CrispEmbed.git", repo])
        command("fetch", ["git", "-C", repo, "fetch", "--no-tags", "origin", SOURCE])
        command("checkout", ["git", "-C", repo, "checkout", "--detach", SOURCE])
        command("submodule", ["git", "-C", repo, "submodule", "update", "--init", "--depth", "1", "ggml"])
        require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == SOURCE,
                "algorithm source differs")
        spec = importlib.util.spec_from_file_location("frozen_process", repo / "tools/ci_process.py")
        process = importlib.util.module_from_spec(spec); spec.loader.exec_module(process)
        accounting = process.run
        # A task-owned 3.12 environment avoids changing the worker's system Python.
        lock = scratch / "uv.lock"
        lock.write_text("uv @ " + UV_URL + " --hash=sha256:" + UV_SHA + "\n")
        tools = scratch / "tools"
        command("uv", [sys.executable, "-I", "-m", "pip", "--isolated", "install", "--no-deps",
                       "--require-hashes", "--only-binary=:all:", "--no-index", "--target", tools, "-r", lock])
        env.update(UV_CACHE_DIR=str(scratch / "uv-cache"), UV_PYTHON_INSTALL_DIR=str(scratch / "python"))
        command("venv", [tools / "bin/uv", "venv", "--python", "3.12", "--seed", scratch / "venv"])
        python = scratch / "venv/bin/python"
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
        command("python-version", [python, "-c", "import sys; assert sys.version_info[:2] == (3,12); print(sys.version)"])
        # Use the exact existing resource planner, separately checking /tmp and
        # the output mount. This is not a reduced-memory/cap/precision contract.
        args = [CASE_ID, "f32,f16,q8_0,q4_k", "2048", "3", "1"]
        report["script_arguments"] = args
        status = command("qualification", ["bash", repo / "tools/ci-heavy/mineru2605_qualify.sh", *args],
                         cwd=repo, timeout=300 * 60, allowed=(0, 1))
        summary = json.loads((out / "panel/summary.json").read_text())
        require(summary["cases"] == [CASE_ID] and len(summary["rows"]) == 4
                and all(row.get("final_output_complete") for row in summary["rows"]),
                "bounded qualification incomplete")
        report.update(completed=True, numerical_pass=summary["all_selected_precision_pass"],
                      qualification_pass=status["exit"] == 0, python_version_log="python-version.log",
                      all_eight_cases_qualified=False, model_upload=False)
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        save()
        # Only this worker's task-owned transient source/model/environment tree.
        shutil.rmtree(scratch)
    return 0 if report["qualification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
