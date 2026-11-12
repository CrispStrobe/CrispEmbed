#!/usr/bin/env python3
"""Kaggle CPU-only, checkpoint-free restoration compile/control worker.

Preparation requires only copying this file; do not execute it on a shared host.
Run on Kaggle with Internet enabled and accelerator=None. Override --source only
with an independently reviewed immutable full commit; no tokens are required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

SOURCE = "7340a6366b57c64c39b746ef53f2a2cb550ded64"
GGML = "3a27b6c665b191cb230be367db5c52d81dd3e2ac"
TARGETS = ["test-lighton4b-source-prefill-chunks", "lighton4b-source-prefill-probe", "lighton4b-native-document"]
FLAGS = ["GGML_NATIVE=OFF", "CRISPEMBED_NATIVE=OFF", "GGML_CUDA=OFF", "GGML_METAL=OFF",
         "GGML_VULKAN=OFF", "GGML_AVX2=ON", "GGML_F16C=ON", "GGML_FMA=ON",
         "GGML_LLAMAFILE=ON", "GGML_OPENMP=OFF", "GGML_AVX512=OFF", "GGML_BLAS=OFF",
         "GGML_BACKEND_DL=OFF", "CRISPEMBED_BUILD_SHARED=OFF", "CMAKE_BUILD_TYPE=Release"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/planner-restoration-controls"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source):
        parser.error("--source must be a full immutable commit")
    if not Path("/kaggle/working").is_dir():
        parser.error("execute only in the authorized Kaggle worker")
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output / "source"
    receipt = {"schema": "lighton4b-planner-restoration-controls/v1", "source": args.source,
               "expected_ggml": GGML, "targets": TARGETS, "flags": FLAGS,
               "checkpoint_loaded": False, "decoded_quality_qualified": False,
               "gpu_qualified": False, "passed": False, "commands": []}
    env = {key: value for key, value in os.environ.items() if not key.startswith("CRISPEMBED_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_CONFIG_NOSYSTEM="1")

    def run(command, cwd=None, timeout=1200):
        index = len(receipt["commands"])
        log = args.output / ("command-%02d.log" % index)
        entry = {"argv": command, "log": log.name}
        receipt["commands"].append(entry)
        started = time.monotonic()
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout)
        entry.update(returncode=result.returncode, seconds=time.monotonic() - started,
                     sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        if result.returncode:
            raise RuntimeError("command %d failed; see %s" % (index, log.name))
        return log.read_text(errors="replace").strip()

    try:
        run(["git", "clone", "--no-checkout", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git", str(root)], timeout=600)
        run(["git", "checkout", "--detach", args.source], root)
        actual = run(["git", "rev-parse", "HEAD"], root)
        if actual != args.source:
            raise RuntimeError("source mismatch")
        run(["git", "submodule", "update", "--init", "--depth", "1", "ggml"], root, timeout=600)
        actual_ggml = run(["git", "rev-parse", "HEAD"], root / "ggml")
        if actual_ggml != GGML:
            raise RuntimeError("ggml mismatch")
        receipt["actual_ggml"] = actual_ggml
        run(["cmake", "-S", ".", "-B", "build"] + ["-D" + flag for flag in FLAGS], root)
        run(["python3", "scripts/check-cpu-baseline.py", "build"], root)
        run(["cmake", "--build", "build", "-j2", "--target"] + TARGETS, root, timeout=3600)
        # Verify the ABI was enabled rather than accepting a capability skip.
        compile_flags = (root / "build/CMakeFiles/crispembed-qwen35-delta.dir/flags.make").read_text()
        if "CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI=1" not in compile_flags:
            raise RuntimeError("required physical F16 ABI was not compiled")
        output = run([str(root / "build" / TARGETS[0])], root, timeout=1200)
        if not re.search(r"SOURCE_PREFILL_CHUNKS_CONTROLS PASS checks=\d+ model_payload=0 decoded=0", output):
            raise RuntimeError("missing actual full control PASS marker")
        receipt["control_output"] = output
        receipt["passed"] = True
    except Exception as error:
        receipt["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
