#!/usr/bin/env python3
"""MinerU2.5-Pro-2604 GPU lane: native CUDA output of every precision.

usage (inside the kernel): run.py BUILD_DIR

For each protocol case (docs/mineru25-pro-publication-v1.json): freeze the
pinned original Python reference on this host's CPU, then run
test-mineru25-run on the CUDA backend for every converted/quantized variant
(the same recipes as tools/ci-heavy/mineru25_qualify.sh) and compare token
IDs, text and EOS against that reference. Writes
/kaggle/working/mineru25-gpu/report.json after every step so a timeout still
leaves evidence. Measured and reported only: CUDA output is not expected to
be bit-identical to CPU, and the publication gate is the hosted CPU receipt.

Exit codes: 0 every arm ran on CUDA; 1 some arm did not run or not on CUDA;
2 setup failure.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_VERSION = "mineru25-kaggle-gpu-v1"
MODEL = "opendatalab/MinerU2.5-Pro-2604-1.2B"
REVISION = "d3f5e08d073c21466bbabe21c71bb1e9c2e595da"
BLUEPRINT_REV = "c9797cde8f54cc48306196fc6aebe73ea31dd4a3"
VARIANTS = {"f32": None, "f16": None, "q8_0": ["q8_0"], "q4_k": ["q4_k"],
            "q4_k-dq8": ["q4_k", "--tensor-type", "l.*=q8_0"], "q4_k-df16": ["q4_k", "--decoder-f16"]}
MAX_TOKENS = 2048
THREADS = 4
SCRATCH = Path("/kaggle/temp/mineru25")
OUT = Path("/kaggle/working/mineru25-gpu")
REPORT = {"script_version": SCRIPT_VERSION, "commit": os.environ.get("MINERU25_COMMIT"),
          "gpu": os.environ.get("MINERU25_GPU"), "model": MODEL, "revision": REVISION,
          "max_tokens": MAX_TOKENS, "variants": {}, "cases": {}}


def save(phase):
    REPORT["phase"] = phase
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False) + "\n")
    print(f"== {phase}", flush=True)


def sh(*command, **kwargs):
    print("+", *command, flush=True)
    return subprocess.run([str(c) for c in command], **kwargs)


def run_on_gpu(command, env):
    """Run command; return (CompletedProcess-like, peak GPU MiB seen for its PID)."""
    print("+", *command, flush=True)
    proc = subprocess.Popen([str(c) for c in command], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    peak = 0
    while proc.poll() is None:
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for line in apps.splitlines():
            pid, _, mib = line.partition(",")
            if pid.strip() == str(proc.pid) and mib.strip().isdigit():
                peak = max(peak, int(mib))
        time.sleep(0.5)
    out, err = proc.communicate()
    return subprocess.CompletedProcess(command, proc.returncode, out, err), peak


def main(argv):
    build = Path(argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path("docs/mineru25-pro-publication-v1.json").read_text())
    run = next(build.rglob("test-mineru25-run"))
    quant = next(build.rglob("crispembed-quantize"))
    save("setup")

    blueprint = SCRATCH / "blueprint"
    if (sh("git", "init", "-q", blueprint).returncode or
            sh("git", "-C", blueprint, "fetch", "-q", "--depth", "1",
               "https://github.com/opendatalab/mineru-vl-utils.git", BLUEPRINT_REV).returncode or
            sh("git", "-C", blueprint, "checkout", "-q", "--detach", "FETCH_HEAD").returncode):
        return 2
    from huggingface_hub import snapshot_download
    checkpoint = Path(snapshot_download(MODEL, revision=REVISION, local_dir=str(SCRATCH / "checkpoint")))

    gguf = {}
    for name, qargs in VARIANTS.items():
        path = SCRATCH / f"mineru25-{name}.gguf"
        if name in ("f32", "f16"):
            done = sh(sys.executable, "models/convert-mineru25-pro-to-gguf.py", "--model", checkpoint,
                      "--dtype", name, "--output", path, capture_output=True, text=True)
        else:
            done = sh(quant, gguf["f32"], path, *qargs, capture_output=True, text=True)
        REPORT["variants"][name] = {"returncode": done.returncode, "recipe": qargs or ["convert", name],
                                    "bytes": path.stat().st_size if path.exists() else None,
                                    "log_tail": (done.stdout + done.stderr)[-1500:]}
        if done.returncode == 0:
            gguf[name] = path
        save(f"variant {name}")
    if "f32" not in gguf:
        return 2

    all_cuda = True
    for case, (image, task) in cfg["case_inputs"].items():
        image_path = Path("tests/regression/images") / image
        ref_path = OUT / f"{case}-python.json"
        t0 = time.time()
        done = sh(sys.executable, "tools/mineru25_validate.py", "reference", "--model", checkpoint,
                  "--blueprint", blueprint, "--image", image_path, "--task", task, "--dtype", "f32",
                  "--device", "cpu", "--threads", THREADS, "--repeat", 2, "--max-tokens", MAX_TOKENS,
                  "--output", ref_path, capture_output=True, text=True)
        entry = {"image": image, "task": task, "reference_returncode": done.returncode,
                 "reference_s": round(time.time() - t0, 1), "arms": {}}
        REPORT["cases"][case] = entry
        if done.returncode or not ref_path.exists():
            entry["reference_tail"] = (done.stdout + done.stderr)[-2000:]
            save(f"{case} reference failed")
            all_cuda = False
            continue
        ref = json.loads(ref_path.read_text())
        entry.update(reference_tokens=len(ref["token_ids"]), reference_eos=ref["eos"],
                     reference_capped=ref["capped"], reference_text=ref["text"])
        save(f"{case} reference")
        env = {k: v for k, v in os.environ.items() if k != "QWEN2VL_OCR_FORCE_CPU"}
        env["CRISPEMBED_MINERU25_PRO"] = "1"
        for name, path in gguf.items():
            done, gpu_mib = run_on_gpu([run, "--model", path, "--image", image_path, "--task", task,
                                        "--max-tokens", MAX_TOKENS, "--threads", THREADS, "--repeat", 2], env)
            arm = {"returncode": done.returncode, "gpu_peak_mib": gpu_mib}
            try:
                native = json.loads(done.stdout)
                ids = native.get("token_ids", [])
                arm.update(device=native.get("device"), tokens=len(ids), ids_exact=ids == ref["token_ids"],
                           text_exact=native.get("text") == ref["text"], eos=native.get("eos"),
                           runs_ms=native.get("timing", {}).get("runs_ms"),
                           first_divergence=next((i for i, (a, b) in enumerate(zip(ids, ref["token_ids"]))
                                                  if a != b), None))
                if arm["ids_exact"] is False and arm["text_exact"] is False:
                    arm["text"] = native.get("text")
            except (json.JSONDecodeError, AttributeError):
                arm["stderr_tail"] = done.stderr[-1500:]
            # The run tool reports device "auto"; GPU residency of the PID is the evidence.
            all_cuda = all_cuda and done.returncode == 0 and gpu_mib > 0
            entry["arms"][name] = arm
            save(f"{case} {name}")
    REPORT["all_arms_on_cuda"] = all_cuda
    save("complete")
    return 0 if all_cuda else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as error:
        REPORT["error"] = f"{type(error).__name__}: {error}"
        OUT.mkdir(parents=True, exist_ok=True)
        save("failed")
        raise
