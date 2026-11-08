#!/usr/bin/env python3
"""OXR-1.0 CPU A/B: decode activation quantization vs the German one-token divergence.

    python tools/kaggle/oxr-cpu-act-ab/run.py BUILD_DIR

Downloads the SHA-pinned published GGUFs (F16, oxr-1.0-q8_0, oxr-1.0-q4_k-head) and, for
every committed case in tests/results/oxr/hosted-parity.json (German first), runs the
native CPU runner for F16 and for each quantized recipe with CRISPEMBED_OXR_DECODE_F32_ACT
off and on. Token IDs are compared with the native F16 arm and the original model's IDs;
prefill stage dumps give cosines vs F16. Writes /kaggle/working/oxr-cpu-act-ab/report.json
after every step; every step has a timeout and output goes to files.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_VERSION = "oxr-kaggle-cpu-act-ab-v1"
HUB = "https://huggingface.co/cstr/OXR-1.0-GGUF/resolve/"
MODELS = {  # same URLs and SHA-256 as examples/cli/model_hashes.h
    "f16": (HUB + "f1e262e276349aa2647a85da6fc6326dd6aaf271/oxr-1.0-f16.gguf",
            "eac8754f4886ba4197416083d49089a3d86c8644f0e5650f44fcf69059d81a57"),
    "q8_0": (HUB + "1fb6277433f0be816b2be1c165e23e0ac2ce3491/oxr-1.0-q8_0.gguf",
             "f0a67a94b9f5416ff35e5c31b7873cea1a15b95743e09a2d032005ead44eca83"),
    "q4_k-head": (HUB + "1fb6277433f0be816b2be1c165e23e0ac2ce3491/oxr-1.0-q4_k-head.gguf",
                  "037a31df623facd71ca96961e48d65d9010bd1b4b16bcd867254d68d57637709"),
}
# arm: (model, gate on)
ARMS = {"f16": ("f16", False), "q8_0": ("q8_0", False), "q8_0+f32act": ("q8_0", True),
        "q4_k-head": ("q4_k-head", False), "q4_k-head+f32act": ("q4_k-head", True)}
ORDER = ["german", "formula", "japanese", "table", "receipt"]
STAGES = "vision.merge,visual.embeddings,llm.norm,llm.logits"
SCRATCH = Path("/kaggle/temp/oxr")
OUT = Path("/kaggle/working/oxr-cpu-act-ab")
ENV = dict(os.environ, CRISPEMBED_OXR="1", CRISPEMBED_ACCEPT_OXR_LICENSE="1", CRISPEMBED_OXR_FORCE_CPU="1")
ENV.pop("CRISPEMBED_OXR_DECODE_F32_ACT", None)
THREADS = str(os.cpu_count() or 4)
REPORT = {"script_version": SCRIPT_VERSION, "commit": os.environ.get("OXR_COMMIT"),
          "cpus": os.environ.get("OXR_CPUS"), "threads": THREADS, "models": {}, "cases": {}}


def save(phase):
    REPORT["phase"] = phase
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False) + "\n")
    print(f"== {phase}", flush=True)


def run(command, timeout=3600, env=None):
    """Run with stdout/stderr in files; returns (returncode, stdout, stderr, seconds)."""
    print("+", *command, flush=True)
    out_p, err_p = SCRATCH / "step.out", SCRATCH / "step.err"
    started = time.monotonic()
    with open(out_p, "w") as out_f, open(err_p, "w") as err_f:
        try:
            rc = subprocess.run([str(c) for c in command], stdout=out_f, stderr=err_f, env=env or ENV,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return rc, out_p.read_text(errors="replace"), err_p.read_text(errors="replace"), time.monotonic() - started


def sha256(path):
    rc, out, _, _ = run(["sha256sum", path], timeout=600)
    return out.split()[0] if rc == 0 and out else None


def first_diff(a, b):
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return min(len(a), len(b)) if d is None and len(a) != len(b) else d


def main(argv):
    build = Path(argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    runner = next(build.rglob("test-oxr-run"))
    dumper = next(build.rglob("test-oxr-stage-dump"))
    parity = json.loads(Path("tests/results/oxr/hosted-parity.json").read_text())
    save("setup")

    paths = {}
    for name, (url, digest) in MODELS.items():
        path = SCRATCH / url.rsplit("/", 1)[1]
        rc, _, err, secs = run(["curl", "-sfL", "-o", path, url], timeout=3600)
        ok = rc == 0 and sha256(path) == digest
        REPORT["models"][name] = {"source": url, "sha256": digest, "sha256_ok": ok, "seconds": round(secs, 1),
                                  "error": err[-500:]}
        if ok:
            paths[name] = path
        save(f"download {name}")

    from PIL import Image
    import numpy as np
    cases = sorted(parity["cases"], key=lambda c: ORDER.index(c["id"]))
    for case in cases:
        cid, lane = case["id"], case["lane"]
        ref = json.loads(Path(f"tests/results/oxr/{lane}/{cid}/python.json").read_text())
        image = SCRATCH / f"{cid}.png"
        source = Path("tests/regression/images") / {
            "formula": "formula_quadratic.png", "table": "cc0/simple_table.jpg", "japanese": "japanese_print.png",
            "receipt": "cc0/receipt_example.png", "german": "cc0/german_official_print.jpg"}[cid]
        Image.open(source).convert("RGB").save(image)  # decode JPEGs once, as the reference did
        entry = REPORT["cases"].setdefault(cid, {"reference_tokens": len(case["reference_ids"]), "arms": {}})
        f16_ids = None
        for arm_name, (model, gate) in ARMS.items():
            if model not in paths:
                continue
            env = dict(ENV, CRISPEMBED_OXR_DECODE_F32_ACT="1") if gate else ENV
            rc, out, err, secs = run([runner, "--model", paths[model], "--image", image, "--prompt", ref["prompt"],
                                      "--max-tokens", ref["max_tokens"], "--threads", THREADS, "--repeat", "1",
                                      "--repetition-penalty", ref["repetition_penalty"], "--hf-generation", "1",
                                      "--min-pixels", ref["min_pixels"], "--max-pixels", ref["max_pixels"]],
                                     timeout=7200, env=env)
            arm = {"model": model, "decode_f32_act": gate, "returncode": rc, "seconds": round(secs, 2)}
            try:
                native = json.loads(out)
                ids = native.get("token_ids", [])
                if arm_name == "f16":
                    f16_ids = ids
                arm.update(ids_exact_reference=ids == case["reference_ids"],
                           first_divergence_reference=first_diff(ids, case["reference_ids"]),
                           ids_exact_f16=None if f16_ids is None else ids == f16_ids,
                           first_divergence_f16=None if f16_ids is None else first_diff(ids, f16_ids),
                           text_exact_reference=native.get("text") == ref["text"], tokens=len(ids),
                           eos=native.get("eos"), inference_ms=native.get("inference_ms"),
                           device=native.get("device"), token_ids=ids)
                d = arm["first_divergence_f16"]
                if f16_ids is not None and d is not None:
                    arm["at_divergence"] = {"f16_id": f16_ids[d] if d < len(f16_ids) else None,
                                            "this_id": ids[d] if d < len(ids) else None}
            except Exception as exc:  # retained verbatim: a failed arm is evidence too
                arm.update(error=f"{type(exc).__name__}: {exc}", stderr_tail=err[-1500:])
            dump = SCRATCH / f"dump-{cid}-{arm_name}"
            rc2, _, err2, _ = run([dumper, "--dump-dir", dump, "--only", STAGES, "--model", paths[model], "--image",
                                   image, "--prompt", ref["prompt"], "--threads", THREADS, "--repeat", "1",
                                   "--hf-generation", "1", "--min-pixels", ref["min_pixels"],
                                   "--max-pixels", ref["max_pixels"]], timeout=3600, env=env)
            arm["stage_dump_rc"] = rc2
            if rc2 != 0:
                arm["stage_dump_stderr"] = err2[-800:]
            base = SCRATCH / f"dump-{cid}-f16"
            if rc2 == 0 and arm_name != "f16" and base.exists():
                cos = {}
                for st in STAGES.split(","):
                    a_, b_ = dump / f"{st}.f32", base / f"{st}.f32"
                    if a_.exists() and b_.exists():
                        x = np.fromfile(a_, np.float32).astype(np.float64)
                        y = np.fromfile(b_, np.float32).astype(np.float64)
                        if x.shape == y.shape:
                            cos[st] = {"cosine": float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-300)),
                                       "max_abs": float(np.abs(x - y).max())}
                arm["prefill_stages_vs_f16"] = cos
            entry["arms"][arm_name] = arm
            save(f"{cid} {arm_name}")
    save("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
