#!/usr/bin/env python
"""Hosted-only: pinned Python blueprint -> GGUF -> native per-layer diff."""

import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import snapshot_download


def main():
    scratch = Path(os.environ["HEAVY_SCRATCH"])
    out = Path(os.environ["HEAVY_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    model = scratch / "model"
    snapshot_download(
        "mrm8488/embeddinggemma-2-ft-typed-decision",
        revision="fcc165b7d7d97827e241e09752e83fd11165a74b",
        local_dir=model,
        ignore_patterns=["*.png"],
    )
    env = dict(
        os.environ,
        USE_TF="0",
        PYTHONNOUSERSITE="1",
        CRISPEMBED_GEMMA_DECISION="1",
        OMP_NUM_THREADS="2",
    )

    def run(cmd, name):
        print("RUN", cmd, flush=True)
        with (out / (name + ".log")).open("w") as log:
            p = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print((out / (name + ".log")).read_text()[-7000:], flush=True)
        if p.returncode:
            raise RuntimeError(f"{name} exited {p.returncode}")

    run(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            ".",
            "-B",
            str(scratch / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCRISPEMBED_NATIVE=OFF",
        ],
        "configure",
    )
    run(
        [
            "cmake",
            "--build",
            str(scratch / "build"),
            "--target",
            "test-gemma-decision-diff",
            "test-gemma-decision-schema",
            "crispembed-shared",
            "crispembed-cli",
            "crispembed-server",
            "crispembed-quantize",
            "-j2",
        ],
        "build",
    )
    gguf = scratch / "gemma-decision-f32.gguf"
    run(
        [
            sys.executable,
            "models/convert-gemma-decision-to-gguf.py",
            "--model",
            str(model),
            "--output",
            str(gguf),
        ],
        "convert",
    )
    refs = scratch / "refs"
    run(
        [
            sys.executable,
            "tools/dump_gemma_decision_reference.py",
            "--model",
            str(model),
            "--output",
            str(refs),
        ],
        "reference",
    )
    for ref in sorted(refs.glob("*.gguf")):
        run(
            [
                str(scratch / "build/test-gemma-decision-diff"),
                str(gguf),
                str(ref),
                str(scratch / "native.gguf"),
            ],
            "diff-" + ref.stem,
        )
    run(
        [
            sys.executable,
            "tests/gemma_decision_parity.py",
            "--model",
            str(gguf),
            "--refs",
            str(refs),
            "--lib",
            str(scratch / "build/libcrispembed.so"),
            "--cli",
            str(scratch / "build/crispembed"),
            "--server",
            str(scratch / "build/crispembed-server"),
            "--schema",
            str(scratch / "build/test-gemma-decision-schema"),
            "--output",
            str(out / "parity-f32.json"),
        ],
        "contracts-f32",
    )
    quant = scratch / "gemma-decision-q8_0.gguf"
    run(
        [
            sys.executable,
            "models/convert-gemma-decision-to-gguf.py",
            "--model",
            str(model),
            "--output",
            str(quant),
            "--dtype",
            "q8_0",
        ],
        "convert-q8",
    )
    env["CRISPEMBED_GEMMA_DECISION_F32_DOT"] = "1"
    run(
        [
            sys.executable,
            "tests/gemma_decision_parity.py",
            "--model",
            str(quant),
            "--refs",
            str(refs),
            "--lib",
            str(scratch / "build/libcrispembed.so"),
            "--cli",
            str(scratch / "build/crispembed"),
            "--server",
            str(scratch / "build/crispembed-server"),
            "--schema",
            str(scratch / "build/test-gemma-decision-schema"),
            "--output",
            str(out / "parity-q8.json"),
            "--measure-only",
        ],
        "contracts-q8",
    )
    (out / "versions.json").write_bytes((refs / "versions.json").read_bytes())
    (out / "decisions.json").write_bytes((refs / "decisions.json").read_bytes())
    (out / "summary.md").write_text(
        "EmbeddingGemma 2: five original-Python per-layer FP32 comparisons passed.\n"
    )


if __name__ == "__main__":
    main()
