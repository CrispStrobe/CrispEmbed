#!/usr/bin/env python3
"""Stage the self-contained Xiaomi layer-0 attribution replay kernel.

The runner imports only the standard library at module scope, downloads or reads
every input at run time from pinned digests, imports the original installed
module, and never carries a copy of upstream code. Staging freezes the exact
runner bytes plus the pinned artifact digests and refuses to freeze a runner
whose declared pins disagree with this file.

usage: python3 stage.py NEW_DIRECTORY
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE_COMMIT = "15adb672f71f8470a23db1284dafa102515adae9"
PINNED = {
    "model_repo": "xiaomi-research/Xiaomi-OCR-0",
    "revision": "bbd8cf8d21f256b3644b44cb860f3a7cecb93c30",
    "checkpoint_sha256": "d4551e9f6127ac9e35cff7b4f7a72a90410e03e6c8676a36f0e2b7eaeb5d5348",
    "config_sha256": "2eb95982069a24336fe66b3a0869ea5a711aba7566787d4a599654907ecc070c",
    "export_sha256": "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235",
    "reference_sha256": "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334",
    "source_module_sha256": "641da34484f6b9769e28615c680cc4e3202afbf5bdbf1da7d56d5a3d678918c7",
    "native_embedded_sha256": "0fa3aa5a49166ae8467e582a9919dc36d4b26bda14775df9984fb95cd43f0d3b",
    "native_layer0_sha256": "dd1d3fa45703dfd92f790c7be8dd947b8c525bcb5bfc6544a911f00a567ddc37",
    "vision_prefill_runner_sha256": "18bfc3e880032e00e7ceadd6f7eea5e2cc2ef3cce28d6723561e2abd155e15d2",
    "vision_prefill_report_sha256": "eae8d973500b2e3890e3dd408f051d67edc26ffa7b5de71f9b3429949470816d",
    "transformers": "5.8.1",
    "tokenizers": "0.22.2",
    "gguf": "0.19.0",
    "safetensors": "0.8.0",
}
RUNNER_CONSTANTS = {
    "model_repo": "MODEL_REPO",
    "revision": "REVISION",
    "checkpoint_sha256": "CHECKPOINT_SHA256",
    "config_sha256": "CONFIG_SHA256",
    "export_sha256": "EXPORT_SHA256",
    "reference_sha256": "REFERENCE_SHA256",
    "source_module_sha256": "SOURCE_MODULE_SHA256",
    "native_embedded_sha256": "NATIVE_EMBEDDED_SHA256",
    "native_layer0_sha256": "NATIVE_LAYER0_SHA256",
    "vision_prefill_runner_sha256": "VISION_PREFILL_RUNNER_SHA256",
    "vision_prefill_report_sha256": "VISION_PREFILL_REPORT_SHA256",
    "transformers": "TRANSFORMERS",
    "tokenizers": "TOKENIZERS",
    "gguf": "GGUF",
    "safetensors": "SAFETENSORS",
}


def load_runner(path):
    spec = importlib.util.spec_from_file_location("staged_xiaomi_layer0_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the runner for pin verification")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1]).resolve()
    if target.exists():
        raise ValueError("staging directory already exists")
    runner = (HERE / "runner.py").read_bytes()
    text = runner.decode()
    compile(text, "staged-runner.py", "exec")
    if "__CRISPEMBED_SOURCE_BUNDLE__" in text:
        raise ValueError("runner must not embed a source bundle; it imports the installed module")
    module = load_runner(HERE / "runner.py")
    drift = {name: {"staged": PINNED[name], "runner": getattr(module, constant)}
             for name, constant in RUNNER_CONSTANTS.items()
             if getattr(module, constant) != PINNED[name]}
    if drift:
        raise ValueError("staged pins disagree with runner constants: "
                         + json.dumps(drift, sort_keys=True))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                          capture_output=True).stdout.strip() or None
    layout = module.block_tensor_layout()
    manifest = {
        "script_version": module.SCRIPT_VERSION,
        "script_version_line": [line for line in text.splitlines()
                                if line.startswith("SCRIPT_VERSION")][0],
        "base_commit": BASE_COMMIT,
        "staging_head": head,
        "pinned": PINNED,
        "strict_gate": {"max_abs": module.STRICT_MAX_ABS, "row_cos_min": module.STRICT_COS_MIN,
                        "unchanged": True},
        "geometry": {"tokens": module.TOKENS, "width": module.WIDTH, "heads": module.HEADS,
                     "head_dim": module.HEAD_DIM, "rotary": module.ROTARY,
                     "layers": module.LAYERS, "cu_seqlens": module.CU_SEQLENS,
                     "grid_thw": module.GRID},
        "block_tensors": {"count": layout["count"], "names": layout["names"],
                          "checkpoint_prefix": module.CHECKPOINT_PREFIX,
                          "export_prefix": module.EXPORT_PREFIX},
        "decomposition": module.REPORT["decomposition"],
        "runner_template_sha256": hashlib.sha256(runner).hexdigest(),
        "staged_runner_sha256": hashlib.sha256(runner).hexdigest(),
        "files_sha256": {"runner.py": hashlib.sha256(runner).hexdigest()},
    }
    target.mkdir(parents=True)
    (target / "runner.py").write_bytes(runner)
    (target / "kernel-metadata.json").write_bytes((HERE / "kernel-metadata.json").read_bytes())
    (target / "source-closure.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
