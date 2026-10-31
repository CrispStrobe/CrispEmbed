#!/usr/bin/env python3
"""One production CLI decode against frozen text IDs, without observer hooks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    workspace, out = args.workspace.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error("output directory is not empty; retain earlier results")
    out.mkdir(parents=True, exist_ok=True)
    cli = workspace / "build-teleocr/crispembed"
    model = workspace / "teleocr-models/teleocr-f16.gguf"
    references = workspace / "validation/reference"
    expected = json.loads((references / "text.json").read_text())
    if sha256(references / "text.png") != expected["image_sha256"]:
        raise ValueError("Frozen text image hash changed")
    for source, name in ((references / "text.png", "text.png"),
                         (references / "text.json", "reference-text.json"),
                         (Path(__file__), "run_cli_observer_control.executed.py"),
                         (workspace / "CrispEmbed-TeleOCR/examples/cli/main.cpp", "cli-main.snapshot.cpp")):
        shutil.copyfile(source, out / name)
    command = [str(cli), "-m", str(model), "--ocr", str(out / "text.png"),
               "--ocr-max-tokens", "1024", "-t", "2", "--json"]
    report = {
        "kind": "teleocr-production-cli-observer-neutrality", "backend": "cpu",
        "scope": "one frozen text case; decoded text, all generated IDs and EOS",
        "numerical_activation_comparison": False,
        "confidence_check": "not exposed by CLI JSON; covered separately by native differential and public C API tests",
        "command": command, "max_tokens": 1024, "threads": 2,
        "model": model.name, "model_sha256": sha256(model), "cli_sha256": sha256(cli),
        "runner_sha256": sha256(__file__), "cli_source_sha256": sha256(out / "cli-main.snapshot.cpp"),
        "reference_sha256": sha256(out / "reference-text.json"), "image_sha256": expected["image_sha256"],
        "expected_token_count": expected["tokens"], "expected_reached_eos": expected["reached_eos"],
        "passed": False,
    }
    environment = dict(os.environ, CRISPEMBED_TELEOCR="1", CRISPEMBED_TELEOCR_GPU="0",
                       QWEN2VL_OCR_FORCE_CPU="1")
    environment.pop("CRISPEMBED_TELEOCR_DUMP_STAGES", None)
    started = time.monotonic()
    try:
        with (out / "cli.stdout.json").open("w", encoding="utf-8") as stdout, \
             (out / "cli.stderr.log").open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, timeout=1200)
        report["exit_code"] = result.returncode
        actual = json.loads((out / "cli.stdout.json").read_text())
        matches = re.findall(r"^\s*gen\[(\d+)\]: token=(\d+)\b", (out / "cli.stderr.log").read_text(), re.M)
        indexes = [int(index) for index, unused in matches]
        ids = [int(token) for unused, token in matches]
        report.update(text=actual["text"], generated_ids=ids, generated_token_count=len(ids),
                      text_exact=actual["text"] == expected["text"],
                      token_ids_exact=ids == expected["generated_ids"],
                      generation_indexes_contiguous=indexes == list(range(len(ids))),
                      reached_eos=bool(ids) and ids[-1] in expected["environment"]["eos_token_ids"])
        report["passed"] = (result.returncode == 0 and report["text_exact"] and report["token_ids_exact"]
                            and report["generation_indexes_contiguous"] and report["reached_eos"]
                            and len(ids) == expected["tokens"] == 66)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["elapsed_seconds"] = time.monotonic() - started
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
