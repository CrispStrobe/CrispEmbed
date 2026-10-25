"""Hosted empty-cache CLI/HTTP audit for a separately published OvisOCR3 F16 alias."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "tests/results/qwen35-ocr"
PUBLICATION = EVIDENCE / "ovisocr3-publication.json"
REGISTRY = ROOT / "examples/cli/model_mgr.cpp"
HASHES = ROOT / "examples/cli/model_hashes.h"
ALIAS = "ovisocr3-f16"
FILENAME = "ovisocr3-f16.gguf"
MAX_TOKENS = 12
GATES = ("CRISPEMBED_QWEN35_OCR", "CRISPEMBED_OVISOCR3_FRONTEND",
         "CRISPEMBED_OVISOCR3_INFERENCE")
ZERO = ("CRISPEMBED_OVISOCR2_FRONTEND", "CRISPEMBED_OVISOCR2_INFERENCE",
        "CRISPEMBED_OVISOCR2_LARGE_IMAGES", "CRISPEMBED_QWEN35_QUANT",
        "CRISPEMBED_QWEN35_PRECISE_RMS", "CRISPEMBED_QWEN35_IMATRIX",
        "CRISPEMBED_QWEN35_PARALLEL_F16", "CRISPEMBED_QWEN35_PROFILE")


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_sha(value, width):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{%d}" % width, value) is not None


def require_publication(receipt, pins, registry, hashes):
    """Pure gate; no network or model files are used."""
    review, producer = receipt.get("independent_review", {}), receipt.get("producer", {})
    if not isinstance(review, dict) or not isinstance(producer, dict):
        raise ValueError("Reviewed OvisOCR3 publication wrapper absent")
    canonical = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    remote = producer.get("remote", {})
    commit = producer.get("published_commit")
    small = remote.get("small_files_sha256", {})
    expected_url = (f'https://huggingface.co/{pins["destination"]}/resolve/{commit}/{FILENAME}'
                    if valid_sha(commit, 40) else "")
    if (review.get("publication_positive") is not True
            or review.get("producer_canonical_sha256") != canonical
            or producer.get("status") != "published and immutable remote file reverified"
            or producer.get("qualification") != pins
            or producer.get("destination") != pins["destination"]
            or not valid_sha(producer.get("source_commit"), 40)
            or not valid_sha(commit, 40)
            or remote.get("repo") != pins["destination"] or remote.get("commit") != commit
            or remote.get("filename") != FILENAME
            or remote.get("sha256") != pins["candidate_sha256"]
            or remote.get("bytes") != pins["candidate_bytes"]
            or set(small) != {"LICENSE", "UPSTREAM_README.md", "README.md", "manifest.json"}
            or not all(valid_sha(value, 64) for value in small.values())
            or not valid_sha(receipt.get("artifact_zip_sha256"), 64)
            or review.get("artifact_zip_sha256") != receipt["artifact_zip_sha256"]
            or type(receipt.get("artifact_id")) is not int or receipt["artifact_id"] < 1
            or type(receipt.get("run")) is not int or receipt["run"] < 1
            or not valid_sha(receipt.get("head_sha"), 40)):
        raise ValueError("Reviewed immutable publication candidate/allowlist differs")
    old = module("ovis3_consumer_registry_parser", "tools/ci-heavy/ovisocr2_published_consumer.py")
    entry = old.registry_entry(registry, hashes, ALIAS, FILENAME,
                               expected_url, pins["candidate_sha256"])
    return {"registry": entry, "commit": commit, "remote": remote,
            "publication_run": receipt["run"], "publication_artifact_id": receipt["artifact_id"],
            "publication_artifact_zip_sha256": receipt["artifact_zip_sha256"],
            "publication_head": receipt["head_sha"]}


def require_public_reference(pins):
    public = json.loads((EVIDENCE / "ovisocr3-public-surfaces.json").read_text())
    native = json.loads((EVIDENCE / "ovisocr3-native-model.json").read_text())
    p, n = public.get("producer", {}), native.get("producer", {})
    c_abi = p.get("arms", {}).get("c-abi", {})
    owned = n.get("owned_readable", {})
    mean = c_abi.get("mean_confidence")
    if (public.get("independent_review", {}).get("public_positive") is not True
            or c_abi.get("passed") is not True
            or type(mean) not in (int, float) or not math.isfinite(mean) or not 0 <= mean <= 1
            or owned.get("png_sha256") != pins["owned_png_sha256"]
            or owned.get("rgb_sha256") != pins["owned_rgb_sha256"]
            or owned.get("gold") != pins["owned_expected_text"]
            or p.get("model_prerequisite", {}).get("receipt_sha256") != pins["receipt_sha256"]["native"]
            or p.get("artifacts", {}).get("model", {}).get("sha256") != pins["candidate_sha256"]):
        raise ValueError("Same-image reviewed public C ABI confidence/source proof differs")
    return {"mean_confidence": mean, "text": pins["owned_expected_text"],
            "png_sha256": pins["owned_png_sha256"],
            "original_generated_ids": pins["owned_generated_ids"],
            "token_ids_exposed_by_public_api": False}


def preflight():
    publisher = module("ovis3_consumer_publisher", "tools/ci-heavy/ovisocr3_publish.py")
    pins = publisher.require_receipts()
    if not PUBLICATION.is_file():
        raise ValueError("Independently reviewed OvisOCR3 publication is absent")
    receipt = json.loads(PUBLICATION.read_text())
    publication = require_publication(receipt, pins, REGISTRY.read_text(), HASHES.read_text())
    if not valid_sha(receipt.get("report_sha256"), 64):
        raise ValueError("Published report SHA absent")
    reference = require_public_reference(pins)
    # Native arithmetic/source closure is unchanged; registry and loader code
    # may differ after publication and are covered by this hosted consumer.
    native = json.loads((EVIDENCE / "ovisocr3-native-model.json").read_text())["producer"]
    source = native.get("native_source_sha256", {})
    if not isinstance(source, dict) or not source or any(sha(ROOT / name) != value
                                                      for name, value in source.items()):
        raise ValueError("Qualified OvisOCR3 native source closure differs")
    return publication, pins, reference


def service_artifact(publication, call=subprocess.check_output):
    artifact_id = publication["publication_artifact_id"]
    value = json.loads(call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/artifacts/{artifact_id}"]))
    run = publication["publication_run"]
    if (value.get("id") != artifact_id
            or value.get("workflow_run", {}).get("id") != run
            or value.get("name") != f"ovisocr3-publication-report-{run}"
            or value.get("digest") != "sha256:" + publication["publication_artifact_zip_sha256"]
            or value.get("expired") is not False):
        raise ValueError("Reviewed publication report artifact service differs")
    return {"id": artifact_id, "run": run, "digest": value["digest"]}


def service_run(publication, call=subprocess.check_output):
    run = publication["publication_run"]
    value = json.loads(call(["gh", "api", f"repos/CrispStrobe/CrispEmbed/actions/runs/{run}"]))
    if (value.get("id") != run or value.get("head_sha") != publication["publication_head"]
            or value.get("path") != ".github/workflows/ovisocr3-publish.yml"
            or value.get("status") != "completed" or value.get("conclusion") != "success"):
        raise ValueError("Reviewed OvisOCR3 publication service run differs")
    return {"run": run, "head_sha": value["head_sha"], "path": value["path"]}


def env_for(cache, selection):
    vision = module("ovis3_consumer_vision", "tools/ci-heavy/ovisocr2_vision_env.py")
    env = vision.apply_selection(os.environ, selection)
    env.update({key: "1" for key in GATES})
    env.update({key: "0" for key in ZERO})
    env.update(CRISPEMBED_CACHE_DIR=str(cache), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", CRISPEMBED_ALLOW_UNPINNED_MODEL="0",
               CRISPEMBED_OFFLINE="0", HF_HUB_OFFLINE="0")
    return env


def http_arm(build, image, alias, cache, expected, selection, out):
    env = env_for(cache, selection)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    path = out / "http-alias.log"
    start = time.monotonic()
    with path.open("wb") as stream:
        proc = subprocess.Popen([str(build / "crispembed-server"), "--ocr-pipeline",
                                 "--ocr-engine", "ovisocr3", "--ocr-rec", alias,
                                 "--ocr-max-tokens", str(MAX_TOKENS), "-t", "2",
                                 "--host", "127.0.0.1", "--port", str(port),
                                 "--image-root", str(image.parent)],
                                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 900
            while True:
                if proc.poll() is not None:
                    raise ValueError("HTTP OvisOCR3 alias service exited before readiness")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                        break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise ValueError("HTTP OvisOCR3 alias service readiness timeout")
                    time.sleep(.3)
            request = urllib.request.Request(f"http://127.0.0.1:{port}/ocr/pipeline",
                                             json.dumps({"image": str(image)}).encode(),
                                             {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                answer = json.load(response)
            confidence = answer.get("mean_confidence")
            if (answer.get("text") != expected or answer.get("n_regions") != 1
                    or type(confidence) not in (int, float) or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1):
                raise ValueError("HTTP OvisOCR3 decoded text/confidence differs")
            return {"text": answer["text"], "n_regions": 1, "mean_confidence": confidence,
                    "seconds": round(time.monotonic() - start, 3),
                    "token_ids_exposed_by_public_api": False}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--publication-run-id", required=True)
    parser.add_argument("--publication-report-dir", type=Path)
    args = parser.parse_args()
    out = Path(os.environ["HEAVY_OUT"])
    scratch = Path(os.environ["HEAVY_SCRATCH"]) / "ovisocr3-consumer"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "ovisocr3-published-consumer.json"
    report = {"status": "started", "phase": "preflight", "source_commit": os.environ.get("GITHUB_SHA"),
              "arms": {}, "limits": ["One reviewed still image at the 12-token request cap; public APIs expose text/confidence only.",
                                       "Native generated IDs and EOS remain in the reviewed low-level receipt.",
                                       "CLI/HTTP confidence uses existing public printing budgets .0006/.00006."]}
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    try:
        publication, pins, reference = preflight()
        if str(publication["publication_run"]) != args.publication_run_id:
            raise ValueError("Explicit publication run differs from reviewed receipt")
        report["publication"] = publication
        report["qualification"] = pins
        report["reference"] = reference
        report["publication_run_service"] = service_run(publication)
        report["publication_artifact_service"] = service_artifact(publication)
        report["phase"] = "preflight-complete"
        save()
        if args.preflight:
            report["status"] = "preflight passed; no build or model access"
            save()
            return
        if args.publication_report_dir is None:
            raise ValueError("Reviewed small publication report artifact required")
        images = list(args.publication_report_dir.rglob("owned-readable.png"))
        if len(images) != 1 or sha(images[0]) != reference["png_sha256"]:
            raise ValueError("Reviewed publication PNG absent or altered")
        old = module("ovis3_consumer_tools", "tools/ci-heavy/ovisocr2_published_consumer.py")
        old.capacity_record(scratch, pins["candidate_bytes"])
        scratch.mkdir(parents=True, exist_ok=True)
        build = scratch / "build"
        report["phase"] = "portable-build"
        save()
        for label, command in (("cmake-configure", ["cmake", "-S", ROOT, "-B", build,
                                                    "-DCMAKE_BUILD_TYPE=Release", "-DCRISPEMBED_NATIVE=OFF",
                                                    "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF"]),
                               ("cpu-baseline", [sys.executable, ROOT / "scripts/check-cpu-baseline.py", build]),
                               ("consumer-build", ["cmake", "--build", build, "--config", "Release",
                                                   "--target", "crispembed-cli", "crispembed-server", "-j2"])):
            result = old.run(label, command, out)
            report["arms"][label] = result
            save()
            if result["exit_code"] != 0 or result["log_bytes"] == 0:
                raise ValueError(label + " failed or returned empty diagnostics")
        cli_cache, http_cache = scratch / "cli-cache", scratch / "http-cache"
        if cli_cache.exists() or http_cache.exists():
            raise ValueError("Fresh independent CLI and HTTP caches required")
        env = env_for(cli_cache, pins["selected_vision_preset"])
        cli = build / "crispembed"
        command = [cli, "--ocr-pipeline", images[0], "--ocr-engine", "ovisocr3",
                   "--ocr-rec", ALIAS, "--ocr-max-tokens", str(MAX_TOKENS), "--json", "-t", "2"]
        report["phase"] = "fresh-cli-alias"
        save()
        first = old.run("cli-fresh", command, out, env=env, timeout=1200)
        report["arms"]["cli-fresh"] = first
        save()
        if first["exit_code"] != 0:
            raise ValueError("Fresh OvisOCR3 CLI alias failed")
        first["result"] = old.parse_cli(out / first["log"], reference["text"])
        first["cache"] = old.check_cached(cli_cache / FILENAME, publication["remote"])
        save()
        env["CRISPEMBED_OFFLINE"] = "1"
        repeat = old.run("cli-offline", [*command, "--offline"], out, env=env, timeout=600)
        report["arms"]["cli-offline"] = repeat
        save()
        if repeat["exit_code"] != 0:
            raise ValueError("Offline OvisOCR3 CLI alias failed")
        repeat["result"] = old.parse_cli(out / repeat["log"], reference["text"])
        repeat["cache"] = old.check_cached(cli_cache / FILENAME, publication["remote"])
        report["phase"] = "fresh-http-alias"
        save()
        try:
            http = http_arm(build, images[0], ALIAS, http_cache, reference["text"],
                            pins["selected_vision_preset"], out)
            report["arms"]["http-fresh"] = http
        finally:
            path = out / "http-alias.log"
            if path.is_file():
                report["arms"].setdefault("http-fresh", {}).update(
                    log=path.name, log_sha256=sha(path), log_bytes=path.stat().st_size)
            save()
        http["cache"] = old.check_cached(http_cache / FILENAME, publication["remote"])
        report["public_comparison"] = old.compare_public_answers(
            first["result"], repeat["result"], http,
            {"mean_confidence": reference["mean_confidence"]})
        report["status"] = "completed published OvisOCR3 alias consumer audit"
        report["phase"] = "complete"
        save()
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


if __name__ == "__main__":
    main()
