#!/usr/bin/env python3
"""Execute TeleOCR through C, Python, stages, CLI and HTTP on one bounded input.

This is an interface consistency test, not an independent model parity test.
The C output is the local anchor; original-model comparisons are separate.
Unexecuted optional surfaces are recorded as skipped, never as passing.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--cli", type=Path)
    ap.add_argument("--server", type=Path)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    if args.max_tokens < 1 or args.threads < 1:
        ap.error("max-tokens and threads must be positive")
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(ROOT / "python"))
    from crispembed import CrispOcrModel, CrispOcrOrchestrator, OCR_ENGINE_TELEOCR
    from crispembed._binding import _setup_ocr_model_signatures

    lib_path, model_path, image_path = (str(p.resolve()) for p in (args.lib, args.model, args.image))
    lib = C.CDLL(lib_path)
    _setup_ocr_model_signatures(lib)
    rows = []
    report = {
        "kind": "teleocr-public-interface-consistency",
        "independent_model_parity": False,
        "model": args.model.name,
        "image": args.image.name,
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "max_tokens": args.max_tokens,
        "threads": args.threads,
        "checks": rows,
    }

    def check(name, condition, **data):
        rows.append({"name": name, "passed": bool(condition), **data})
        if not condition:
            raise AssertionError(name)

    def command(argv):
        run = subprocess.run([str(x) for x in argv], capture_output=True, text=True,
                             timeout=args.timeout, env=os.environ.copy())
        check("command-exit", run.returncode == 0, executable=Path(argv[0]).name,
              returncode=run.returncode, stderr_tail=run.stderr[-3000:] if run.returncode else "")
        return run

    # This harness exercises interfaces, not the license gate (tests/test_teleocr_loader.py does).
    os.environ["CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE"] = "1"
    previous_gate = os.environ.get("CRISPEMBED_TELEOCR")
    try:
        for gate in (None, "", "0"):
            if gate is None:
                os.environ.pop("CRISPEMBED_TELEOCR", None)
            else:
                os.environ["CRISPEMBED_TELEOCR"] = gate
            ctx = lib.crispembed_ocr_model_init(model_path.encode(), args.threads)
            if ctx:
                lib.crispembed_ocr_model_free(ctx)
            check("gate-reject-" + str(gate), not ctx)
        os.environ["CRISPEMBED_TELEOCR"] = "1"
        pixels = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8).copy()
        height, width = pixels.shape[:2]
        ctx = lib.crispembed_ocr_model_init(model_path.encode(), args.threads)
        check("c-load", bool(ctx))
        try:
            initial_cap = lib.crispembed_ocr_model_get_max_tokens(ctx)
            check("c-default-cap", initial_cap == 4096, actual=initial_cap)
            lib.crispembed_ocr_model_set_max_tokens(ctx, args.max_tokens)
            check("c-cap-roundtrip", lib.crispembed_ocr_model_get_max_tokens(ctx) == args.max_tokens)
            check("c-custom-prompt", lib.crispembed_ocr_model_set_prompt(ctx, b"Read the image.") == 1)
            check("c-reset-prompt", lib.crispembed_ocr_model_set_prompt(ctx, None) == 1)
            repetition, presence, frequency, ngram = C.c_float(), C.c_float(), C.c_float(), C.c_int()
            check("c-generation-get", lib.crispembed_ocr_model_get_generation_params(
                ctx, C.byref(repetition), C.byref(presence), C.byref(frequency), C.byref(ngram)) == 1)
            check("c-generation-defaults", abs(repetition.value - 1.05) < 1e-6 and
                  presence.value == frequency.value == 0 and ngram.value == 0)
            check("c-generation-set", lib.crispembed_ocr_model_set_generation_params(ctx, 1.0, 1.0, .05, 100) == 1)
            check("c-generation-invalid", lib.crispembed_ocr_model_set_generation_params(
                ctx, float("nan"), 0, 0, 0) == 0)
            check("c-generation-reset", lib.crispembed_ocr_model_set_generation_params(ctx, 1.05, 0, 0, 0) == 1)
            out_len = C.c_int()
            text = lib.crispembed_ocr_model_recognize(
                ctx, pixels.ctypes.data_as(C.POINTER(C.c_uint8)), width, height, 3, C.byref(out_len))
            check("c-recognize", bool(text) and out_len.value > 0)
            anchor = text.decode("utf-8")
            n = C.c_int()
            ptr = lib.crispembed_ocr_model_confidences(ctx, C.byref(n))
            conf = [ptr[i] for i in range(n.value)] if ptr else []
            check("c-token-cap", 0 < n.value <= args.max_tokens, tokens=n.value)
            check("c-confidence", all(math.isfinite(x) and 0 <= x <= 1 for x in conf), values=conf)
            report["anchor_text"] = anchor
        finally:
            lib.crispembed_ocr_model_free(ctx)

        ocr = CrispOcrModel(model_path, n_threads=args.threads, lib_path=lib_path)
        try:
            ocr.set_max_tokens(args.max_tokens)
            check("python-generation-get", ocr.get_generation_params()["no_repeat_ngram"] == 0)
            check("python-generation-invalid", not ocr.set_generation_params(repetition_penalty=0))
            check("python-generation-set", ocr.set_generation_params())
            check("python-text", ocr.recognize(pixels) == anchor)
            check("python-confidences", np.allclose(ocr.confidences(), conf, rtol=1e-6, atol=1e-6))
        finally:
            ocr.__del__()

        orch = CrispOcrOrchestrator(router=False, lib_path=lib_path, n_threads=args.threads,
                                   stages=[{"engine": OCR_ENGINE_TELEOCR, "model_a": model_path,
                                            "vlm_max_tokens": 1, "vlm_prompt": "Read the image.",
                                            "min_chars": 100000, "min_confidence": 0},
                                           {"engine": OCR_ENGINE_TELEOCR, "model_a": model_path,
                                            "vlm_max_tokens": args.max_tokens,
                                            "min_chars": 1, "min_confidence": 0}])
        try:
            result = orch.run(image_path)
            check("orchestrator-text", result["text"] == anchor, regions=result["n_regions"])
            check("orchestrator-region", result["n_regions"] == 1)
        finally:
            orch.__del__()

        # A syntactically valid metadata-only GGUF from the wrong architecture
        # must be rejected before entering a Qwen graph. No model tensors needed.
        with tempfile.TemporaryDirectory() as temp:
            wrong_model = Path(temp) / "wrong-profile.gguf"
            def gguf_string(value):
                encoded = value.encode("utf-8")
                return struct.pack("<Q", len(encoded)) + encoded
            metadata = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
            metadata += gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("qwen2vl")
            metadata += gguf_string("teleocr.gguf_version") + struct.pack("<II", 4, 1)
            metadata += b"\0" * (-len(metadata) % 32)
            wrong_model.write_bytes(metadata)
            orch = CrispOcrOrchestrator(router=False, lib_path=lib_path, n_threads=args.threads,
                                       stages=[{"engine": OCR_ENGINE_TELEOCR, "model_a": str(wrong_model),
                                                "vlm_max_tokens": 1, "min_chars": 1, "min_confidence": 0}])
            try:
                result = orch.run(image_path)
                check("orchestrator-wrong-model", result["n_regions"] == 0 and not result["text"])
            finally:
                orch.__del__()

        if args.cli:
            run = command([args.cli.resolve(), "-m", model_path, "--ocr", image_path,
                           "--ocr-max-tokens", str(args.max_tokens), "-t", str(args.threads), "--json"])
            check("cli-text", json.loads(run.stdout)["text"] == anchor)
            run = command([args.cli.resolve(), "--ocr-pipeline", image_path, "--ocr-engine", "teleocr",
                           "--ocr-rec", model_path, "--ocr-max-tokens", str(args.max_tokens),
                           "-t", str(args.threads), "--json"])
            payload = json.loads(run.stdout)
            check("cli-stage-text", payload.get("text", payload.get("full_text")) == anchor)
        else:
            rows.append({"name": "cli", "skipped": "--cli not supplied"})

        if args.server:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            # Ignore ambient HTTP proxy settings for the loopback test server.
            client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with tempfile.TemporaryFile(mode="w+") as logs:
                proc = subprocess.Popen([str(args.server.resolve()), "--ocr", model_path,
                                         "--ocr-max-tokens", str(args.max_tokens), "-t", str(args.threads),
                                         "--host", "127.0.0.1", "--port", str(port),
                                         "--image-root", str(args.image.resolve().parent)],
                                        stdout=logs, stderr=logs, env=os.environ.copy())
                try:
                    deadline = time.monotonic() + args.timeout
                    while time.monotonic() < deadline:
                        if proc.poll() is not None:
                            logs.seek(0)
                            raise RuntimeError("server exited: " + logs.read()[-3000:])
                        try:
                            with client.open(base + "/health", timeout=1) as response:
                                health = json.load(response)
                            if health.get("ocr_model"):
                                break
                        except (OSError, urllib.error.URLError):
                            time.sleep(0.1)
                    else:
                        raise TimeoutError("server did not become ready")

                    def post(payload):
                        request = urllib.request.Request(base + "/ocr/model", json.dumps(payload).encode(),
                                                         {"Content-Type": "application/json"})
                        with client.open(request, timeout=args.timeout) as response:
                            return json.load(response)

                    # An error after applying request knobs must still restore
                    # the startup prompt and cap for the next valid request.
                    try:
                        post({"image": str(args.image.resolve().parent / "does-not-exist.png"),
                              "max_tokens": 1, "prompt": "Read the image.",
                              "repetition_penalty": 1.0, "presence_penalty": 1.0,
                              "frequency_penalty": .05, "no_repeat_ngram": 100})
                        check("http-invalid-image", False)
                    except urllib.error.HTTPError as exc:
                        check("http-invalid-image", exc.code == 400)
                    data = post({"image": image_path})
                    check("http-text", data["latex"] == anchor)
                    check("http-cap-restored", len(data["token_confidences"]) == len(conf))
                    check("http-confidence", np.allclose(data["token_confidences"], conf, atol=0.000051))
                finally:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        else:
            rows.append({"name": "http", "skipped": "--server not supplied"})
        report["passed"] = True
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if previous_gate is None:
            os.environ.pop("CRISPEMBED_TELEOCR", None)
        else:
            os.environ["CRISPEMBED_TELEOCR"] = previous_gate
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
