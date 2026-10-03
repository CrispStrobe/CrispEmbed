#!/usr/bin/env python
"""Compare actual HTTP encoder routes with Python C-ABI results, serially.

The server is stopped before loading the binding model to bound memory use.
No torch required; use a published model with the desired arithmetic env gate.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from crispembed import CrispEmbed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--log", required=True, type=Path)
    args = ap.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    def post(route, payload, status=200):
        req = urllib.request.Request(
            url + route,
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as res:
                assert res.status == status
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            assert exc.code == status, (route, exc.code, exc.read())
            return json.loads(exc.read().decode("utf-8"))

    texts = [
        "The capital of France is [MASK].",
        'Escaped "quote", bracket ] and newline\n[MASK] / [MASK].',
        "日本の首都は[MASK]です。中国的首都是[MASK]。",
        "The [MASK][MASK] sat on the mat.",
    ]
    outputs = []
    with args.log.open("w") as log:
        process = subprocess.Popen(
            [
                args.server,
                "-m",
                args.model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-t",
                "2",
            ],
            stdout=log,
            stderr=log,
            env=os.environ.copy(),
        )
        try:
            for _ in range(600):
                assert process.poll() is None, (
                    "server exited during startup; inspect --log"
                )
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as res:
                        health = json.load(res)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            else:
                raise AssertionError("server startup timeout")
            assert health["masked_lm"] is True and "colbert" not in health
            for route in ("/tokens", "/masked-logits", "/fill-mask"):
                for payload in ({}, {"text": ""}, {"text": "bad\0input"}):
                    assert "error" in post(route, payload, 400)
            for k in (0, 101, 1.5):
                post("/fill-mask", {"text": texts[0], "top_k": k}, 400)
            for route in ("/masked-logits", "/fill-mask"):
                post(route, {"text": "no mask here"}, 400)
            for text in texts:
                outputs.append(
                    {
                        "raw": post(
                            "/tokens", {"text": text.replace("[MASK]", "<|mask|>")}
                        ),
                        "normalized": post(
                            "/tokens",
                            {
                                "text": text.replace("[MASK]", "<|mask|>"),
                                "normalize": True,
                            },
                        ),
                        "logits": post("/masked-logits", {"text": text}),
                        "fill": post("/fill-mask", {"text": text, "top_k": 100}),
                    }
                )
            # Alternate shapes and routes, then repeat the first request.
            assert (
                post("/fill-mask", {"text": texts[0], "top_k": 100})
                == outputs[0]["fill"]
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    # Explicitly disable the encoder to exercise capability discovery and guards.
    disabled_env = dict(os.environ, CRISPEMBED_LFM2_ENCODER="0")
    with args.log.with_suffix(".disabled.log").open("w") as log:
        process = subprocess.Popen(
            [
                args.server,
                "-m",
                args.model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "-t",
                "2",
            ],
            stdout=log,
            stderr=log,
            env=disabled_env,
        )
        try:
            for _ in range(600):
                assert process.poll() is None
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as res:
                        health = json.load(res)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            else:
                raise AssertionError("disabled server startup timeout")
            assert "masked_lm" not in health
            for route in ("/masked-logits", "/fill-mask"):
                assert (
                    "no masked LM head" in post(route, {"text": texts[0]}, 400)["error"]
                )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    print("PASS HTTP masked-LM capability guards when encoder explicitly disabled")

    model = CrispEmbed(args.model, n_threads=2, lib_path=args.lib)
    for text, result in zip(texts, outputs):
        encoded = text.replace("[MASK]", "<|mask|>")
        for key, normalize in (("raw", False), ("normalized", True)):
            ids, embeddings = model.encode_tokens(encoded, normalize=normalize)
            actual = result[key]
            assert actual["normalized"] == normalize
            np.testing.assert_array_equal(actual["token_ids"], ids)
            np.testing.assert_allclose(
                actual["embeddings"], embeddings, atol=1e-6, rtol=1e-6
            )
            assert actual["dim"] == embeddings.shape[1] == 1024
        positions, logits = model.masked_logits(encoded)
        np.testing.assert_array_equal(result["logits"]["positions"], positions)
        np.testing.assert_allclose(
            result["logits"]["logits"], logits, atol=1e-6, rtol=1e-6
        )
        expected = model.fill_mask(text, top_k=100)
        actual = result["fill"]["masks"]
        assert result["fill"]["text"] == text
        for a, e in zip(actual, expected):
            assert a["position"] == e["position"]
            assert len(a["predictions"]) == len(e["predictions"]) == 100
            for apred, epred in zip(a["predictions"], e["predictions"]):
                assert (
                    apred["token_id"] == epred["token_id"]
                    and apred["token"] == epred["token"]
                )
                np.testing.assert_allclose(
                    apred["logit"], epred["logit"], atol=1e-6, rtol=1e-6
                )
                np.testing.assert_allclose(
                    apred["score"], epred["score"], atol=1e-10, rtol=1e-6
                )
        assert len(actual) == len(expected)
        print(
            f"PASS HTTP/Python: tokens={len(ids)} masks={len(positions)}, top-100 decoded predictions"
        )
    print(
        "PASS HTTP health, escaping, invalid inputs, normalization, raw logits and repeat contracts"
    )


if __name__ == "__main__":
    main()
