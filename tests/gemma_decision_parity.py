#!/usr/bin/env python
"""Replay hosted original-Python typed decisions through native API/CLI/HTTP."""

import argparse
import ctypes
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--measure-only", action="store_true")
    args = ap.parse_args()
    fixtures = json.loads((args.refs / "decisions.json").read_text())
    os.environ["CRISPEMBED_GEMMA_DECISION"] = "1"
    model = CrispEmbed(args.model, n_threads=2, lib_path=args.lib)
    assert model.has_decision and model.dim == 768
    pointer = ctypes.POINTER(ctypes.c_float)
    score = model._lib.crispembed_decision_scores
    score.argtypes = [
        ctypes.c_void_p,
        pointer,
        pointer,
        pointer,
        ctypes.c_int,
        ctypes.c_int,
        pointer,
        pointer,
    ]
    score.restype = ctypes.c_int
    reports = []
    native_answers = []

    def check_answers(actual, reference):
        assert list(actual) == list(reference)
        for qid, answer in reference.items():
            result = actual[qid]
            assert result["kind"] == answer["kind"]
            if not args.measure_only:
                assert (
                    result["label"] == answer["label"]
                    and result["value"] == answer["value"]
                ), (qid, result, answer)
            assert type(result["value"]) is type(answer["value"])
            assert list(result["probabilities"]) == list(answer["probabilities"])
            p, r = (
                np.array(list(result["probabilities"].values())),
                np.array(list(answer["probabilities"].values())),
            )
            assert np.isfinite(p).all() and abs(p.sum() - 1) < 2e-6
            assert abs(result["confidence"] - p.max()) < 2e-6
            delta = float(np.abs(p - r).max())
            reports.append(
                {
                    "question": qid,
                    "native": result,
                    "reference": answer,
                    "max_probability_error": delta,
                }
            )
            if not args.measure_only:
                np.testing.assert_allclose(p, r, atol=3e-4, rtol=3e-4)
                if answer["expected_level"] is not None:
                    np.testing.assert_allclose(
                        result["expected_level"], answer["expected_level"], atol=3e-3
                    )

    for case, fixture in enumerate(fixtures):
        request = fixture["request"]
        payload = json.dumps(request, ensure_ascii=False, allow_nan=False)
        path = args.output.with_name(f"schema-{case}.json")
        path.write_text(payload)
        rendered = subprocess.check_output([args.schema, str(path)], text=True)
        assert json.loads(rendered) == fixture["texts"], (
            "rendering differs from Python blueprint"
        )
        embeddings = model.encode(fixture["texts"])
        reference = np.asarray(fixture["text_embeds"], dtype=np.float32)
        cosine = np.sum(embeddings * reference, axis=1) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(reference, axis=1)
        )
        assert np.isfinite(embeddings).all()
        if not args.measure_only:
            assert cosine.min() >= 0.99999 and np.linalg.norm(
                embeddings - reference
            ) <= 0.003 * np.linalg.norm(reference)
        for i, options in enumerate(fixture["indices"]["option_index"]):
            n = len(list(fixture["answers"].values())[i]["probabilities"])
            s = np.ascontiguousarray(reference[fixture["indices"]["state_index"][i]])
            q = np.ascontiguousarray(reference[fixture["indices"]["question_index"][i]])
            o = np.ascontiguousarray(reference[np.array(options[:n])])
            logits, probabilities = np.zeros(n, np.float32), np.zeros(n, np.float32)
            assert score(
                model._ctx,
                s.ctypes.data_as(pointer),
                q.ctypes.data_as(pointer),
                o.ctypes.data_as(pointer),
                n,
                fixture["indices"]["question_type"][i],
                logits.ctypes.data_as(pointer),
                probabilities.ctypes.data_as(pointer),
            )
            np.testing.assert_allclose(
                logits, fixture["logits"][i][:n], atol=3e-5, rtol=3e-5
            )
            np.testing.assert_allclose(
                probabilities, fixture["probabilities"][i][:n], atol=3e-6, rtol=3e-5
            )
        answers = model.decide(
            request["state"], request["questions"], workflow=request["workflow"]
        )
        check_answers(answers, fixture["answers"])
        native_answers.append(answers)
        print(
            f"PASS API/schema/head: scenario {case}, embeddings min cosine={cosine.min():.9f}"
        )
    for questions in ({}, {"bad": {"criteria": []}}, {"bad": "not an object"}):
        try:
            model.decide("state", questions)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid question schema accepted")
    assert model.decide(**fixtures[0]["request"]) == native_answers[0], (
        "repeat changed output"
    )
    model._lib.crispembed_free(model._ctx)
    model._ctx = None
    # All subsequent processes are serial, after freeing the binding model.
    for fixture, expected in zip(fixtures, native_answers):
        payload = json.dumps(fixture["request"], ensure_ascii=False)
        cli = subprocess.check_output(
            [args.cli, "-m", args.model, "--decide", payload], text=True
        )
        assert json.loads(cli) == expected
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with args.output.with_suffix(".server.log").open("w") as log:
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
        )
        try:
            for _ in range(600):
                assert process.poll() is None
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as res:
                        health = json.load(res)
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise AssertionError("server did not start")
            assert health["typed_decision"] is True
            for fixture, expected in zip(fixtures, native_answers):
                req = urllib.request.Request(
                    url + "/decide",
                    json.dumps(fixture["request"]).encode(),
                    {"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as res:
                    assert json.load(res) == expected
            req = urllib.request.Request(
                url + "/decide",
                b'{"state":"x","questions":{}}',
                {"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req)
            except urllib.error.HTTPError as exc:
                assert exc.code == 400 and "error" in json.load(exc)
            else:
                raise AssertionError("server accepted empty questions")
        finally:
            process.terminate()
            process.wait(timeout=10)
    args.output.write_text(
        json.dumps(
            {
                "model": Path(args.model).name,
                "questions": reports,
                "contracts": "API/CLI/HTTP/schema/head/invalid/repeat PASS",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("PASS actual CLI/HTTP, probabilities, typed values and failure contracts")


if __name__ == "__main__":
    main()
