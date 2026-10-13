#!/usr/bin/env python
"""Verify published pinned aliases by fresh downloads and real CLI decisions."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from huggingface_hub import hf_hub_download


def main():
    scratch, out = Path(os.environ["HEAVY_SCRATCH"]), Path(os.environ["HEAVY_OUT"])
    build, cache = scratch / "build", scratch / "alias-cache"
    assert not cache.exists(), "This check requires a fresh model cache"
    subprocess.run(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            ".",
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCRISPEMBED_NATIVE=OFF",
            "-DGGML_NATIVE=OFF",
            "-DGGML_BLAS=OFF",
            "-DGGML_METAL=OFF",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "crispembed-cli", "-j2"],
        check=True,
    )
    publication = json.loads(
        Path("tests/results/gemma-decision/publication.json").read_text()
    )
    refs = hf_hub_download(
        publication["fixture_repo"],
        "gemma-decision/original-fp32/decisions.json",
        repo_type="dataset",
        revision=publication["fixture_revision"],
    )
    fixtures = json.loads(Path(refs).read_text())
    env = dict(
        os.environ,
        CRISPEMBED_CACHE_DIR=str(cache),
        CRISPEMBED_GEMMA_DECISION="1",
        CRISPEMBED_GEMMA_DECISION_F32_DOT="1",
    )
    reports = []
    for alias, precision, tolerance in [
        ("embeddinggemma-2-ft-typed-decision", "f32", 3e-4),
        ("embeddinggemma-2-ft-typed-decision-q8", "q8_0", 3e-3),
    ]:
        answers = 0
        max_error = 0.0
        for fixture in fixtures:
            actual = json.loads(
                subprocess.check_output(
                    [
                        str(build / "crispembed"),
                        "-m",
                        alias,
                        "-t",
                        "2",
                        "--decide",
                        json.dumps(fixture["request"], ensure_ascii=False),
                    ],
                    env=env,
                    text=True,
                )
            )
            assert list(actual) == list(fixture["answers"])
            for key, reference in fixture["answers"].items():
                result = actual[key]
                for field in ["kind", "label", "value"]:
                    assert result[field] == reference[field], (alias, key, field)
                assert type(result["value"]) is type(reference["value"])
                assert list(result["probabilities"]) == list(reference["probabilities"])
                ps = list(result["probabilities"].values())
                assert all(math.isfinite(p) for p in ps) and abs(sum(ps) - 1) < 2e-6
                assert abs(result["confidence"] - max(ps)) < 2e-6
                error = max(
                    abs(p - reference["probabilities"][option])
                    for option, p in result["probabilities"].items()
                )
                assert error < tolerance, (alias, key, error)
                max_error = max(max_error, error)
                answers += 1
        filename = "embeddinggemma-2-ft-typed-decision-" + precision + ".gguf"
        path = cache / filename
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(block)
        identity = {"sha256": sha.hexdigest(), "bytes": path.stat().st_size}
        assert identity == publication["artifacts"][filename]
        reports.append(
            {
                "alias": alias,
                "answers": answers,
                "max_probability_error": max_error,
                **identity,
            }
        )
    (out / "registry.json").write_text(
        json.dumps(
            {
                "contracts": "fresh pinned alias downloads and real CLI decisions PASS",
                "native_commit": os.environ["GITHUB_SHA"],
                "workflow_run": os.environ["GITHUB_RUN_ID"],
                "models": reports,
            },
            indent=2,
        )
        + "\n"
    )
    print("PASS", reports, flush=True)


if __name__ == "__main__":
    main()
