"""Hosted Rust/Dart consumers: real shared-library scores, JSON and error/repeat contracts."""

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import numpy as np


def main():
    scratch = Path(os.environ["HEAVY_SCRATCH"])
    out = Path(os.environ["HEAVY_OUT"])
    root = Path.cwd()
    lib = scratch / (
        "build/libcrispembed.dylib"
        if sys.platform == "darwin"
        else "build/libcrispembed.so"
    )
    env = dict(
        os.environ,
        CRISPEMBED_SYS_LIB_DIR=str(lib.parent),
        CARGO_BUILD_JOBS="2",
        CRISPEMBED_GEMMA_DECISION="1",
    )
    rust = scratch / "rust-decision-consumer"
    (rust / "src").mkdir(parents=True, exist_ok=True)
    (rust / "Cargo.toml").write_text(
        '[package]\nname = "decision-probe"\nversion = "0.0.0"\nedition = "2021"\n[dependencies]\ncrispembed = { path = '
        + json.dumps(str(root / "crispembed"))
        + " }\n"
    )
    shutil.copy("tests/gemma_decision_bindings.rs", rust / "src/main.rs")
    subprocess.run(
        ["cargo", "build", "--manifest-path", str(rust / "Cargo.toml")],
        env=env,
        check=True,
    )
    dart = scratch / "dart-decision-consumer"
    (dart / "lib").mkdir(parents=True, exist_ok=True)
    for name in ("crispembed.dart", "crispembed_bindings.dart"):
        shutil.copy(root / "flutter/crispembed/lib/src" / name, dart / "lib" / name)
    (dart / "pubspec.yaml").write_text(
        "name: decision_probe\nenvironment:\n  sdk: '>=3.0.0 <4.0.0'\ndependencies:\n  ffi: ^2.0.0\n"
    )
    shutil.copy("tests/gemma_decision_bindings.dart", dart / "main.dart")
    arch = "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "x64"
    system = "macos" if sys.platform == "darwin" else "linux"
    url = f"https://storage.googleapis.com/dart-archive/channels/stable/release/3.13.5/sdk/dartsdk-{system}-{arch}-release.zip"
    archive = scratch / "dart-sdk.zip"
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(scratch)
        for entry in z.infolist():
            mode = entry.external_attr >> 16
            if mode & 0o111:
                (scratch / entry.filename).chmod(mode & 0o777)
    sdk = scratch / "dart-sdk/bin/dart"
    subprocess.run([str(sdk), "--disable-analytics"], check=True)
    subprocess.run([str(sdk), "pub", "get"], cwd=dart, check=True)
    subprocess.run(
        [str(sdk), "analyze", "--no-fatal-warnings", "--no-fatal-infos"],
        cwd=dart,
        check=True,
    )
    subprocess.run([str(sdk), "format", str(dart)], check=True)
    # Retain formatted copies as small reports; never ship SDK or build artifacts.
    for name in ("crispembed.dart", "crispembed_bindings.dart"):
        shutil.copy(dart / "lib" / name, out / ("formatted-" + name))
    fixtures = json.loads((scratch / "refs/decisions.json").read_text())
    for index, fixture in enumerate(fixtures):
        request = scratch / f"binding-request-{index}.json"
        request.write_text(json.dumps(fixture["request"], ensure_ascii=False))
        n = len(next(iter(fixture["answers"].values()))["probabilities"])
        indices = fixture["indices"]
        select = [
            indices["state_index"][0],
            indices["question_index"][0],
            *indices["option_index"][0][:n],
        ]
        vectors = scratch / f"binding-vectors-{index}.bin"
        np.asarray(fixture["text_embeds"], dtype="<f4")[select].tofile(vectors)
        kind = str(indices["question_type"][0])
        model = str(scratch / "gemma-decision-f32.gguf")
        expected = json.loads(
            subprocess.check_output(
                [
                    str(scratch / "build/crispembed"),
                    "-m",
                    model,
                    "-t",
                    "2",
                    "--decide",
                    request.read_text(),
                ],
                env=env,
                text=True,
            )
        )
        commands = {
            "Rust": [
                str(rust / "target/debug/decision-probe"),
                model,
                str(request),
                str(vectors),
                kind,
            ],
            "Dart": [
                str(sdk),
                "run",
                str(dart / "main.dart"),
                model,
                str(request),
                str(vectors),
                kind,
                str(lib),
            ],
        }
        for language, command in commands.items():
            lines = subprocess.check_output(command, env=env, text=True).splitlines()
            scores, answers = map(json.loads, lines)
            np.testing.assert_allclose(
                scores["logits"], fixture["logits"][0][:n], atol=3e-5, rtol=3e-5
            )
            np.testing.assert_allclose(
                scores["probabilities"],
                fixture["probabilities"][0][:n],
                atol=3e-6,
                rtol=3e-5,
            )
            assert answers == expected, (language, index, answers, expected)
            print(
                "PASS", language, "scores/typed JSON/errors/repeat", index, flush=True
            )
    (out / "bindings.json").write_text(
        json.dumps(
            {
                "contracts": "Rust/Dart real FFI scores/typed JSON/errors/repeat PASS",
                "scenarios": len(fixtures),
                "dart": "3.13.5",
                "rust": subprocess.check_output(
                    ["rustc", "--version"], text=True
                ).strip(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
