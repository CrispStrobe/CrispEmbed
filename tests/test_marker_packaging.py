#!/usr/bin/env python3
"""Build local package archives and verify native Apache notice preservation.

This is a packaging-only test. The supplied native binary is copied as an
opaque fixture; it is never loaded. No model, network or publication is used.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTICES = {
    "NOTICE.marker": ROOT / "NOTICE.marker",
    "NOTICE.teleocr": ROOT / "NOTICE.teleocr",
    "Apache-2.0.txt": ROOT / "models/licenses/apache-2.0/LICENSE",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    native = args.native_library.resolve()
    if not native.is_file():
        parser.error("--native-library must name an existing native binary")
    work, output = args.work_dir.resolve(), args.output.resolve()
    if work.exists() and any(work.iterdir()):
        parser.error("--work-dir must be absent or empty")
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be absent or empty")
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    expected = {name: path.read_bytes() for name, path in NOTICES.items()}
    binary_bytes = native.read_bytes()
    native_name = "libcrispembed.so"
    checkout = work / "checkout"
    staged = checkout / "python"
    shutil.copytree(
        ROOT / "python", staged,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "build", "dist", "*.egg-info",
            "*.so", "*.so.*", "*.dylib", "*.dll", "*.pyd",
        ),
    )
    for name, data in expected.items():
        destination = checkout / name if name != "Apache-2.0.txt" else checkout / "models/licenses/apache-2.0/LICENSE"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        # A checkout must replace stale staged copies with the source notice.
        (staged / "crispembed" / name).write_text("stale test-only notice\n")
    if (ROOT / "POLICY.md").is_file():
        shutil.copyfile(ROOT / "POLICY.md", checkout / "POLICY.md")
    (staged / "crispembed" / native_name).write_bytes(binary_bytes)

    checks, commands, artifacts = [], [], []
    source_files = ["python/setup.py", "python/MANIFEST.in", "python/pyproject.toml",
                    "tests/test_marker_packaging.py", "NOTICE.marker", "NOTICE.teleocr",
                    "models/licenses/apache-2.0/LICENSE"]
    source_hashes = {name: digest((ROOT / name).read_bytes()) for name in source_files}

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    env = os.environ.copy()
    env.update(PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1")

    def run(label, cwd, arguments, expected_success=True):
        command = [sys.executable, "setup.py", *arguments]
        result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output / (label + ".log")).write_text(result.stdout, encoding="utf-8")
        commands.append(dict(label=label, command=["python", "setup.py", *arguments],
                             exit_code=result.returncode))
        check(label + ": command status", (result.returncode == 0) == expected_success)
        if expected_success and result.returncode:
            raise RuntimeError(label + " failed; see its log")
        return result

    def verify_wheel(path, label):
        with zipfile.ZipFile(path) as archive:
            for name, data in expected.items():
                check(label + ": " + name, archive.read("crispembed/" + name) == data,
                      expected_sha256=digest(data))
            check(label + ": native bytes", digest(archive.read("crispembed/" + native_name)) == digest(binary_bytes))
        artifacts.append(dict(kind=label, filename=path.name, bytes=path.stat().st_size,
                              sha256=digest(path.read_bytes())))

    def extract_owned_sdist(path, destination):
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            prefix = next(member.name.split("/", 1)[0] for member in members)
            for name, data in expected.items():
                handle = archive.extractfile(prefix + "/crispembed/" + name)
                check("sdist: " + name, handle is not None and handle.read() == data,
                      expected_sha256=digest(data))
            handle = archive.extractfile(prefix + "/crispembed/" + native_name)
            check("sdist: native bytes", handle is not None and digest(handle.read()) == digest(binary_bytes))
            # Extract only ordinary members of this freshly produced archive.
            for member in members:
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                    raise RuntimeError("unexpected unsafe sdist member")
                target = destination / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
                else:
                    raise RuntimeError("unexpected special sdist member")
        artifacts.append(dict(kind="sdist", filename=path.name, bytes=path.stat().st_size,
                              sha256=digest(path.read_bytes())))
        return destination / prefix

    error = None
    try:
        run("checkout-wheel", staged, ["bdist_wheel", "--dist-dir", "artifacts-wheel"])
        wheel = next((staged / "artifacts-wheel").glob("*.whl"))
        verify_wheel(wheel, "checkout wheel")
        run("checkout-sdist", staged, ["sdist", "--dist-dir", "artifacts-sdist"])
        sdist = next((staged / "artifacts-sdist").glob("*.tar.gz"))
        isolated = extract_owned_sdist(sdist, work / "isolated")
        check("isolated sdist has no repository-parent notices",
              not (isolated.parent / "NOTICE.marker").exists()
              and not (isolated.parent / "NOTICE.teleocr").exists()
              and not (isolated.parent / "models/licenses/apache-2.0/LICENSE").exists())
        run("isolated-sdist-wheel", isolated, ["bdist_wheel", "--dist-dir", "rebuilt-wheel"])
        verify_wheel(next((isolated / "rebuilt-wheel").glob("*.whl")), "isolated sdist wheel")
        for name, data in expected.items():
            destination = isolated / "crispembed" / name
            destination.unlink()
            result = run("reject-missing-" + name.replace(".", "-"), isolated, ["--name"], False)
            check("missing " + name + ": explicit diagnostic",
                  "Cannot bundle native CrispEmbed libraries" in result.stdout and name in result.stdout)
            destination.write_bytes(data)
        destination = isolated / "crispembed/Apache-2.0.txt"
        destination.write_bytes(b"")
        result = run("reject-empty-apache-license", isolated, ["--name"], False)
        check("empty license: explicit diagnostic", "Apache-2.0.txt" in result.stdout)
        destination.write_bytes(expected["Apache-2.0.txt"])
        check("input native binary unchanged", digest(native.read_bytes()) == digest(binary_bytes))
        check("packaging sources unchanged", source_hashes == {
            name: digest((ROOT / name).read_bytes()) for name in source_files})
    except Exception as exc:
        error = repr(exc)
        check("unexpected packaging exception", False, error=error)
    versions = {}
    for name in ("setuptools", "wheel"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    receipt = dict(
        schema_version=1, completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="Local wheel/sdist attribution and binary-byte preservation only. Native library not loaded; no models, network or publication.",
        python_version=sys.version.split()[0], build_packages=versions,
        input_native_sha256=digest(binary_bytes), input_native_bytes=len(binary_bytes),
        source_sha256=source_hashes,
        notices_sha256={name: digest(data) for name, data in expected.items()},
        environment_overrides={key: env[key] for key in ("PIP_NO_INDEX", "PIP_DISABLE_PIP_VERSION_CHECK", "PYTHONNOUSERSITE")},
        commands=commands, archives=artifacts, checks=checks,
        passed=error is None and all(item["passed"] for item in checks),
    )
    receipt["logs_sha256"] = {path.name: digest(path.read_bytes()) for path in output.glob("*.log")}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print("Marker packaging:", len(checks), "checks,", sum(not item["passed"] for item in checks), "failures")
    if error:
        print(error)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
