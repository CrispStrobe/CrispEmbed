"""Hosted iOS library build plus native Qwen positions regression, no models."""
import json
import os
import subprocess
import time
from pathlib import Path


def run():
    repo = Path(__file__).resolve().parents[2]
    out, scratch = Path(os.environ["HEAVY_OUT"]), Path(os.environ["HEAVY_SCRATCH"])
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "initial_failure": "https://github.com/CrispStrobe/CrispEmbed/actions/runs/38681503635",
              "scope": "model-free iOS cross-build and native standalone probe", "checks": {}}
    try:
        sdk = subprocess.check_output(["xcrun", "--sdk", "iphoneos", "--show-sdk-path"], text=True).strip()
        ios = scratch / "ios-library"
        query = ios / ".cmake/api/v1/query"
        query.mkdir(parents=True, exist_ok=True)
        (query / "codemodel-v2").touch()
        started = time.perf_counter()
        subprocess.run(["cmake", "-S", str(repo), "-B", str(ios), "-DCMAKE_BUILD_TYPE=Release",
                        "-DCMAKE_SYSTEM_NAME=iOS", "-DCMAKE_OSX_ARCHITECTURES=arm64",
                        "-DCMAKE_OSX_SYSROOT=" + sdk, "-DCMAKE_OSX_DEPLOYMENT_TARGET=15.0",
                        "-DGGML_NATIVE=OFF", "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON",
                        "-DGGML_BLAS=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"], check=True)
        reply = ios / ".cmake/api/v1/reply"
        index = json.loads(next(reply.glob("index-*.json")).read_text())
        model = json.loads((reply / index["reply"]["codemodel-v2"]["jsonFile"]).read_text())
        targets = [json.loads((reply / target["jsonFile"]).read_text())
                   for configuration in model["configurations"] for target in configuration["targets"]]
        executables = [target["name"] for target in targets if target["type"] == "EXECUTABLE"]
        if executables:
            raise ValueError("Unexpected iOS standalone executables: " + ", ".join(executables))
        if not any(target["name"] == "crispembed" for target in targets):
            raise ValueError("iOS library target missing")
        subprocess.run(["cmake", "--build", str(ios), "-j", "4"], check=True)
        library = ios / "libcrispembed-static.a"
        arch = subprocess.check_output(["xcrun", "lipo", "-archs", str(library)], text=True).strip()
        if arch != "arm64":
            raise ValueError("Unexpected iOS archive architecture")
        report["checks"]["ios"] = dict(passed=True, standalone_executables=executables,
                                       library_bytes=library.stat().st_size, architecture=arch,
                                       build_seconds=time.perf_counter() - started)
        native = scratch / "native-qwen-positions"
        subprocess.run(["cmake", "-S", str(repo), "-B", str(native), "-DCMAKE_BUILD_TYPE=Release",
                        "-DGGML_NATIVE=OFF", "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF"], check=True)
        subprocess.run(["cmake", "--build", str(native), "--target", "test-qwen35-positions", "-j", "4"], check=True)
        result = subprocess.run([str(native / "test-qwen35-positions")], capture_output=True, text=True)
        (out / "native-positions.log").write_text(result.stdout + result.stderr)
        print(result.stdout + result.stderr)
        result.check_returncode()
        report["checks"]["native_positions"] = dict(passed=True, log="native-positions.log")
        report["result"] = "passed iOS library-only target graph/full build and native probe"
    finally:
        (out / "ios-build-check.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    run()
