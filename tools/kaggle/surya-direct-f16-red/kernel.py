"""Private, weight-free hosted RED probe for direct-F16 projection.

stage.py embeds the exact test bytes and records their sha256. The repository
base and ggml gitlink remain unchanged. This probe does not use a checkpoint.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT = "@COMMIT@"
TEST_B64 = "@TEST_B64@"
TEST_SHA256 = "@TEST_SHA256@"
SOURCE_B64 = "@SOURCE_B64@"
SOURCE_SHA256 = "@SOURCE_SHA256@"
MODE = "@MODE@"
ROOT = Path("/kaggle/temp/direct-f16-repo")
BUILD = Path("/kaggle/temp/direct-f16-build")
OUT = Path("/kaggle/working/direct-f16-red")


def command(args, **kw):
    print("+", *map(str, args), flush=True)
    return subprocess.run(args, check=True, **kw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"phase": "setup", "commit": COMMIT, "test_sha256": TEST_SHA256,
              "scope": "weight-free direct-F16 operator test", "mode": MODE,
              "source_sha256": SOURCE_SHA256, "status": "failed"}
    def save():
        (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    try:
        command(["git", "clone", "-q", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git", str(ROOT)])
        command(["git", "checkout", "-q", COMMIT], cwd=ROOT)
        command(["git", "submodule", "update", "-q", "--init", "--depth", "1", "ggml"], cwd=ROOT)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if head != COMMIT:
            raise ValueError("repository commit mismatch")
        data = base64.b64decode(TEST_B64, validate=True)
        if hashlib.sha256(data).hexdigest() != TEST_SHA256:
            raise ValueError("embedded test digest mismatch")
        (ROOT / "tests/test_qwen35_direct_f16.cpp").write_bytes(data)
        if MODE == "green":
            source = base64.b64decode(SOURCE_B64, validate=True)
            if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
                raise ValueError("embedded source digest mismatch")
            (ROOT / "src/qwen35_linear.cpp").write_bytes(source)
        elif MODE != "red" or SOURCE_B64 or SOURCE_SHA256:
            raise ValueError("invalid test mode")
        cmake = ROOT / "CMakeLists.txt"
        text = cmake.read_text()
        anchor = 'target_link_libraries(test-qwen35-weight-cache PRIVATE crispembed-qwen35-delta crispembed-core)'
        if text.count(anchor) != 1:
            raise ValueError("CMake anchor is not unique")
        cmake.write_text(text.replace(anchor, anchor + '\nadd_executable(test-qwen35-direct-f16 tests/test_qwen35_direct_f16.cpp)\ntarget_link_libraries(test-qwen35-direct-f16 PRIVATE crispembed-qwen35-delta crispembed-core)'))
        report["phase"] = "build"
        save()
        command(["cmake", "-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF"])
        command(["cmake", "--build", str(BUILD), "--parallel", "4", "--target", "test-qwen35-direct-f16"])
        report["phase"] = "test"
        save()
        result = subprocess.run([str(BUILD / "test-qwen35-direct-f16")], text=True, capture_output=True)
        (OUT / "test.stdout").write_text(result.stdout)
        (OUT / "test.stderr").write_text(result.stderr)
        report.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                      expected_failure=(result.returncode != 0 and "FAIL direct multi-token" in result.stderr),
                      status="complete", phase="complete")
        save()
        report["expected_result"] = (report["expected_failure"] if MODE == "red" else result.returncode == 0)
        save()
        print("DIRECT_F16_RESULT", json.dumps(report), flush=True)
        return 0 if report["expected_result"] else 1
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == "__main__":
    sys.exit(main())
