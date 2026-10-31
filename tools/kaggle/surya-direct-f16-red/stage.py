"""Stage the weight-free direct F16 RED probe without modifying the base commit."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--green", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("commit must be a full lowercase SHA")
    if args.directory.exists():
        parser.error("stage directory already exists")
    test = (ROOT / "tests/test_qwen35_direct_f16.cpp").read_bytes()
    source = (ROOT / "src/qwen35_linear.cpp").read_bytes() if args.green else b""
    kernel = (HERE / "kernel.py").read_text()
    for placeholder, value in (("@COMMIT@", args.commit), ("@TEST_B64@", base64.b64encode(test).decode()),
                               ("@TEST_SHA256@", sha(test)),
                               ("@SOURCE_B64@", base64.b64encode(source).decode()),
                               ("@SOURCE_SHA256@", sha(source) if source else ""),
                               ("@MODE@", "green" if args.green else "red")):
        if kernel.count(placeholder) != 1:
            raise ValueError("invalid placeholder: " + placeholder)
        kernel = kernel.replace(placeholder, value)
    compile(kernel, "kernel.py", "exec")
    metadata = (HERE / "kernel-metadata.json").read_bytes()
    json.loads(metadata)
    args.directory.mkdir(parents=True)
    (args.directory / "kernel.py").write_text(kernel)
    (args.directory / "kernel-metadata.json").write_bytes(metadata)
    (args.directory / "test.cpp").write_bytes(test)
    print(json.dumps({"commit": args.commit, "test_sha256": sha(test),
                      "source_sha256": sha(source) if source else None,
                      "mode": "green" if args.green else "red",
                      "kernel_sha256": sha(kernel.encode()), "metadata_sha256": sha(metadata)}, indent=2))


if __name__ == "__main__":
    main()
