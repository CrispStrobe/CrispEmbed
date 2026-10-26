#!/usr/bin/env python3
"""Hosted-only, original-first MI-GAN capture and optional lossless conversion."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from migan_reference_contract import load_contract, sha256


def download(url, path, expected_bytes, expected_sha=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("download destination already exists")
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    count = 0
    request = urllib.request.Request(url, headers={"User-Agent": "CrispEmbed-MIGAN-reference/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as stream:
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ValueError("HTML download response; checkpoint confirmation/auth must not be treated as weights")
        for block in iter(lambda: response.read(1024 * 1024), b""):
            count += len(block)
            if count > expected_bytes:
                raise ValueError("download exceeds pinned byte count")
            digest.update(block)
            stream.write(block)
    if count != expected_bytes or (expected_sha is not None and digest.hexdigest() != expected_sha):
        raise ValueError("download bytes/hash differ")
    temporary.rename(path)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default="", help="Reviewed digest on subsequent captures; empty only for first observation")
    parser.add_argument("--convert-after-original", action="store_true")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise RuntimeError("Run model download/capture only on hosted GitHub Actions, not the shared development host")
    args.output.mkdir(parents=True, exist_ok=False)
    review = args.output / "review"
    review.mkdir()
    shutil.copy2(ROOT / "docs/migan-reference-contract.json", review / "contract.json")
    contract = load_contract()
    source = args.output / "source"
    upstream = contract["original_source"]
    for name, record in upstream["files"].items():
        download(f"https://raw.githubusercontent.com/Picsart-AI-Research/MI-GAN/{upstream['commit']}/{name}",
                 source / name, record["bytes"], record["sha256"])
    for notice in ("LICENSE", "LICENSE-WEIGHTS"):
        shutil.copy2(source / notice, review / notice)
    checkpoint_record = contract["checkpoint"]
    checkpoint = args.output / checkpoint_record["filename"]
    observed = download(checkpoint_record["download_url"], checkpoint, checkpoint_record["declared_bytes"], args.checkpoint_sha256 or None)
    if not zipfile.is_zipfile(checkpoint):
        raise ValueError("checkpoint is not a ZIP; do not deserialize an HTML/error page")
    with zipfile.ZipFile(checkpoint) as archive:
        if archive.testzip() is not None:
            raise ValueError("checkpoint ZIP integrity check failed")
    identity = {"schema": "crispembed-migan-checkpoint-first-observation-v1", "filename": checkpoint.name,
                "bytes": checkpoint.stat().st_size, "sha256": observed, "upstream_sha256_declared": False,
                "source_commit": upstream["commit"], "reviewed_digest_supplied": bool(args.checkpoint_sha256),
                "github_sha": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                "artifact_publication_authorized_by_this_receipt": False}
    # Freeze transport identity before any framework load or reference execution.
    (args.output / "checkpoint-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    shutil.copy2(args.output / "checkpoint-identity.json", review / "checkpoint-identity.json")
    original = args.output / "original"
    subprocess.run([sys.executable, str(ROOT / "tools/dump_migan_reference.py"),
                    "--source-dir", str(source), "--checkpoint", str(checkpoint),
                    "--checkpoint-sha256", observed, "--output", str(original), "--threads", "4"], check=True)
    # The original is complete before the converter is imported or invoked.
    if args.convert_after_original:
        subprocess.run([sys.executable, str(ROOT / "models/convert-migan-to-gguf.py"),
                        "--source-dir", str(source), "--checkpoint", str(checkpoint),
                        "--checkpoint-sha256", observed, "--original-report", str(original / "report.json"),
                        "--output", str(args.output / "migan-256-places2-f32-experimental.gguf")], check=True)
    for path in (args.output / "checkpoint-identity.json", original / "report.json", original / "tensor-inventory.json",
                 args.output / "migan-256-places2-f32-experimental.json"):
        if path.exists():
            shutil.copy2(path, review / path.name)
    for case in contract["fixtures"]:
        destination = review / case["id"]
        destination.mkdir()
        for name in ("input.png", "fill-mask.png", "generated.png", "composited.png"):
            shutil.copy2(original / case["id"] / name, destination / name)
    members = [{"path": str(path.relative_to(args.output)), "bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in sorted(args.output.rglob("*")) if path.is_file()]
    (args.output / "artifact-members.json").write_text(json.dumps(members, indent=2) + "\n")
    shutil.copy2(args.output / "artifact-members.json", review / "artifact-members.json")


if __name__ == "__main__":
    main()
