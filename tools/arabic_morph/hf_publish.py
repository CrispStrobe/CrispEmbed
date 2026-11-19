#!/usr/bin/env python3
"""Upload the Camel Morph bundles to Hugging Face and verify an immutable readback.

Developer tooling. Reads the token from the HF_TOKEN environment variable and
never prints it.

    HF_TOKEN=... python tools/arabic_morph/hf_publish.py upload --dir out/ --repo cstr/camel-morph-msa-v1.0
    python tools/arabic_morph/hf_publish.py verify --dir out/ --repo cstr/camel-morph-msa-v1.0 --revision <sha>

verify downloads every local file from the given commit (anonymously unless
HF_TOKEN is set) into a fresh cache and compares size and SHA-256 with the
local tree, and checks that the remote tree holds no other files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def local_files(root: Path) -> dict[str, tuple[int, str]]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            out[p.relative_to(root).as_posix()] = (len(data), hashlib.sha256(data).hexdigest())
    return out


def cmd_upload(args):
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    info = api.upload_folder(folder_path=args.dir, repo_id=args.repo, repo_type="model",
                             commit_message=args.message)
    print(json.dumps({"repo": args.repo, "commit": info.oid, "url": info.commit_url}))


def cmd_verify(args):
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN") or None
    expected = local_files(Path(args.dir))
    api = HfApi(token=token)
    remote = sorted(f for f in api.list_repo_files(args.repo, revision=args.revision) if f != ".gitattributes")
    result = {"repo": args.repo, "revision": args.revision, "files": {}, "extra_remote": [], "missing_remote": []}
    result["extra_remote"] = [f for f in remote if f not in expected]
    result["missing_remote"] = [f for f in expected if f not in remote]
    with tempfile.TemporaryDirectory() as cache:
        for name, (size, sha) in expected.items():
            path = hf_hub_download(args.repo, name, revision=args.revision, cache_dir=cache, token=token)
            h = hashlib.sha256()
            n = 0
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
                    n += len(chunk)
            result["files"][name] = {"size": n, "sha256": h.hexdigest(), "match": n == size and h.hexdigest() == sha}
    result["all_match"] = (all(v["match"] for v in result["files"].values()) and not result["extra_remote"]
                           and not result["missing_remote"])
    print(json.dumps(result, indent=2))
    if not result["all_match"]:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("upload")
    p.add_argument("--dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--message", default="Camel Morph MSA v1.0 bundles (full, noprob)")
    p = sub.add_parser("verify")
    p.add_argument("--dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    args = ap.parse_args()
    {"upload": cmd_upload, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
