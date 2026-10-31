#!/usr/bin/env python3
"""Fail closed unless a stable package matches this run's published assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ASSETS = {
    "linux-x86_64": "crispembed-linux-x86_64.tar.gz",
    "linux-arm64": "crispembed-linux-arm64.tar.gz",
    "macos-arm64": "crispembed-macos-arm64.tar.gz",
    "windows-x86_64": "crispembed-windows-x86_64.zip",
    "android-arm64-v8a": "crispembed-android-arm64-v8a.tar.gz",
    "android-armeabi-v7a": "crispembed-android-armeabi-v7a.tar.gz",
    "ios-arm64": "crispembed-ios-arm64.tar.gz",
}
STABLE_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")


def sha256(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if not size:
        raise ValueError(f"empty asset: {path}")
    return digest.hexdigest()


def check(args):
    version = args.version_file.read_text(encoding="utf-8").strip()
    if not STABLE_VERSION.fullmatch(version) or args.tag != "v" + version:
        raise ValueError("tag must exactly match a stable VERSION")
    pubspec = args.pubspec.read_text(encoding="utf-8")
    versions = re.findall(r"^version:[ \t]*([^\s#]+)[ \t]*(?:#.*)?$", pubspec, re.MULTILINE)
    names = re.findall(r"^name:[ \t]*([^\s#]+)[ \t]*(?:#.*)?$", pubspec, re.MULTILINE)
    if versions != [version] or names != ["crispembed"]:
        raise ValueError("pubspec name or version does not match VERSION")
    headers = re.findall(r"^##[ \t]+([^\s]+)[ \t]*$", args.changelog.read_text(encoding="utf-8"), re.MULTILINE)
    if not headers or headers[0] != version:
        raise ValueError("the current package changelog entry is missing")
    release = json.loads(args.release_json.read_text(encoding="utf-8"))
    if release.get("tag_name") != args.tag or release.get("draft") is not False or release.get("prerelease") is not False:
        raise ValueError("GitHub release tag or publication state is invalid")
    assets = release.get("assets")
    if not isinstance(assets, list):
        raise ValueError("GitHub release asset list is missing")
    for artifact, filename in ASSETS.items():
        matching = [entry for entry in assets if isinstance(entry, dict) and entry.get("name") == filename]
        if len(matching) != 1 or matching[0].get("state") != "uploaded":
            raise ValueError(f"missing, duplicated, or unuploaded release asset: {filename}")
        built = sha256(args.workflow_assets / artifact / filename)
        published = sha256(args.downloaded_assets / filename)
        if built != published:
            raise ValueError(f"release asset differs from this workflow run: {filename}")
        print(f"{filename}: sha256 {built}")
    print(f"pub.dev preflight passed for {args.tag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    for option in ("version-file", "pubspec", "changelog", "release-json", "workflow-assets", "downloaded-assets"):
        parser.add_argument("--" + option, type=Path, required=True)
    try:
        check(parser.parse_args())
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"pub.dev preflight failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
