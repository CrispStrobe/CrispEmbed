#!/usr/bin/env python3
"""Conservative, model-free routing for the ordinary CI build workflows.

Keep the dependency map small: only explicitly isolated changes skip lanes.
Unknown inputs or incomplete git history select every lane. This is not used by
the source-pinned model/evidence/publication workflows.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess
import sys


LANES = (
    "native", "native_tests", "mobile", "sibling", "wasm_ocr", "wasm_embed", "rust",
    "inpaint_python", "inpaint_native", "bundle",
)
SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
ROOT = Path(__file__).resolve().parents[1]

# These controls execute Python only. In particular, changes to their schemas
# must not be ignored merely because the JSON lives under docs/.
INPAINT_PYTHON = (
    "tools/prepare_migan_inpaint.py", "tools/migan_reference_contract.py",
    "tools/migan512_reference_contract.py", "tools/dump_migan_photo_reference.py",
    "tools/latent_inpaint_contract.py", "tools/dump_latent_inpaint_reference.py",
    "tools/ci-heavy/migan_native.py", "tools/ci-heavy/inpaint_gguf_publish.py",
    "models/convert-migan-to-gguf.py", "models/convert-latent-inpaint-to-gguf.py",
    "tests/test_prepare_migan_inpaint.py", "tests/test_inpaint_png.py",
    "tests/test_inpaint_engine_live_controls.py", "tests/test_inpaint_engine_live.py",
    "tests/test_migan_reference_contract.py", "tests/test_migan_native_driver.py",
    "tests/test_migan512_contract.py", "tests/test_latent_inpaint_contract.py",
    "tests/test_latent_inpaint_original_import.py", "tests/test_inpaint_gguf_publish.py",
    "tests/test_inpaint_video_quality.py",
    "docs/*inpaint*contract*.json", "docs/migan512-reference-contract.json",
    "docs/inpaint-publication-v1.json", "tests/results/migan/original-reference.json",
)
INPAINT_NATIVE = (
    "src/inpaint.*", "src/inpaint_*.cpp", "src/inpaint_*.h", "src/migan.*",
    "src/latent_inpaint*", "examples/cli/inpaint_cli.*",
    "tests/*inpaint*.cpp", "tests/*migan*.cpp", "tests/inpaint_cli_driver.cpp",
    "tests/test_inpaint_cli.py", "tests/inpaint_video_quality.py",
    "tests/test_inpaint_model.py",
)
WORKFLOW_LANES = {
    ".github/workflows/build.yml": ("native", "native_tests", "mobile", "sibling"),
    ".github/workflows/build-wasm.yml": ("wasm_ocr",),
    ".github/workflows/build-wasm-embed.yml": ("wasm_embed",),
    ".github/workflows/rust-crates.yml": ("rust",),
    ".github/workflows/inpaint-cli.yml": ("inpaint_python", "inpaint_native"),
}


def matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def selection(paths: list[str], *, force_full: str | None = None) -> dict:
    """Return the union of all changed paths; deletions/renames are inputs too."""
    lanes = dict.fromkeys(LANES, False)
    reason = force_full
    for path in paths:
        if not path or path.startswith("/") or ".." in path.split("/"):
            reason = "invalid changed path; full CI required"
            break
        if path.endswith(".md"):
            # The selected native/WASM builds do not consume Markdown. Known
            # machine-readable contracts/fixtures are treated separately.
            continue
        if path in ("CMakeLists.txt", "VERSION", "LICENSE", ".gitmodules", "ggml") or path.startswith(("cmake/", "ggml/", "src/core/")):
            reason = "shared build/runtime dependency changed"
            break
        if path in ("tools/ci_selection.py", "tools/ci_cpu_signature.py", "tests/test_ci_selection.py") or path.startswith((".github/actions/ci-select/", ".github/actions/compiler-cache/")):
            reason = "CI selection or compiler-cache implementation changed"
            break
        if path in WORKFLOW_LANES:
            for lane in WORKFLOW_LANES[path]:
                lanes[lane] = True
        elif matches(path, INPAINT_PYTHON):
            lanes["inpaint_python"] = True
        elif matches(path, INPAINT_NATIVE):
            lanes["inpaint_python"] = lanes["inpaint_native"] = True
            # Source edits also receive the ordinary desktop backend build.
            # Tests alone use the three-platform, real-CLI inpaint job.
            if path.startswith(("src/", "examples/")):
                lanes["native"] = True
        elif path == "examples/cli/main.cpp":
            lanes["native"] = lanes["inpaint_python"] = lanes["inpaint_native"] = True
            lanes["native_tests"] = True
        elif path.startswith(("crispembed/", "crispembed-sys/")) or path in ("Cargo.toml", "Cargo.lock", "scripts/vendor_rust_sources.sh"):
            lanes["rust"] = True
        elif path == "build-embed-wasm.sh" or path == "wasm/embed_wrapper.c":
            lanes["wasm_embed"] = True
        elif path in ("build-wasm.sh", "wasm/ocr_wrapper.c", "wasm/crispembed-ocr.js") or path.startswith(("examples/wasm-ocr/", "tests/wasm-browser/")) or matches(path, ("tests/test_wasm_ocr*.js",)):
            lanes["wasm_ocr"] = True
        else:
            # This includes unreviewed src/, test fixtures, docs/*.json and
            # workflow files. New functionality cannot silently miss CI.
            reason = "unclassified change; full CI required"
            break
    if reason:
        lanes = dict.fromkeys(LANES, True)
    lanes["inpaint_python"] |= lanes["inpaint_native"]
    lanes["bundle"] = lanes["native"] and lanes["mobile"]
    return {"lanes": lanes, "full": bool(reason), "reason": reason or "reviewed dependency map", "paths": sorted(set(paths))}


def git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=120,
    ).stdout


def commit(repo: Path, value: str) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value) or value == "0" * 40:
        raise ValueError("missing or invalid comparison commit")
    try:
        git(repo, "cat-file", "-e", value + "^{commit}")
    except subprocess.CalledProcessError:
        git(repo, "fetch", "--no-tags", "--no-recurse-submodules", "--depth=64", "origin", value)
        git(repo, "cat-file", "-e", value + "^{commit}")
    return value


def parse_name_status(data: bytes) -> list[str]:
    """Parse git --name-status -z without losing spaces, newlines or old names."""
    if not data:
        return []
    if not data.endswith(b"\0"):
        raise ValueError("truncated git diff")
    fields = data[:-1].decode("utf-8", errors="strict").split("\0")
    paths = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if status in ("A", "D", "M", "T"):
            count = 1
        elif re.fullmatch(r"[RC][0-9]{1,3}", status):
            count = 2
        else:
            raise ValueError("unrecognized git diff status")
        if index + count > len(fields) or any(not p for p in fields[index:index + count]):
            raise ValueError("incomplete git diff record")
        paths.extend(fields[index:index + count])
        index += count
    return paths


def changed_paths(event_name: str, event: dict, repo: Path) -> list[str]:
    if event_name == "pull_request":
        pr = event["pull_request"]
        base = commit(repo, pr["base"]["sha"])
        head = commit(repo, pr["head"]["sha"])
        try:
            git(repo, "merge-base", base, head)
        except subprocess.CalledProcessError:
            # A shallow merge checkout often lacks both parent histories. One
            # bounded fetch suffices for ordinary PRs; older branches fall back
            # to full CI if their merge base still cannot be established.
            git(repo, "fetch", "--no-tags", "--no-recurse-submodules", "--depth=64", "origin", base, head)
            git(repo, "merge-base", base, head)
        comparison = base + "..." + head
    elif event_name == "push":
        # Use the whole push, not HEAD^; a push can contain several commits.
        comparison = commit(repo, event["before"]) + ".." + commit(repo, event["after"])
    else:
        raise ValueError("event has no reviewed incremental comparison")
    return parse_name_status(git(repo, "diff", "--no-ext-diff", "--name-status", "-z", "--find-renames", comparison, "--"))


def select_event(event_name: str, event: dict, repo: Path = ROOT) -> dict:
    if event_name in ("workflow_dispatch", "merge_group"):
        return selection([], force_full="manual or merge-queue validation requests full CI")
    try:
        return selection(changed_paths(event_name, event, repo))
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError):
        # A missing object, API/event mismatch, encoding failure or shallow
        # history is never evidence that no work is necessary.
        return selection([], force_full="change detection incomplete; full CI required")


def check_needs(needs: dict, selector: str, requirements: dict[str, str]) -> list[str]:
    """Reject failed/cancelled jobs and every skip not authorized by selection."""
    errors = []
    if set(needs) != {selector, *requirements}:
        errors.append("aggregate dependency list does not match its checked jobs")
    selected = needs.get(selector, {})
    if selected.get("result") != "success":
        errors.append(f"{selector} did not succeed")
    outputs = selected.get("outputs", {})
    if any(outputs.get(lane) not in ("true", "false") for lane in LANES):
        errors.append("selection did not provide a complete set of lane outputs")
    for job, lane in requirements.items():
        choice = outputs.get(lane)
        if choice not in ("true", "false"):
            errors.append(f"missing/invalid selection for {job}")
            continue
        expected = "success" if choice == "true" else "skipped"
        actual = needs.get(job, {}).get("result")
        if actual != expected:
            errors.append(f"{job}: expected {expected}, observed {actual}")
    return errors


def emit(report: dict) -> None:
    print(json.dumps(report, indent=2))
    if report["reason"].startswith("change detection incomplete"):
        print("::warning::Change detection was incomplete; every CI lane is selected.")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            for name, value in {**report["lanes"], "full": report["full"]}.items():
                handle.write(f"{name}={str(value).lower()}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write("## CI selection\n\n" + report["reason"] + f" ({len(report['paths'])} changed paths).\n\n")
            handle.write("| Lane | Selected |\n| --- | --- |\n")
            for name, value in report["lanes"].items():
                handle.write(f"| {name} | {str(value).lower()} |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("select", help="select from the GitHub event and complete git diff")
    classify = commands.add_parser("classify", help="preview routing for explicit changed paths")
    classify.add_argument("paths", nargs="*")
    gate = commands.add_parser("check", help="validate an always-run aggregate job")
    gate.add_argument("--selector", required=True)
    gate.add_argument("jobs", nargs="+", help="job_id=selection_output")
    args = parser.parse_args()
    if args.command == "check":
        try:
            requirements = dict(item.split("=", 1) for item in args.jobs)
            errors = check_needs(json.loads(os.environ["CI_NEEDS"]), args.selector, requirements)
        except (KeyError, TypeError, ValueError):
            errors = ["invalid aggregate job configuration"]
        for error in errors:
            print("::error::" + error)
        if not errors:
            print("All selected jobs succeeded; every skipped job was explicitly unselected.")
        return int(bool(errors))
    if args.command == "classify":
        emit(selection(args.paths))
    else:
        try:
            event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
            report = select_event(os.environ.get("GITHUB_EVENT_NAME", ""), event)
        except (KeyError, ValueError, OSError):
            report = selection([], force_full="change detection incomplete; full CI required")
        emit(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
