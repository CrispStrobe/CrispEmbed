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


# "native" is the Linux desktop build. "native_windows"/"native_macos" add the
# other two desktop legs to the same matrix job and always imply "native".
LANES = (
    "native", "native_windows", "native_macos", "native_tests", "mobile", "sibling",
    "wasm_ocr", "wasm_embed", "rust", "inpaint_python", "inpaint_native", "bundle",
)
DESKTOP = (("native", "linux-x86_64"), ("native_macos", "macos-arm64"), ("native_windows", "windows-x86_64"))
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
    # Imported or read by the contracts above (see the dependency-closure test).
    "tools/dump_migan_reference.py", "tools/dump_migan512_reference.py",
    "tools/inpaint_hash_evidence.py", "tools/latent_inpaint_original_import.py",
    "docs/migan-reference-contract.json", "docs/teleocr-publication-v1.json",
    "docs/*inpaint*contract*.json", "docs/migan512-reference-contract.json",
    "docs/inpaint-publication-v1.json", "tests/results/migan/original-reference.json",
)
INPAINT_NATIVE = (
    "src/inpaint.*", "src/inpaint_*.cpp", "src/inpaint_*.h", "src/migan.*",
    "src/latent_inpaint*", "examples/cli/inpaint_cli.*",
    "tests/*inpaint*.cpp", "tests/*migan*.cpp", "tests/*migan*.h", "tests/inpaint_cli_driver.cpp",
    "tests/test_inpaint_cli.py", "tests/inpaint_video_quality.py",
    "tests/test_inpaint_model.py",
)
# Python entry points the Linux desktop job runs after its native build, plus
# what they import or read. Linux only: they are not run on Windows/macOS.
NATIVE_PYTHON = (
    "tests/test_ultrabert_api_log.py", "tests/test_vendored_kaggle_harness.py",
    "kaggle_harness.py", "tools/kaggle/*/kaggle_harness.py",
    "tests/test_ultrabert_*.py", "tools/ultrabert*.py", "tools/dump_ultrabert*.py", "tools/ci-heavy/*ultrabert*.py",
    "models/convert-ultrabert-to-gguf.py", "tests/results/ultrabert/*.json",
)
# Paths that no ordinary build lane compiles, packages or reads once the
# consumed exceptions above have been matched: evidence receipts, other
# workflows and their private actions, machine-readable docs, and Python
# tooling. Each has its own path-filtered workflow where one exists. A guard
# test recomputes the import/read closure of every Python entry point the
# ordinary lanes run, so a new dependency cannot silently fall in here.
NOT_BUILT = (
    "tests/results/*", "docs/*", "handover-prompts/*", "hf_readmes/*",
    ".github/workflows/*", ".github/actions/scifact-setup/*",
    "tools/ci-heavy/*", "tools/kaggle/*", "tools/colab/*", "tools/*.sh",
    "tools/*.py", "models/*", "tests/*.py",
)
# scripts/vendor_rust_sources.sh packages every C/C++ file under tests/ and
# tools/ into the Rust crate. Inside the folders above CMake compiles none of
# them (snapshots, Kaggle benches), so they select only the crate package.
COMPILED_EXTENSIONS = (".c", ".cc", ".cpp", ".h", ".hpp")
# Compiled sources outside the shared core. A PR gets the Linux build, the
# model-free suite and the MSVC leg (POSIX-only headers are the recurring
# cross-platform break); macOS joins when the file is Apple/Metal-specific. A
# push to main gets every lane. The public C ABI and its implementation keep
# the full fallback because Rust, Flutter and both WASM wrappers bind to them.
CXX_SOURCE = (
    "src/*", "examples/*", "tests/*.cpp", "tests/*.h", "tests/*.hpp", "tests/*.c",
    "tools/*.cpp", "tools/*.h", "tools/*.c",
)
# Compiled sources that an inpaint Python contract also names (source-identity
# lists); they keep their native lanes and add the Python contracts.
CXX_READ_BY_INPAINT_PYTHON = ("src/crispembed_diff.h",)
FULL_ABI = ("src/crispembed.h", "src/crispembed.cpp")
# Compile-time Apple/Metal markers. Comments that merely mention Metal do not
# change what the macOS compiler sees.
APPLE_MARKERS = ("__APPLE__", "__MACH__", "TARGET_OS_", "GGML_USE_METAL", "GGML_METAL", "ggml-metal", "ggml_backend_metal")
WORKFLOW_LANES = {
    ".github/workflows/build.yml": ("native", "native_windows", "native_macos", "native_tests", "mobile", "sibling"),
    ".github/workflows/build-wasm.yml": ("wasm_ocr",),
    ".github/workflows/build-wasm-embed.yml": ("wasm_embed",),
    ".github/workflows/rust-crates.yml": ("rust",),
    ".github/workflows/inpaint-cli.yml": ("inpaint_python", "inpaint_native"),
}


def matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def apple_specific(path: str, root: Path = ROOT) -> bool:
    if path.endswith((".m", ".mm", ".metal")) or "metal" in path.lower() or "apple" in path.lower():
        return True
    try:
        text = (root / path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False  # deleted/renamed away: the main push still runs everything
    return any(marker in text for marker in APPLE_MARKERS)


def selection(paths: list[str], *, force_full: str | None = None, event_name: str = "pull_request") -> dict:
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
        if path in FULL_ABI:
            reason = "public C ABI changed"
            break
        if path in WORKFLOW_LANES:
            for lane in WORKFLOW_LANES[path]:
                lanes[lane] = True
        elif matches(path, INPAINT_PYTHON):
            lanes["inpaint_python"] = True
        elif matches(path, INPAINT_NATIVE):
            lanes["inpaint_python"] = lanes["inpaint_native"] = True
            # Source edits also receive the Linux desktop backend build; the
            # inpaint workflow itself compiles them on all three platforms.
            if path.startswith(("src/", "examples/")):
                lanes["native"] = True
        elif path == "examples/cli/main.cpp":
            lanes["native"] = lanes["inpaint_python"] = lanes["inpaint_native"] = True
            lanes["native_tests"] = True
        elif path.startswith(("crispembed/", "crispembed-sys/")) or path in ("Cargo.toml", "Cargo.lock", "scripts/vendor_rust_sources.sh"):
            lanes["rust"] = True
        elif path == "build-embed-wasm.sh" or path == "wasm/embed_wrapper.c":
            lanes["wasm_embed"] = True
        elif path in ("build-wasm.sh", "wasm/ocr_wrapper.c", "wasm/crispembed-ocr.js", "tools/wasm_size_report.py") or path.startswith(("examples/wasm-ocr/", "tests/wasm-browser/")) or matches(path, ("tests/test_wasm_ocr*.js",)):
            lanes["wasm_ocr"] = True
        elif path == "scripts/check-cpu-baseline.py":
            lanes["native"] = lanes["native_windows"] = lanes["native_macos"] = True
        elif matches(path, NATIVE_PYTHON):
            lanes["native"] = lanes["native_tests"] = True
        elif matches(path, NOT_BUILT):
            if path.endswith(COMPILED_EXTENSIONS) and path.startswith(("tests/", "tools/")):
                lanes["rust"] = True
        elif matches(path, CXX_SOURCE) or ("/" not in path and path.endswith(".cpp")):
            if event_name == "push":
                # Main gets complete merged-source proof for every code change
                # (mobile, sibling, WASM, Rust package, all desktops).
                reason = "compiled source changed on main; full CI required"
                break
            lanes["native"] = lanes["native_tests"] = lanes["native_windows"] = True
            if path in CXX_READ_BY_INPAINT_PYTHON:
                lanes["inpaint_python"] = True
            if apple_specific(path):
                lanes["native_macos"] = True
        else:
            # Unreviewed paths (fixtures, scripts, flutter, python package,
            # wrappers, ...). New functionality cannot silently miss CI.
            reason = "unclassified change; full CI required"
            break
    if reason:
        lanes = dict.fromkeys(LANES, True)
    lanes["inpaint_python"] |= lanes["inpaint_native"]
    lanes["native"] |= lanes["native_windows"] or lanes["native_macos"]
    lanes["bundle"] = lanes["native"] and lanes["native_windows"] and lanes["native_macos"] and lanes["mobile"]
    return {"lanes": lanes, "full": bool(reason), "reason": reason or "reviewed dependency map", "paths": sorted(set(paths))}


def desktop_platforms(lanes: dict) -> list[str]:
    """Desktop matrix legs for build.yml, in a fixed order."""
    return [name for lane, name in DESKTOP if lanes[lane]]


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
        return selection(changed_paths(event_name, event, repo), event_name=event_name)
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
            handle.write("desktop=" + json.dumps(desktop_platforms(report["lanes"])) + "\n")
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
