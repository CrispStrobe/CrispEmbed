#!/usr/bin/env python3
"""Select the bounded LightOn4B original/native component reference job.

Only a complete pull-request or push diff can skip the reference job. Reuse the
ordinary CI helper for full PR merge-base comparisons, complete pushes, and both
sides of renames; this selector does not inspect only the tip commit. A manual,
merge-queue, unknown, malformed, or unreadable event always selects the job.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import sys

from ci_selection import changed_paths


ROOT = Path(__file__).resolve().parents[1]
EXACT = frozenset((
    ".github/workflows/controls.yml", "CMakeLists.txt", ".gitmodules", "VERSION", "ggml",
    "src/crispembed_diff.h", "tools/ci_selection.py", "tools/lighton4b_ci.py",
    "tests/test_lighton4b_ci.py", "tools/lighton4b-gdn-requirements.txt",
    "models/lightonocr3_4b_profile.py", "models/convert-qwen35-ocr-to-gguf.py",
    "tests/test_lighton4b_gdn_reference.py", "tests/test_qwen35_gdn_heads.cpp",
))
# CMake reads VERSION and its cmake modules/templates before any target build.
# Its C2PA module also discovers an optional local library for crispembed-core.
PREFIXES = (
    "ggml/", "cmake/", "third_party/c2pa/", "src/core/", "include/",
    "tests/fixtures/lightonocr3-4b-source/",
)
PATTERNS = (
    "src/qwen35_*.cpp", "src/qwen35_*.h", "tools/ci-heavy/lighton4b_*.py",
    "tests/test_lightonocr3_4b_export*.py",
)


def forced(reason: str) -> dict:
    return {"run_reference": True, "reason": reason, "paths": [], "relevant_paths": []}


def valid_path(path: str) -> bool:
    return (isinstance(path, str) and bool(path) and not path.startswith("/")
            and "\\" not in path and "\0" not in path
            and not (len(path) > 1 and path[1] == ":")
            and all(part not in ("", ".", "..") for part in path.split("/")))


def relevant(path: str) -> bool:
    return (path in EXACT
            or any(path == prefix[:-1] or path.startswith(prefix) for prefix in PREFIXES)
            or any(fnmatch.fnmatchcase(path, pattern) for pattern in PATTERNS))


def select_paths(paths: list[str]) -> dict:
    """Classify a successfully resolved complete diff, including old names."""
    if not isinstance(paths, list) or any(not valid_path(path) for path in paths):
        return forced("invalid changed-path evidence; reference required")
    paths = sorted(set(paths))
    affected = [path for path in paths if relevant(path)]
    return {
        "run_reference": bool(affected),
        "reason": ("reference dependency changed" if affected
                   else "complete diff contains no reference dependency change"),
        "paths": paths,
        "relevant_paths": affected,
    }


def select_event(event_name: str, event: dict, repo: Path = ROOT) -> dict:
    if event_name in ("workflow_dispatch", "merge_group"):
        return forced("manual or merge-queue validation requires the reference job")
    if event_name not in ("pull_request", "push"):
        return forced("event has no reviewed incremental comparison; reference required")
    try:
        if not isinstance(event, dict):
            raise ValueError("event must be an object")
        return select_paths(changed_paths(event_name, event, repo))
    except Exception as error:
        # A diff failure, including an unexpected helper error, is not evidence
        # that the reference job can be skipped. The type remains visible.
        return forced(f"change detection incomplete ({type(error).__name__}); reference required")


def emit(report: dict, github_output: Path) -> int:
    """Write one explicit decision; an output write failure fails the step."""
    print(json.dumps(report, indent=2))
    try:
        with github_output.open("a", encoding="utf-8") as handle:
            handle.write(f"run_reference={str(report['run_reference']).lower()}\n")
    except (OSError, ValueError) as error:
        print(f"::error::Cannot write explicit run_reference output ({type(error).__name__}).", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--event-path", type=Path, default=os.environ.get("GITHUB_EVENT_PATH") or None)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--github-output", type=Path, required=True,
                        help="explicit GitHub step output file; never selected implicitly from the environment")
    args = parser.parse_args(argv)
    if args.event_name not in ("pull_request", "push"):
        report = select_event(args.event_name, {}, args.repo)
    else:
        try:
            if args.event_path is None:
                raise ValueError("event path is missing")
            event = json.loads(args.event_path.read_text(encoding="utf-8"))
        except Exception as error:
            report = forced(f"event read incomplete ({type(error).__name__}); reference required")
        else:
            report = select_event(args.event_name, event, args.repo)
    return emit(report, args.github_output)


if __name__ == "__main__":
    raise SystemExit(main())
