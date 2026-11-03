#!/usr/bin/env python3
"""Report recorded, owned-main CI outcomes without inferring current-tree coverage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPOSITORY = "CrispStrobe/CrispEmbed"
REPOSITORY_ID = 1210669686
PREFIX = f"repos/{REPOSITORY}"
WATCHED = (
    (262006165, "build.yml", "Build"),
    (292897227, "build-wasm.yml", "Build WASM"),
    (306995154, "deploy-pages.yml", "Deploy WASM demo (GitHub Pages)"),
    (305945697, "lint.yml", "Lint"),
    (293073967, "build-wasm-embed.yml", "Build WASM (text embeddings)"),
)
MAIN_EVENTS = {"push", "workflow_dispatch", "schedule"}
FAILED = {"failure", "timed_out", "action_required", "startup_failure"}
SHA = re.compile(r"[0-9a-f]{40}\Z")
PER_PAGE = 100
MAX_PAGES = 10  # GitHub's filtered workflow-run searches return at most 1,000 rows.


class Unverified(ValueError):
    """The API did not establish a usable recorded outcome."""


def positive_int(value):
    return type(value) is int and value > 0


def timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise Unverified("missing or malformed UTC timestamp")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Unverified("malformed UTC timestamp") from exc
    return value


def sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value) or value == "0" * 40:
        raise Unverified("missing or malformed commit SHA")
    return value


def own_repository(value):
    if not isinstance(value, dict) or "id" not in value or "full_name" not in value:
        raise Unverified("missing repository identity")
    return value["id"] == REPOSITORY_ID and value["full_name"] == REPOSITORY


class GitHub:
    def __init__(self, token):
        if not token:
            raise Unverified("GH_TOKEN is required")
        self.token = token

    def __call__(self, path):
        if path != PREFIX and not path.startswith(PREFIX + "/"):
            raise Unverified("API request escaped the pinned repository")
        request = Request("https://api.github.com/" + path, headers={
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        try:
            with urlopen(request, timeout=20) as response:
                return json.load(response)
        except HTTPError as exc:
            raise Unverified(f"GitHub API returned HTTP {exc.code}") from exc
        except (URLError, OSError, ValueError) as exc:
            raise Unverified("GitHub API request or JSON decoding failed") from exc


def owned_run(raw, workflow):
    if not isinstance(raw, dict) or not isinstance(raw.get("event"), str):
        raise Unverified("malformed workflow-run record")
    # PR heads named main, including forks, are not validation of our main.
    if raw["event"] not in MAIN_EVENTS:
        return None
    if not isinstance(raw.get("head_branch"), str):
        raise Unverified("missing or malformed head branch")
    if raw["head_branch"] != "main":
        return None
    if not own_repository(raw.get("repository")):
        raise Unverified("workflow-run repository does not match the pinned repository")
    if not own_repository(raw.get("head_repository")):
        return None
    workflow_id, filename, _ = workflow
    path = raw.get("path")
    if (raw.get("workflow_id") != workflow_id or not isinstance(path, str)
            or path.split("@", 1)[0] != ".github/workflows/" + filename):
        raise Unverified("workflow-run identity does not match the declared workflow")
    if not positive_int(raw.get("id")) or not positive_int(raw.get("run_attempt")):
        raise Unverified("missing or malformed run ID/attempt")
    if not isinstance(raw.get("status"), str) or "conclusion" not in raw:
        raise Unverified("missing run status/conclusion")
    if raw["conclusion"] is not None and not isinstance(raw["conclusion"], str):
        raise Unverified("malformed run conclusion")
    if raw["status"] != "completed" and raw["conclusion"] is not None:
        raise Unverified("unfinished run has a conclusion")
    return {
        "id": raw["id"], "attempt": raw["run_attempt"], "sha": sha(raw.get("head_sha")),
        "created_at": timestamp(raw.get("created_at")), "event": raw["event"],
        "status": raw["status"], "conclusion": raw["conclusion"],
    }


def pages(api, workflow, params):
    """Fully consume a bounded query, rejecting truncation and unstable pagination."""
    seen, expected = set(), None
    for page in range(1, MAX_PAGES + 1):
        query = urlencode({**params, "branch": "main", "per_page": PER_PAGE, "page": page})
        data = api(f"{PREFIX}/actions/workflows/{workflow[0]}/runs?{query}")
        if not isinstance(data, dict) or type(data.get("total_count")) is not int:
            raise Unverified("malformed workflow-run page")
        total, runs = data["total_count"], data.get("workflow_runs")
        if total < 0 or not isinstance(runs, list) or len(runs) > PER_PAGE:
            raise Unverified("malformed workflow-run page")
        if expected is not None and total != expected:
            raise Unverified("workflow-run pagination changed during the report")
        expected = total
        rows = []
        for raw in runs:
            if not isinstance(raw, dict) or not positive_int(raw.get("id")) or raw["id"] in seen:
                raise Unverified("malformed or duplicate paginated run")
            seen.add(raw["id"])
            row = owned_run(raw, workflow)
            if row is not None:
                rows.append(row)
        if len(seen) > total or (not runs and len(seen) < total):
            raise Unverified("incomplete workflow-run pagination")
        yield rows
        if len(seen) == total:
            return
    raise Unverified("workflow-run query exceeds the bounded pagination limit")


def newest_run(api, workflow, captured_at, *, successful=False):
    # Find an owned-main lower bound without scanning the repository's lifetime
    # history. Then fully paginate ALL runs at/after that bound. Any seed works:
    # no assumption about API ordering or a rerun's start time is needed.
    params = {"created": "<=" + captured_at}
    if successful:
        params["status"] = "success"  # Historical context only; never the latest-attempt query.
    seed = None
    for rows in pages(api, workflow, params):
        if rows:
            seed = max(rows, key=lambda row: (row["created_at"], row["id"]))
            break
    if seed is None:
        return None
    params["created"] = seed["created_at"] + ".." + captured_at
    recent = [row for group in pages(api, workflow, params) for row in group]
    if not recent or any(not seed["created_at"] <= row["created_at"] <= captured_at for row in recent):
        raise Unverified("workflow-run date range changed or was incomplete")
    if successful and any(row["status"] != "completed" or row["conclusion"] != "success" for row in recent):
        raise Unverified("successful-run query returned an unsuccessful run")
    return max(recent, key=lambda row: (row["created_at"], row["id"]))


def report(api, captured_at=None):
    captured_at = timestamp(captured_at or datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
    result = {"repository": REPOSITORY, "captured_at": captured_at, "main_sha": None,
              "state": "UNVERIFIED", "workflows": [], "errors": []}
    try:
        if not own_repository(api(PREFIX)):
            raise Unverified("API repository does not match the pinned repository")
        branch = api(PREFIX + "/branches/main")
        if not isinstance(branch, dict) or branch.get("name") != "main" or not isinstance(branch.get("commit"), dict):
            raise Unverified("malformed main-branch response")
        result["main_sha"] = sha(branch["commit"].get("sha"))
    except Unverified as exc:
        result["errors"].append(str(exc))
        return result
    for workflow in WATCHED:
        row = {"workflow": workflow[2], "path": ".github/workflows/" + workflow[1],
               "state": "UNVERIFIED", "latest": None, "last_success": None, "reason": ""}
        try:
            latest = row["latest"] = newest_run(api, workflow, captured_at)
            if latest is None:
                raise Unverified("no owned-main run was found")
            if latest["status"] == "completed" and latest["conclusion"] == "success":
                row["last_success"] = latest
                row["state"] = "RECORDED_SUCCESS" if latest["sha"] == result["main_sha"] else "HISTORICAL_SUCCESS"
                row["reason"] = "latest recorded run succeeded; current-main coverage is not inferred"
            else:
                row["state"] = "FAILED" if latest["status"] == "completed" and latest["conclusion"] in FAILED else "UNVERIFIED"
                row["reason"] = f"latest recorded run is {latest['status']} / {latest['conclusion'] or 'no conclusion'}"
                row["last_success"] = newest_run(api, workflow, captured_at, successful=True)
        except Unverified as exc:
            if row["state"] != "FAILED":
                row["state"] = "UNVERIFIED"
            row["reason"] = (row["reason"] + "; " if row["reason"] else "") + str(exc)
        result["workflows"].append(row)
    states = {row["state"] for row in result["workflows"]}
    result["state"] = "FAILED" if "FAILED" in states else "UNVERIFIED" if "UNVERIFIED" in states else "RECORDED_SUCCESS"
    return result


def markdown(result):
    lines = ["## Recorded CI status", "", f"Outcome: **{result['state']}**.",
             f"Captured main: `{result['main_sha'] or 'unavailable'}` at {result['captured_at']}.", "",
             "This reports recorded workflow outcomes, not coverage of every change at current main.",
             "Older successes are historical; path-selective CI need not run every workflow on each commit.", "",
             "| Workflow | Outcome | Latest run / source | Last recorded success / source |",
             "|---|---|---|---|"]
    def link(run):
        return "none" if run is None else f"[{run['id']}](https://github.com/{REPOSITORY}/actions/runs/{run['id']}) / `{run['sha']}`"
    for row in result["workflows"]:
        lines.append(f"| {row['workflow']} | {row['state']} | {link(row['latest'])} | {link(row['last_success'])} |")
    lines += ["", *[f"- {row['workflow']}: {row['reason']}" for row in result["workflows"]],
              *[f"- {error}" for error in result["errors"]], ""]
    return "\n".join(lines)


def main(argv=None, *, api=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    args = parser.parse_args(argv)
    try:
        if os.environ.get("GITHUB_REPOSITORY", REPOSITORY) != REPOSITORY:
            raise Unverified("workflow context does not match the pinned repository")
        outcome = report(api or GitHub(os.environ.get("GH_TOKEN")))
    except Unverified as exc:
        outcome = {"repository": REPOSITORY, "captured_at": datetime.now(timezone.utc).isoformat(),
                   "main_sha": None, "state": "UNVERIFIED", "workflows": [], "errors": [str(exc)]}
    print(json.dumps(outcome, indent=2))
    print(markdown(outcome))
    if args.summary:
        try:
            with Path(args.summary).open("a", encoding="utf-8") as stream:
                stream.write(markdown(outcome))
        except OSError:
            print("UNVERIFIED: could not write the workflow summary")
            return 1
    return 0 if outcome["state"] == "RECORDED_SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
