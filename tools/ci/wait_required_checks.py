#!/usr/bin/env python3
"""Wait until every check on ONE pull-request head commit is green; exit nonzero otherwise.

    python3 tools/ci/wait_required_checks.py --pr 123 [--repo owner/name] [--sha <40-hex>]
        [--timeout 7200] [--interval 30] [--settle 60] [--allow-stalled "<check name>"]...
    # then merge only that commit:
    gh pr merge 123 --merge --match-head-commit <sha>

Why: a merge was once made on a stale poll that returned before the new head's
checks had even been created. This helper pins the head SHA (the --sha you
reviewed, or the PR head at start), re-reads the PR head on every poll and right
before exiting, and fails if it moved. It reads both check runs and commit
statuses for exactly that SHA, keeps the newest run per check name, and
requires the branch-protection required contexts to be present when the API
exposes them.

Green means: at least one check exists, and every check run, commit status and
GitHub Actions workflow run for that SHA is success or skipped. Workflow runs
count because a run can stay queued or pending without having created any
check run yet.
queued / in_progress / pending / waiting keep it waiting. cancelled, failure,
timed_out, action_required, neutral, stale and startup_failure fail at once.
Once everything is green it waits --settle seconds and checks again, so late
workflows (path-filtered, chained) are not missed.

--allow-stalled NAME ignores a check with exactly that name ONLY while it is
still queued/pending (a known-stalled runner queue). It never turns a failed or
cancelled check green. Use it only when the owner has accepted that job as
stalled, and say so in the PR.

Exit codes: 0 green, 1 a check failed / none exist, 2 timeout, 3 head moved,
4 gh/API error.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

GREEN = {"success", "skipped"}
PENDING = {"queued", "in_progress", "pending", "waiting", "requested", "expected", None, ""}


class GitHubAPIError(RuntimeError):
    """Keep gh's error contract while exposing status for optional API reads."""

    def __init__(self, path: str, detail: str):
        detail = detail.strip()
        status = re.search(r"\bHTTP (\d{3})\b", detail, re.I)
        self.status_code = int(status.group(1)) if status else None
        self.rate_limited = bool(re.search(r"rate[\s_-]*limit|retry[\s_-]*after|abuse", detail, re.I))
        super().__init__(f"gh api {path}: {detail[:300]}")


def gh_api(path: str, paginate: bool = False):
    cmd = ["gh", "api", "-H", "Accept: application/vnd.github+json", path]
    if paginate:
        cmd[2:2] = ["--paginate", "--slurp"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise GitHubAPIError(path, r.stderr or r.stdout)
    return json.loads(r.stdout or "null")


def current_repo() -> str:
    r = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                       capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        sys.exit("wait_required_checks: pass --repo owner/name (gh repo view failed)")
    return r.stdout.strip()


def pr_info(repo: str, pr: int) -> dict:
    p = gh_api(f"repos/{repo}/pulls/{pr}")
    return {"head": p["head"]["sha"], "base": p["base"]["ref"], "state": p["state"], "merged": p.get("merged")}


def required_contexts(repo: str, base: str) -> list[str] | None:
    try:
        r = gh_api(f"repos/{repo}/branches/{base}/protection/required_status_checks")
    except GitHubAPIError as e:
        if e.status_code in {403, 404} and not e.rate_limited:
            return None  # no protection, or not visible with this token
        raise  # throttling, service and network errors must not erase requirements
    names = set(r.get("contexts") or [])
    names.update(c.get("context") for c in r.get("checks") or [] if c.get("context"))
    return sorted(names)


def checks_for(repo: str, sha: str) -> dict[str, str]:
    """name -> state word for the newest check run / status per name."""
    out: dict[str, tuple] = {}
    pages = gh_api(f"repos/{repo}/commits/{sha}/check-runs?per_page=100", paginate=True)
    for page in pages:
        for run in page.get("check_runs", []):
            key = run.get("started_at") or "", run.get("id") or 0
            state = run.get("conclusion") if run.get("status") == "completed" else run.get("status")
            # Several workflows may each have a job with the same name; keep them apart
            # by check suite, and keep only the newest run (re-runs) within a suite.
            name = run["name"]
            suite = (run.get("check_suite") or {}).get("id")
            ident = name if suite is None else f"{name} [suite {suite}]"
            if ident not in out or key > out[ident][0]:
                out[ident] = (key, state)
    # A workflow run can sit in queued/pending (runner queue, concurrency group)
    # before it has created any check run, so check runs alone can look green
    # while whole workflows have not started. Count every Actions run for the SHA.
    runs = gh_api(f"repos/{repo}/actions/runs?head_sha={sha}&per_page=100", paginate=True)
    newest: dict[str, tuple] = {}
    for page in runs:
        for wr in page.get("workflow_runs", []):
            # Attempts belong to one run: an old run's attempt 2 must never
            # supersede a newer run's queued/failed attempt 1 at the same SHA.
            key = wr.get("created_at") or "", wr.get("id") or 0, wr.get("run_attempt") or 0
            wf = f"workflow {wr.get('name') or wr.get('workflow_id')} [{wr.get('event')}]"
            if wf not in newest or key > newest[wf][0]:
                state = wr.get("conclusion") if wr.get("status") == "completed" else wr.get("status")
                newest[wf] = (key, state)
    for wf, (_, state) in newest.items():
        out[wf] = (("", 0), state)
    status = gh_api(f"repos/{repo}/commits/{sha}/status?per_page=100")
    for st in status.get("statuses", []):  # newest first per the API
        name = st["context"]
        if name not in out:
            out[name] = (("", 0), st["state"])
    return {k: v[1] for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pr", type=int, required=True)
    ap.add_argument("--repo")
    ap.add_argument("--sha", help="head commit you reviewed; default: the PR head at start")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--settle", type=int, default=60, help="re-check delay after first all-green")
    ap.add_argument("--allow-stalled", action="append", default=[], metavar="NAME")
    a = ap.parse_args(argv)
    repo = a.repo or current_repo()
    try:
        info = pr_info(repo, a.pr)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 4
    sha = a.sha or info["head"]
    if info["merged"] or info["state"] != "open":
        print(f"PR #{a.pr} is {('merged' if info['merged'] else info['state'])}")
        return 1
    print(f"waiting on {repo}#{a.pr} head {sha}", flush=True)
    t0, green_since = time.time(), None
    while True:
        try:
            info = pr_info(repo, a.pr)
            if info["head"] != sha:
                print(f"HEAD MOVED: PR head is now {info['head']}, expected {sha}; re-review and rerun")
                return 3
            states = checks_for(repo, sha)
            required = required_contexts(repo, info["base"]) or []
        except RuntimeError as e:
            print(e, file=sys.stderr)
            return 4
        present = {k.split(" [suite ")[0] for k in states}
        for name in required:
            if name not in present:
                states[name] = "expected"
        bad = {n: s for n, s in states.items() if s not in GREEN and s not in PENDING}
        def allowed(n, s):
            return s in {"queued", "pending"} and n.split(" [suite ")[0] in a.allow_stalled
        pending = {n: s for n, s in states.items() if s in PENDING and not allowed(n, s)}
        stalled = sorted(n for n, s in states.items() if allowed(n, s))
        el = time.time() - t0
        if bad:
            print("NOT GREEN:", json.dumps(bad, sort_keys=True))
            return 1
        if not states:
            print(f"no checks yet on {sha[:12]} ({el:.0f}s)", flush=True)
            green_since = None
        elif len(stalled) == len(states):
            print(f"no non-stalled green checks yet on {sha[:12]} ({el:.0f}s)", flush=True)
            green_since = None
        elif pending:
            print(f"pending {len(pending)}/{len(states)}: {', '.join(sorted(pending)[:8])} ({el:.0f}s)", flush=True)
            green_since = None
        elif green_since is None:
            green_since = time.time()
            print(f"all {len(states)} green; settling {a.settle}s", flush=True)
        elif time.time() - green_since >= a.settle:
            # The head may have moved while the check/status APIs were read.
            # Fail closed on this last read; --match-head-commit also protects
            # the later merge from changes after this helper exits.
            try:
                final_info = pr_info(repo, a.pr)
            except RuntimeError as e:
                print(e, file=sys.stderr)
                return 4
            if final_info["head"] != sha:
                print(f"HEAD MOVED: PR head is now {final_info['head']}, expected {sha}; re-review and rerun")
                return 3
            if final_info["merged"] or final_info["state"] != "open":
                print(f"PR #{a.pr} is {('merged' if final_info['merged'] else final_info['state'])}")
                return 1
            if stalled:
                print("IGNORED (explicit --allow-stalled, still pending):", ", ".join(stalled))
            print(f"GREEN: {len(states) - len(stalled)} checks on {sha}")
            print(f"merge with: gh pr merge {a.pr} --repo {repo} --match-head-commit {sha} --merge")
            return 0
        if el > a.timeout:
            print(f"TIMEOUT after {el:.0f}s; pending: {json.dumps(pending, sort_keys=True)}")
            return 2
        time.sleep(min(a.interval, a.settle) if green_since else a.interval)


if __name__ == "__main__":
    sys.exit(main())
