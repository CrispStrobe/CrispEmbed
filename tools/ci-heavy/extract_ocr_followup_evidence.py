#!/usr/bin/env python3
"""Hosted read-only extraction of the explicitly pinned corrected OCR archives."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
import urllib.request

import extract_ocr_nov4_evidence as archive

SELECTOR = "tools/ci-heavy/extract_ocr_followup_evidence.py"
HELPER_SHA256 = "5b52a1b14eb8658c3371fff920fade76c2df1346c0659b2d372322144bab84e6"
TARGETS = (
    {"label": "mineru2605-formula", "run_id": 41877117676, "artifact_id": 13347927913,
     "bytes": 136053962, "sha256": "7a052a5a73b47d90431c9fabdcea9af10ada9239a105ecd642d21d98a8d0705e",
     "source": "1bb0ddb1d9ac12ded51eb5bd0edd89be0a8259e6",
     "branch": "audit/mineru2605-aligned-q4-20261104", "conclusion": "success"},
    {"label": "mineru2605-japanese", "run_id": 41877121416, "artifact_id": 13348396823,
     "bytes": 249816387, "sha256": "8919d773dc544e30084d3b3da14519fdd884297548bdf5e91d08d993688a868d",
     "source": "1bb0ddb1d9ac12ded51eb5bd0edd89be0a8259e6",
     "branch": "audit/mineru2605-aligned-q4-20261104", "conclusion": "success"},
    {"label": "mineru2605-arabic", "run_id": 41877125216, "artifact_id": 13348311566,
     "bytes": 68835208, "sha256": "fad1d207c52c1b37e0c6f65c10ac10a701e31b4cec84c0392f85792d42a3ae95",
     "source": "1bb0ddb1d9ac12ded51eb5bd0edd89be0a8259e6",
     "branch": "audit/mineru2605-aligned-q4-20261104", "conclusion": "success"},
    {"label": "mineru2605-table", "run_id": 41877129385, "artifact_id": 13347724003,
     "bytes": 386986678, "sha256": "1efd735469a8956aa6364e6fbd9587acf9f93958a48c39ad6a51393edf1164b5",
     "source": "1bb0ddb1d9ac12ded51eb5bd0edd89be0a8259e6",
     "branch": "audit/mineru2605-aligned-q4-20261104", "conclusion": "success"},
    {"label": "mineru2605-layout-page", "run_id": 41877137082, "artifact_id": 13350809558,
     "bytes": 1549053766, "sha256": "701decb5245cc0bdf5e1993370eec2da95fa66e96e94043ac594c3947989aba4",
     "source": "1bb0ddb1d9ac12ded51eb5bd0edd89be0a8259e6",
     "branch": "audit/mineru2605-aligned-q4-20261104", "conclusion": "success"},
)


def validate_metadata(target, run, artifact):
    archive.require(target in TARGETS, "archive outside the reviewed follow-up list")
    expected = {"id": target["run_id"], "head_sha": target["source"],
                "head_branch": target["branch"], "event": "workflow_dispatch",
                "path": ".github/workflows/heavy-cpu.yml", "status": "completed",
                "conclusion": target["conclusion"], "run_attempt": 1}
    for key, value in expected.items():
        archive.require(run.get(key) == value, "follow-up producer identity differs: " + key)
    for key in ("repository", "head_repository"):
        archive.require(run.get(key, {}).get("full_name") == archive.REPOSITORY, "producer repository differs")
    expected = {"id": target["artifact_id"], "name": "heavy-" + str(target["run_id"]),
                "size_in_bytes": target["bytes"], "digest": "sha256:" + target["sha256"], "expired": False,
                "url": archive.API + "/actions/artifacts/" + str(target["artifact_id"]),
                "archive_download_url": archive.API + "/actions/artifacts/" + str(target["artifact_id"]) + "/zip"}
    for key, value in expected.items():
        archive.require(artifact.get(key) == value, "follow-up artifact identity differs: " + key)
    producer = artifact.get("workflow_run", {})
    for key, value in (("id", target["run_id"]), ("head_sha", target["source"]),
                       ("head_branch", target["branch"]),
                       ("repository_id", 1210669686), ("head_repository_id", 1210669686)):
        archive.require(producer.get(key) == value, "artifact producer differs: " + key)


def main():
    # Reuse the exact authenticated archive/security implementation, not a more
    # permissive reader. Its original two targets/diagnostic guards stay frozen.
    archive.require(archive.sha256(Path(archive.__file__).read_bytes()) == HELPER_SHA256,
                    "reviewed extraction helper changed")
    env = os.environ
    for key, value in {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": archive.REPOSITORY,
                       "RUNNER_OS": "Linux", "OCR_EVIDENCE_SCRIPT": SELECTOR,
                       "OCR_EVIDENCE_ARGS": "", "OCR_EVIDENCE_PIP": "",
                       "OCR_EVIDENCE_RUNNER": "ubuntu-latest", "OCR_EVIDENCE_BUILD_MODE": "portable"}.items():
        archive.require(env.get(key, "") == value, "closed follow-up dispatch differs: " + key)
    token = env["GH_TOKEN"]
    archive.require(bool(token), "Actions read token required")
    out, scratch = Path(env["OCR_EVIDENCE_OUT"]), Path(env["OCR_EVIDENCE_SCRATCH"])
    archive.require(not out.exists() and not scratch.exists(), "fresh extraction destinations required")
    out.mkdir(parents=True); scratch.mkdir(parents=True)
    deadline = time.monotonic() + archive.DEADLINE_SECONDS
    report = {"schema": "crispembed.ocr-followup.retained-text.v1", "source": env["GITHUB_SHA"],
              "model_execution": False, "model_qualified": False, "targets": [], "complete": False}
    try:
        for target in TARGETS:
            record = dict(target, complete=False); report["targets"].append(record)
            archive.require(shutil.disk_usage(scratch).free >= target["bytes"] + 512 * archive.CHUNK,
                            "insufficient archive scratch")
            run = archive.fetch_metadata("/actions/runs/" + str(target["run_id"]), token)
            artifact = archive.fetch_metadata("/actions/artifacts/" + str(target["artifact_id"]), token)
            validate_metadata(target, run, artifact)
            destination = out / target["label"]; destination.mkdir()
            archive.write_json(destination / "run-metadata.json", run)
            archive.write_json(destination / "artifact-metadata.json", artifact)
            local_zip = scratch / (str(target["artifact_id"]) + ".zip")
            try:
                with urllib.request.build_opener(archive.NoRedirect).open(
                        archive.archive_location(target, token), timeout=60) as response:
                    record["verified_archive"] = archive.copy_verified(
                        response, local_zip, target["bytes"], target["sha256"], deadline)
                directory, members = archive.extract_text(local_zip, destination / "text")
                archive.write_json(destination / "archive-directory.json", directory)
                archive.write_json(destination / "selected-text-sha256.json", members)
                record.update(complete=True, text_members=len(members), text_bytes=sum(row["bytes"] for row in members))
            finally:
                local_zip.unlink(missing_ok=True)
        report["complete"] = True
    finally:
        archive.write_json(out / "extraction.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
