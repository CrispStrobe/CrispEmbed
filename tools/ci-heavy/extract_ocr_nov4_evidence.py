#!/usr/bin/env python3
"""Hosted, read-only extraction of two immutable failed OCR run reports.

Dispatch heavy-cpu.yml with script=tools/ci-heavy/extract_ocr_nov4_evidence.py,
empty args/pip, portable mode and the Linux x64 runner. The dedicated job has
Actions read permission only. Original ZIPs stay remote; its new artifact
contains verified small text members and diagnostics, never model/array files.
No model execution, original recapture, guard change or qualification occurs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "CrispStrobe/CrispEmbed"
SOURCE = "2f1bbdb11c3612acf7399e31b33ac715027f3184"
SELECTOR = "tools/ci-heavy/extract_ocr_nov4_evidence.py"
API = "https://api.github.com/repos/" + REPOSITORY
TARGETS = (
    {
        "label": "mineru2605-receipt",
        "run_id": 41810161887,
        "artifact_id": 13313715962,
        "bytes": 745003580,
        "sha256": "36fc4d014373c8b116d48afd1105f3e3a55f9869c4bea8e8af2785c8b3f52c16",
    },
    {
        "label": "lighton08-inventory-table",
        "run_id": 41810175624,
        "artifact_id": 13312356863,
        "bytes": 3459070891,
        "sha256": "7517525e72d29d1e535257458cc11942d3df360a6fb7b1ab0182ce8fff2138da",
    },
)
FROZEN = {
    "tests/results/qwen35-ocr/quality.json":
        "fad6669a452c0cbfd93db8d393c062e806337c5793e8cf7f664270a2bc83a4e3",
    "tools/ci-heavy/qwen35_quality_quant.py":
        "984dafaef7c20f42ed7da421e9fd315731b1e2a730c5f0a180bdb26a1107f0b1",
    "tests/fixtures/lighton08-portable/protocol.json":
        "d4771d104d6b592642dacb9596aceefa0470ad4b94a0b63b1ff43e7b7f90c402",
}
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".md", ".tsv"}
MAX_MEMBERS = 20000
MAX_MEMBER_BYTES = 16 * 1024**2
MAX_SELECTED_BYTES = 32 * 1024**2
MAX_ARCHIVE_EXPANDED_BYTES = 32 * 1024**3
MAX_TEXT_RATIO = 200
MAX_DIFFS = 20000
CHUNK = 1024**2
DEADLINE_SECONDS = 20 * 60


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("non-finite JSON value: " + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def write_json(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    require(len(data) <= MAX_MEMBER_BYTES, "generated report exceeds size cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def validate_dispatch(env):
    require(env.get("GITHUB_ACTIONS") == "true", "archive transfer is hosted-only")
    require(env.get("GITHUB_REPOSITORY") == REPOSITORY, "unexpected repository")
    require(env.get("RUNNER_OS") == "Linux", "Linux hosted runner required")
    require(env.get("OCR_EVIDENCE_SCRIPT") == SELECTOR, "closed script selector differs")
    require(env.get("OCR_EVIDENCE_ARGS", "") == "", "extraction accepts no args")
    require(env.get("OCR_EVIDENCE_PIP", "") == "", "extraction accepts no packages")
    require(env.get("OCR_EVIDENCE_RUNNER") == "ubuntu-latest", "Linux x64 runner required")
    require(env.get("OCR_EVIDENCE_BUILD_MODE") == "portable", "portable selector required")
    require(bool(env.get("GH_TOKEN")), "Actions read token required")


def validate_metadata(target, run, artifact):
    require(target in TARGETS, "artifact is outside the closed source list")
    required_run = {
        "id": target["run_id"], "head_sha": SOURCE, "head_branch": "main",
        "event": "workflow_dispatch", "path": ".github/workflows/heavy-cpu.yml",
        "status": "completed", "conclusion": "failure", "run_attempt": 1,
    }
    for key, value in required_run.items():
        require(run.get(key) == value, "producer run identity differs: " + key)
    for key in ("repository", "head_repository"):
        require(run.get(key, {}).get("full_name") == REPOSITORY, "producer repository differs")
    expected_artifact = {
        "id": target["artifact_id"], "name": "heavy-" + str(target["run_id"]),
        "size_in_bytes": target["bytes"], "digest": "sha256:" + target["sha256"],
        "expired": False,
    }
    for key, value in expected_artifact.items():
        require(artifact.get(key) == value, "artifact identity differs: " + key)
    producer = artifact.get("workflow_run", {})
    for key, value in (("id", target["run_id"]), ("head_sha", SOURCE), ("head_branch", "main"),
                       ("repository_id", 1210669686), ("head_repository_id", 1210669686)):
        require(producer.get(key) == value, "artifact producer differs: " + key)
    expected_url = API + "/actions/artifacts/" + str(target["artifact_id"])
    require(artifact.get("url") == expected_url, "artifact URL differs")
    require(artifact.get("archive_download_url") == expected_url + "/zip",
            "archive API URL differs")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def api_request(path, token):
    require(path.startswith("/actions/"), "read endpoint outside Actions")
    return urllib.request.Request(API + path, headers={
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "CrispEmbed-immutable-OCR-report-extraction",
    })


def fetch_metadata(path, token):
    # Refuse API redirects so an Authorization header never follows another host.
    with urllib.request.build_opener(NoRedirect).open(api_request(path, token), timeout=60) as response:
        data = response.read(2 * CHUNK + 1)
    require(len(data) <= 2 * CHUNK, "API metadata exceeds size cap")
    return read_json(data)


def archive_location(target, token):
    try:
        with urllib.request.build_opener(NoRedirect).open(
                api_request("/actions/artifacts/" + str(target["artifact_id"]) + "/zip", token),
                timeout=60):
            raise ValueError("expected a signed archive redirect")
    except urllib.error.HTTPError as error:
        require(error.code == 302, "archive redirect was not HTTP 302")
        location = error.headers.get("Location", "")
    parsed = urllib.parse.urlsplit(location)
    require(parsed.scheme == "https" and bool(parsed.hostname) and
            parsed.username is None and parsed.password is None and
            parsed.port in (None, 443), "unsafe archive redirect")
    # A signed provider URL is used once, never logged or retained in the reports.
    return location


def copy_verified(stream, destination, expected_bytes, expected_sha, deadline):
    count = 0
    digest = hashlib.sha256()
    created = False
    try:
        with destination.open("xb") as output:
            created = True
            while True:
                require(time.monotonic() <= deadline, "archive transfer deadline exceeded")
                data = stream.read(min(CHUNK, expected_bytes - count + 1))
                if not data:
                    break
                count += len(data)
                require(count <= expected_bytes, "archive download exceeds frozen size")
                digest.update(data)
                output.write(data)
        require(count == expected_bytes, "archive download size differs")
        require(digest.hexdigest() == expected_sha, "archive SHA256 differs")
    except Exception:
        # This helper owns this new path; no pre-existing data is removed.
        if created and destination.exists():
            destination.unlink()
        raise
    return {"bytes": count, "sha256": digest.hexdigest()}


def safe_members(archive):
    members = archive.infolist()
    require(len(members) <= MAX_MEMBERS, "archive member count exceeds cap")
    names, files = set(), set()
    selected, directory = [], []
    expanded = selected_bytes = 0
    for info in members:
        name = info.filename
        require(name == info.orig_filename and name and len(name.encode()) <= 1024 and "\\" not in name and
                not any(ord(char) < 32 for char in name), "unsafe archive name")
        path = PurePosixPath(name)
        canonical = path.as_posix()
        require(not path.is_absolute() and ".." not in path.parts and ":" not in name and
                canonical not in ("", ".") and name.rstrip("/") == canonical,
                "unsafe archive path")
        require(canonical not in names, "duplicate archive member")
        names.add(canonical)
        require(not (info.flag_bits & 1), "encrypted archive member")
        kind = stat.S_IFMT(info.external_attr >> 16)
        require(kind in (0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG),
                "archive link or special file")
        require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                "unsupported ZIP compression")
        require(info.file_size >= 0 and info.compress_size >= 0, "invalid archive size")
        expanded += info.file_size
        require(expanded <= MAX_ARCHIVE_EXPANDED_BYTES, "archive expanded bytes exceed cap")
        is_text = not info.is_dir() and path.suffix.lower() in TEXT_SUFFIXES
        if not info.is_dir():
            files.add(canonical)
        if is_text:
            require(info.file_size <= MAX_MEMBER_BYTES, "text member exceeds expanded size cap")
            require(info.file_size <= MAX_TEXT_RATIO * max(1, info.compress_size),
                    "text member compression ratio exceeds cap")
            selected_bytes += info.file_size
            require(selected_bytes <= MAX_SELECTED_BYTES, "selected expanded bytes exceed cap")
            selected.append(info)
        directory.append({"name": name, "bytes": info.file_size,
                          "compressed_bytes": info.compress_size,
                          "crc32": f"{info.CRC:08x}", "selected_text": is_text})
    for name in names:
        require(not any(parent.as_posix() in files for parent in PurePosixPath(name).parents
                        if parent.as_posix() != "."), "file/directory archive collision")
    return selected, directory


def extract_text(archive_path, destination):
    require(not destination.exists(), "text destination must be new")
    with zipfile.ZipFile(archive_path) as archive:
        selected, directory = safe_members(archive)
        destination.mkdir(parents=True)
        retained = []
        for info in selected:
            with archive.open(info) as stream:
                data = stream.read(MAX_MEMBER_BYTES + 1)
            require(len(data) == info.file_size and len(data) <= MAX_MEMBER_BYTES,
                    "expanded member size differs")
            data.decode("utf-8")  # Refuse binary payloads disguised with a text suffix.
            path = destination / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(data)
            retained.append({"name": info.filename, "bytes": len(data),
                             "sha256": sha256(data), "crc32": f"{info.CRC:08x}"})
    return directory, retained


def exact_differences(expected, observed):
    """Report differences under Python equality, exactly as frozen_original does."""
    rows = []

    def visit(left, right, path):
        if left == right:
            return
        require(len(rows) < MAX_DIFFS, "original diagnostic difference count exceeds cap")
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                pointer = path + "/" + str(key).replace("~", "~0").replace("/", "~1")
                if key not in left or key not in right:
                    rows.append({"path": pointer, "expected_present": key in left,
                                 "observed_present": key in right,
                                 "expected": left.get(key), "observed": right.get(key)})
                else:
                    visit(left[key], right[key], pointer)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                rows.append({"path": path + "/length", "expected": len(left), "observed": len(right)})
            for index in range(max(len(left), len(right))):
                if index >= len(left) or index >= len(right):
                    rows.append({"path": path + "/" + str(index),
                                 "expected_present": index < len(left),
                                 "observed_present": index < len(right),
                                 "expected": left[index] if index < len(left) else None,
                                 "observed": right[index] if index < len(right) else None})
                else:
                    visit(left[index], right[index], path + "/" + str(index))
                require(len(rows) <= MAX_DIFFS, "original diagnostic difference count exceeds cap")
        else:
            rows.append({"path": path, "expected": left, "observed": right})
        require(len(rows) <= MAX_DIFFS, "original diagnostic difference count exceeds cap")

    visit(expected, observed, "")
    return rows


def original_diagnostic(report, prior):
    expected = next(case["reference"] for case in prior["cases"] if case["id"] == "inventory-table")
    require(report.get("case_id") == "inventory-table" and report.get("source_commit") == SOURCE,
            "retained original report identity differs")
    observed = report.get("reference")
    require(isinstance(observed, dict), "retained original reference is missing")
    stable_expected = {key: value for key, value in expected.items() if key != "generation_seconds"}
    stable_observed = {key: value for key, value in observed.items() if key != "generation_seconds"}
    differences = exact_differences(stable_expected, stable_observed)
    equal = stable_expected == stable_observed
    require(equal == (not differences), "original diagnostic disagrees with exact equality")
    return {
        "schema": "crispembed.lighton08.original-failure-diagnostic.v1",
        "case_id": "inventory-table", "producer_source": SOURCE,
        "ignored_fields": ["generation_seconds"], "guard_result_exact": equal,
        "observed_failure": report.get("error", report.get("failure")),
        "differences": differences, "difference_count": len(differences),
        "historical_verdicts_changed": False, "model_qualified": False,
        "publication_qualified": False, "diagnostic_only": True,
    }


def frozen_sources(destination):
    records = {}
    for name, digest in FROZEN.items():
        data = (ROOT / name).read_bytes()
        require(sha256(data) == digest, "frozen diagnostic source differs: " + name)
        records[name] = {"bytes": len(data), "sha256": digest}
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as output:
            output.write(data)
    return records


def main():
    require(len(sys.argv) == 1, "extraction accepts no command arguments")
    validate_dispatch(os.environ)
    out = Path(os.environ["OCR_EVIDENCE_OUT"])
    scratch = Path(os.environ["OCR_EVIDENCE_SCRATCH"])
    require(not out.exists() and not scratch.exists(), "extraction destinations must be new")
    out.mkdir(parents=True)
    scratch.mkdir(parents=True)
    deadline = time.monotonic() + DEADLINE_SECONDS
    token = os.environ["GH_TOKEN"]
    summary = {"schema": "crispembed.ocr-nov4.retained-text.v1",
               "extractor_source": os.environ["GITHUB_SHA"], "producer_source": SOURCE,
               "model_execution": False, "originals_modified": False,
               "model_qualified": False, "publication_qualified": False, "targets": []}
    status = 0
    try:
        summary["frozen_diagnostic_sources"] = frozen_sources(out / "frozen-source")
        for target in TARGETS:
            record = dict(target, complete=False)
            summary["targets"].append(record)
            target_out = out / target["label"]
            target_out.mkdir()
            try:
                require(time.monotonic() <= deadline, "extraction deadline exceeded")
                require(shutil.disk_usage(scratch).free >= target["bytes"] + 512 * CHUNK,
                        "insufficient scratch space for one verified archive")
                run = fetch_metadata("/actions/runs/" + str(target["run_id"]), token)
                artifact = fetch_metadata("/actions/artifacts/" + str(target["artifact_id"]), token)
                validate_metadata(target, run, artifact)
                write_json(target_out / "run-metadata.json", run)
                write_json(target_out / "artifact-metadata.json", artifact)
                location = archive_location(target, token)
                archive_path = scratch / (str(target["artifact_id"]) + ".zip")
                # No Authorization header is sent to the signed storage endpoint.
                with urllib.request.build_opener(NoRedirect).open(location, timeout=60) as response:
                    header_size = response.headers.get("Content-Length")
                    require(header_size is None or int(header_size) == target["bytes"],
                            "archive Content-Length differs")
                    record["verified_archive"] = copy_verified(
                        response, archive_path, target["bytes"], target["sha256"], deadline)
                directory, members = extract_text(archive_path, target_out / "text")
                write_json(target_out / "archive-directory.json", directory)
                write_json(target_out / "selected-text-sha256.json", members)
                record["text_members"] = len(members)
                record["text_bytes"] = sum(member["bytes"] for member in members)
                if target["label"] == "lighton08-inventory-table":
                    report_path = target_out / "text/lighton08-portable-inventory-table/report.json"
                    diagnostic = original_diagnostic(
                        read_json(report_path.read_bytes()),
                        read_json((ROOT / "tests/results/qwen35-ocr/quality.json").read_bytes()))
                    write_json(target_out / "original-exact-diff.json", diagnostic)
                    record["original_guard_result_exact"] = diagnostic["guard_result_exact"]
                    record["original_difference_count"] = diagnostic["difference_count"]
                record["complete"] = True
                print(target["label"] + ": verified archive; retained " +
                      str(record["text_members"]) + " small text members", flush=True)
            except Exception as error:
                # HTTP errors contain a status, never retain their signed request URL.
                record["failure"] = (f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError)
                                     else type(error).__name__ + ": " + str(error))
                status = 1
            finally:
                archive_path = scratch / (str(target["artifact_id"]) + ".zip")
                if archive_path.exists():
                    archive_path.unlink()
    except Exception as error:
        summary["failure"] = type(error).__name__ + ": " + str(error)
        status = 1
    finally:
        summary["complete"] = status == 0 and all(row["complete"] for row in summary["targets"])
        write_json(out / "extraction.json", summary)
        text = ("# Retained OCR report extraction\n\n" +
                "\n".join(f"- {row['label']}: complete={row['complete']}" for row in summary["targets"]) +
                "\n\nRead-only diagnostics. Original failures and qualification gates remain unchanged.\n")
        (out / "summary.md").write_text(text)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
