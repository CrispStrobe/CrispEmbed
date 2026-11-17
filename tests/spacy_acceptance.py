#!/usr/bin/env python3
"""Freeze, then qualify the previously unused eight-case spaCy acceptance split.

Freeze hashes input bytes without interpreting acceptance cases. Publish the
candidate and resulting protocol before exporting the original acceptance
reference. Run verifies that reference and its saved Git blobs before loading
the native library. It never edits goldens, downloads assets, queries CI, or
weakens the separate development gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "crispembed.spacy.acceptance.protocol.v1"
REPORT_SCHEMA = "crispembed.spacy.acceptance.result.v1"
ORACLE_COMMIT = "dcf2ec9b4d6fba05fa5eec2634c6654266332cc9"
CASES_SHA256 = "26c7e9e1e18f7423f6844729099a49e071d714c5095f613e311e950c9c880063"
BUNDLE_MANIFEST_SHA256 = "d599768375de3dc56abcca5cf1d49da0015857f9d3a10640aebb5317ef5d427e"
ATOL, RTOL = 2e-4, 2e-5
CASE_COUNT = 8
SOURCES = [
    "src/spacy_pipeline.h", "src/spacy_pipeline.cpp", "src/spacy_numeric.h", "src/spacy_numeric.cpp",
    "src/spacy_lexical.h", "src/spacy_lexical.cpp", "src/spacy_lexical_data.h",
    "src/spacy_decode.h", "src/spacy_decode.cpp", "src/core/json.h", "src/core/json_value.h",
    "src/core/sha256.h", "src/core/gguf_loader.h", "src/core/gguf_loader.cpp", "src/core/env_gate.h",
    "tools/spacy/CMakeLists.txt", "tools/spacy/generate_lexical_data.py",
    "tools/spacy/requirements-reference.txt", "tests/spacy_parity.py", "tests/spacy_text_parity.py",
    "tests/test_spacy_text_api.py", "tests/spacy_acceptance.py",
]
ORACLE_FILES = [
    "tools/proofing/spacy/export_reference.py", "tools/proofing/spacy/verify_reference.py",
    "tools/proofing/spacy/fixtures/cases.json",
]
POLICY = {
    "split": "acceptance", "expected_cases": CASE_COUNT,
    "absolute_tolerance": ATOL, "relative_tolerance": RTOL,
    "all_cases_required": True, "nonzero_denominators_required": True,
    "discrete_annotations_features_actions_and_classifier_choices": "exact",
    "missing_or_extra_numeric_stages": "fail",
    "reference_completed_and_saved_before_native_load": True,
    "native_inputs": "original UTF-8 only",
    "failure_policy": "Preserve failure; do not tune this candidate, reference, scorer or tolerances on these eight cases. If used for debugging, reclassify the split as development evidence.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"Missing or linked file: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=nonfinite)


def verify_sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Expected a trusted lowercase SHA-256")
    return value


def verify_commit(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value), "Expected a full Git commit SHA")
    return value


def relative_path(value):
    path = PurePosixPath(value)
    require(isinstance(value, str) and value and not path.is_absolute() and
            all(part not in (".", "..") for part in value.split("/")) and "\\" not in value,
            "Expected a safe repository-relative path")
    return path.as_posix()


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    require(result.returncode == 0, "Required Git source object is unavailable; refresh source refs before qualification")
    return result.stdout


def hashes(root, paths):
    return {name: sha256(root / name) for name in paths}


def match_committed_sources(root, commit, expected):
    verify_commit(commit)
    require(git(root, "cat-file", "-t", commit).strip() == b"commit", "Candidate pin is not a Git commit")
    for name, digest in expected.items():
        relative_path(name)
        verify_sha(digest)
        require(sha256(root / name) == digest, f"Working source drift: {name}")
        saved = git(root, "show", commit + ":" + name)
        require(hashlib.sha256(saved).hexdigest() == digest, f"Source differs from pinned commit: {name}")


def ggml_identity(candidate_commit):
    entry = git(ROOT, "ls-tree", candidate_commit, "ggml").decode("utf-8").strip().split()
    require(len(entry) == 4 and entry[:2] == ["160000", "commit"] and entry[3] == "ggml",
            "Missing pinned ggml submodule")
    require(git(ROOT / "ggml", "rev-parse", "HEAD").decode("ascii").strip() == entry[2],
            "Checked-out ggml differs from candidate gitlink")
    require(not git(ROOT / "ggml", "status", "--porcelain", "--untracked-files=no").strip(),
            "Modified tracked ggml source")
    return entry[2]


def validate_bundle(bundle):
    require(sha256(bundle / "manifest.json") == BUNDLE_MANIFEST_SHA256,
            "Bundle manifest differs from the externally frozen W1 identity")
    manifest = read_json(bundle / "manifest.json")
    require(set(manifest["files"]) == {"model.gguf", "resources.json", "LICENSE", "LICENSES_SOURCES"},
            "Incomplete bundle asset inventory")
    for name, entry in manifest["files"].items():
        require((bundle / name).stat().st_size == entry["size_bytes"] and
                sha256(bundle / name) == entry["sha256"], f"Changed bundle asset: {name}")
    return manifest


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def freeze(args):
    require(not args.protocol.exists(), "Refusing to replace a frozen protocol")
    verify_commit(args.candidate_commit)
    native_sources = hashes(ROOT, SOURCES)
    oracle_sources = hashes(args.quds_root, ORACLE_FILES)
    require(oracle_sources[ORACLE_FILES[2]] == CASES_SHA256, "Previously frozen case-input bytes changed")
    match_committed_sources(ROOT, args.candidate_commit, native_sources)
    match_committed_sources(args.quds_root, ORACLE_COMMIT, oracle_sources)
    bundle = validate_bundle(args.bundle)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "repository": "https://github.com/CrispStrobe/CrispEmbed",
            "commit": args.candidate_commit,
            "sources_sha256": native_sources,
            "ggml_commit": ggml_identity(args.candidate_commit),
            "library_sha256": sha256(args.library),
            "target": {"system": platform.system(), "machine": platform.machine()},
        },
        "oracle": {
            "repository": "https://github.com/CrispStrobe/quds_office_suite",
            "source_commit": ORACLE_COMMIT, "sources_sha256": oracle_sources,
            "cases_file_sha256": CASES_SHA256,
            "environment_lock_sha256": native_sources["tools/spacy/requirements-reference.txt"],
        },
        "bundle_manifest_sha256": BUNDLE_MANIFEST_SHA256,
        "model_sha256": bundle["files"]["model.gguf"]["sha256"],
        "resources_sha256": bundle["files"]["resources.json"]["sha256"],
        "policy": POLICY,
        "acceptance_inputs_interpreted_by_freeze": False,
        "acceptance_reference_exported_at_freeze": False,
        "acceptance_candidate_executed_at_freeze": False,
        "publication_sequence": [
            "Publish complete candidate code, development evidence and this gate; verify the remote commit.",
            "Freeze this protocol against that commit and the installed native library; publish and verify the protocol checkpoint.",
            "Run the unchanged original exporter with --split acceptance into a new directory.",
            "Save the complete original acceptance reference in Quds and verify its remote commit and manifest hash.",
            "Run the frozen native candidate with explicit protocol/reference hashes and the saved reference commit/path.",
            "Preserve the complete result, including failures, without tuning on these eight cases.",
        ],
    }
    write_new(args.protocol, protocol)
    print(json.dumps({"protocol_sha256": sha256(args.protocol), "candidate_commit": args.candidate_commit,
                      "library_sha256": protocol["candidate"]["library_sha256"], "acceptance_executed": False}))


def check_protocol(args, protocol):
    require(protocol.get("schema") == PROTOCOL_SCHEMA and protocol.get("policy") == POLICY,
            "Acceptance protocol/schema/policy differs")
    candidate, oracle = protocol["candidate"], protocol["oracle"]
    require(set(candidate["sources_sha256"]) == set(SOURCES) and set(oracle["sources_sha256"]) == set(ORACLE_FILES),
            "Incomplete frozen source inventory")
    require(oracle["source_commit"] == ORACLE_COMMIT and oracle["cases_file_sha256"] == CASES_SHA256,
            "Acceptance input/oracle identity differs")
    require(sha256(args.library) == candidate["library_sha256"], "Frozen native library bytes changed")
    require(candidate["target"] == {"system": platform.system(), "machine": platform.machine()},
            "Native candidate target differs")
    match_committed_sources(ROOT, candidate["commit"], candidate["sources_sha256"])
    match_committed_sources(args.quds_root, ORACLE_COMMIT, oracle["sources_sha256"])
    require(ggml_identity(candidate["commit"]) == candidate["ggml_commit"], "Frozen ggml identity changed")
    require(protocol["bundle_manifest_sha256"] == BUNDLE_MANIFEST_SHA256, "Unexpected frozen bundle")
    bundle = validate_bundle(args.bundle)
    require(bundle["files"]["model.gguf"]["sha256"] == protocol["model_sha256"] and
            bundle["files"]["resources.json"]["sha256"] == protocol["resources_sha256"],
            "Frozen model/resource identities differ")
    return bundle


def verify_reference_first(args, protocol):
    require(sha256(args.reference / "manifest.json") == verify_sha(args.reference_sha256),
            "Original reference manifest differs from its external pin")
    manifest = read_json(args.reference / "manifest.json")
    oracle = protocol["oracle"]
    require(manifest["split"] == "acceptance" and manifest["case_count"] == CASE_COUNT,
            "The complete eight-case acceptance reference is required")
    require(manifest["cases_file_sha256"] == CASES_SHA256 and
            manifest["exporter_sha256"] == oracle["sources_sha256"][ORACLE_FILES[0]],
            "Original exporter or input bytes differ")
    require(manifest["environment"]["lock"]["sha256"] == oracle["environment_lock_sha256"],
            "Original reference environment lock differs")
    require(manifest["numerical_policy"]["absolute_tolerance"] == ATOL and
            manifest["numerical_policy"]["relative_tolerance"] == RTOL,
            "Acceptance reference changed the frozen numerical budget")
    # The pinned verifier authenticates every case, stage, span and file before
    # any NativeAPI construction or CDLL loading in this process.
    verifier_path = args.quds_root / ORACLE_FILES[1]
    spec = importlib.util.spec_from_file_location("acceptance_reference_integrity", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(verifier_path.parent))
    try:
        spec.loader.exec_module(verifier)
        verified = verifier.verify(args.reference, args.quds_root / ORACLE_FILES[2])
    finally:
        sys.path.pop(0)
    require(verified["cases"] == CASE_COUNT and verified["tokens"] > 0 and
            verified["trace_stages"] > 0 and verified["parser_steps"] > 0,
            "Missing/empty acceptance denominators")
    commit = verify_commit(args.reference_commit)
    prefix = relative_path(args.reference_repo_path)
    require(git(args.quds_root, "cat-file", "-t", commit).strip() == b"commit",
            "Reference publication pin is not a Git commit")
    for path in sorted(args.reference.rglob("*")):
        if not path.is_file():
            continue
        name = relative_path(path.relative_to(args.reference).as_posix())
        saved = git(args.quds_root, "show", f"{commit}:{prefix}/{name}")
        require(hashlib.sha256(saved).hexdigest() == sha256(path),
                f"Original reference differs from saved Git blob: {name}")
    return manifest, verified


def run(args):
    require(not args.report.exists(), "Refusing to overwrite retained acceptance evidence")
    report = {"schema": REPORT_SCHEMA, "passed": False, "acceptance_inputs_executed": False,
              "native_candidate_loaded": False, "reference_completed_and_verified_first": False}
    try:
        require(sha256(args.protocol) == verify_sha(args.protocol_sha256), "Frozen protocol SHA-256 differs")
        protocol = read_json(args.protocol)
        bundle = check_protocol(args, protocol)
        manifest, verified = verify_reference_first(args, protocol)
        report.update(protocol_sha256=args.protocol_sha256, candidate_commit=protocol["candidate"]["commit"],
                      library_sha256=protocol["candidate"]["library_sha256"],
                      reference_manifest_sha256=args.reference_sha256, reference_commit=args.reference_commit,
                      reference_repo_path=args.reference_repo_path,
                      reference_completed_and_verified_first=True, expected_denominators=verified)
        import spacy_text_parity as raw
        from test_spacy_text_api import NativeAPI

        os.environ["CRISPEMBED_SPACY"] = "1"
        api = NativeAPI(args.library, args.bundle, bundle)
        report["native_candidate_loaded"] = True
        context, error = api.open()
        require(context, "Frozen native text context admission failed: " + error)
        cases = []
        try:
            require(api.lib.crispembed_spacy_text_abi_version() == 1 and
                    api.lib.crispembed_spacy_context_capabilities(context) == 15,
                    "Complete native text capabilities are required")
            for entry in manifest["cases"]:
                case_dir = args.reference / entry["id"]
                case = read_json(case_dir / "case.json")
                report["acceptance_inputs_executed"] = True
                try:
                    cases.append(raw.run_case(api, context, case, (case_dir / "traces.f32").read_bytes()))
                except Exception as exception:
                    cases.append({"id": entry["id"], "passed": False, "error": str(exception)})
        finally:
            api.lib.crispembed_spacy_free(context)
        counts = {
            "cases": len(cases), "tokens": sum(case.get("tokens", 0) for case in cases),
            "sentences": sum(case.get("sentences", 0) for case in cases),
            "parser_steps": sum(case.get("parser_steps", 0) for case in cases),
            "classifier_decisions": sum(case.get("classifier_decisions", 0) for case in cases),
            "numeric_stages": sum(len(case.get("stages", [])) for case in cases),
            "float_elements": sum(stage["elements"] for case in cases for stage in case.get("stages", [])),
        }
        complete = (counts["cases"] == CASE_COUNT and counts["tokens"] == verified["tokens"] and
                    counts["parser_steps"] == verified["parser_steps"] and
                    counts["numeric_stages"] == verified["trace_stages"] and
                    counts["classifier_decisions"] == 3 * verified["tokens"] and counts["float_elements"] > 0)
        report.update(counts=counts, cases=cases, complete_denominators=complete,
                      numerical_budget={"absolute_tolerance": ATOL, "relative_tolerance": RTOL},
                      reference_data_supplied_to_native=False, policy=POLICY,
                      sources_sha256=protocol["candidate"]["sources_sha256"],
                      environment={"system": platform.system(), "machine": platform.machine(),
                                   "python": platform.python_version()})
        # Recheck every mutable input after inference. Documentation/evidence
        # may advance, but all frozen source blobs, library and model must match.
        check_protocol(args, protocol)
        require(sha256(args.protocol) == args.protocol_sha256, "Protocol changed during qualification")
        verify_reference_first(args, protocol)
        report["passed"] = complete and all(case["passed"] for case in cases)
    except Exception as exception:
        report["error"] = str(exception)
    # Keep public reports independent of this host's paths.
    encoded = json.dumps(report, ensure_ascii=False)
    mappings = ((ROOT, "$CRISPEMBED_CHECKOUT"), (args.quds_root, "$QUDS_CHECKOUT"),
                (args.bundle, "$SPACY_BUNDLE"), (args.reference, "$ACCEPTANCE_REFERENCE"),
                (args.library.parent, "$SPACY_LIBRARY_DIR"))
    for actual, logical in sorted(mappings, key=lambda pair: len(str(pair[0].resolve())), reverse=True):
        encoded = encoded.replace(str(actual.resolve()), logical)
    write_new(args.report, json.loads(encoded))
    print(json.dumps({name: report[name] for name in ("passed", "counts", "error", "acceptance_inputs_executed")
                      if name in report}))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "run"):
        command = commands.add_parser(name)
        command.add_argument("--library", type=Path, required=True)
        command.add_argument("--bundle", type=Path, required=True)
        command.add_argument("--quds-root", type=Path, required=True)
        command.add_argument("--protocol", type=Path, required=True)
        if name == "freeze":
            command.add_argument("--candidate-commit", required=True)
        else:
            command.add_argument("--protocol-sha256", required=True)
            command.add_argument("--reference", type=Path, required=True)
            command.add_argument("--reference-sha256", required=True)
            command.add_argument("--reference-commit", required=True)
            command.add_argument("--reference-repo-path", required=True)
            command.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
        return 0
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Acceptance gate failed: {error}", file=sys.stderr)
        sys.exit(2)
