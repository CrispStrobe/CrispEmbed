#!/usr/bin/env python3
"""Recompute W7 retained Python/package evidence checks without native loading.

Usage: python -B audit_retained_python_package.py --repo PATH --output FILE
Only stdlib; reads reports, current source files and the retained native tarball.
It does not execute the tested consumer, rebuild a model or query remote services.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = repo / "docs/validation/native-spacy-w7-20261118"
    resume = root / "resume-20261118"
    package = resume / "package"
    consumer = resume / "python-consumer"
    findings, checks, evidence = [], [], {}

    def check(label, value, expected=True):
        ok = value == expected
        checks.append({"check": label, "passed": ok})
        if not ok:
            describe = lambda item: ({"sha256": sha(item), "bytes": len(item)}
                                     if isinstance(item, bytes) else item)
            findings.append({"check": label, "actual": describe(value), "expected": describe(expected)})
        return ok

    def read(path):
        data = path.read_bytes()
        evidence[str(path.relative_to(repo))] = {"sha256": sha(data), "bytes": len(data)}
        return data

    def load(path):
        return json.loads(read(path))

    def git(*argv):
        return subprocess.check_output(["git", "-C", str(repo), *argv])

    inspected_head = git("rev-parse", "HEAD").decode().strip()
    protocol_data = read(root / "operational-protocol.json")
    protocol = json.loads(protocol_data)
    archive_hash = "4e3b627a614cd21851e6e354335e2e08adeb4cedd4d7778df0cc68c1ce2edf01"
    manifest_hash = "e043af51e247368931f23fa0d9b931453e174f105b021fe2053e8e3029fd3d8d"
    protocol_hash = "60f85df220bb05ff467c6036ce465c841cff3480103720d5139af0644cfd3934"
    check("protocol external pin", sha(protocol_data), protocol_hash)
    check("protocol equals pre-execution freeze", protocol_data,
          git("show", "c0f9837c2af318a10f907531b41e3269757efe64:docs/validation/native-spacy-w7-20261118/operational-protocol.json"))

    index = load(package / "evidence-index.json")
    for item in index["files"]:
        data = read(package / item["path"])
        check("package index hash: " + item["path"], sha(data), item["sha256"])
        check("package index size: " + item["path"], len(data), item["size_bytes"])

    original = load(root / "distribution-build.json")
    first = load(package / "distribution-build-a.json")
    second = load(package / "distribution-build-b.json")
    manifest = first["manifest"]
    check("manifest recomputed digest", sha(canonical(manifest)), manifest_hash)
    for label, receipt in [("original", original), ("rebuild-a", first), ("rebuild-b", second)]:
        check(label + " archive identity", receipt["archive_sha256"], archive_hash)
        check(label + " archive length", receipt["archive_size_bytes"], 7808129)
        check(label + " manifest identity", receipt["manifest_sha256"], manifest_hash)
        check(label + " manifest content", receipt["manifest"], manifest)
        check(label + " payload count", receipt["payload_file_count"], 20)
        check(label + " verification outcome", receipt["status"], "verified")
    comparison = load(package / "distribution-byte-comparison.json")
    check("two complete rebuilds compared byte for byte", comparison["complete_bytes_equal"])
    check("both rebuilds match original frozen digest", comparison["both_match_original_frozen_archive_sha256"])
    for label in ["first_restored_build", "second_restored_build"]:
        check(label + " digest", comparison[label]["sha256"], archive_hash)
        check(label + " length", comparison[label]["size_bytes"], 7808129)

    result = load(package / "package-result.json")
    for path, pin in result["sources"].items():
        data = read(repo / path)
        check("current package source hash: " + path, sha(data), pin["sha256"])
        check("current package source size: " + path, len(data), pin["size_bytes"])
    for filename in ["distribution-install.json", "distribution-verify-install.json"]:
        receipt = load(package / filename)
        check(filename + " digest", receipt["archive_sha256"], archive_hash)
        check(filename + " manifest", receipt["manifest_sha256"], manifest_hash)
        check(filename + " payload count", receipt["payload_file_count"], 20)
        check(filename + " result", receipt["status"], "verified")
    install = load(package / "distribution-install.json")
    check("install transport authenticated", install["transport_verified"])
    check("install newly published", install["installed"])
    package_log = read(package / "distribution-controls.txt").decode()
    check("retained 25 package controls", "Ran 25 tests" in package_log and package_log.rstrip().endswith("OK"))

    restoration = load(root / "runtime-restoration.json")
    native = load(package / "native-restore.json")
    native_path = repo / "docs/validation/native-spacy-20261117/acceptance-candidate-linux-x86_64.tar.gz"
    native_data = read(native_path)
    check("native archive actual bytes", sha(native_data), native["native_archive_sha256"])
    check("native archive actual length", len(native_data), native["native_archive_size_bytes"])
    check("native archive frozen digest", sha(native_data), manifest["native"]["archive_sha256"])
    observed_native = {}
    with tarfile.open(native_path, "r:gz") as archive:
        members = archive.getmembers()
        check("native archive file count", len(members), 15)
        check("native archive distinct regular files", len({m.name for m in members}) == 15 and all(m.isfile() for m in members))
        for member in members:
            data = archive.extractfile(member).read()
            pin = {"sha256": sha(data), "size_bytes": len(data)}
            observed_native[member.name] = pin
            check("native bytes: " + member.name,
                  {k: native["native_files"][member.name][k] for k in pin}, pin)
            destination = (manifest["entrypoints"]["library"] if member.name == "lib/libcrispembed-spacy.so"
                           else "native/" + member.name)
            check("native-to-package pin: " + member.name, manifest["files"][destination], pin)
    bundle = load(package / "bundle-files.json")
    check("bundle five members", len(bundle["files"]), 5)
    for name, row in bundle["files"].items():
        pin = {"sha256": row["sha256"], "size_bytes": row["size_bytes"]}
        check("bundle frozen pin: " + name, pin, restoration["bundle"][name])
        check("bundle-to-package pin: " + name, pin, manifest["files"]["bundle/" + name])
    model = load(package / "model-wheel-download.json")
    for key, source in [("sha256", "sha256"), ("size_bytes", "bytes"), ("url", "url")]:
        check("official wheel frozen " + key, model[key], restoration["model_wheel"][source])

    control = load(resume / "python-controls/initial-receipt.json")
    check("21-control exit", control["controls"]["returncode"], 0)
    control_log = read(resume / "python-controls" / control["controls"]["stderr"]).decode()
    check("retained 21 consumer controls", "Ran 21 tests" in control_log and control_log.rstrip().endswith("OK"))
    check("printed protocol exact bytes", read(resume / "python-controls" / control["protocol"]["printed"]), protocol_data)
    execution = load(consumer / "first-execution.json")
    launch = load(consumer / "first-launch.json")
    report_data = read(consumer / "first.json")
    report = json.loads(report_data)
    check("retained driver report digest", sha(report_data), execution["report_sha256"])
    retained_audit = load(consumer / "first-retained-audit.json")
    check("prior retained audit report digest", retained_audit["report_sha256"], sha(report_data))
    driver_stdout = json.loads(read(consumer / execution["stdout"]))
    check("driver stdout successful", driver_stdout["passed"] is True and driver_stdout["status"] == "passed")
    check("driver stderr empty", read(consumer / execution["stderr"]), b"")
    for receipt_name, receipt in [("controls", control), ("execution", execution)]:
        check(receipt_name + " source unchanged during run", receipt["source_hashes_before"], receipt["source_hashes_after"])
        for path, digest in receipt["source_hashes_before"].items():
            check(receipt_name + " current source: " + path, sha(read(repo / path)), digest)
            check(receipt_name + " recorded commit source: " + path,
                  sha(git("show", receipt["source_commit"] + ":" + path)), digest)
    check("launch source commit", launch["source_commit"], execution["source_commit"])
    check("launch source hashes", launch["source_hashes_before"], execution["source_hashes_before"])
    check("driver successful", report["passed"] is True and report["status"] == "passed")
    check("driver native not loaded", report["native_loaded_in_driver"], False)
    check("driver external return code", execution["returncode"], 0)
    check("driver external timeout absent", execution["external_timed_out"], False)
    check("payload reverified after all children", report["archive_and_installed_payloads_unchanged"])
    source_hashes = {Path(k).name: v for k, v in execution["source_hashes_before"].items() if k.startswith("tools/")}
    check("driver initial source binding", report["source_hashes_before"], source_hashes)
    check("driver final source binding", report["source_hashes_after"], source_hashes)
    dependencies = {row["soname"]: row for row in protocol["dependencies"]}
    check("manifest and protocol dependency identity", manifest["system_dependencies"]["qualified_resolved_files"], protocol["dependencies"])
    mapped_expected = {name: row["sha256"] for name, row in dependencies.items()}
    mapped_expected["libcrispembed-spacy.so"] = manifest["native"]["library_sha256"]

    def admission_checks(label, admission):
        check(label + " verified transport/payload", admission["transport_and_installed_payloads_verified"])
        check(label + " archive", admission["archive_sha256"], archive_hash)
        check(label + " manifest", admission["manifest_sha256"], manifest_hash)
        check(label + " payloads", admission["payload_file_count"], 20)
        actual = {row["soname"]: {key: row[key] for key in ("soname", "sha256", "bytes")}
                  for row in admission["resolved_dependencies"]}
        check(label + " six resolved dependency pins", actual, dependencies)
        check(label + " dependencies all matched", all(row["matches_frozen"] for row in admission["resolved_dependencies"]))
        elf = admission["elf"]
        check(label + " ELF floors", elf["required_symbol_floors"], protocol["required_elf_symbol_floors"])
        check(label + " ELF no rpath/runpath", elf["rpath"] is None and elf["runpath"] is None)
        check(label + " ELF closure", sorted(elf["needed"]), sorted(dependencies))
        target = admission["target"]
        check(label + " target", {k: target[k] for k in ("system", "machine")}, {k: protocol["platform"][k] for k in ("system", "machine")})
        check(label + " cpu flags", target["required_cpu_flags_present"], protocol["platform"]["required_cpu_flags"])
        check(label + " loader overrides absent", target["loader_overrides_absent"])

    admission_checks("driver", report["admission"])
    token_count, sentence_count = 0, 0

    def annotation_checks(label, case, text):
        nonlocal token_count, sentence_count
        raw = text.encode()
        check(label + " input hash", case["input_sha256"], sha(raw))
        check(label + " input length", case["input_utf8_bytes"], len(raw))
        check(label + " annotation digest", case["annotations_sha256"], sha(canonical(case["annotations"])))
        check(label + " freed-before-read ownership", case["native_context_freed_before_view_read"] and case["native_document_freed_before_validation"])
        annotation = case["annotations"]
        tokens = annotation["tokens"]
        token_count += len(tokens)
        sentence_count += len(annotation["sentences"])
        cursor, sentence_start, derived_sentences = 0, 0, []
        for n, token in enumerate(tokens):
            start, end, flags = token["byte_start"], token["byte_end"], token["flags"]
            valid = start == cursor and start < end <= len(raw) and token["text"].encode() == raw[start:end]
            for boundary in ("start", "end"):
                before = raw[:token["byte_" + boundary]].decode()
                valid &= token["scalar_" + boundary] == len(before) and token["utf16_" + boundary] == len(before.encode("utf-16-le")) // 2
            valid &= flags & ~15 == 0 and bool(flags & 4) == (n == sentence_start)
            valid &= token["features"][4:6] == [str(int(bool(flags & 1))), str(int(bool(flags & 2)))]
            valid &= 0 <= token["head"] < len(tokens)
            cursor = end
            if flags & 1:
                valid &= raw[cursor:cursor + 1] == b" "
                cursor += 1
            check(label + " token coordinates/flags " + str(n), bool(valid))
            if flags & 8:
                stop = n + 1
                roots = [k for k in range(sentence_start, stop) if tokens[k]["head"] == k]
                check(label + " sentence root " + str(sentence_start), len(roots), 1)
                for k in range(sentence_start, stop):
                    visited, position = set(), k
                    while sentence_start <= position < stop and position not in visited:
                        visited.add(position)
                        if tokens[position]["head"] == position:
                            break
                        position = tokens[position]["head"]
                    check(label + " tree reaches sentence root " + str(k), position in roots)
                first_token = tokens[sentence_start]
                derived_sentences.append({"start_token": sentence_start, "end_token": stop,
                                          **{field + "_start": first_token[field + "_start"] for field in ("byte", "scalar", "utf16")},
                                          **{field + "_end": token[field + "_end"] for field in ("byte", "scalar", "utf16")},
                                          "text": raw[first_token["byte_start"]:end].decode()})
                sentence_start = stop
        check(label + " source fully partitioned", cursor == len(raw) and sentence_start == len(tokens))
        check(label + " derived sentences", annotation["sentences"], derived_sentences)

    children, metrics = [], []
    for number, embedded in enumerate(report["children"]):
        label = "child " + str(number)
        process = embedded["process"]
        raw = read(consumer / process["report"])
        child = json.loads(raw)
        children.append(child)
        check(label + " file digest", sha(raw), process["report_sha256"])
        check(label + " embedded report exact", {k: embedded[k] for k in child}, child)
        report_path = consumer / process["report"]
        stdout = json.loads(read(report_path.with_suffix(".stdout.txt")))
        check(label + " stdout successful", stdout["passed"] is True and stdout["status"] == "passed")
        check(label + " stderr empty", read(report_path.with_suffix(".stderr.txt")), b"")
        check(label + " return code", process["returncode"], 0)
        check(label + " timeout absent", process["timed_out"], False)
        check(label + " frozen timeout", process["timeout_seconds"], protocol["child_timeout_seconds"])
        check(label + " expected index", child["index"], number)
        check(label + " expected mode", child["mode"], "disabled" if number == 0 else "enabled")
        check(label + " passed", child["passed"] is True and child["status"] == "passed")
        check(label + " before sources", child["source_hashes_before"], source_hashes)
        check(label + " after sources", child["source_hashes_after"], source_hashes)
        check(label + " protocol", child["protocol_sha256"], protocol_hash)
        check(label + " installed payload recheck", child["installed_payloads_unchanged_after_execution"])
        admission_checks(label, child["admission"])
        for closure in ["actual_loaded_closure"] + (["actual_loaded_closure_after_cycles"] if number else []):
            rows = child[closure]
            check(label + " " + closure + " pins", {row["soname"]: row["sha256"] for row in rows}, mapped_expected)
            check(label + " " + closure + " inode checks", all(row["mapped_inode_matches_hashed_file"] and row["mapped_segments"] > 0 for row in rows))
        if number == 0:
            check(label + " refused disabled context", child["explicitly_unavailable"] and "disabled" in child["diagnostic"].lower())
            check(label + " disabled zero ownership", child["ownership"], dict.fromkeys(["contexts_created", "contexts_freed", "documents_created", "documents_freed"], 0))
            check(label + " disabled no inference", child["inference_executed"], False)
            check(label + " disabled no cases/cycles", child["cases"] == [] and child["cycles"] == [])
            continue
        check(label + " case order", [c["id"] for c in child["cases"]], [c["id"] for c in protocol["cases"]])
        for case, declared in zip(child["cases"], protocol["cases"]):
            annotation_checks(label + " " + case["id"], case, declared["text"])
        annotation_checks(label + " warmup", child["warmup"], protocol["cycle_text"])
        check(label + " cycle count", len(child["cycles"]), 25)
        check(label + " cycle index sequence", [row["index"] for row in child["cycles"]], list(range(25)))
        check(label + " cycle repeat exact hashes", all(row["annotations_sha256"] == child["warmup"]["annotations_sha256"] for row in child["cycles"]))
        check(label + " full drained ownership", child["ownership"], dict.fromkeys(["contexts_created", "contexts_freed", "documents_created", "documents_freed"], 34))
        resources = child["resources"]
        baseline = resources["post_warmup_baseline"]
        samples = [row["post_free_resources"] for row in child["cycles"]]
        growth = max(row["rss_bytes"] for row in samples) - baseline["rss_bytes"]
        check(label + " derived RSS growth", growth, resources["maximum_sampled_rss_growth_bytes"])
        check(label + " RSS budget", growth <= protocol["maximum_sampled_post_free_rss_growth_bytes"])
        check(label + " final sample", samples[-1], resources["final"])
        check(label + " final FD bound", samples[-1]["open_file_descriptors"] <= baseline["open_file_descriptors"])
        metrics.append({"index": number, "case_executions": len(child["cases"]), "measured_cycles": len(samples),
                        "warmup_cycles": 1, "context_document_pairs": child["ownership"]["contexts_created"],
                        "maximum_sampled_rss_growth_bytes": growth, "baseline_fds": baseline["open_file_descriptors"],
                        "final_fds": samples[-1]["open_file_descriptors"], "final_threads": samples[-1]["threads"]})
    check("four children retained", len(children), 4)
    enabled = children[1:]
    reference = [(row["input_sha256"], row["annotations"]) for row in enabled[0]["cases"]]
    check("cross-process full annotations exact", all([(row["input_sha256"], row["annotations"]) for row in child["cases"]] == reference for child in enabled))
    check("cross-process warmup full annotations exact", all(child["warmup"]["annotations"] == enabled[0]["warmup"]["annotations"] for child in enabled))
    totals = {"case_executions": sum(m["case_executions"] for m in metrics),
              "short_cycle_executions_including_warmup": sum(m["measured_cycles"] + m["warmup_cycles"] for m in metrics),
              "native_contexts_created_and_freed": sum(m["context_document_pairs"] for m in metrics),
              "native_documents_created_and_freed": sum(m["context_document_pairs"] for m in metrics)}
    for key, total in totals.items():
        check("independently derived " + key, total, report["repeat_checks"][key])
    output = {
        "schema": "crispembed.spacy.w7.independent_retained_python_package_audit.v1",
        "status": "passed" if not findings else "failed",
        "scope": "Retained evidence audit; no consumer, model, native or CI execution",
        "inspected_head": inspected_head,
        "auditor_sha256": sha(Path(__file__).read_bytes()),
        "native_executions": 0, "model_builds": 0, "tests_rerun": 0,
        "github_actions_queries_waits_reruns_dispatches": 0,
        "checked": len(checks), "failed": len(findings), "findings": findings,
        "checks": checks, "evidence": evidence, "child_metrics": metrics,
        "totals": {**totals, "full_annotation_documents_recomputed": sum(len(child["cases"]) + 1 for child in enabled),
                   "token_annotation_rows_recomputed": token_count, "sentence_rows_recomputed": sentence_count},
        "package": {"archive_sha256": archive_hash, "manifest_sha256": manifest_hash,
                    "archive_size_bytes": 7808129, "native_payload_files_actually_rehashed": len(observed_native),
                    "bundle_files_bound_by_retained_receipts": 5, "retained_package_controls": 25,
                    "retained_python_consumer_controls": 21},
        "documentation_findings": ["Top-level W7 README, CONTINUE and checkpoint-status still describe pending consumers; newer retained passing evidence supersedes that status."],
        "targeted_source_review": {
            "actionable_findings": [],
            "areas": [
                "External archive hash precedes decompression; canonical USTAR and all payload pins are closed and bounded.",
                "Installation stages read-only files, verifies the closed tree and publishes with Linux no-replace rename.",
                "Current consumer authenticates protocol, installed tree, ELF baseline, six dependency hashes and actual loaded mappings.",
                "Real calls increment ownership counters; contexts free before document views and documents free before copied-data validation.",
                "Child launch identity, current source hashes and original report bytes remain bound to the parent receipt."
            ],
            "boundary": "Private caller-owned immutable cache and trusted host; no hostile concurrent-writer security claim."
        },
        "limitations": [
            "Package/model binary archives are not retained in ordinary Git; this audit verifies retained build/install identities, input pins, source identity and native archive bytes, not a new full package reconstruction.",
            "RSS, inode mapping, allocation/free and execution observations are retained reports tied to audited source, not a new runtime observation.",
            "Cycle reports retain annotation digests rather than full annotations; all 75 measured digests bind to the fully retained warmups.",
            "Bounds cover Linux x86-64/AVX2 and six fixed dependency hashes; no other platform, target device, hosted distribution, activation or added linguistic-quality qualification follows."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(output))
    print(json.dumps({"status": output["status"], "checked": len(checks), "failed": len(findings), "totals": output["totals"], "findings": findings}, sort_keys=True))
    return bool(findings)


if __name__ == "__main__":
    raise SystemExit(main())
