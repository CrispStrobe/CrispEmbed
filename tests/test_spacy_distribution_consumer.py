#!/usr/bin/env python3
"""Model-free negative admission/accounting controls for the W7 consumer."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/spacy"))
import distribution_consumer as consumer


def valid_cpuinfo():
    return "flags : " + " ".join(consumer.REQUIRED_CPU_FLAGS) + "\n"


def elf_fixture():
    header = "Class: ELF64\nData: 2's complement, little endian\nMachine: Advanced Micro Devices X86-64\n"
    dynamic = "(SONAME) Library soname: [libcrispembed-spacy.so]\n"
    dynamic += "".join("(NEEDED) Shared library: [" + row["soname"] + "]\n"
                       for row in consumer.distribution.QUALIFIED_SYSTEM_DEPENDENCIES)
    versions = "Version needs section\n" + "\n".join(
        "Name: " + tag + "_" + value for tag, value in consumer.ELF_FLOORS.items())
    return header, dynamic, versions


def ldd_fixture():
    lines = [" linux-vdso.so.1 (0x1234)"]
    for pin in consumer.distribution.QUALIFIED_SYSTEM_DEPENDENCIES:
        name = pin["soname"]
        lines.append(" /lib64/" + name + " (0xabcd)" if name.startswith("ld-")
                     else " " + name + " => /lib/" + name + " (0xabcd)")
    return "\n".join(lines)


def one_token(text):
    return {**{name: text if name == "text" else "" for name in consumer.TEXT_FIELDS},
            "byte_start": 0, "byte_end": len(text.encode("utf-8")),
            "scalar_start": 0, "scalar_end": len(text),
            "utf16_start": 0, "utf16_end": len(text.encode("utf-16-le")) // 2,
            "features": ["0"] * 6, "orth_id": "0", "head": 0, "flags": 12}


class EnvironmentAdmissionTests(unittest.TestCase):
    def test_rejects_loader_overrides_including_empty_values(self):
        for key in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "LD_ORIGIN_PATH"):
            for value in ("", "/untrusted"):
                with self.subTest(key=key, value=value), self.assertRaises(consumer.ConsumerError):
                    consumer.check_environment({key: value}, system="Linux", machine="x86_64",
                                               cpuinfo=valid_cpuinfo())

    def test_rejects_different_platform_and_incomplete_cpu(self):
        for system, machine, info in (("Darwin", "x86_64", valid_cpuinfo()),
                                     ("Linux", "aarch64", valid_cpuinfo()),
                                     ("Linux", "x86_64", "flags : sse4_2\n"),
                                     ("Linux", "x86_64", valid_cpuinfo() + "flags : avx\n")):
            with self.subTest(system=system, machine=machine), self.assertRaises(consumer.ConsumerError):
                consumer.check_environment({}, system=system, machine=machine, cpuinfo=info)

    def test_external_protocol_hash_and_exact_frozen_bytes_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            payload = consumer.canonical(consumer.protocol())
            path.write_bytes(payload)
            sha = hashlib.sha256(payload).hexdigest()
            self.assertEqual(consumer.verify_protocol(path, sha), sha)
            with self.assertRaises(consumer.ConsumerError):
                consumer.verify_protocol(path, "0" * 64)
            changed = consumer.protocol()
            changed["measured_create_analyze_free_cycles"] = 0
            payload = consumer.canonical(changed)
            path.write_bytes(payload)
            with self.assertRaises(consumer.ConsumerError):
                consumer.verify_protocol(path, hashlib.sha256(payload).hexdigest())

    def test_transport_failure_precedes_candidate_or_elf_loading(self):
        with patch.object(consumer, "check_environment", return_value={}), \
             patch.object(consumer.distribution, "verify_archive", side_effect=ValueError("bad SHA")), \
             patch.object(consumer, "inspect_elf") as inspect, \
             patch.object(consumer.C, "CDLL") as load:
            with self.assertRaisesRegex(ValueError, "bad SHA"):
                consumer.preflight("archive", "0" * 64, "installation")
            inspect.assert_not_called()
            load.assert_not_called()

    def test_changed_installed_payload_precedes_candidate_loading(self):
        with patch.object(consumer, "check_environment", return_value={}), \
             patch.object(consumer.distribution, "verify_archive", return_value={}), \
             patch.object(consumer.distribution, "verify_tree", side_effect=ValueError("changed file")), \
             patch.object(consumer, "inspect_elf") as inspect, \
             patch.object(consumer.C, "CDLL") as load:
            with self.assertRaisesRegex(ValueError, "changed file"):
                consumer.preflight("archive", "0" * 64, "installation")
            inspect.assert_not_called()
            load.assert_not_called()


class DependencyAdmissionTests(unittest.TestCase):
    def test_actual_elf_contract_and_rejects_search_paths(self):
        header, dynamic, versions = elf_fixture()
        self.assertEqual(consumer.parse_elf(header, dynamic, versions)["required_symbol_floors"],
                         consumer.ELF_FLOORS)
        for extra in ("(RPATH) /tmp", "(RUNPATH) $ORIGIN"):
            with self.subTest(extra=extra), self.assertRaises(consumer.ConsumerError):
                consumer.parse_elf(header, dynamic + extra, versions)

    def test_rejects_wrong_elf_architecture_dependency_and_floor(self):
        header, dynamic, versions = elf_fixture()
        for values in ((header.replace("ELF64", "ELF32"), dynamic, versions),
                       (header, dynamic.replace("libc.so.6", "libextra.so.6"), versions),
                       (header, dynamic, versions.replace("GLIBC_2.38", "GLIBC_2.39"))):
            with self.subTest(values=values), self.assertRaises(consumer.ConsumerError):
                consumer.parse_elf(*values)

    def test_loader_inventory_is_exact_and_complete(self):
        output = ldd_fixture()
        self.assertEqual(len(consumer.parse_ldd(output)), 6)
        for changed in (output + "\nlibmissing.so => not found", output + "\nunknown line",
                        output + "\n libc.so.6 => /lib/libc.so.6 (0x1234)",
                        output.replace("/lib/libc.so.6", "relative/libc.so.6")):
            with self.subTest(changed=changed), self.assertRaises(consumer.ConsumerError):
                consumer.parse_ldd(changed)

    def test_dependency_mismatch_cannot_repin_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "libexample.so.1"
            path.write_bytes(b"accepted")
            pin = {"soname": path.name, "bytes": 8, "sha256": hashlib.sha256(b"accepted").hexdigest()}
            with patch.object(consumer.distribution, "QUALIFIED_SYSTEM_DEPENDENCIES", [pin]):
                self.assertTrue(consumer.verify_dependencies({path.name: path})[0][0]["matches_frozen"])
                path.write_bytes(b"modified")
                with self.assertRaises(consumer.ConsumerError):
                    consumer.verify_dependencies({path.name: path})

    def test_actual_mapping_rejects_alternate_loaded_library(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            expected = directory / "libexample.so.1"
            expected.write_bytes(b"expected")
            alternate = directory / "other/libexample.so.1"
            alternate.parent.mkdir()
            alternate.write_bytes(b"expected")
            pin = {"soname": expected.name, "bytes": 8, "sha256": hashlib.sha256(b"expected").hexdigest()}
            def row(path):
                info = path.stat()
                return {"path": path, "device": (os.major(info.st_dev), os.minor(info.st_dev)),
                        "inode": info.st_ino}
            with patch.object(consumer.distribution, "QUALIFIED_SYSTEM_DEPENDENCIES", [pin]):
                self.assertEqual(len(consumer.mapped_closure({pin["soname"]: expected}, mappings=[row(expected)])), 1)
                with self.assertRaises(consumer.ConsumerError):
                    consumer.mapped_closure({pin["soname"]: expected}, mappings=[row(expected), row(alternate)])
                invalid_inode = row(expected)
                invalid_inode["inode"] += 1
                with self.assertRaises(consumer.ConsumerError):
                    consumer.mapped_closure({pin["soname"]: expected}, mappings=[invalid_inode])

    def test_missing_or_deleted_actual_mapping_fails(self):
        with self.assertRaises(consumer.ConsumerError):
            consumer.mapped_closure({"libc.so.6": Path("/missing")}, mappings=[])
        with self.assertRaises(consumer.ConsumerError):
            consumer.proc_maps("1000-2000 r--p 0000 01:02 123 /lib/libc.so.6 (deleted)")


class SourceAndResourceTests(unittest.TestCase):
    def test_preserves_empty_emoji_and_counted_nul(self):
        self.assertEqual(consumer.validate_annotations("", []), {"tokens": [], "sentences": []})
        for text in ("🧭", "ab\0cd"):
            annotations = consumer.validate_annotations(text, [one_token(text)])
            self.assertEqual(annotations["tokens"][0]["text"], text)
            self.assertEqual(annotations["sentences"][0]["text"], text)

    def test_rejects_incorrect_utf16_source_and_dependency_spans(self):
        for field, value in (("utf16_end", 1), ("byte_end", 3), ("head", 1), ("flags", 4)):
            token = one_token("🧭")
            token[field] = value
            with self.subTest(field=field), self.assertRaises((consumer.ConsumerError, UnicodeDecodeError)):
                consumer.validate_annotations("🧭", [token])
        with self.assertRaises(consumer.ConsumerError):
            consumer.validate_annotations("missing", [])

    def test_rejects_false_space_flag_or_changed_text(self):
        token = one_token("a")
        token["flags"] |= 1
        with self.assertRaises(consumer.ConsumerError):
            consumer.validate_annotations("a", [token])
        token = one_token("a")
        token["text"] = "b"
        with self.assertRaises(consumer.ConsumerError):
            consumer.validate_annotations("a", [token])

    def test_resource_bounds_and_incomplete_denominator(self):
        baseline = {"rss_bytes": 100, "open_file_descriptors": 3, "threads": 1}
        samples = [dict(baseline) for _ in range(consumer.CYCLE_COUNT)]
        self.assertTrue(consumer.resource_verdict(baseline, samples)["rss_within_budget"])
        samples[4]["rss_bytes"] += consumer.RSS_GROWTH_BUDGET_BYTES + 1
        samples[-1]["open_file_descriptors"] = 4
        result = consumer.resource_verdict(baseline, samples)
        self.assertFalse(result["rss_within_budget"])
        self.assertFalse(result["fd_final_within_baseline"])
        with self.assertRaises(consumer.ConsumerError):
            consumer.resource_verdict(baseline, samples[:-1])

    def test_reports_cannot_overwrite_an_earlier_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            report = consumer.Report(path)
            report.write({"passed": False, "error": "retained first failure"})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                consumer.Report(path)
            self.assertEqual(path.read_bytes(), before)

    def test_timeout_retains_all_failed_children_without_native_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            protocol = directory / "protocol.json"
            protocol.write_bytes(consumer.canonical(consumer.protocol()))
            args = SimpleNamespace(archive=directory / "archive", sha256="0" * 64,
                                   installed=directory / "installed", protocol=protocol,
                                   protocol_sha256=consumer.digest(protocol), report=directory / "report.json")
            sink = consumer.Report(args.report)
            with patch.object(consumer, "preflight", return_value=({}, None, None, None)), \
                 patch.object(consumer.subprocess, "run", side_effect=subprocess.TimeoutExpired("child", 60)), \
                 patch.object(consumer.distribution, "verify_archive", return_value={}), \
                 patch.object(consumer.distribution, "verify_tree", return_value={}), \
                 patch.object(consumer.C, "CDLL") as load:
                result = consumer.run_driver(args, sink)
                self.assertFalse(result["passed"])
                self.assertEqual(len(result["children"]), 4)
                self.assertTrue(all(child["process"]["timed_out"] for child in result["children"]))
                self.assertTrue(args.report.is_file())
                load.assert_not_called()

    def test_missing_children_never_qualify(self):
        result = consumer.repeat_verdict([])
        self.assertFalse(result["complete"])
        self.assertFalse(result["annotations_exact_across_processes"])


class ChildIdentityTests(unittest.TestCase):
    def fixture(self, *, mode="enabled", index=1):
        sources = {"distribution.py": "a" * 64, "distribution_consumer.py": "b" * 64}
        admission = {"archive_sha256": "c" * 64, "manifest_sha256": "d" * 64,
                     "payload_file_count": 20, "transport_and_installed_payloads_verified": True}
        child = {"schema": consumer.SCHEMA, "role": "fresh_process", "mode": mode,
                 "index": index, "passed": True, "status": "passed",
                 "archive_sha256": "c" * 64, "protocol_sha256": "e" * 64,
                 "source_hashes_before": dict(sources), "source_hashes_after": dict(sources),
                 "admission": dict(admission)}
        launch = {"mode": mode, "index": index, "archive_sha256": "c" * 64,
                  "protocol_sha256": "e" * 64, "parent_sources": sources,
                  "parent_admission": admission}
        return child, launch

    def test_exact_child_identity_passes_without_mutating_evidence(self):
        child, launch = self.fixture()
        before = copy.deepcopy(child)
        self.assertTrue(consumer.child_identity(child, **launch)["passed"])
        self.assertEqual(child, before)

    def test_mixed_source_and_mismatched_launch_identity_fail(self):
        for field, value in (("role", "subprocess_driver"), ("mode", "disabled"),
                             ("index", True), ("archive_sha256", "f" * 64),
                             ("protocol_sha256", "f" * 64),
                             ("source_hashes_before", {"different": "f" * 64}),
                             ("source_hashes_after", {"different": "f" * 64})):
            child, launch = self.fixture()
            child[field] = value
            with self.subTest(field=field):
                self.assertFalse(consumer.child_identity(child, **launch)["passed"])
        child, launch = self.fixture()
        child["source_hashes_before"] = child["source_hashes_after"] = {"same_wrong_source": "f" * 64}
        self.assertFalse(consumer.child_identity(child, **launch)["passed"])
        child, launch = self.fixture()
        child["admission"]["archive_sha256"] = "f" * 64
        self.assertFalse(consumer.child_identity(child, **launch)["passed"])

    def test_driver_rejects_forged_pass_and_retains_original_child_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            protocol_path = directory / "protocol.json"
            protocol_path.write_bytes(consumer.canonical(consumer.protocol()))
            template, launch = self.fixture()
            args = SimpleNamespace(archive=directory / "archive", sha256="c" * 64,
                                   installed=directory / "installed", protocol=protocol_path,
                                   protocol_sha256=consumer.digest(protocol_path),
                                   report=directory / "driver.json")
            retained = []
            def forged_child(command, **_kwargs):
                mode = command[command.index("--child") + 1]
                index = int(command[command.index("--child-index") + 1])
                path = Path(command[command.index("--report") + 1])
                child = copy.deepcopy(template)
                child.update(mode=mode, index=index, protocol_sha256=args.protocol_sha256,
                             source_hashes_before={"mixed_source": "f" * 64},
                             source_hashes_after={"mixed_source": "f" * 64})
                payload = consumer.canonical(child)
                path.write_bytes(payload)
                retained.append((path, payload))
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch.object(consumer, "source_hashes", return_value=launch["parent_sources"]), \
                 patch.object(consumer, "preflight", return_value=(launch["parent_admission"], None, None, None)), \
                 patch.object(consumer.subprocess, "run", side_effect=forged_child), \
                 patch.object(consumer.distribution, "verify_archive", return_value={}), \
                 patch.object(consumer.distribution, "verify_tree", return_value={}), \
                 patch.object(consumer.C, "CDLL") as load:
                result = consumer.run_driver(args, consumer.Report(args.report))
            self.assertFalse(result["passed"])
            self.assertEqual(len(result["children"]), 4)
            for child in result["children"]:
                self.assertTrue(child["reported_passed"])
                self.assertFalse(child["passed"])
                self.assertFalse(child["identity_check"]["passed"])
            for path, payload in retained:
                self.assertEqual(path.read_bytes(), payload)
                self.assertTrue(json.loads(payload)["passed"])
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
