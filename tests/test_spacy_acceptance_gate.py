#!/usr/bin/env python3
"""Model-free admission controls; never reads or executes acceptance inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import spacy_acceptance as gate


class AcceptanceGateControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="spacy-gate-controls-", dir=Path.cwd())
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def git_fixture(self):
        repository = self.root / "source"
        repository.mkdir()
        environment = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

        def git(*args):
            command = ["git", "-C", str(repository), "-c", "core.hooksPath=" + os.devnull,
                       "-c", "commit.gpgsign=false", "-c", "user.name=Native gate test",
                       "-c", "user.email=native-gate-test@example.invalid", *args]
            return subprocess.check_output(command, env=environment, stderr=subprocess.PIPE)

        git("init", "-q")
        source = repository / "source.cpp"
        source.write_bytes(b"frozen source\n")
        git("add", "source.cpp")
        git("commit", "-q", "-m", "Temporary admission fixture")
        return repository, source, git("rev-parse", "HEAD").decode("ascii").strip()

    def test_current_file_must_match_committed_bytes(self):
        repository, source, commit = self.git_fixture()
        original = gate.sha256(source)
        gate.match_committed_sources(repository, commit, {"source.cpp": original})
        source.write_bytes(b"changed candidate\n")
        with self.assertRaisesRegex(ValueError, "Working source drift"):
            gate.match_committed_sources(repository, commit, {"source.cpp": original})
        with self.assertRaisesRegex(ValueError, "Source differs from pinned commit"):
            gate.match_committed_sources(repository, commit, {"source.cpp": gate.sha256(source)})

    def test_missing_commit_object_rejects(self):
        repository, source, _commit = self.git_fixture()
        with self.assertRaisesRegex(ValueError, "Git source object is unavailable"):
            gate.match_committed_sources(repository, "0" * 40, {"source.cpp": gate.sha256(source)})

    def test_unsafe_source_paths_and_linked_files_reject(self):
        for value in ("", "/absolute", "../escape", "a/../escape", "./path", "a\\escape"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                gate.relative_path(value)
        source = self.root / "target"
        source.write_bytes(b"fixture")
        link = self.root / "link"
        link.symlink_to(source)
        with self.assertRaisesRegex(ValueError, "linked file"):
            gate.sha256(link)

    def test_bad_protocol_pin_stops_before_reference_or_candidate(self):
        protocol = self.root / "protocol.json"
        protocol.write_text("{}\n", encoding="utf-8")
        args = argparse.Namespace(protocol=protocol, protocol_sha256="0" * 64,
                                  library=self.root / "native.so", bundle=self.root / "bundle",
                                  quds_root=self.root / "quds", reference=self.root / "reference",
                                  report=self.root / "result.json")
        with patch.object(gate, "check_protocol") as check, patch.object(gate, "verify_reference_first") as verify:
            self.assertEqual(gate.run(args), 1)
            check.assert_not_called()
            verify.assert_not_called()
        report = json.loads(args.report.read_text(encoding="utf-8"))
        self.assertFalse(report["native_candidate_loaded"])
        self.assertFalse(report["acceptance_inputs_executed"])
        self.assertFalse(report["reference_completed_and_verified_first"])
        self.assertIn("protocol SHA-256 differs", report["error"])

    def test_bad_reference_pin_stops_before_reference_contents(self):
        reference = self.root / "reference"
        reference.mkdir()
        # Intentionally not JSON: authentication must fail before parsing.
        (reference / "manifest.json").write_bytes(b"not a reference")
        args = argparse.Namespace(reference=reference, reference_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "reference manifest differs"):
            gate.verify_reference_first(args, {})

    def test_retained_protocol_and_report_cannot_be_overwritten(self):
        target = self.root / "retained.json"
        gate.write_new(target, {"retained": True})
        original = hashlib.sha256(target.read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            gate.write_new(target, {"retained": False})
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
