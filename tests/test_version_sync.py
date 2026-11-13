"""Exercise release version propagation without native builds or package imports."""

import ast
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def exported_version(source):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("Python package does not export __version__")


class VersionSyncTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for relative in (
            "VERSION", "scripts/sync-version.py", "scripts/bump-version.sh",
            "python/pyproject.toml", "python/crispembed/__init__.py",
            "crispembed/Cargo.toml", "crispembed-sys/Cargo.toml",
            "flutter/crispembed/pubspec.yaml",
        ):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)

    def run_command(self, *args):
        return subprocess.run(
            args, cwd=self.root, check=True, capture_output=True, text=True
        ).stdout

    def test_checked_in_python_version_matches_release(self):
        version = (ROOT / "VERSION").read_text().strip()
        self.assertEqual(exported_version(
            (ROOT / "python/crispembed/__init__.py").read_text()
        ), version)
        self.assertIn(f'version = "{version}"',
                      (ROOT / "python/pyproject.toml").read_text())

    def test_sync_updates_both_python_versions_and_is_idempotent(self):
        (self.root / "VERSION").write_text("98.7.6\n")
        self.run_command(sys.executable, "scripts/sync-version.py")
        self.assertEqual(exported_version(
            (self.root / "python/crispembed/__init__.py").read_text()
        ), "98.7.6")
        self.assertIn('version = "98.7.6"',
                      (self.root / "python/pyproject.toml").read_text())
        before = {p.relative_to(self.root): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        self.run_command(sys.executable, "scripts/sync-version.py")
        after = {p.relative_to(self.root): p.read_bytes()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_bump_commits_export_and_leaves_unrelated_work_unstaged(self):
        # Run the real release script only in a disposable repository. This
        # catches missing stage-list entries that a direct sync test misses.
        self.run_command("git", "init", "-q")
        self.run_command("git", "config", "user.email", "test@example.invalid")
        self.run_command("git", "config", "user.name", "Version sync test")
        self.run_command("git", "add", ".")
        self.run_command("git", "commit", "-qm", "fixture")
        (self.root / "unrelated.txt").write_text("keep untracked\n")
        self.run_command("bash", "scripts/bump-version.sh", "98.7.6")
        committed = self.run_command(
            "git", "show", "HEAD:python/crispembed/__init__.py"
        )
        self.assertEqual(exported_version(committed), "98.7.6")
        self.assertEqual(self.run_command("git", "status", "--porcelain").strip(),
                         "?? unrelated.txt")
        self.assertEqual(self.run_command("git", "tag", "--list").strip(),
                         "v98.7.6")


if __name__ == "__main__":
    unittest.main()
