"""Run a historical source guard inside a clean checkout of its own frozen commit.

Producer-time guards (source_guard) require HEAD to be the reviewed frozen commit with a
clean tree; by design they reject every later tree, including main. Tests that exercise
them on a later tree therefore check out the frozen commit into a temporary detached
worktree and run the guard there, unchanged. The guard itself is never relaxed and its
expected sources are never rewritten (docs/handoff-2026-10-29.md).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_guard(commit, code):
    """Run `code` (Python, cwd = frozen checkout root) with -I; return CompletedProcess."""
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / 'frozen'
        subprocess.run(['git', 'worktree', 'add', '-q', '--detach', str(tree), commit], cwd=ROOT, check=True)
        try:
            return subprocess.run([sys.executable, '-I', '-c', code], cwd=tree, capture_output=True, text=True)
        finally:
            subprocess.run(['git', 'worktree', 'remove', '--force', str(tree)], cwd=ROOT, check=False)
            subprocess.run(['git', 'worktree', 'prune'], cwd=ROOT, check=False)


def head_is(commit):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == commit
