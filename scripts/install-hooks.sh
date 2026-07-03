#!/usr/bin/env bash
# scripts/install-hooks.sh — point git at the tracked hooks/ directory.
# core.hooksPath is stored in the shared repo config, so this covers every
# worktree of this clone. Run once after cloning.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git config core.hooksPath hooks
chmod +x hooks/* 2>/dev/null || true
echo "Installed git hooks (core.hooksPath=hooks). pre-commit will auto-format staged C/C++."
