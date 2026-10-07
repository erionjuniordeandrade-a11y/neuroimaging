#!/usr/bin/env bash
# Point git at the tracked hooks dir (applies to all worktrees of this repo).
set -euo pipefail
cd "$(dirname "$0")/.."
git config core.hooksPath hooks
echo "core.hooksPath=$(git config core.hooksPath)"
