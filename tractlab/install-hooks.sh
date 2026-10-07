#!/usr/bin/env bash
# Install tracked hooks into this clone. Run once after cloning.
set -euo pipefail
cd "$(dirname "$0")"
dest="$(git rev-parse --git-path hooks)"
install -m 0755 hooks/pre-commit "$dest/pre-commit"
echo "installed: $dest/pre-commit"
