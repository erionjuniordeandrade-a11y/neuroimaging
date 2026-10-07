#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_APP="$REPO_ROOT/build/Case Capsules.app"
APPLICATIONS_DIR="${HOME}/Applications"
DESTINATION_APP="$APPLICATIONS_DIR/Case Capsules.app"

if [[ ! -d "$SOURCE_APP" ]]; then
  printf '%s\n' "Build Case Capsules.app before installing." >&2
  exit 1
fi

mkdir -p "$APPLICATIONS_DIR"
ditto "$SOURCE_APP" "$DESTINATION_APP"
