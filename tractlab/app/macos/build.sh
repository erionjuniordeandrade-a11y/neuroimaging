#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_BUNDLE="$REPO_ROOT/app/macos/build/TractLab.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS="$CONTENTS/MacOS"
SOURCE="$REPO_ROOT/app/macos/TractLab.swift"
CANDIDATE="$MACOS/TractLab.new"

mkdir -p "$MACOS"
cp "$REPO_ROOT/app/macos/Info.plist" "$CONTENTS/Info.plist"

# Apple's toolchain carries the AppKit/WebKit SDK; Homebrew swiftc does not.
if ! command -v xcrun >/dev/null 2>&1; then
  echo "xcrun not found. Install the Xcode Command Line Tools (xcode-select --install)." >&2
  exit 127
fi
xcrun swiftc -parse-as-library -O "$SOURCE" -framework AppKit -framework WebKit -o "$CANDIDATE"

chmod 755 "$CANDIDATE"
mv -f "$CANDIDATE" "$MACOS/TractLab"
codesign -s - --deep --force "$APP_BUNDLE"
echo "Built $APP_BUNDLE"
