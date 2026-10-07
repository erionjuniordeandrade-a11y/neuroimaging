#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
STAGING_APP="$BUILD_DIR/.Case Capsules.app.tmp.$$"
FINAL_APP="$BUILD_DIR/Case Capsules.app"
PREVIOUS_APP=""

mkdir -p "$BUILD_DIR"

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "$PREVIOUS_APP" && -e "$PREVIOUS_APP" && ! -e "$FINAL_APP" ]]; then
    mv "$PREVIOUS_APP" "$FINAL_APP" || true
  fi
  if [[ -e "$STAGING_APP" ]]; then
    rm -rf "$STAGING_APP"
  fi
  exit "$status"
}
trap cleanup EXIT

# Apple toolchain: Homebrew swift lacks the AppKit/WebKit SDK.
xcrun swift build -c release

mkdir -p "$STAGING_APP/Contents/MacOS"
cp "$REPO_ROOT/.build/release/CaseCapsules" "$STAGING_APP/Contents/MacOS/CaseCapsules"
cp "$REPO_ROOT/Info.plist" "$STAGING_APP/Contents/Info.plist"
codesign -s - --force --deep "$STAGING_APP"

if [[ -e "$FINAL_APP" ]]; then
  PREVIOUS_APP="$BUILD_DIR/.Case Capsules.app.previous.$$"
  mv "$FINAL_APP" "$PREVIOUS_APP"
fi

mv "$STAGING_APP" "$FINAL_APP"
if [[ -n "$PREVIOUS_APP" ]]; then
  rm -rf "$PREVIOUS_APP"
  PREVIOUS_APP=""
fi

trap - EXIT
