#!/bin/bash
# Verify gate: unit tests, bundle build, offscreen render of every public capsule.
set -euo pipefail
cd "$(dirname "$0")/.."
xcrun swift test
scripts/build.sh
BIN="build/Case Capsules.app/Contents/MacOS/CaseCapsules"
codesign -v "build/Case Capsules.app"
mkdir -p build/selftest/dl
fail=0
for f in "$HOME"/case-capsule/out/*.capsule.html; do
  out="build/selftest/$(basename "$f" .capsule.html).png"
  "$BIN" --selftest "$f" --out "$out" || fail=1
done
small=$(ls -S "$HOME"/case-capsule/out/*.capsule.html | tail -1)
rm -f build/selftest/dl/*
cp "$small" build/selftest/dl/copy.capsule.html
"$BIN" --selftest build/selftest/dl/copy.capsule.html --out build/selftest/dl/copy.png --selftest-download || fail=1
"$BIN" --scan-summary
exit $fail
