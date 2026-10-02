#!/bin/bash
# Build the prebuilt macOS package for Qwen3.8-Flash-Next: the binaries, the
# Metal shaders they load at start and the dstar launcher, in one tarball.
#
#   ./package-macos.sh [output directory]
#
# The build runs on a clean export of HEAD, targets the baseline Apple
# Silicon CPU and macOS 15, so the result does not depend on the machine
# that made it.
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$ROOT/dist}
NAME=ds4-flash-next
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$WORK/src" "$WORK/$NAME" "$OUT"
git -C "$ROOT" archive HEAD | tar -x -C "$WORK/src"
MACOSX_DEPLOYMENT_TARGET=15.0 make -C "$WORK/src" ds4 ds4-server \
    NATIVE_CPU_FLAG=-mcpu=apple-m1 DEBUG_FLAGS= >/dev/null

cp "$WORK/src/ds4" "$WORK/src/ds4-server" "$WORK/src/dstar" "$WORK/src/download_model.sh" "$WORK/src/LICENSE" "$WORK/$NAME/"
mkdir "$WORK/$NAME/metal"
cp "$WORK/src"/metal/*.metal "$WORK/$NAME/metal/"
strip -x "$WORK/$NAME/ds4" "$WORK/$NAME/ds4-server"
codesign -s - -f "$WORK/$NAME/ds4" "$WORK/$NAME/ds4-server" 2>/dev/null
git -C "$ROOT" rev-parse HEAD > "$WORK/$NAME/VERSION"

TARBALL=$OUT/$NAME-macos-arm64.tar.gz
tar -czf "$TARBALL" -C "$WORK" "$NAME"
(cd "$OUT" && shasum -a 256 "$(basename "$TARBALL")" > "$(basename "$TARBALL").sha256")
echo "$TARBALL"
