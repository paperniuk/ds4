#!/bin/bash
# Install the prebuilt dstar package (ds4 and its launcher) for Apple Silicon Macs:
#
#   curl -fsSL https://raw.githubusercontent.com/paperniuk/ds4/m1-flash-next/install.sh | bash
#
# It unpacks the latest release into ~/dstar (DSTAR_DIR to change),
# links the launcher into ~/.local/bin (DSTAR_BIN to change) and downloads
# nothing else; `dstar pull` fetches the model.
set -e
DIR=${DSTAR_DIR:-$HOME/dstar}
URL=https://github.com/paperniuk/ds4/releases/latest/download/dstar-macos-arm64.tar.gz

if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo "install: this package is for Apple Silicon Macs" >&2
    exit 1
fi
if [ "$(sw_vers -productVersion | cut -d. -f1)" -lt 15 ]; then
    echo "install: the prebuilt binaries need macOS 15 or newer; build from source instead" >&2
    exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "Downloading $URL"
curl -fL --progress-bar -o "$TMP/pkg.tar.gz" "$URL"
curl -fsL -o "$TMP/pkg.sha256" "$URL.sha256"
if [ "$(shasum -a 256 "$TMP/pkg.tar.gz" | cut -d' ' -f1)" != "$(cut -d' ' -f1 "$TMP/pkg.sha256")" ]; then
    echo "install: checksum mismatch, nothing was installed" >&2
    exit 1
fi
mkdir -p "$DIR"
tar -xzf "$TMP/pkg.tar.gz" -C "$DIR" --strip-components 1

"$DIR/dstar" link

cat <<MSG

Installed in $DIR. Next:

  dstar pull      # 67 GB: Qwen3.8-Flash-Next, the MTP block, the vision encoder
  dstar serve     # server on http://127.0.0.1:8010/v1
  dstar opencode  # provider block for OpenCode

DeepSeek V4 Flash and the other ds4 models: dstar models
MSG
