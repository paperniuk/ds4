#!/bin/bash
# Install the prebuilt Qwen3.8-Flash-Next package for Apple Silicon Macs:
#
#   curl -fsSL https://raw.githubusercontent.com/paperniuk/ds4/m1-flash-next/install.sh | bash
#
# It unpacks the latest release into ~/ds4-flash-next (FLASH_DIR to change)
# and downloads nothing else; `flash pull` fetches the model.
set -e
DIR=${FLASH_DIR:-$HOME/ds4-flash-next}
URL=https://github.com/paperniuk/ds4/releases/latest/download/ds4-flash-next-macos-arm64.tar.gz

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

cat <<MSG

Installed in $DIR. Next:

  cd $DIR
  ./flash pull      # 69 GB: the model, the MTP block, the vision encoder
  ./flash serve     # server on http://127.0.0.1:8010/v1
  ./flash opencode  # provider block for OpenCode
MSG
