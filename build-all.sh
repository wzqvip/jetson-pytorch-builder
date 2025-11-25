#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SCRIPT="$SCRIPT_DIR/build.sh"

if [[ ! -x "$BUILD_SCRIPT" ]]; then
  echo "Missing build.sh. Did you clone the project correctly?" >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  set -- 3.8 3.9 3.10 3.11 3.12
fi

STATUS=0
for version in "$@"; do
  echo "===== Building PyTorch for Python $version ====="
  if ! "$BUILD_SCRIPT" "$version"; then
    echo "Build for Python $version failed." >&2
    STATUS=1
  fi
done

exit "$STATUS"
