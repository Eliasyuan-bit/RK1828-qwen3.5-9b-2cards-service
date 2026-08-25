#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="${1:-$ROOT/dist/RK1828-qwen3.5-9b-2cards-service}"
REMOTE_ROOT="${2:-/userdata/RK1828-qwen3.5-9b-2cards-service}"
: "${ADB_SERIAL:?Set ADB_SERIAL to the target board serial or host:port}"

test -x "$PACKAGE_DIR/bin/rk1828_qwen35_9b_2cards_daemon" || {
  echo "Invalid package directory: $PACKAGE_DIR" >&2
  exit 1
}

adb -s "$ADB_SERIAL" shell "mkdir -p '$REMOTE_ROOT/bin' '$REMOTE_ROOT/lib' '$REMOTE_ROOT/config'"
adb -s "$ADB_SERIAL" push "$PACKAGE_DIR/bin/." "$REMOTE_ROOT/bin/" >/dev/null
adb -s "$ADB_SERIAL" push "$PACKAGE_DIR/lib/." "$REMOTE_ROOT/lib/" >/dev/null
adb -s "$ADB_SERIAL" push "$PACKAGE_DIR/config/qwen35-9b.json" "$REMOTE_ROOT/config/" >/dev/null
adb -s "$ADB_SERIAL" shell "chmod 755 '$REMOTE_ROOT/bin/rk1828_qwen35_9b_2cards_daemon'"

echo "Deployed to $REMOTE_ROOT"
echo "Model files are not copied. Ensure paths in $REMOTE_ROOT/config/qwen35-9b.json exist on the board."
