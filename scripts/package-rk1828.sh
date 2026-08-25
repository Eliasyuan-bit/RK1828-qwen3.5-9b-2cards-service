#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/rk1828}"
PACKAGE_DIR="${PACKAGE_DIR:-$ROOT/dist/RK1828-qwen3.5-9b-2cards-service}"

test -f "$BUILD_DIR/rk1828_qwen35_9b_2cards_daemon" || {
  echo "Missing build artifact; run scripts/build-rk1828.sh first." >&2
  exit 1
}

mkdir -p "$PACKAGE_DIR"
cmake --install "$BUILD_DIR" --prefix "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR/config"
cp "$ROOT/config/qwen35-9b.json" "$PACKAGE_DIR/config/"

echo "Package complete: $PACKAGE_DIR"
