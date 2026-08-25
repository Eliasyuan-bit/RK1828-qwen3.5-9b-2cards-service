#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/rk1828}"

: "${RKNN3_MODEL_ZOO_ROOT:?Set RKNN3_MODEL_ZOO_ROOT to the rknn3-model-zoo checkout}"
: "${RK1828_C_COMPILER:?Set RK1828_C_COMPILER to the aarch64 cross C compiler}"
: "${RK1828_CXX_COMPILER:?Set RK1828_CXX_COMPILER to the aarch64 cross C++ compiler}"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DRKNN3_MODEL_ZOO_ROOT="$RKNN3_MODEL_ZOO_ROOT" \
  -DTARGET_SOC=rk3588 \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER="$RK1828_C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$RK1828_CXX_COMPILER" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel

echo "Build complete: $BUILD_DIR/rk1828_qwen35_9b_2cards_daemon"
