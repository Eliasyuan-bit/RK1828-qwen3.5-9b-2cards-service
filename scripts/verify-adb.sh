#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_ROOT="${1:-/userdata/RK1828-qwen3.5-9b-2cards-service}"
: "${ADB_SERIAL:?Set ADB_SERIAL to the target board serial or host:port}"

REQUEST='{"id":"smoke-001","messages":[{"role":"user","content":"请只回答：服务已就绪。"}],"max_new_tokens":16,"enable_thinking":false}'
printf '%s\n' "$REQUEST" | adb -s "$ADB_SERIAL" shell \
  "export LD_LIBRARY_PATH='$REMOTE_ROOT/lib':\$LD_LIBRARY_PATH; '$REMOTE_ROOT/bin/rk1828_qwen35_9b_2cards_daemon' --config '$REMOTE_ROOT/config/qwen35-9b.json' --daemon"
