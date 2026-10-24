#!/usr/bin/env bash
# 把一个 .tlx 安装到 TuneLab 扩展目录（默认取系统用户数据目录，可用 TUNELAB_DATA_DIR / --data-dir 覆盖）。
# 用法：
#   scripts/install.sh                     # 安装最新打好的 artifacts/tlx/*.tlx
#   scripts/install.sh path/to/x.tlx
#   TUNELAB_DATA_DIR=/tmp/tunelab-data scripts/install.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_DIR="${TUNELAB_DATA_DIR:-}"
TLX=""
while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir) DATA_DIR="${2:?--data-dir needs a value}"; shift 2 ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) TLX="$1"; shift ;;
    esac
done

if [ -z "$TLX" ]; then
    TLX="$(ls -t "$ROOT"/artifacts/tlx/*.tlx 2>/dev/null | head -1 || true)"
fi
if [ -z "$TLX" ] || [ ! -f "$TLX" ]; then
    echo "No .tlx found. Run scripts/pack.sh first, or pass a path." >&2
    exit 1
fi

if [ -z "$DATA_DIR" ]; then
    case "$(uname -s)" in
        Darwin) DATA_DIR="$HOME/Library/Application Support/TuneLab" ;;
        MINGW*|MSYS*|CYGWIN*) DATA_DIR="${APPDATA:-$HOME/AppData/Roaming}/TuneLab" ;;
        *) DATA_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/TuneLab" ;;
    esac
fi

TARGET="$DATA_DIR/Extensions/tunelab-cloudvoice"
echo "==> Installing $(basename "$TLX")"
echo "    -> $TARGET"
mkdir -p "$DATA_DIR/Extensions"
rm -rf "$TARGET"
mkdir -p "$TARGET"
unzip -q "$TLX" -d "$TARGET"
echo "==> Installed. Restart TuneLab if it is running (extension dlls are locked while open)."
