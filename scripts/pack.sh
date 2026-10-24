#!/usr/bin/env bash
# 编译并打包成 TuneLab 扩展包 .tlx（zip，manifest.json 在根目录）。
# 产物：artifacts/tlx/TuneLab.CloudVoice-<version>.tlx
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-Release}"

CONFIG="$CONFIG" "$ROOT/scripts/build.sh"

VERSION="$(grep -o '"version": *"[^"]*"' "$ROOT/src/TuneLab.CloudVoice/manifest.json" | head -1 | cut -d'"' -f4)"
OUT_DIR="$ROOT/artifacts/tlx"
TLX="$OUT_DIR/TuneLab.CloudVoice-$VERSION.tlx"
mkdir -p "$OUT_DIR"
rm -f "$TLX"

# 包内只放必要文件：SDK dll 由宿主共享（Private=false，不会出现在这里），
# pdb / deps.json 是本地调试产物，不随包分发。
(
    cd "$ROOT/artifacts/package"
    zip -q -r "$TLX" . -x '*.pdb' '*.deps.json' '.DS_Store'
)

# 注意别用 unzip | grep -q：grep 提前退出会给 unzip 发 SIGPIPE，set -o pipefail 下会被误判成失败。
LISTING="$(unzip -l "$TLX")"
case "$LISTING" in
    *" manifest.json"*) ;;
    *) echo "Packaging failed: manifest.json is not at the .tlx root." >&2; exit 1 ;;
esac

echo "==> Packed: $TLX"
unzip -l "$TLX" | sed -n '1,20p'