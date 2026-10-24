#!/usr/bin/env bash
# 拉取 / 更新 TuneLab 最新源码到仓库根目录的 TuneLab/（插件编译所依赖的 SDK 来源）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TUNELAB_DIR="$ROOT/TuneLab"
REPO="https://github.com/LiuYunPlayer/TuneLab.git"

if [ -d "$TUNELAB_DIR/.git" ]; then
    echo "==> Updating TuneLab source in $TUNELAB_DIR"
    git -C "$TUNELAB_DIR" fetch --prune origin
    git -C "$TUNELAB_DIR" pull --ff-only
else
    echo "==> Cloning TuneLab into $TUNELAB_DIR"
    git clone "$REPO" "$TUNELAB_DIR"
fi

git -C "$TUNELAB_DIR" log -1 --format='==> TuneLab HEAD: %h %ad %s' --date=iso
