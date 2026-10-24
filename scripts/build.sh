#!/usr/bin/env bash
# 编译插件（Release 默认），输出到 artifacts/package/。
# 依赖：.NET 8 SDK（仓库内 TuneLab/ 源码会自动一起编译；缺失时先跑 scripts/update-tunelab.sh）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-Release}"

export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1
export AVALONIA_TELEMETRY_OPTOUT=1

if [ ! -d "$ROOT/TuneLab/TuneLab.SDK" ]; then
    echo "TuneLab source is missing. Run scripts/update-tunelab.sh first." >&2
    exit 1
fi

# 清掉旧输出，避免已删除的内容文件留在包里。
rm -rf "$ROOT/artifacts/package"
echo "==> Building plugin ($CONFIG)"
dotnet build "$ROOT/src/TuneLab.CloudVoice/TuneLab.CloudVoice.csproj" -c "$CONFIG"
echo "==> Package folder: $ROOT/artifacts/package"
ls -1 "$ROOT/artifacts/package"
