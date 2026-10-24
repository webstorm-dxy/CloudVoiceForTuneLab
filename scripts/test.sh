#!/usr/bin/env bash
# 跑单元测试（协议 / 音频解码 / G2P / 设置 / 声库目录 / cloud provider / mock provider）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-Debug}"

export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1

dotnet test "$ROOT/tests/TuneLab.CloudVoice.Tests/TuneLab.CloudVoice.Tests.csproj" -c "$CONFIG" "$@"
