#!/usr/bin/env bash
# 启动 tunelab-cloudvoice/1 服务端（DiffSinger / OpenUTAU 声库）。
#
#   demo/DiffSingerCloud/scripts/run.sh                        # --backend auto --port 18791
#   demo/DiffSingerCloud/scripts/run.sh --port 8080 --backend cpu
#   demo/DiffSingerCloud/scripts/run.sh --voice-dir /path/to/OpenUTAU --api-key secret
#
# 除下面写死的默认值（--backend auto --port 18791）外，所有参数原样传给 python -m server；
# 位置靠后的参数优先，所以显式传 --backend / --port 会覆盖默认值。
# 环境变量 PYTHON 可覆盖解释器（默认优先用同目录 .venv/bin/python）。
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "${PYTHON:-}" ]; then
    if [ -x "$HERE/.venv/bin/python" ]; then
        PYTHON="$HERE/.venv/bin/python"
    else
        PYTHON="python3"
        echo "warning: $HERE/.venv not found; falling back to python3" >&2
        echo "         run $HERE/scripts/setup.sh to create the venv" >&2
    fi
fi

cd "$HERE"
exec "$PYTHON" -m server --backend auto --port 18791 "$@"
