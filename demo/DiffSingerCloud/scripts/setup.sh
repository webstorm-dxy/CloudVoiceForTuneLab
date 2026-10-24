#!/usr/bin/env bash
# 创建 / 复用 demo/DiffSingerCloud/.venv 并安装服务端依赖。
#
#   demo/DiffSingerCloud/scripts/setup.sh              # 首次创建或补齐依赖（幂等）
#   demo/DiffSingerCloud/scripts/setup.sh --recreate   # 删除旧 venv 重建
#
# 可用环境变量：
#   PYTHON=/path/to/python3.12   指定用于创建 venv 的解释器（要求 >= 3.10）
#   PIP_INDEX_URL=...            pip 镜像（透传给 pip）
#   SKIP_DEV=1                   不安装验证/测试用的轻量包（onnx / pytest）
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HERE/.venv"
REQUIREMENTS="$HERE/requirements.txt"
RECREATE=0
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=1 ;;
        -h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -f "$REQUIREMENTS" ] || die "requirements.txt not found: $REQUIREMENTS"

pick_python() {
    if [ -n "${PYTHON:-}" ]; then
        command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON=$PYTHON is not executable"
        printf '%s\n' "$PYTHON"
        return 0
    fi
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    die "no python3 interpreter found; install Python >= 3.10 or set PYTHON=..."
}

if [ "$RECREATE" = "1" ] && [ -d "$VENV" ]; then
    say "removing existing venv: $VENV"
    rm -rf "$VENV"
fi

if [ ! -x "$VENV/bin/python" ]; then
    BASE_PYTHON="$(pick_python)"
    "$BASE_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "error: Python >= 3.10 required, got " + sys.version.split()[0])' \
        || die "Python >= 3.10 is required for $BASE_PYTHON"
    say "creating venv with $BASE_PYTHON: $VENV"
    "$BASE_PYTHON" -m venv "$VENV" || die "failed to create venv (on Debian/Ubuntu install python3-venv)"
else
    say "reusing venv: $VENV"
fi
PY="$VENV/bin/python"

say "python $("$PY" -V 2>&1)"
say "installing runtime requirements: requirements.txt"
"$PY" -m pip install --disable-pip-version-check -r "$REQUIREMENTS"

if [ "${SKIP_DEV:-0}" != "1" ]; then
    # 验证设施需要 onnx（scripts/make_test_voicebank.py 手工构建测试模型）与 pytest（tests/）。
    if ! "$PY" -c 'import onnx, pytest' >/dev/null 2>&1; then
        say "installing verification extras: onnx pytest"
        "$PY" -m pip install --disable-pip-version-check onnx pytest \
            || say "warning: could not install onnx/pytest; synthetic-bank verification and tests will not run"
    fi
fi

say "checking imports"
"$PY" - <<'PY' || die "dependency check failed"
import fastapi, numpy, onnxruntime, pydantic, pypinyin, uvicorn, yaml
print("  fastapi %s | uvicorn %s | pydantic %s | numpy %s | onnxruntime %s | pypinyin %s"
      % (fastapi.__version__, uvicorn.__version__, pydantic.__version__,
         numpy.__version__, onnxruntime.__version__, pypinyin.__version__))
try:
    import onnx
    print("  onnx %s (test voicebank tooling)" % onnx.__version__)
except ImportError:
    print("  onnx missing (test voicebank tooling will not run)")
PY

say "ready"
cat <<EOF
    python:      $PY
    start:       $HERE/scripts/run.sh --port 18791
    verify:      $PY $HERE/scripts/verify.py
    test voicebank: $PY $HERE/scripts/make_test_voicebank.py
EOF
