#!/usr/bin/env bash
# 端到端集成验证：真实 TuneLab 无头宿主加载插件 → 建声源片段 → 触发合成 → 读出合成音素 / 渲染音频。
# 覆盖两条路径：
#   1) 内置 mock 声库（无需网络）
#   2) HTTP 云端 provider（指向 tools/mock_cloud_server.py 起的本地 mock 服务）：
#      只配置 service URL，声库目录由插件自动 GET /voices 拉取，再校验协议请求/响应与音素回填
#
# 前置：
#   - TuneLab 源码在 TuneLab/（scripts/update-tunelab.sh）
#   - 已编译 TuneLab.Cli：dotnet build TuneLab/TuneLab.Cli/TuneLab.Cli.csproj -c Debug
#   - python3（跑 mock 云服务）；可用环境变量 SKIP_HTTP=1 跳过第 2 段
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="${TUNELAB_CLI:-$ROOT/TuneLab/TuneLab.Cli/bin/Debug/net8.0/TuneLab.Cli}"
CONFIG="${CONFIG:-Release}"
SANDBOX="$ROOT/.sandbox/integration"
PORT="${MOCK_CLOUD_PORT:-18766}"
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
assert_contains() {
    case "$1" in
        *"$2"*) ;;
        *) fail "expected '$2' in output: $1" ;;
    esac
}

[ -x "$CLI" ] || fail "TuneLab.Cli not found at $CLI - build it: dotnet build TuneLab/TuneLab.Cli/TuneLab.Cli.csproj -c Debug"
command -v python3 >/dev/null 2>&1 || fail "python3 is required for the mock cloud server"
rm -rf "$SANDBOX"
mkdir -p "$SANDBOX/data"

echo "==> Build + pack plugin"
CONFIG="$CONFIG" "$ROOT/scripts/pack.sh" >/dev/null
TLX="$(ls -t "$ROOT"/artifacts/tlx/*.tlx | head -1)"

echo "==> Install into sandbox data dir"
TUNELAB_DATA_DIR="$SANDBOX/data" "$ROOT/scripts/install.sh" "$TLX" >/dev/null
export TUNELAB_DATA_DIR="$SANDBOX/data"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1

echo "==> Host loads the extension"
LIST="$($CLI --headless extension list --json 2>&1)"
assert_contains "$LIST" '"status": "Loaded"'
assert_contains "$LIST" '"CloudVoice"'
echo "    loaded OK"

echo "==> Mock provider: synthesize a note and read its phonemes"
PROBE="$($CLI --headless sandbox run --code '
const v = sandbox.voices().find(x => x.type === "CloudVoice" && x.id === "mock-soprano");
const track = tl.currentProject().addTrack();
const part = track.addPart({pos: 0, endOffset: 1920});
part.setSoundSource({kind: "voice", type: v.Type, id: v.Id});
const note = part.addNote({pos: 0, dur: 480, pitch: 60, lyric: "la"});
const result = sandbox.synthesize(part);
const syllable = sandbox.syllable(note);
print("SYNTH=" + JSON.stringify(result));
print("SYLLABLE=" + JSON.stringify(syllable));
' 2>&1)"
assert_contains "$PROBE" '"Done":true'
assert_contains "$PROBE" 'leading'
assert_contains "$PROBE" 'body'
echo "    mock synthesis OK"

echo "==> Render audio through the real host"
mkdir -p "$SANDBOX/project"
cat > "$SANDBOX/project/commands.txt" <<EOF
script run --code "const t = tl.currentProject().addTrack(); const p = t.addPart({pos: 0, endOffset: 1920}); p.setSoundSource({kind: 'voice', type: 'CloudVoice', id: 'mock-soprano'}); p.addNote({pos: 0, dur: 480, pitch: 60, lyric: 'la'}); p.addNote({pos: 480, dur: 480, pitch: 62, lyric: 'ma'});" --yes
project synthesize
project export --path $SANDBOX/project/probe.tlpx --yes
EOF
$CLI --headless --commands "$SANDBOX/project/commands.txt" >"$SANDBOX/project/build.log" 2>&1 || fail "building the probe project failed: $(cat "$SANDBOX/project/build.log")"
$CLI --headless --project "$SANDBOX/project/probe.tlpx" project export-audio --path "$SANDBOX/project/probe.wav" --yes >"$SANDBOX/project/export.log" 2>&1 \
    || fail "audio export failed: $(cat "$SANDBOX/project/export.log")"
[ -s "$SANDBOX/project/probe.wav" ] || fail "audio export produced no file"
WAV_BYTES="$(wc -c < "$SANDBOX/project/probe.wav" | tr -d ' ')"
[ "$WAV_BYTES" -gt 10000 ] || fail "audio export too small ($WAV_BYTES bytes)"
echo "    audio export OK ($WAV_BYTES bytes)"

if [ "${SKIP_HTTP:-0}" = "1" ]; then
    echo "==> Skipping cloud HTTP provider probe (SKIP_HTTP=1)"
    echo "ALL INTEGRATION CHECKS PASSED"
    exit 0
fi

echo "==> Cloud HTTP provider: local mock cloud server on port $PORT"
python3 "$ROOT/tools/mock_cloud_server.py" --port "$PORT" --log "$SANDBOX/cloud-requests.jsonl" >"$SANDBOX/cloud-server.log" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 50); do
    if python3 -c "import socket,sys; s=socket.socket(); sys.exit(0 if s.connect_ex(('127.0.0.1', $PORT))==0 else 1)"; then break; fi
    sleep 0.1
done

# The plugin only needs the service URL: the voice catalog is pulled from GET /voices automatically.
$CLI --headless extension set-setting --extension CloudVoice --key endpoint --value "http://127.0.0.1:$PORT/synthesize" --yes >/dev/null
CATALOG="$($CLI --headless source list --engine CloudVoice 2>&1)"
assert_contains "$CATALOG" "test-voice"

CLOUD_PROBE="$($CLI --headless sandbox run --code '
const v = sandbox.voices().find(x => x.type === "CloudVoice" && x.id === "test-voice");
const track = tl.currentProject().addTrack();
const part = track.addPart({pos: 0, endOffset: 1920});
part.setSoundSource({kind: "voice", type: v.Type, id: v.Id});
const note = part.addNote({pos: 0, dur: 480, pitch: 60, lyric: "la"});
const result = sandbox.synthesize(part);
print("SYNTH=" + JSON.stringify(result));
print("SYLLABLE=" + JSON.stringify(sandbox.syllable(note)));
' 2>&1)"
assert_contains "$CLOUD_PROBE" '"Done":true'
# mock 云服务故意回 srv- 前缀音素，证明走的是云端响应而不是本地 G2P 兜底。
assert_contains "$CLOUD_PROBE" 'srv-la'

REQUESTS="$(cat "$SANDBOX/cloud-requests.jsonl" 2>/dev/null || true)"
[ -n "$REQUESTS" ] || fail "the mock cloud server received no request"
assert_contains "$REQUESTS" '"protocol": "tunelab-cloudvoice"'
assert_contains "$REQUESTS" '"voice": "test-voice"'
assert_contains "$REQUESTS" '"pitch"'
# schema 驱动的 part / note 属性确实随请求发出；后端声明 auto pitch 时 pitchMode 为 auto。
assert_contains "$REQUESTS" '"partProperties"'
assert_contains "$REQUESTS" '"style": "natural"'
assert_contains "$REQUESTS" '"tone": 0.5'
assert_contains "$REQUESTS" '"pitchMode": "auto"'
echo "    cloud provider OK (request logged, schema properties sent, response phonemes used)"

echo "ALL INTEGRATION CHECKS PASSED"