#!/usr/bin/env bash
# DiffSinger Cloud ↔ TuneLab 宿主端到端验证（无头 TuneLab.Cli + 通用 CloudVoice 插件）。
#
# 覆盖：服务端启动 -> 插件安装到沙箱 TuneLab -> 设置 Service URL -> 断言 /voices 目录被插件拉到
#       -> 无头宿主合成单音符并读回音素 -> 建两音符工程 synthesize + export-audio -> 断言 WAV 非空。
#
# 前置（脚本会尽量自动准备，失败时给出提示）：
#   - TuneLab/ 源码已克隆：scripts/update-tunelab.sh
#   - TuneLab.Cli 已编译：dotnet build TuneLab/TuneLab.Cli/TuneLab.Cli.csproj -c Debug
#   - 插件 .tlx 已打包：scripts/pack.sh（缺失时脚本会尝试自动打包）
#   - demo/DiffSingerCloud/.venv 已就绪：scripts/setup.sh
#   - OpenUTAU/ 下有真实声库更好；没有时自动生成一个微型合成测试声库
#
# 用法：
#   demo/DiffSingerCloud/scripts/tunelab-e2e.sh
#   PORT=18794 BACKEND=cpu ./scripts/tunelab-e2e.sh
#   KEEP=1 ./scripts/tunelab-e2e.sh          # 保留沙箱目录便于排查
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$HERE/.venv/bin/python"
CLI="${TUNELAB_CLI:-$ROOT/TuneLab/TuneLab.Cli/bin/Debug/net8.0/TuneLab.Cli}"
SANDBOX="${DIFFSINGER_E2E_DIR:-$ROOT/.sandbox/diffsinger-e2e}"
PORT="${PORT:-18793}"
BACKEND="${BACKEND:-cpu}"
SERVER_LOG="$SANDBOX/server.log"
SERVER_PID=""

say() { printf '==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ "${KEEP:-0}" != "1" ]; then rm -rf "$SANDBOX"; fi
}
trap cleanup EXIT

wait_health() {
    for _ in $(seq 1 200); do
        if "$PY" - "$PORT" <<'PY' >/dev/null 2>&1
import sys, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:%s/health" % sys.argv[1], timeout=1) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
PY
        then return 0; fi
        sleep 0.3
    done
    return 1
}

[ -x "$PY" ] || fail "Python venv not found at $PY - run demo/DiffSingerCloud/scripts/setup.sh first"
[ -f "$ROOT/scripts/install.sh" ] || fail "not a TuneLabRemotePlugin checkout: $ROOT"

# ---- 1) TuneLab.Cli ----------------------------------------------------------------
if [ ! -x "$CLI" ]; then
    say "TuneLab.Cli not built, trying dotnet build"
    command -v dotnet >/dev/null 2>&1 || fail "dotnet not found and $CLI missing (install .NET 8 SDK, see README)"
    [ -d "$ROOT/TuneLab/TuneLab.Cli" ] || fail "TuneLab/ source missing - run scripts/update-tunelab.sh first"
    dotnet build "$ROOT/TuneLab/TuneLab.Cli/TuneLab.Cli.csproj" -c Debug >/dev/null || fail "TuneLab.Cli build failed"
fi
[ -x "$CLI" ] || fail "TuneLab.Cli still missing at $CLI"

# ---- 2) 插件 .tlx ------------------------------------------------------------------
TLX="$(ls -t "$ROOT"/artifacts/tlx/*.tlx 2>/dev/null | head -1 || true)"
if [ -z "$TLX" ] || [ ! -f "$TLX" ]; then
    say "plugin .tlx not found, running scripts/pack.sh"
    "$ROOT/scripts/pack.sh" >/dev/null 2>&1 || fail "scripts/pack.sh failed"
    TLX="$(ls -t "$ROOT"/artifacts/tlx/*.tlx 2>/dev/null | head -1 || true)"
fi
[ -n "$TLX" ] && [ -f "$TLX" ] || fail "no .tlx package available"

# ---- 3) 声库：真实优先，没有就生成合成测试声库 --------------------------------------
rm -rf "$SANDBOX"
mkdir -p "$SANDBOX/data"
VOICE_ARGS=(--voice-dir "$HERE/OpenUTAU")
VOICE_COUNT="$("$PY" -m server --list-voices --voice-dir "$HERE/OpenUTAU" 2>/dev/null | "$PY" -c 'import json,sys; print(len(json.load(sys.stdin)["voices"]))' 2>/dev/null || echo 0)"
if [ "$VOICE_COUNT" = "0" ]; then
    say "OpenUTAU/ has no voicebank; generating a tiny synthetic test voicebank"
    "$PY" "$HERE/scripts/make_test_voicebank.py" --output "$SANDBOX/voicebanks/TestVoice" --force --quiet
    VOICE_ARGS=(--voice-dir "$SANDBOX/voicebanks")
fi

# ---- 4) 启动 DiffSinger 服务端 -----------------------------------------------------
say "starting DiffSinger server on 127.0.0.1:$PORT (backend=$BACKEND)"
( cd "$HERE" && exec "$PY" -m server --port "$PORT" --backend "$BACKEND" "${VOICE_ARGS[@]}" ) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
wait_health || { tail -40 "$SERVER_LOG"; fail "server did not become healthy"; }
say "server ready: $(grep -c 'ready:' "$SERVER_LOG" 2>/dev/null || true)"

# ---- 5) 安装插件到沙箱 TuneLab -----------------------------------------------------
export TUNELAB_DATA_DIR="$SANDBOX/data"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
say "installing CloudVoice plugin into sandbox TuneLab"
"$ROOT/scripts/install.sh" "$TLX" >/dev/null

LIST="$("$CLI" --headless extension list --json 2>&1)"
case "$LIST" in *'"status": "Loaded"'*) ;; *) fail "plugin not loaded: $LIST" ;; esac
say "plugin loaded"

say "pointing CloudVoice at the DiffSinger server"
"$CLI" --headless extension set-setting --extension CloudVoice --key endpoint --value "http://127.0.0.1:$PORT/synthesize" --yes >/dev/null

CATALOG="$("$CLI" --headless source list --engine CloudVoice 2>&1)"
case "$CATALOG" in *DiffSinger*) ;; *) fail "server voices not visible in TuneLab source list: $CATALOG" ;; esac
printf '    %s\n' "$(echo "$CATALOG" | head -3)"
VOICE="$("$PY" - "$PORT" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen("http://127.0.0.1:%s/voices" % sys.argv[1], timeout=30) as response:
    voices = json.loads(response.read())["voices"]
preferred = next((v["id"] for v in voices if "JiHeHong" in v["id"]), None)
print(preferred or voices[0]["id"])
PY
)"
say "using voice: $VOICE"

# ---- 6) 无头宿主合成 + 读回音素 -----------------------------------------------------
PROBE="$("$CLI" --headless sandbox run --code '
const v = sandbox.voices().find(x => x.type === "CloudVoice" && x.id === "'"$VOICE"'") || sandbox.voices().find(x => x.type === "CloudVoice" && !x.id.startsWith("mock-"));
if (!v) throw new Error("no cloud voice available");
const track = tl.currentProject().addTrack();
const part = track.addPart({pos: 0, endOffset: 1920});
part.setSoundSource({kind: "voice", type: v.Type, id: v.Id});
const note = part.addNote({pos: 0, dur: 480, pitch: 69, lyric: "la"});
const result = sandbox.synthesize(part);
print("SYNTH=" + JSON.stringify(result));
print("SYLLABLE=" + JSON.stringify(sandbox.syllable(note)));
' 2>&1)"
case "$PROBE" in *'"Done":true'*) ;; *) fail "headless synthesis failed: $PROBE" ;; esac
case "$PROBE" in *'"Symbols"'*) ;; *) fail "no phonemes returned: $PROBE" ;; esac
say "headless synthesis OK: $(echo "$PROBE" | grep -o '"Ms":[0-9]*' | head -1)"

# ---- 7) 工程 synthesize + export-audio ---------------------------------------------
mkdir -p "$SANDBOX/project"
cat > "$SANDBOX/project/commands.txt" <<EOF
script run --code "const t = tl.currentProject().addTrack(); const p = t.addPart({pos: 0, endOffset: 1920}); p.setSoundSource({kind: 'voice', type: 'CloudVoice', id: '$VOICE'}); p.addNote({pos: 0, dur: 480, pitch: 69, lyric: 'la'}); p.addNote({pos: 480, dur: 480, pitch: 72, lyric: 'ma'});" --yes
project synthesize
project export --path $SANDBOX/project/probe.tlpx --yes
EOF
"$CLI" --headless --commands "$SANDBOX/project/commands.txt" >"$SANDBOX/project/build.log" 2>&1 \
    || { tail -30 "$SANDBOX/project/build.log"; fail "building the probe project failed"; }
"$CLI" --headless --project "$SANDBOX/project/probe.tlpx" project export-audio --path "$SANDBOX/project/probe.wav" --yes \
    >"$SANDBOX/project/export.log" 2>&1 || { tail -30 "$SANDBOX/project/export.log"; fail "audio export failed"; }
[ -s "$SANDBOX/project/probe.wav" ] || fail "audio export produced no file"
WAV_BYTES="$(wc -c < "$SANDBOX/project/probe.wav" | tr -d ' ')"
[ "$WAV_BYTES" -gt 10000 ] || fail "audio export too small ($WAV_BYTES bytes)"
"$PY" - "$SANDBOX/project/probe.wav" <<'PY'
import sys, wave
import numpy as np
with wave.open(sys.argv[1], "rb") as wav_file:
    rate = wav_file.getframerate()
    frames = wav_file.getnframes()
    channels = wav_file.getnchannels()
    width = wav_file.getsampwidth()
    raw = wav_file.readframes(frames)
dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
if width == 2:
    data /= 32768.0
if channels > 1:
    data = data.reshape(-1, channels).mean(axis=1)
rms = float(np.sqrt(np.mean(data * data)))
if rms <= 1e-3:
    raise SystemExit("exported audio is silent (rms=%.5f)" % rms)
print("    wav: %.2fs @ %dHz, rms=%.4f, peak=%.3f" % (frames / rate, rate, rms, float(np.max(np.abs(data)))))
PY

# ---- 8) 服务端日志抽查 --------------------------------------------------------------
grep -q "synthesized voice=" "$SERVER_LOG" || fail "server did not log any synthesis"
if grep -q "dsdur timing failed" "$SERVER_LOG"; then
    echo "    warning: dsdur 时长预测发生了回退（真实声库上不应出现；合成声库正常）"
fi

echo "ALL TUNELAB E2E CHECKS PASSED (voice=$VOICE backend=$BACKEND wav=$WAV_BYTES bytes)"
