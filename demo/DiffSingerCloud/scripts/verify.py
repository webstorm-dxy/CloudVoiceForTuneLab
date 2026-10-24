#!/usr/bin/env python3
"""tunelab-cloudvoice/1 服务端端到端验证（只依赖标准库 + numpy）。

覆盖三件事：

1. HTTP 契约：GET /health、GET /voices、GET /schema 的结构与协议字段；
2. 真实合成：单音 2s、pitch=69（A4=440Hz）、lyric=啊，检查
   HTTP 200、base64 音频可解码、时长约等于块长、RMS 非静音、FFT 主频 440 +-5%、
   phonemes 含元音、pitch / parameters 回显结构合法；
3. 错误语义：错误 protocol / 错误 version / 未知 voice / endTime<=startTime 都返回 4xx。
4. 自动音高：/schema pitch.auto=true + partProperties.PitchMode；有 auto-pitch 声库（如 yousa 的
   dspitch）时验证 predict（meta.pitchSource=auto、pitch echo 非平坦、音频基频与 echo 中位数差 <1
   半音）与 tunelab（meta.pitchSource=tunelab、echo 恒等于请求曲线）两条路径；没有 auto-pitch
   声库时该步骤 SKIP（合成声库/干净环境仍 PASS）。

声库来源：默认自动探测 demo/DiffSingerCloud/OpenUTAU/ 下的真实声库（优先 JiHeHong），
没有真实声库时用 scripts/make_test_voicebank.py 现场生成一个微型合成声库（干净环境可用）。

服务端来源：默认用 .venv/bin/python -m server 起一个子进程（--backend 透传，便于
CPU/MPS/CUDA 矩阵验证）；也可以用 --url 连已经跑起来的服务，或用 --server-cmd 换命令。

用法::

    # 默认：自动选声库（有真实声库优先）+ 自己起服务（backend=auto, port=18791）
    scripts/verify.py

    # 强制合成声库（交付验证路径 / 干净环境）+ CPU 后端
    scripts/verify.py --voice-source synthetic --backend cpu

    # 真实声库 + 指定 GPU 后端
    scripts/verify.py --voice-source real --backend mps

    # 连已有服务
    scripts/verify.py --url http://127.0.0.1:18791 --voice JiHeHong-diffsinger-openutau-V2

退出码：0 = 全部通过；1 = 有断言失败 / 服务启动失败。
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = DEMO_DIR.parent.parent
DEFAULT_PYTHON = DEMO_DIR / ".venv" / "bin" / "python"
DEFAULT_VOICE_ROOT = DEMO_DIR / "OpenUTAU"
MAKE_TEST_VOICEBANK = DEMO_DIR / "scripts" / "make_test_voicebank.py"

PROTOCOL = "tunelab-cloudvoice"
VERSION = 1

TEST_PITCH = 69          # MIDI 半音：A4 = 440 Hz
TEST_HZ = 440.0
TEST_START = 0.0
TEST_END = 2.0           # 单音 2s
TEST_NOTE_KEY = "n1"
TEST_LYRIC = "啊"
REQUEST_SAMPLE_RATE = 48000

SUPPORTED_AUDIO_FORMATS = ("pcm_f32le", "pcm_s16le", "wav")
# 判定"响应音素里有元音"用的集合（覆盖中文拼音韵母 / 日语罗马字 / 英语 ARPABET 常见元音）。
VOWEL_SYMBOLS = {
    "a", "e", "i", "o", "u", "v", "y",
    "aa", "ae", "ah", "ao", "aw", "ax", "ay",
    "eh", "er", "ey", "ih", "iy", "oh", "oo", "ou", "ow", "oy", "uh", "uw",
    "ai", "an", "ang", "ei", "en", "eng", "ia", "ian", "iang", "iao", "ie", "in", "ing",
    "iong", "iu", "ong", "ua", "uai", "uan", "uang", "ui", "un", "uo", "van", "ve", "vn",
}


# --------------------------------------------------------------------------------------
# 错误 / 输出
# --------------------------------------------------------------------------------------
class VerifyFailure(AssertionError):
    """一条断言失败（会被打印成 FAIL，并以退出码 1 结束）。"""


class TransportError(ConnectionError):
    """网络层失败（连接拒绝 / 超时等），与 HTTP 4xx 区分。"""


class StepSkipped(Exception):
    """步骤跳过（环境里没有对应能力，例如没有 auto-pitch 声库）：计入 skipped，不算失败。"""


def check(condition: Any, message: str) -> None:
    if not condition:
        raise VerifyFailure(message)


class Harness:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.started = time.perf_counter()
        self.index = 0
        self.total = 9 if args.url else 10
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.warnings: list[str] = []
        self._details: list[str] = []

        self.python = self._resolve_python()
        self.url = (args.url or ("http://" + args.host + ":" + str(args.port))).rstrip("/")
        self.tmpdir: Optional[Path] = None
        self._server_log_handle: Any = None
        self.log_path: Optional[Path] = None
        self.process: Optional[subprocess.Popen] = None

        self.source = "remote" if args.url else "unknown"
        self.voice_dirs: list[Path] = []
        self.real_banks: list[Path] = []
        self.synthetic_bank: Optional[Path] = None
        self.voice_id: Optional[str] = None
        self.voices: list[dict[str, Any]] = []
        self.schema: dict[str, Any] = {}
        self.health: dict[str, Any] = {}
        self.response: dict[str, Any] = {}
        self.samples: Optional[np.ndarray] = None
        self.sample_rate = 0

    # ---- 小工具 ---------------------------------------------------------------------
    def _resolve_python(self) -> Path:
        if self.args.python:
            path = Path(self.args.python).expanduser()
            check(path.is_file(), "--python 不存在: " + str(path))
            return path
        if DEFAULT_PYTHON.is_file():
            return DEFAULT_PYTHON
        return Path(sys.executable)

    def detail(self, message: str) -> None:
        self._details.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        self._details.append("warning: " + message)

    @contextmanager
    def step(self, title: str) -> Iterator[None]:
        self.index += 1
        self._details = []
        started = time.perf_counter()
        label = "[%d/%d] %s " % (self.index, self.total, title)
        try:
            yield
        except StepSkipped as reason:
            self.skipped += 1
            elapsed = time.perf_counter() - started
            dots = "." * max(3, 66 - len(label))
            print(label + dots + " SKIP (%.2fs)" % elapsed)
            if str(reason):
                print("    " + str(reason))
            for line in self._details:
                print("    " + line)
        except BaseException as error:
            self.failed += 1
            elapsed = time.perf_counter() - started
            dots = "." * max(3, 66 - len(label))
            print(label + dots + " FAIL (%.2fs)" % elapsed)
            for line in self._details:
                print("    " + line)
            raise
        else:
            self.passed += 1
            elapsed = time.perf_counter() - started
            dots = "." * max(3, 66 - len(label))
            print(label + dots + " PASS (%.2fs)" % elapsed)
            for line in self._details:
                print("    " + line)

    # ---- 主流程 ---------------------------------------------------------------------
    def run(self) -> None:
        self.print_header()
        if not self.args.url:
            with self.step("准备测试声库（真实优先，否则合成）"):
                self.prepare_voicebank()
        with self.step("启动 / 连接服务端"):
            self.start_server()
        with self.step("GET /health"):
            self.check_health()
        with self.step("GET /voices + GET /schema"):
            self.check_catalog()
        with self.step("选择声库"):
            self.select_voice()
        with self.step("POST /synthesize（单音 2s, pitch=69/A4）"):
            self.synthesize()
        with self.step("校验音频（长度 / RMS / 音高）"):
            self.check_audio()
        with self.step("校验回显（phonemes / pitch / parameters）"):
            self.check_echoes()
        with self.step("自动音高（PitchMode=predict / tunelab）"):
            self.check_auto_pitch()
        with self.step("错误路径（protocol / version / voice / endTime）"):
            self.check_errors()

    def print_header(self) -> None:
        print("tunelab-cloudvoice/1 end-to-end verification")
        print("  service:   " + self.url + ("  (remote --url)" if self.args.url else "  (managed subprocess)"))
        print("  backend:   " + self.args.backend + ("  (passed to server)" if not self.args.url else "  (reported by service)"))
        print("  python:    " + str(self.python))
        print("  voicebank: " + ("auto (real first, synthetic fallback)" if self.args.voice_source == "auto"
                                 else self.args.voice_source))
        print()

    # ---- 1. 声库 --------------------------------------------------------------------
    @staticmethod
    def scan_voicebanks(roots: list[Path]) -> list[Path]:
        """扫描每个 root 的直接子目录，找有 character.txt + dsconfig.yaml 的声库。"""
        banks: list[Path] = []
        for root in roots:
            root = Path(root).expanduser()
            if not root.is_dir():
                continue
            try:
                children = sorted(root.iterdir())
            except OSError:
                continue
            for child in children:
                try:
                    if not child.is_dir() or child.name.startswith("."):
                        continue
                    if (child / "character.txt").is_file() and (child / "dsconfig.yaml").is_file():
                        banks.append(child)
                except OSError:
                    continue
        return banks

    def make_tempdir(self) -> Path:
        if self.tmpdir is None:
            self.tmpdir = Path(tempfile.mkdtemp(prefix="tunelab-verify-"))
        return self.tmpdir

    def require_tempdir(self, owner: str) -> Path:
        if self.tmpdir is None:
            self.tmpdir = Path(tempfile.mkdtemp(prefix=owner + "-"))
        return self.tmpdir

    def prepare_voicebank(self) -> None:
        args = self.args
        roots = [Path(item).expanduser() for item in args.voice_dir] if args.voice_dir else [DEFAULT_VOICE_ROOT]
        real_banks = self.scan_voicebanks(roots)
        if args.voice_source != "synthetic" and real_banks:
            self.source = "real"
            self.voice_dirs = roots
            # JiHeHong 优先用于默认选声库。
            ordered = sorted(real_banks, key=lambda bank: (0 if "jihehong" in bank.name.lower() else 1, bank.name))
            self.real_banks = ordered
            self.detail("发现 %d 个真实声库: %s" % (len(real_banks), ", ".join(bank.name for bank in ordered)))
            self.detail("服务端 voice 目录: " + ", ".join(str(path) for path in roots))
            return

        if args.voice_source == "real":
            raise VerifyFailure(
                "--voice-source real 但这些目录下没有真实声库（需要 character.txt + dsconfig.yaml）: "
                + ", ".join(str(path) for path in roots)
                + "；不指定 --voice-source 会自动回退到合成声库"
            )
        if args.voice_dir:
            self.warn("--voice-source synthetic：忽略 --voice-dir（合成声库会生成在临时目录）")
        self.source = "synthetic"
        temp = self.require_tempdir("tunelab-verify-voicebank-")
        if args.synthetic_dir:
            parent = Path(args.synthetic_dir).expanduser()
            parent.mkdir(parents=True, exist_ok=True)
        else:
            parent = temp / "voicebanks"
        bank_dir = parent / "TestVoice"
        command = [
            str(self.python),
            str(MAKE_TEST_VOICEBANK),
            "--output", str(bank_dir),
            "--name", "TestVoice",
            "--force",
            "--quiet",
        ]
        started = time.perf_counter()
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise VerifyFailure("生成合成声库失败（无法执行 make_test_voicebank.py）: " + str(error))
        if result.returncode != 0:
            raise VerifyFailure(
                "生成合成声库失败（exit %d）:\n%s\n%s\n提示：先跑 scripts/setup.sh 安装依赖（onnx/numpy）"
                % (result.returncode, result.stdout.strip()[-1000:], result.stderr.strip()[-2000:])
            )
        self.synthetic_bank = bank_dir
        self.voice_dirs = [parent]
        self.detail("合成声库: " + str(bank_dir) + "（%.2fs）" % (time.perf_counter() - started))
        self.detail("服务端 voice 目录: " + str(parent))

    # ---- 2. 服务端 ------------------------------------------------------------------
    def build_server_command(self) -> list[str]:
        args = self.args
        self.require_tempdir("tunelab-verify-server-")
        voice_dir = str(self.voice_dirs[0]) if self.voice_dirs else ""
        if args.server_cmd:
            template = args.server_cmd
            command = [
                token.replace("{port}", str(args.port))
                     .replace("{host}", args.host)
                     .replace("{backend}", args.backend)
                     .replace("{voice_dir}", voice_dir)
                     .replace("{python}", str(self.python))
                for token in shlex.split(template)
            ]
            overridden = {key for key in ("{port}", "{host}", "{backend}", "{voice_dir}") if key in template}
        else:
            command = [str(self.python), "-m", "server"]
            overridden = set()
        if "--port" not in command and "{port}" not in overridden and not any(t.startswith("--port=") for t in command):
            command += ["--port", str(args.port)]
        if "--host" not in command and "{host}" not in overridden and not any(t.startswith("--host=") for t in command):
            command += ["--host", args.host]
        if "--backend" not in command and "{backend}" not in overridden and not any(t.startswith("--backend=") for t in command):
            command += ["--backend", args.backend]
        if args.api_key and "--api-key" not in command:
            command += ["--api-key", args.api_key]
        if "{voice_dir}" not in overridden:
            for directory in self.voice_dirs:
                command += ["--voice-dir", str(directory)]
        for directory in args.vocoder_dir:
            command += ["--vocoder-dir", str(Path(directory).expanduser())]
        return command

    def ensure_port_free(self) -> None:
        """启动前确认端口没有被别的服务占用。

        否则 health 探活会连到别人启动的服务：verify 的 PASS/FAIL 就不可信了
        （声库、后端都可能不是本次 --backend/--voice-dir 指定的）。
        """
        probe_host = str(self.args.host)
        if probe_host in ("", "0.0.0.0", "::", "::0"):
            probe_host = "127.0.0.1"
        import socket

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(1.0)
        try:
            result = probe.connect_ex((probe_host, int(self.args.port)))
        finally:
            probe.close()
        if result == 0:
            raise VerifyFailure(
                "端口 " + probe_host + ":" + str(self.args.port) + " 已有服务在监听："
                "verify 会误连到它而不是自己启动的服务（声库/后端都可能不是预期的）。"
                "请换一个 --port，或改用 --url 直接验证那个服务。"
            )

    def start_server(self) -> None:
        if self.args.url:
            self.wait_health(float(self.args.startup_timeout), require_process=False)
            self.detail("连接已有服务成功: " + self.url)
            return
        self.ensure_port_free()
        command = self.build_server_command()
        log_path = Path(self.args.server_log).expanduser() if self.args.server_log else self.require_tempdir("x") / "server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = log_path
        self._server_log_handle = open(log_path, "wb")
        self.detail("命令: " + " ".join(shlex.quote(part) for part in command))
        try:
            self.process = subprocess.Popen(
                command,
                cwd=str(DEMO_DIR),
                stdin=subprocess.DEVNULL,
                stdout=self._server_log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            raise VerifyFailure("无法启动服务端进程: " + str(error))
        self.detail("pid=%d, log=%s" % (self.process.pid, log_path))
        started = time.perf_counter()
        self.wait_health(float(self.args.startup_timeout), require_process=True)
        self.detail("服务端就绪（%.2fs）: %s" % (time.perf_counter() - started, self.url))

    def wait_health(self, timeout: float, *, require_process: bool) -> None:
        deadline = time.time() + timeout
        last_error = "no attempt"
        while time.time() < deadline:
            if require_process and self.process is not None and self.process.poll() is not None:
                raise VerifyFailure(
                    "服务端进程在启动阶段退出（exit=%s）" % (self.process.returncode,)
                )
            try:
                status, body, _text = http_request(self.url + "/health", timeout=3.0)
                if status == 200:
                    self.health = body if isinstance(body, dict) else {}
                    return
                last_error = "HTTP %d" % status
            except TransportError as error:
                last_error = str(error)
            time.sleep(0.2)
        raise VerifyFailure("服务端 %.0fs 内未就绪（最后错误: %s）" % (timeout, last_error))

    def check_health(self) -> None:
        status, body, text = self.http_json("/health")
        check(status == 200, "GET /health 期望 200，实际 HTTP %d: %s" % (status, text[:300]))
        check(isinstance(body, dict), "GET /health 响应不是 JSON object")
        self.health = body
        check(body.get("status") == "ok", "health.status 应为 'ok'，实际 " + repr(body.get("status")))
        check(body.get("protocol") == PROTOCOL, "health.protocol 应为 %r，实际 %r" % (PROTOCOL, body.get("protocol")))
        check(body.get("version") == VERSION, "health.version 应为 %d，实际 %r" % (VERSION, body.get("version")))
        check("backend" in body and body.get("backend"), "health 缺少 backend 信息")
        voices = body.get("voices")
        check(isinstance(voices, int) and voices >= 1, "health.voices 应为 >=1 的整数，实际 " + repr(voices))

        backend = body.get("backend")
        resolved = ""
        if isinstance(backend, dict):
            resolved = str(backend.get("resolved") or backend.get("name") or "")
        else:
            resolved = str(backend)
        providers = backend.get("providers") if isinstance(backend, dict) else None
        self.detail("status=ok protocol=%s version=%d voices=%d" % (PROTOCOL, VERSION, voices))
        self.detail("backend: requested=%s resolved=%s providers=%s"
                    % (self.args.backend, resolved or "?", providers if providers else "?"))
        if not self.args.url and self.args.backend != "auto" and resolved and resolved != self.args.backend:
            self.warn("请求 --backend %s，但 /health 报告 resolved=%s（后端可能已回退 CPU）"
                      % (self.args.backend, resolved))

    def check_catalog(self) -> None:
        status, body, text = self.http_json("/voices")
        check(status == 200, "GET /voices 期望 200，实际 HTTP %d: %s" % (status, text[:300]))
        voices: Any
        if isinstance(body, dict):
            voices = body.get("voices")
            embedded_schema = body.get("schema")
            if isinstance(embedded_schema, dict):
                self.schema = embedded_schema
        else:
            voices = body
        check(isinstance(voices, list) and voices, "GET /voices 响应没有 voices 数组（或为空）")
        parsed: list[dict[str, Any]] = []
        for index, voice in enumerate(voices):
            check(isinstance(voice, dict), "voices[%d] 不是 object" % index)
            voice_id = voice.get("id")
            check(isinstance(voice_id, str) and voice_id.strip(), "voices[%d] 缺少非空 id" % index)
            parsed.append(voice)
        self.voices = parsed
        schema_count = len(self.schema)

        status, schema_body, text = self.http_json("/schema")
        check(status == 200, "GET /schema 期望 200，实际 HTTP %d: %s" % (status, text[:300]))
        check(isinstance(schema_body, dict) and schema_body, "GET /schema 应返回非空 JSON object")
        self.schema = schema_body
        schema_count = len(schema_body)

        health_voices = self.health.get("voices")
        if isinstance(health_voices, int):
            check(health_voices == len(self.voices),
                  "health.voices=%d 与 /voices 数量 %d 不一致" % (health_voices, len(self.voices)))
        self.detail("voices: " + ", ".join(voice["id"] for voice in self.voices[:8])
                    + (" ..." if len(self.voices) > 8 else ""))
        self.detail("schema keys: " + ", ".join(sorted(str(key) for key in self.schema.keys())))
        # 自动音高能力声明：pitch.auto=true，partProperties 里有 PitchMode（default=predict）。
        pitch_capability = self.schema.get("pitch")
        check(isinstance(pitch_capability, dict) and pitch_capability.get("auto") is True,
              "/schema.pitch.auto 应为 true（服务端具备自动音高能力），实际 %r" % (pitch_capability,))
        part_properties = self.schema.get("partProperties")
        check(isinstance(part_properties, list), "/schema.partProperties 应为数组")
        pitch_mode_prop = next(
            (item for item in part_properties
             if isinstance(item, dict) and item.get("id") == "PitchMode"),
            None,
        )
        check(pitch_mode_prop is not None, "/schema.partProperties 缺少 PitchMode 属性")
        check(pitch_mode_prop.get("default") == "predict",
              "/schema PitchMode.default 应为 'predict'，实际 %r" % (pitch_mode_prop.get("default"),))
        pitch_modes = [
            str(option.get("value")).lower()
            for option in (pitch_mode_prop.get("options") or [])
            if isinstance(option, dict)
        ]
        check("predict" in pitch_modes and "tunelab" in pitch_modes,
              "/schema PitchMode.options 应同时包含 predict / tunelab，实际 %r" % (pitch_modes,))
        self.detail("auto-pitch schema: pitch.auto=true, PitchMode default=predict options=%s"
                    % (pitch_modes,))
        if not isinstance(self.schema.get("synthesizedParameters"), list):
            self.warn("schema 没有 synthesizedParameters（响应 parameters 回显仍会被校验）")

    def check_catalog_identity(self, ids: list[str]) -> None:
        """防止连错服务：合成模式必须能看到合成声库，真实模式必须能看到探测到的真实声库。"""
        lowered = [voice_id.lower() for voice_id in ids]
        names = [str(voice.get("name") or "").lower() for voice in self.voices]
        if self.source == "synthetic" and self.synthetic_bank is not None:
            expected = self.synthetic_bank.name.lower()
            found = any(expected in voice_id for voice_id in lowered) or any(
                "testvoice" in name for name in names
            )
            if not found:
                raise VerifyFailure(
                    "服务端 /voices 里没有合成声库 '" + self.synthetic_bank.name + "'（实际: "
                    + ", ".join(ids[:8]) + "）：可能连到了别的服务（端口被占用 / --url 指错），"
                    "或 --voice-dir 没有生效。"
                )
        elif self.source == "real" and self.real_banks and self.args.voice is None:
            bank_names = [bank.name.lower() for bank in self.real_banks]
            matched = [
                voice_id for voice_id in ids
                if any(name in voice_id.lower() for name in bank_names)
            ]
            if not matched:
                raise VerifyFailure(
                    "服务端 /voices 里没有探测到的真实声库（"
                    + ", ".join(bank.name for bank in self.real_banks[:3])
                    + "）；实际: " + ", ".join(ids[:8])
                )

    def select_voice(self) -> None:
        ids = [voice["id"] for voice in self.voices]
        self.check_catalog_identity(ids)
        if self.args.voice:
            check(self.args.voice in ids, "请求的 --voice %r 不在服务端目录里: %s" % (self.args.voice, ids))
            self.voice_id = self.args.voice
        else:
            hints: list[str] = []
            if self.synthetic_bank is not None:
                hints.append(self.synthetic_bank.name)
                hints.append("TestVoice")
            # self.real_banks 已按 JiHeHong 优先排序，按顺序作为选声库偏好。
            for bank in self.real_banks:
                hints.append(bank.name)
            hints.append("JiHeHong")
            chosen: Optional[str] = None
            for hint in hints:
                lowered = hint.lower()
                for voice in self.voices:
                    name = str(voice.get("name") or "")
                    if lowered in voice["id"].lower() or lowered in name.lower():
                        chosen = voice["id"]
                        break
                if chosen:
                    break
            if chosen is None:
                chosen = ids[0]
            self.voice_id = chosen
        voice_entry = next(voice for voice in self.voices if voice["id"] == self.voice_id)
        self.detail("voice=%s name=%s" % (self.voice_id, voice_entry.get("name")))
        if self.source == "real":
            self.detail("使用真实声库路径验证（%s）" % (", ".join(bank.name for bank in self.real_banks[:3])))
        elif self.source == "synthetic":
            self.detail("使用合成声库验证: " + str(self.synthetic_bank))

    # ---- HTTP -----------------------------------------------------------------------
    def http_json(self, path: str, *, method: str = "GET", payload: Any = None,
                  timeout: Optional[float] = None) -> tuple[int, Any, str]:
        url = self.url + path
        try:
            return http_request(
                url,
                method=method,
                payload=payload,
                api_key=self.args.api_key,
                timeout=float(timeout if timeout is not None else self.args.timeout),
            )
        except TransportError as error:
            raise VerifyFailure("请求 %s 失败: %s" % (url, error))

    def build_request(self, **overrides: Any) -> dict[str, Any]:
        points = [
            {"t": round(TEST_START + index * 0.01, 4), "v": float(TEST_PITCH)}
            for index in range(int(round((TEST_END - TEST_START) / 0.01)) + 1)
        ]
        payload: dict[str, Any] = {
            "protocol": PROTOCOL,
            "version": VERSION,
            "voice": self.voice_id,
            "sampleRate": REQUEST_SAMPLE_RATE,
            "startTime": TEST_START,
            "endTime": TEST_END,
            "pitchMode": "manual",
            "partProperties": {},
            "notes": [
                {
                    "key": TEST_NOTE_KEY,
                    "start": TEST_START,
                    "duration": TEST_END - TEST_START,
                    "pitch": TEST_PITCH,
                    "lyric": TEST_LYRIC,
                    "rest": False,
                    "properties": {},
                }
            ],
            "pitch": points,
            "parameters": {
                "DYN": [
                    {"t": TEST_START, "v": 0.0},
                    {"t": TEST_END, "v": 0.0},
                ]
            },
        }
        payload.update(overrides)
        return payload

    # ---- 3. 合成 --------------------------------------------------------------------
    def synthesize(self) -> None:
        payload = self.build_request()
        started = time.perf_counter()
        status, body, text = self.http_json("/synthesize", method="POST", payload=payload)
        elapsed = time.perf_counter() - started
        check(status == 200, "POST /synthesize 期望 200，实际 HTTP %d: %s" % (status, text[:500]))
        check(isinstance(body, dict), "POST /synthesize 响应不是 JSON object: " + text[:300])
        if body.get("error"):
            raise VerifyFailure("POST /synthesize 返回 error: " + str(body["error"])[:500])
        self.response = body
        self.detail("HTTP 200, %.0f ms（含首次加载）" % (elapsed * 1000.0))
        if body.get("protocol") not in (None, PROTOCOL):
            raise VerifyFailure("响应 protocol 应为 %r，实际 %r" % (PROTOCOL, body.get("protocol")))
        if body.get("version") not in (None, VERSION):
            raise VerifyFailure("响应 version 应为 %d，实际 %r" % (VERSION, body.get("version")))
        meta = body.get("meta")
        if isinstance(meta, dict):
            interesting = {key: meta[key] for key in ("engine", "backend", "phones", "voice") if key in meta}
            if interesting:
                self.detail("meta: " + json.dumps(interesting, ensure_ascii=False))

    # ---- 4. 音频 --------------------------------------------------------------------
    def check_audio(self) -> None:
        response = self.response
        audio = response.get("audio")
        check(isinstance(audio, dict), "响应缺少 audio object")
        samples, wav_sample_rate, audio_format = decode_audio(audio)
        declared_rate = response.get("sampleRate")
        check(isinstance(declared_rate, (int, float)) and declared_rate > 0,
              "响应 sampleRate 非法: " + repr(declared_rate))
        if wav_sample_rate:
            if int(wav_sample_rate) != int(declared_rate):
                self.warn("wav 头 sampleRate=%d 与响应 sampleRate=%d 不一致" % (wav_sample_rate, declared_rate))
            sample_rate = int(wav_sample_rate)
        else:
            sample_rate = int(declared_rate)
        self.samples = samples
        self.sample_rate = sample_rate

        check(samples.size > 0, "解码后的音频为空")
        duration = samples.size / float(sample_rate)
        audio_start_value = response.get("audioStart", TEST_START)
        check(isinstance(audio_start_value, (int, float)) and math.isfinite(float(audio_start_value)),
              "audioStart 非法: " + repr(audio_start_value))
        audio_start = float(audio_start_value)
        # 协议允许用 audioStart 覆盖默认对齐（例如包含块首之前的辅音 pre-roll）：
        # 音频应覆盖 [audioStart, endTime]，所以时长按 audioStart 起算；同时必须覆盖块首。
        check(audio_start <= TEST_START + 1e-3,
              "audioStart=%.3fs 晚于请求块起点 %.3fs：请求块开头的音频会缺失" % (audio_start, TEST_START))
        check(audio_start >= TEST_START - 0.5,
              "audioStart=%.3fs 早于请求块起点超过 500ms：前置音频异常" % audio_start)
        expected = TEST_END - audio_start
        check(abs(duration - expected) <= 0.05,
              "音频时长 %.3fs != (endTime - audioStart) %.3fs（format=%s, %d samples @ %d Hz, audioStart=%.3fs）"
              % (duration, expected, audio_format, samples.size, sample_rate, audio_start))
        if abs(audio_start - TEST_START) > 1e-3:
            self.detail("audioStart=%.3fs（比 startTime 早 %.0fms，按 endTime-audioStart 校验时长）"
                        % (audio_start, (TEST_START - audio_start) * 1000.0))
        check(np.isfinite(samples).all(), "音频里含 NaN/Infinity")
        check(int(audio.get("channels") or 1) == 1, "响应 channels 应为 1，实际 " + repr(audio.get("channels")))

        quarter = samples.size // 4
        middle = samples[quarter:samples.size - quarter] if samples.size >= 8 else samples
        rms = float(np.sqrt(np.mean(np.square(middle.astype(np.float64)))))
        peak = float(np.max(np.abs(samples)))
        check(rms >= float(self.args.min_rms),
              "音频 RMS %.5f 低于阈值 %.5f（几乎是静音？format=%s）" % (rms, float(self.args.min_rms), audio_format))

        f0_estimate, fft_peak = estimate_pitch(samples, sample_rate)
        tolerance = self.args.pitch_tolerance
        if tolerance is None:
            # 合成声库是确定性的正弦（5% 足够严格）；真实模型受音素/方差影响，
            # 默认放宽到 10% 且不硬失败（避免真实声库的误报，见 lead 要求）。
            tolerance = 0.10 if self.source == "real" else 0.05
        tolerance = float(tolerance)
        within = TEST_HZ * tolerance
        if abs(fft_peak - TEST_HZ) <= within:
            pitch_ok = True
        elif f0_estimate > 0 and abs(f0_estimate - TEST_HZ) <= within and is_harmonic(fft_peak, f0_estimate):
            pitch_ok = True
            self.detail("FFT 主频 %.1f Hz 是谐波，基频估计 %.1f Hz" % (fft_peak, f0_estimate))
        else:
            pitch_ok = False
        if not pitch_ok:
            if self.source == "real" and not self.args.strict_pitch:
                self.warn(
                    "音高估计偏离 %.0f Hz +-%.0f%%（FFT 主频 %.1f Hz / 基频估计 %.1f Hz）；"
                    "真实声库不做硬断言，可用 --strict-pitch 强制检查"
                    % (TEST_HZ, tolerance * 100.0, fft_peak, f0_estimate)
                )
            else:
                raise VerifyFailure(
                    "音高不符：FFT 主频 %.1f Hz / 基频估计 %.1f Hz，期望 %.0f Hz +-%.0f%%（pitch=%d）"
                    % (fft_peak, f0_estimate, TEST_HZ, tolerance * 100.0, TEST_PITCH)
                )

        self.detail("audio: format=%s %d samples @ %d Hz = %.3fs, RMS=%.3f, peak=%.3f"
                    % (audio_format, samples.size, sample_rate, duration, rms, peak))
        self.detail("pitch: FFT 主频 %.1f Hz (期望 %.0f +-%.0f%%), 基频估计 %.1f Hz"
                    % (fft_peak, TEST_HZ, tolerance * 100.0, f0_estimate))

    # ---- 5. 回显 --------------------------------------------------------------------
    def check_echoes(self) -> None:
        response = self.response
        phonemes = response.get("phonemes")
        check(isinstance(phonemes, dict), "响应缺少 phonemes object")
        check(TEST_NOTE_KEY in phonemes, "响应 phonemes 缺少 note key %r（实际 keys: %s）"
              % (TEST_NOTE_KEY, list(phonemes.keys())))
        entry = phonemes[TEST_NOTE_KEY]
        check(isinstance(entry, dict), "phonemes[%r] 不是 object" % TEST_NOTE_KEY)
        leading = entry.get("leading")
        body = entry.get("body")
        check(isinstance(leading, list), "phonemes[%r].leading 不是数组" % TEST_NOTE_KEY)
        check(isinstance(body, list), "phonemes[%r].body 不是数组" % TEST_NOTE_KEY)
        symbols: list[str] = []
        for group_name, group in (("leading", leading), ("body", body)):
            for index, item in enumerate(group):
                check(isinstance(item, dict), "phonemes.%s[%d] 不是 object" % (group_name, index))
                symbol = item.get("symbol")
                check(isinstance(symbol, str) and symbol.strip(),
                      "phonemes.%s[%d].symbol 缺失/为空" % (group_name, index))
                duration = item.get("duration")
                check(isinstance(duration, (int, float)) and math.isfinite(float(duration)) and float(duration) >= 0,
                      "phonemes.%s[%d].duration 非法: %r" % (group_name, index, duration))
                weight = item.get("stretchWeight", 0.0)
                check(isinstance(weight, (int, float)) and math.isfinite(float(weight)),
                      "phonemes.%s[%d].stretchWeight 非法: %r" % (group_name, index, weight))
                symbols.append(symbol.strip())
        check(symbols, "phonemes 为空：单音 %r 至少应产生一个音素" % TEST_LYRIC)
        check(body, "phonemes.body 为空：元音歌词 %r 的主体应包含元音音素" % TEST_LYRIC)
        body_offset = entry.get("bodyOffset", 0.0)
        check(isinstance(body_offset, (int, float)) and math.isfinite(float(body_offset)),
              "phonemes.bodyOffset 非法: %r" % (body_offset,))
        vowels = [symbol for symbol in symbols if is_vowel_symbol(symbol)]
        check(vowels, "响应音素里没有元音音素: " + " ".join(symbols))
        self.detail("phonemes=%r leading=%d body=%d vowels=%s bodyOffset=%.3f"
                    % (symbols, len(leading), len(body), vowels, float(body_offset)))

        allow_missing = bool(self.args.allow_missing_echo)
        pitch = response.get("pitch")
        if pitch is None or pitch == []:
            if allow_missing:
                self.warn("响应没有 pitch 回显（协议里可选；已用 --allow-missing-echo 放行）")
            else:
                raise VerifyFailure(
                    "响应缺少 pitch 回显：应返回渲染出的 F0 折线（协议 §响应；"
                    "如只验证协议最小实现可加 --allow-missing-echo）"
                )
        else:
            check(isinstance(pitch, list), "响应 pitch 不是数组")
            check(len(pitch) >= 2, "响应 pitch 点数过少: %d" % len(pitch))
            times: list[float] = []
            values: list[float] = []
            for index, point in enumerate(pitch):
                check(isinstance(point, dict), "pitch[%d] 不是 object" % index)
                t = point.get("t")
                v = point.get("v")
                check(isinstance(t, (int, float)) and math.isfinite(float(t)),
                      "pitch[%d].t 非法: %r" % (index, t))
                check(isinstance(v, (int, float)) and math.isfinite(float(v)),
                      "pitch[%d].v 非法: %r" % (index, v))
                times.append(float(t))
                values.append(float(v))
            check(all(times[i] <= times[i + 1] + 1e-9 for i in range(len(times) - 1)),
                  "响应 pitch 的 t 不是单调不减")
            median = float(np.median(np.asarray(values, dtype=np.float64)))
            check(abs(median - TEST_PITCH) <= 2.0,
                  "pitch 回显中位数 %.2f 半音，与请求 pitch=%d 相差超过 2 半音" % (median, TEST_PITCH))
            self.detail("pitch echo: %d 点, %.3fs..%.3fs, 中位 %.2f 半音"
                        % (len(pitch), times[0], times[-1], median))

        parameters = response.get("parameters")
        check(isinstance(parameters, dict), "响应缺少 parameters object")
        energy = parameters.get("energy")
        check(isinstance(energy, list) and energy, "响应 parameters.energy 缺失/为空（schema 声明了 energy 回显轨）")
        energy_times: list[float] = []
        energy_values: list[float] = []
        for index, point in enumerate(energy):
            check(isinstance(point, dict), "parameters.energy[%d] 不是 object" % index)
            t = point.get("t")
            v = point.get("v")
            check(isinstance(t, (int, float)) and math.isfinite(float(t)),
                  "parameters.energy[%d].t 非法: %r" % (index, t))
            check(isinstance(v, (int, float)) and math.isfinite(float(v)) and 0.0 <= float(v) <= 100.0,
                  "parameters.energy[%d].v 非法（应为 0..100）: %r" % (index, v))
            energy_times.append(float(t))
            energy_values.append(float(v))
        check(all(energy_times[i] <= energy_times[i + 1] + 1e-9 for i in range(len(energy_times) - 1)),
              "parameters.energy 的 t 不是单调不减")
        self.detail("energy echo: %d 点, %.1f..%.1f" % (len(energy), min(energy_values), max(energy_values)))

    # ---- 6. 自动音高（dspitch） ------------------------------------------------------
    @staticmethod
    def pitch_echo_values(body: dict[str, Any]) -> list[float]:
        pitch = body.get("pitch")
        check(isinstance(pitch, list) and len(pitch) >= 2, "自动音高响应缺少 pitch 回显")
        values: list[float] = []
        for index, point in enumerate(pitch):
            check(isinstance(point, dict), "pitch[%d] 不是 object" % index)
            value = point.get("v")
            check(isinstance(value, (int, float)) and math.isfinite(float(value)),
                  "pitch[%d].v 非法: %r" % (index, value))
            values.append(float(value))
        return values

    def audio_pitch_and_rms(self, body: dict[str, Any]) -> tuple[float, float, float, int]:
        """返回 (音频基频对应的 MIDI 半音, FFT 主频 Hz, RMS, sampleRate)。"""
        audio = body.get("audio")
        check(isinstance(audio, dict), "响应缺少 audio object")
        samples, wav_sample_rate, audio_format = decode_audio(audio)
        sample_rate = int(wav_sample_rate or body.get("sampleRate") or 0)
        check(sample_rate > 0, "响应 sampleRate 非法: " + repr(body.get("sampleRate")))
        check(samples.size > 0, "解码后的音频为空（format=%s）" % audio_format)
        check(np.isfinite(samples).all(), "音频里含 NaN/Infinity")
        quarter = samples.size // 4
        middle = samples[quarter:samples.size - quarter] if samples.size >= 8 else samples
        rms = float(np.sqrt(np.mean(np.square(middle.astype(np.float64)))))
        f0_estimate, fft_peak = estimate_pitch(samples, sample_rate)
        check(f0_estimate > 0, "无法从音频估计基频（可能是静音）")
        midi = TEST_PITCH + 12.0 * math.log2(f0_estimate / TEST_HZ)
        return midi, fft_peak, rms, sample_rate

    def synthesize_checked(self, payload: dict[str, Any]) -> dict[str, Any]:
        status, body, text = self.http_json("/synthesize", method="POST", payload=payload)
        check(status == 200, "自动音高请求期望 HTTP 200，实际 %d: %s" % (status, text[:400]))
        check(isinstance(body, dict), "自动音高响应不是 JSON object")
        if body.get("error"):
            raise VerifyFailure("自动音高响应 error: " + str(body["error"])[:300])
        return body

    def find_auto_pitch_voice(self) -> Optional[dict[str, Any]]:
        """description 里带 auto-pitch 的声库（优先 yousa），没有则 None。"""
        candidates = [
            voice for voice in self.voices
            if "auto-pitch" in str(voice.get("description") or "").lower()
        ]
        if not candidates:
            return None
        for voice in candidates:
            if "yousa" in voice["id"].lower():
                return voice
        return candidates[0]

    def check_auto_pitch(self) -> None:
        voice = self.find_auto_pitch_voice()
        if voice is None:
            raise StepSkipped(
                "catalog 里没有 description 含 'auto-pitch' 的声库（合成声库 / 无 yousa），"
                "跳过自动音高场景（不计失败，干净环境仍 PASS）"
            )
        self.detail("auto-pitch voice: " + voice["id"] + "（" + str(voice.get("description")) + "）")

        # a) PitchMode=predict：dspitch 预测 F0；echo 非平坦，音频真实基频与 echo 中位数一致。
        payload = self.build_request(
            voice=voice["id"], partProperties={"PitchMode": "predict", "language": "zh"}
        )
        body = self.synthesize_checked(payload)
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        check(meta.get("pitchSource") == "auto",
              "PitchMode=predict 的 meta.pitchSource 应为 'auto'，实际 %r" % (meta.get("pitchSource"),))
        predict_values = self.pitch_echo_values(body)
        predict_median = float(np.median(np.asarray(predict_values, dtype=np.float64)))
        predict_std = float(np.std(np.asarray(predict_values, dtype=np.float64)))
        check(predict_std > 0.05,
              "predict 回显应非平坦（dspitch 预测），实际 std=%.3f（是否回退成画线了？）" % predict_std)
        midi_est, fft_peak, predict_rms, sample_rate = self.audio_pitch_and_rms(body)
        check(predict_rms >= float(self.args.min_rms),
              "predict 音频 RMS %.5f 低于阈值 %.5f" % (predict_rms, float(self.args.min_rms)))
        difference = abs(midi_est - predict_median)
        check(difference < 1.0,
              "predict 音频基频 %.2f 半音与 pitch 回显中位数 %.2f 相差 %.2f（应 < 1 半音）"
              % (midi_est, predict_median, difference))
        f0_hz = TEST_HZ * (2.0 ** ((midi_est - TEST_PITCH) / 12.0))
        self.detail(
            "predict: pitchSource=auto (autoPitchModel=%r), echo %d 点 median=%.2f std=%.2f, "
            "音频自相关基频 %.1f Hz = %.2f 半音（FFT 主频 %.1f Hz，与 echo 差 %.2f）, RMS=%.3f, %d Hz"
            % (meta.get("autoPitchModel"), len(predict_values), predict_median, predict_std,
               f0_hz, midi_est, fft_peak, difference, predict_rms, sample_rate)
        )

        # b) PitchMode=tunelab：始终跟随请求曲线；echo 恒等于请求 pitch（平坦 69）。
        payload = self.build_request(
            voice=voice["id"], partProperties={"PitchMode": "tunelab", "language": "zh"}
        )
        body = self.synthesize_checked(payload)
        meta = body.get("meta") if isinstance(body.get("meta"), dict) else {}
        check(meta.get("pitchSource") == "tunelab",
              "PitchMode=tunelab 的 meta.pitchSource 应为 'tunelab'，实际 %r" % (meta.get("pitchSource"),))
        tunelab_values = self.pitch_echo_values(body)
        tunelab_median = float(np.median(np.asarray(tunelab_values, dtype=np.float64)))
        tunelab_std = float(np.std(np.asarray(tunelab_values, dtype=np.float64)))
        check(tunelab_std <= 0.05,
              "tunelab 回显应恒等于请求曲线（std 约 0），实际 std=%.3f" % tunelab_std)
        check(abs(tunelab_median - TEST_PITCH) <= 0.2,
              "tunelab 回显中位数 %.2f 应约等于请求 pitch=%d" % (tunelab_median, TEST_PITCH))
        _unused_midi, _unused_fft, tunelab_rms, _unused_rate = self.audio_pitch_and_rms(body)
        check(tunelab_rms >= float(self.args.min_rms),
              "tunelab 音频 RMS %.5f 低于阈值 %.5f" % (tunelab_rms, float(self.args.min_rms)))
        self.detail("tunelab: pitchSource=tunelab, echo %d 点 median=%.2f std=%.3f, RMS=%.3f"
                    % (len(tunelab_values), tunelab_median, tunelab_std, tunelab_rms))

    # ---- 7. 错误路径 ----------------------------------------------------------------
    def check_errors(self) -> None:
        base = self.build_request()
        cases: list[tuple[str, dict[str, Any], str]] = []
        payload = copy.deepcopy(base)
        payload["protocol"] = "tunelab-cloudvoice-bogus"
        cases.append(("错误 protocol", payload, "protocol should be rejected"))
        payload = copy.deepcopy(base)
        payload["version"] = 99
        cases.append(("错误 version", payload, "version should be rejected"))
        payload = copy.deepcopy(base)
        payload["voice"] = "__no_such_voice__"
        cases.append(("未知 voice", payload, "unknown voice should be rejected"))
        payload = copy.deepcopy(base)
        payload["endTime"] = payload["startTime"]
        cases.append(("endTime <= startTime", payload, "empty block should be rejected"))

        for label, payload, _hint in cases:
            started = time.perf_counter()
            status, body, text = self.http_json(
                "/synthesize", method="POST", payload=payload, timeout=None
            )
            elapsed = time.perf_counter() - started
            check(400 <= status < 500,
                  "%s 期望 4xx，实际 HTTP %d: %s" % (label, status, text[:300]))
            error = body.get("error") if isinstance(body, dict) else None
            check(isinstance(error, str) and error.strip(),
                  "%s 的 4xx 响应缺少可读 error 字段: %s" % (label, text[:300]))
            self.detail("%s -> HTTP %d (%s, %.0f ms)"
                        % (label, status, error.strip()[:70], elapsed * 1000.0))

    # ---- 收尾 ----------------------------------------------------------------------
    def log_tail(self, lines: int = 40) -> str:
        if not self.log_path or not self.log_path.is_file():
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        rows = [row for row in text.strip().splitlines() if row.strip()]
        if not rows:
            return ""
        return "---- server log (last %d lines) ----\n%s" % (len(rows[-lines:]), "\n".join(rows[-lines:]))

    def print_summary(self, ok: bool) -> None:
        elapsed = time.perf_counter() - self.started
        print()
        print("=" * 78)
        print("结果: %d passed, %d failed, %d skipped, %d warning(s), 总耗时 %.2fs"
              % (self.passed, self.failed, self.skipped, len(self.warnings), elapsed))
        if self.voice_id:
            print("  voice:     " + str(self.voice_id) + " (" + self.source + ")")
        if self.synthetic_bank is not None:
            print("  test bank: " + str(self.synthetic_bank))
        for warning in self.warnings:
            print("  warning:   " + warning)
        if not ok and self.log_path and self.log_path.is_file():
            print("  server log: " + str(self.log_path))
        if self.args.keep_temp and self.tmpdir is not None:
            print("  temp dir:   " + str(self.tmpdir))
        print("PASS" if ok else "FAIL")
        print("=" * 78)

    def cleanup(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self._server_log_handle is not None:
            try:
                self._server_log_handle.close()
            except OSError:
                pass
            self._server_log_handle = None
        if self.tmpdir is not None and not self.args.keep_temp:
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            self.tmpdir = None


# --------------------------------------------------------------------------------------
# HTTP / 音频 / 音高工具
# --------------------------------------------------------------------------------------
def http_request(
    url: str,
    *,
    method: str = "GET",
    payload: Any = None,
    api_key: Optional[str] = None,
    timeout: float = 300.0,
) -> tuple[int, Any, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = int(error.code)
        raw = error.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
        raise TransportError(str(getattr(error, "reason", error))) from error
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"_raw": text[:1000]}
    return status, parsed, text


def decode_audio(audio: dict[str, Any]) -> tuple[np.ndarray, int, str]:
    """把响应 audio 对象解码成 (mono float32 数组, wav 头采样率或 0, format)。"""
    audio_format = str(audio.get("format") or "")
    check(audio_format in SUPPORTED_AUDIO_FORMATS,
          "不支持的 audio.format=%r（支持 %s）" % (audio_format, "/".join(SUPPORTED_AUDIO_FORMATS)))
    channels = int(audio.get("channels") or 1)
    check(channels >= 1, "audio.channels 非法: %r" % (audio.get("channels"),))
    encoded = audio.get("data")
    if not encoded:
        if audio.get("url"):
            raise VerifyFailure("响应使用 audio.url 模式，verify.py 只校验 base64 内联音频（data）")
        raise VerifyFailure("audio.data 缺失/为空")
    check(isinstance(encoded, str), "audio.data 不是 base64 字符串")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as error:  # noqa: BLE001 - binascii.Error 等
        raise VerifyFailure("audio.data 不是合法 base64: " + str(error))
    check(raw, "audio.data 解码后为空")

    if audio_format == "wav":
        samples, sample_rate = decode_wav(raw)
        check(samples.size > 0, "wav 里没有采样")
        check(samples.size % channels == 0, "wav 采样数不能被 channels 整除")
        return samples.reshape(-1, channels).astype(np.float32).mean(axis=1), sample_rate, audio_format

    if audio_format == "pcm_f32le":
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:  # pcm_s16le
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    check(samples.size > 0, "pcm 里没有采样")
    check(samples.size % channels == 0, "pcm 采样数不能被 channels 整除")
    return samples.reshape(-1, channels).astype(np.float32).mean(axis=1), 0, audio_format


def decode_wav(raw: bytes) -> tuple[np.ndarray, int]:
    """最小 RIFF/WAVE 解析：支持 PCM 16/32-bit 与 IEEE float32。"""
    check(len(raw) >= 44 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "不是合法的 RIFF/WAVE 数据")
    position = 12
    fmt: Optional[bytes] = None
    data: Optional[bytes] = None
    while position + 8 <= len(raw):
        chunk_id = raw[position:position + 4]
        size = struct.unpack("<I", raw[position + 4:position + 8])[0]
        body = raw[position + 8:position + 8 + size]
        if chunk_id == b"fmt ":
            fmt = body
        elif chunk_id == b"data":
            data = body
        position += 8 + size + (size & 1)
    check(fmt is not None and len(fmt) >= 16, "wav 缺少 fmt chunk")
    check(data is not None, "wav 缺少 data chunk")
    audio_format, channels, sample_rate, _byte_rate, _block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    check(channels >= 1, "wav channels 非法: %d" % channels)
    check(sample_rate > 0, "wav sample_rate 非法: %d" % sample_rate)
    if audio_format == 3 and bits == 32:
        samples = np.frombuffer(data, dtype="<f4").astype(np.float32)
    elif audio_format == 1 and bits == 16:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 1 and bits == 32:
        samples = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise VerifyFailure("不支持的 wav 编码: format=%d bits=%d" % (audio_format, bits))
    return samples, int(sample_rate)


def estimate_pitch(samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """返回 (基频估计, 裸 FFT 主频)。只在能量最大的 0.5s 窗口上算。"""
    signal = np.asarray(samples, dtype=np.float64).reshape(-1)
    if signal.size < 32 or sample_rate <= 0:
        return 0.0, 0.0
    window = min(int(0.5 * sample_rate), signal.size)
    window = max(window, min(signal.size, int(0.075 * sample_rate)))
    step = max(1, window // 4)
    best_start = 0
    best_energy = -1.0
    for start in range(0, signal.size - window + 1, step):
        segment = signal[start:start + window]
        energy = float(np.dot(segment, segment))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    segment = signal[best_start:best_start + window]
    segment = segment - segment.mean()
    if float(np.std(segment)) < 1e-9:
        return 0.0, 0.0
    taper = np.hanning(segment.size)
    spectrum = np.abs(np.fft.rfft(segment * taper))
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    valid = freqs >= 50.0
    fft_peak = float(freqs[valid][int(np.argmax(spectrum[valid]))])

    fmin, fmax = 55.0, 1200.0
    size = 1
    while size < 2 * segment.size:
        size <<= 1
    transform = np.fft.rfft(segment, size)
    autocorrelation = np.fft.irfft(transform * np.conj(transform), size)[:segment.size]
    if autocorrelation[0] <= 0:
        return fft_peak, fft_peak
    autocorrelation = autocorrelation / autocorrelation[0]
    min_lag = max(1, int(math.floor(sample_rate / fmax)))
    max_lag = min(segment.size - 2, int(math.ceil(sample_rate / fmin)))
    if max_lag <= min_lag:
        return fft_peak, fft_peak
    lag = min_lag + int(np.argmax(autocorrelation[min_lag:max_lag]))
    shift = 0.0
    if 0 < lag < segment.size - 1:
        left, center, right = autocorrelation[lag - 1], autocorrelation[lag], autocorrelation[lag + 1]
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            shift = 0.5 * (left - right) / denominator
    period = lag + shift
    fundamental = sample_rate / period if period > 0 else 0.0
    return float(fundamental), fft_peak


def is_harmonic(frequency: float, fundamental: float) -> bool:
    if fundamental <= 0:
        return False
    for multiple in range(1, 7):
        target = fundamental * multiple
        if abs(frequency - target) <= fundamental * 0.05:
            return True
    return False


def is_vowel_symbol(symbol: str) -> bool:
    final = symbol.strip().split("/")[-1].strip().lower()
    return final in VOWEL_SYMBOLS


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify.py",
        description="tunelab-cloudvoice/1 服务端端到端验证（真实或合成声库，std-lib + numpy）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 服务端来源
    parser.add_argument("--url", default=None, help="连接已运行的服务（不再自己启动进程）")
    parser.add_argument("--host", default="127.0.0.1", help="自启动服务端时监听地址")
    parser.add_argument("--port", type=int, default=18791, help="自启动服务端时监听端口")
    parser.add_argument("--backend", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="透传给服务端的推理后端")
    parser.add_argument("--server-cmd", default=None,
                        help="覆盖启动命令（可含 {python} {port} {host} {backend} {voice_dir} 占位符），"
                             "例如 'python3 -m server --config my.yaml'")
    parser.add_argument("--server-log", default=None, help="服务端日志文件路径（默认临时目录）")
    parser.add_argument("--python", default=None, help="跑服务端/生成声库用的解释器（默认 .venv/bin/python）")
    parser.add_argument("--api-key", default=None, help="服务端 API key（请求带 Bearer，启动时也传给服务端）")
    parser.add_argument("--startup-timeout", type=float, default=120.0, help="等待服务端就绪的秒数")
    parser.add_argument("--timeout", type=float, default=300.0, help="单次 /synthesize 请求超时秒数")
    # 声库来源
    parser.add_argument("--voice-source", default="auto", choices=["auto", "real", "synthetic"],
                        help="auto = 有真实声库用真实，否则合成；real = 只允许真实；synthetic = 强制合成")
    parser.add_argument("--voice", default=None, help="指定 voice id（默认按目录名/JiHeHong 自动挑）")
    parser.add_argument("--voice-dir", action="append", default=[], metavar="DIR",
                        help="真实声库父目录（可重复；默认 demo/DiffSingerCloud/OpenUTAU）")
    parser.add_argument("--vocoder-dir", action="append", default=[], metavar="DIR",
                        help="内建声码器目录（透传给服务端，可重复）")
    parser.add_argument("--synthetic-dir", default=None,
                        help="合成声库落盘目录（默认临时目录；调试时想看文件可指定）")
    parser.add_argument("--keep-temp", action="store_true", help="保留临时目录（合成声库 / 服务端日志）")
    # 断言阈值
    parser.add_argument("--min-rms", type=float, default=0.005, help="音频中段 RMS 下限")
    parser.add_argument("--pitch-tolerance", type=float, default=None,
                        help="FFT 主频允许偏差比例（默认：合成声库 0.05，真实声库 0.10）")
    parser.add_argument("--strict-pitch", action="store_true",
                        help="真实声库也把 440Hz 主频校验变成硬失败（默认只 warning）")
    parser.add_argument("--allow-missing-echo", action="store_true",
                        help="允许响应缺少 pitch 回显（默认要求返回，协议里它是可选的）")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    harness = Harness(args)
    ok = False
    try:
        harness.run()
        ok = True
    except VerifyFailure as error:
        print()
        print("FAIL: " + str(error))
    except KeyboardInterrupt:
        print()
        print("FAIL: interrupted")
    except Exception as error:  # noqa: BLE001 - 顶层兜底，输出可读错误
        print()
        print("FAIL: unexpected " + type(error).__name__ + ": " + str(error))
        import traceback
        traceback.print_exc()
    finally:
        if not ok:
            tail = harness.log_tail(60)
            if tail:
                print(tail)
        try:
            harness.print_summary(ok)
        finally:
            harness.cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
