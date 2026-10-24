"""服务端配置：默认值 / YAML 配置文件 / 命令行覆盖。"""
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEMO_DIR = Path(__file__).resolve().parent.parent          # demo/DiffSingerCloud
DEFAULT_VOICE_DIR = DEMO_DIR / "OpenUTAU"
ORIGINAL_CWD = Path.cwd()

BACKENDS = ("auto", "cpu", "cuda", "mps")


def _split_env_paths(name: str) -> list[Path]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip()]


def default_vocoder_dirs() -> list[Path]:
    """OpenUTAU / TuneLab 内建声码器可能出现的目录（不存在的会被忽略）。"""
    home = Path.home()
    candidates: list[Path] = [
        DEMO_DIR / "OpenUTAU" / "Vocoders",
        home / "Library/OpenUtau/Dependencies",
        home / "Documents/Library/OpenUtau/Dependencies",
        home / "Library/Application Support/TuneLab/Extensions/ChoristaDiffsinger/Vocoders",
        home / ".local/share/OpenUtau/Dependencies",
        home / "AppData/Roaming/OpenUtau/Dependencies",
    ]
    candidates.extend(_split_env_paths("TUNELAB_DIFFSINGER_VOCODER_DIRS"))
    seen: set[str] = set()
    result: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen and path.is_dir():
            seen.add(key)
            result.append(path)
    return result


def default_voice_dirs() -> list[Path]:
    dirs = [DEFAULT_VOICE_DIR]
    dirs.extend(_split_env_paths("TUNELAB_DIFFSINGER_VOICE_DIRS"))
    return [path for path in dirs if path.is_dir()] or [DEFAULT_VOICE_DIR]


@dataclass
class ServerConfig:
    """所有可配置项。命令行 / YAML 配置文件里的 key 与本类字段同名。"""

    # ---- HTTP ----
    host: str = "127.0.0.1"
    port: int = 18791
    api_key: str | None = None
    log_level: str = "INFO"

    # ---- 推理后端：auto | cpu | cuda | mps ----
    backend: str = "auto"
    device_id: int = 0
    threads: int = 0                     # 0 = ONNX Runtime 默认（CPU 线程数）
    max_cached_sessions: int = 12

    # ---- 声库 ----
    voice_dirs: list[Path] = field(default_factory=default_voice_dirs)
    vocoder_dirs: list[Path] = field(default_factory=default_vocoder_dirs)
    scan_depth: int = 3
    language: str = "auto"               # auto | zh | ja | en
    # 音高来源默认值（part 属性 PitchMode 会覆盖）：
    #   auto    = 声库有 dspitch 时自动预测 F0（用户画线作为相对偏差），无 dspitch 回退画线
    #   tunelab = 始终按 TuneLab 的 pitch 曲线渲染
    pitch_mode: str = "auto"

    # ---- DiffSinger 渲染参数（与 OpenUTAU 语义一致） ----
    depth: float = 1.0                   # 浅扩散深度（voicebank 有 use_variable_depth 时生效）
    steps: int = 20                      # 扩散步数
    # 首音节 leading 辅音在 note 头之前：True = 响应音频向前多覆盖（audioStart 早于 startTime），
    # 保留辅音；False = 严格裁到请求块（旧行为）。
    include_leading: bool = True

    # ---- 协议 / 并发限制 ----
    max_concurrent: int = 1
    queue_timeout_seconds: float = 30.0
    max_request_bytes: int = 16 * 1024 * 1024
    max_block_seconds: float = 600.0
    max_notes: int = 512

    def __post_init__(self) -> None:
        self.voice_dirs = [Path(p).expanduser() for p in self.voice_dirs]
        self.vocoder_dirs = [Path(p).expanduser() for p in self.vocoder_dirs]
        self.validate()

    def validate(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError("backend must be one of " + ", ".join(BACKENDS) + " (got " + str(self.backend) + ")")
        if self.depth <= 0:
            raise ValueError("depth must be > 0")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        if self.threads < 0:
            raise ValueError("threads must be >= 0")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port out of range")
        if self.language not in ("auto", "zh", "ja", "en"):
            raise ValueError("language must be auto/zh/ja/en")
        if self.pitch_mode not in ("auto", "tunelab"):
            raise ValueError("pitch_mode must be auto/tunelab")

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["voice_dirs"] = [str(path) for path in self.voice_dirs]
        result["vocoder_dirs"] = [str(path) for path in self.vocoder_dirs]
        return result


# 兼容 YAML 里可能出现的连字符 / 大小写差异
_ALIASES = {
    "service_url": None,
    "device": "backend",
    "num_threads": "threads",
    "max_loaded_sessions": "max_cached_sessions",
}


def _normalize_key(key: str) -> str:
    normalized = str(key).strip().replace("-", "_")
    return _ALIASES.get(normalized) or normalized


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ServerConfig:
    """读取 YAML 配置并叠加命令行覆盖（overrides 里值为 None 的键会被忽略）。

    配置里的相对路径按配置文件所在目录解析（没有配置文件时按当前工作目录）。
    """
    data: dict[str, Any] = {}
    config_dir = Path.cwd()

    if path is not None:
        config_path = Path(path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError("config file not found: " + str(config_path))
        config_dir = config_path.resolve().parent
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("config file must contain a YAML mapping: " + str(config_path))
        data = {_normalize_key(key): value for key, value in raw.items()}

    known_fields = {item.name for item in dataclasses.fields(ServerConfig)}
    unknown = sorted(key for key in data if key not in known_fields)
    if unknown:
        print("warning: unknown config key(s), ignored: " + ", ".join(unknown), file=sys.stderr)
        for key in unknown:
            data.pop(key, None)

    for key in ("voice_dirs", "vocoder_dirs"):
        if key in data and data[key] is not None:
            values = data[key]
            if isinstance(values, (str, Path)):
                values = [values]
            resolved: list[Path] = []
            for value in values:
                item = Path(str(value)).expanduser()
                if not item.is_absolute():
                    item = (config_dir / item).resolve()
                resolved.append(item)
            data[key] = resolved

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                data[_normalize_key(key)] = value

    return ServerConfig(**data)
