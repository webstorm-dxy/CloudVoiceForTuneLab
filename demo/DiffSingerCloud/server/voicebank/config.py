"""OpenUTAU 声库配置文件解析。

负责把 dsconfig.yaml / vocoder.yaml / character.txt / character.yaml 解析成
server.voicebank.types 里的数据类。目录扫描在 discovery.py，字典/G2P 分别在
dictionary.py / g2p.py；本模块只做"文件 -> 数据类"，不加载 ONNX、不解析 dsdict。

字段命名与 OpenUTAU 的 DsConfig / DsVocoderConfig 对齐（snake_case）。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import types

LOG = logging.getLogger(__name__)

#: 文本编码候选。OpenUTAU 声库来自 Windows/macOS 各地，按顺序尝试（latin-1 永远成功）。
TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp932", "gbk", "big5", "latin-1")

#: 声库内的子模型目录名；扫描声库时既不当作声库也不继续递归。
MODEL_SUBDIR_NAMES: tuple[str, ...] = ("dsdur", "dspitch", "dsvariance", "dsvocoder")


# --------------------------------------------------------------------------------------
# 通用读取工具
# --------------------------------------------------------------------------------------
def read_text(path: str | Path) -> str:
    """按候选编码读取文本（兼容 BOM / Windows 编码）。"""
    data = Path(path).read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
        return data[3:].decode("utf-8", errors="replace")
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML mapping；空文件返回 {}。"""
    data = yaml.safe_load(read_text(path))
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"expected a YAML mapping in {path}, got {type(data).__name__}")
    return dict(data)


def load_json(path: str | Path) -> Any:
    return json.loads(read_text(path))


_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _normalized_keys(data: Mapping[str, Any]) -> dict[str, Any]:
    """同时接受 snake_case 与 camelCase 键（OpenUTAU 不同版本/工具导出不一致）。"""
    normalized: dict[str, Any] = {}
    for key, value in data.items():
        text = str(key)
        normalized[text] = value
        snake = _CAMEL_RE.sub("_", text).lower()
        normalized.setdefault(snake, value)
    return normalized


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0", ""):
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


# --------------------------------------------------------------------------------------
# dsconfig.yaml -> types.DsConfig
# --------------------------------------------------------------------------------------
_DS_STRING_FIELDS: tuple[str, ...] = (
    "phonemes",
    "languages",
    "acoustic",
    "vocoder",
    "dur",
    "linguistic",
    "pitch",
    "variance",
    "mel_base",
    "mel_scale",
)
_DS_BOOL_FIELDS: tuple[str, ...] = (
    "use_key_shift_embed",
    "use_speed_embed",
    "use_energy_embed",
    "use_breathiness_embed",
    "use_voicing_embed",
    "use_tension_embed",
    "use_continuous_acceleration",
    "use_lang_id",
    "use_expr",
    "use_note_rest",
    "predict_dur",
    "predict_energy",
    "predict_breathiness",
    "predict_voicing",
    "predict_tension",
)
_DS_INT_FIELDS: tuple[str, ...] = (
    "hidden_size",
    "sample_rate",
    "hop_size",
    "win_size",
    "fft_size",
    "num_mel_bins",
)
_DS_FLOAT_FIELDS: tuple[str, ...] = ("mel_fmin", "mel_fmax")


def parse_dsconfig(path: str | Path) -> types.DsConfig:
    """解析一份 dsconfig.yaml（声学 / dsdur / dsvariance / dspitch 通用）。"""
    path = Path(path).resolve()
    raw = load_yaml(path)
    keys = _normalized_keys(raw)

    config = types.DsConfig(raw=raw, source_path=path)

    for name in _DS_STRING_FIELDS:
        value = keys.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            setattr(config, name, text)

    for name in _DS_BOOL_FIELDS:
        if name in keys:
            setattr(config, name, _as_bool(keys[name], getattr(config, name)))
    for name in _DS_INT_FIELDS:
        if name in keys:
            setattr(config, name, _as_int(keys[name], getattr(config, name)))
    for name in _DS_FLOAT_FIELDS:
        if name in keys:
            setattr(config, name, _as_float(keys[name], getattr(config, name)))

    # OpenUTAU: use_variable_depth 与旧字段 use_shallow_diffusion 二选一
    if "use_variable_depth" in keys:
        config.use_variable_depth = _as_bool(keys["use_variable_depth"], False)
    elif "use_shallow_diffusion" in keys:
        config.use_variable_depth = _as_bool(keys["use_shallow_diffusion"], False)
    if "max_depth" in keys:
        config.max_depth = _as_float(keys["max_depth"], config.max_depth)

    speakers = keys.get("speakers")
    if isinstance(speakers, (list, tuple)):
        config.speakers = [str(item) for item in speakers if str(item).strip()]
    elif speakers is not None:
        text = str(speakers).strip()
        config.speakers = [text] if text else None

    augmentation = keys.get("augmentation_args")
    if isinstance(augmentation, Mapping):
        pitch = augmentation.get("random_pitch_shifting")
        if isinstance(pitch, Mapping):
            rng = pitch.get("range")
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                config.pitch_shifting_range = (_as_float(rng[0], -5.0), _as_float(rng[1], 5.0))
    return config


# --------------------------------------------------------------------------------------
# vocoder.yaml -> types.VocoderConfig
# --------------------------------------------------------------------------------------
def parse_vocoder_yaml(path: str | Path) -> types.VocoderConfig:
    """解析一份 vocoder.yaml；model 按 yaml 所在目录解析相对路径。"""
    path = Path(path).resolve()
    raw = load_yaml(path)
    keys = _normalized_keys(raw)

    config = types.VocoderConfig(raw=raw, source_path=path)
    config.name = _as_str(keys.get("name"), path.parent.name) or path.parent.name
    config.model = _as_str(keys.get("model"), "model.onnx") or "model.onnx"
    model_path = Path(config.model)
    if not model_path.is_absolute():
        model_path = path.parent / model_path
    config.model_path = model_path

    if "sample_rate" in keys:
        config.sample_rate = _as_int(keys["sample_rate"], config.sample_rate)
    if "hop_size" in keys:
        config.hop_size = _as_int(keys["hop_size"], config.hop_size)
    if "win_size" in keys:
        config.win_size = _as_int(keys["win_size"], config.win_size)
    if "fft_size" in keys:
        config.fft_size = _as_int(keys["fft_size"], config.fft_size)
    if "num_mel_bins" in keys:
        config.num_mel_bins = _as_int(keys["num_mel_bins"], config.num_mel_bins)
    if "mel_fmin" in keys:
        config.mel_fmin = _as_float(keys["mel_fmin"], config.mel_fmin)
    if "mel_fmax" in keys:
        config.mel_fmax = _as_float(keys["mel_fmax"], config.mel_fmax)
    if "mel_base" in keys:
        config.mel_base = _as_str(keys["mel_base"], config.mel_base) or config.mel_base
    if "mel_scale" in keys:
        config.mel_scale = _as_str(keys["mel_scale"], config.mel_scale) or config.mel_scale
    if "pitch_controllable" in keys:
        config.pitch_controllable = _as_bool(keys["pitch_controllable"], config.pitch_controllable)
    return config


def vocoder_from_model_file(
    model_path: str | Path,
    *,
    dsconfig: types.DsConfig | None = None,
    name: str = "",
) -> types.VocoderConfig:
    """dsconfig.vocoder 直接指向本地 .onnx 文件时，构造一个 VocoderConfig。

    mel 参数优先继承声学 dsconfig（保证与声学模型一致），否则用默认值。
    """
    model_path = Path(model_path)
    config = types.VocoderConfig(raw={}, source_path=None)
    config.name = name or model_path.stem
    config.model = model_path.name
    config.model_path = model_path
    if dsconfig is not None:
        config.sample_rate = dsconfig.sample_rate
        config.hop_size = dsconfig.hop_size
        config.win_size = dsconfig.win_size
        config.fft_size = dsconfig.fft_size
        config.num_mel_bins = dsconfig.num_mel_bins
        config.mel_fmin = dsconfig.mel_fmin
        config.mel_fmax = dsconfig.mel_fmax
        config.mel_base = dsconfig.mel_base
        config.mel_scale = dsconfig.mel_scale
    return config


# --------------------------------------------------------------------------------------
# character.txt / character.yaml
# --------------------------------------------------------------------------------------
@dataclass
class CharacterInfo:
    """一个声库的 character.txt + character.yaml 合并结果。"""

    directory: Path | None = None
    name: str = ""
    author: str = ""
    version: str = ""
    voice: str = ""
    web: str = ""
    description: str = ""
    image: str = ""
    portrait: str = ""
    singer_type: str = ""
    default_phonemizer: str = ""
    text_file_encoding: str = "utf-8"
    subbanks: list[types.Subbank] = field(default_factory=list)
    raw_txt: dict[str, str] = field(default_factory=dict)
    raw_yaml: dict[str, Any] = field(default_factory=dict)

    @property
    def is_diffsinger(self) -> bool:
        return self.singer_type.lower() in ("diffsinger", "diff_singer") or not self.singer_type


def parse_character_txt(path: str | Path) -> dict[str, str]:
    """解析 character.txt 的 key=value 格式（忽略空行/注释）。"""
    result: dict[str, str] = {}
    for raw_line in read_text(path).splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().lower()
        if not key:
            continue
        result[key] = value.strip()
    return result


def parse_character_yaml(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def parse_subbanks(value: Any) -> list[types.Subbank]:
    """character.yaml 的 subbanks -> types.Subbank（tone_ranges 当前版本不需要）。"""
    result: list[types.Subbank] = []
    if not isinstance(value, (list, tuple)):
        return result
    for item in value:
        if not isinstance(item, Mapping):
            continue
        result.append(
            types.Subbank(
                color=str(item.get("color") or ""),
                suffix=str(item.get("suffix") or ""),
                prefix=str(item.get("prefix") or ""),
            )
        )
    return result


def load_character(directory: str | Path) -> CharacterInfo:
    """读取声库目录下的 character.txt / character.yaml（都缺失时也返回空对象）。"""
    directory = Path(directory)
    info = CharacterInfo(directory=directory)

    txt_path = directory / "character.txt"
    if txt_path.is_file():
        try:
            info.raw_txt = parse_character_txt(txt_path)
        except OSError:  # pragma: no cover - 权限/IO 异常
            LOG.warning("failed to read %s", txt_path)

    yaml_path = directory / "character.yaml"
    if yaml_path.is_file():
        try:
            info.raw_yaml = parse_character_yaml(yaml_path)
        except Exception as error:  # noqa: BLE001 - 坏 yaml 只记录
            LOG.warning("failed to read %s: %s", yaml_path, error)

    txt = info.raw_txt
    yaml_data = info.raw_yaml
    info.name = txt.get("name") or _as_str(yaml_data.get("name")) or directory.name
    info.author = txt.get("author") or _as_str(yaml_data.get("author"))
    info.version = txt.get("version") or _as_str(yaml_data.get("version"))
    info.voice = txt.get("voice") or _as_str(yaml_data.get("voice"))
    info.web = txt.get("web") or _as_str(yaml_data.get("web"))
    info.description = txt.get("description") or _as_str(yaml_data.get("description"))
    info.image = txt.get("image") or _as_str(yaml_data.get("image"))
    info.portrait = _as_str(yaml_data.get("portrait"))
    info.singer_type = _as_str(yaml_data.get("singer_type") or info.raw_txt.get("singer_type"))
    info.default_phonemizer = _as_str(
        yaml_data.get("default_phonemizer") or info.raw_txt.get("default_phonemizer")
    )
    info.text_file_encoding = _as_str(yaml_data.get("text_file_encoding"), "utf-8") or "utf-8"
    info.subbanks = parse_subbanks(yaml_data.get("subbanks"))
    return info


_PHONEMIZER_LANGUAGE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("zh", ("chinese", "mandarin", "jyutping", "cantonese", "pinyin")),
    ("ja", ("japanese", "kana")),
    ("en", ("english", "arpabet")),
)


def language_from_phonemizer(name: str | None) -> str:
    """从 character.yaml 的 default_phonemizer 类名推断默认歌词语言（auto/zh/ja/en）。"""
    if not name:
        return "auto"
    lowered = str(name).lower()
    for language, keywords in _PHONEMIZER_LANGUAGE_KEYWORDS:
        for keyword in keywords:
            if keyword in lowered:
                return language
    return "auto"
