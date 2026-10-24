"""扫描 OpenUTAU 声库目录，构建可推理的 Voicebank 列表（VoiceCatalog）。

识别规则:
  * 目录里有 dsconfig.yaml -> 声库根目录；只有 character.txt/character.yaml 但没有
    dsconfig.yaml 的目录记录 warning 后跳过；
  * 不进入声库内部（父目录已有 dsconfig.yaml 的嵌套目录自然被排除），
    也不进入 dsdur / dspitch / dsvariance / dsvocoder 这些子模型目录；
  * 跟随目录符号链接（用 realpath 去重防止环）；
  * 单个坏声库只记录 warning，不影响其他声库；
  * 多说话人声库按 dsconfig.speakers 展开成多个 Voicebank（id = "<bank_id>::<speaker>"），
    单说话人 / 无说话人时 id = bank_id。

词典只构造懒加载对象（见 dictionary.py），扫描阶段不解析 dsdict；也不加载 ONNX。
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from . import config as config_parser
from . import dictionary as dictionary_module
from . import types

LOG = logging.getLogger(__name__)

#: 子模型目录：既不当作声库，也不继续递归。
SKIP_DIR_NAMES: frozenset[str] = frozenset({"dsdur", "dspitch", "dsvariance", "dsvocoder"})

_MODEL_FIELDS: dict[str, str] = {"dur": "dur", "variance": "variance", "pitch": "pitch"}
_MODEL_SUBDIRS: tuple[tuple[str, str], ...] = (
    ("dur", "dsdur"),
    ("variance", "dsvariance"),
    ("pitch", "dspitch"),
)
_PHONEME_MARKERS: tuple[str, ...] = (
    "phonemes.txt",
    "phonemes.json",
    "dsdict.yaml",
    "dsdict.txt",
    "character.yaml",
)


# --------------------------------------------------------------------------------------
# 基础读取
# --------------------------------------------------------------------------------------
def _as_path_list(value: Iterable[str | Path] | str | Path | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value).expanduser()]
    return [Path(item).expanduser() for item in value]


def _load_phonemes(path: str | Path | None) -> dict[str, int]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        text = config_parser.read_text(path)
    except OSError:
        return {}
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}
        result: dict[str, int] = {}
        for symbol, token in data.items():
            try:
                result[str(symbol)] = int(token)
            except (TypeError, ValueError):
                continue
        return result
    result = {}
    for index, line in enumerate(text.splitlines()):
        symbol = line.strip()
        if symbol:
            result[symbol] = index
    return result


def _load_json_dict(path: str | Path | None) -> dict[str, int]:
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(config_parser.read_text(path))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in data.items():
        try:
            result[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return result


def _is_diffsinger_dir(directory: Path) -> bool:
    """character.yaml 是否声明了 singer_type: diffsinger。"""
    character_yaml = directory / "character.yaml"
    if not character_yaml.is_file():
        return False
    try:
        data = config_parser.parse_character_yaml(character_yaml)
    except Exception:  # noqa: BLE001 - 坏 character.yaml 不影响识别
        return False
    singer_type = str(data.get("singer_type") or "").strip().lower()
    return singer_type in ("diffsinger", "diff_singer")


def _looks_like_bank(directory: Path) -> bool:
    if (directory / "dsconfig.yaml").is_file():
        return True
    if (directory / "character.txt").is_file() and any(
        (directory / name).is_file() for name in _PHONEME_MARKERS
    ):
        return True
    if _is_diffsinger_dir(directory):
        return True
    return False


# --------------------------------------------------------------------------------------
# 目录遍历
# --------------------------------------------------------------------------------------
def _iter_bank_dirs(root: Path, max_depth: int) -> Iterator[tuple[Path, str]]:
    """深度优先遍历 root，产出 (声库目录, 相对 id) 候选。"""
    root = Path(root)
    stack: list[tuple[Path, int]] = [(root, 0)]
    seen: set[str] = set()
    while stack:
        directory, depth = stack.pop()
        try:
            real = os.path.realpath(directory)
        except OSError:  # pragma: no cover - 极少见
            real = str(directory)
        if real in seen:
            continue
        seen.add(real)

        if _looks_like_bank(directory):
            try:
                relative = directory.relative_to(root)
                bank_id = relative.as_posix()
            except ValueError:  # root 本身
                bank_id = directory.name
            if bank_id in ("", "."):
                bank_id = directory.name
            yield directory, bank_id
            continue  # 不递归进声库内部

        if depth >= max_depth:
            continue
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            continue
        children: list[Path] = []
        for entry in entries:
            if entry.name.startswith(".") or entry.name in SKIP_DIR_NAMES:
                continue
            try:
                if entry.is_dir(follow_symlinks=True):
                    children.append(Path(entry.path))
            except OSError:  # pragma: no cover - 断链符号链接
                continue
        for child in reversed(children):
            stack.append((child, depth + 1))


# --------------------------------------------------------------------------------------
# 子模型（dsdur / dsvariance / dspitch）
# --------------------------------------------------------------------------------------
def _find_model_candidate(directory: Path, kind: str) -> Path | None:
    patterns = (f"*{kind}*.onnx", "*.onnx")
    for pattern in patterns:
        for candidate in sorted(directory.glob(pattern)):
            if not candidate.is_file():
                continue
            if "linguistic" in candidate.name.lower():
                continue
            return candidate
    return None


def _build_model_group(
    kind: str,
    directory: Path,
    config: types.DsConfig,
    *,
    root_config: types.DsConfig,
    root_phonemes: dict[str, int],
    root_languages: dict[str, int],
    warnings: list[str],
    label: str,
) -> types.ModelGroup:
    model_field = _MODEL_FIELDS[kind]
    phonemes: dict[str, int] = {}
    phoneme_name = config.phonemes or root_config.phonemes
    phoneme_path = config.resolve(phoneme_name) if phoneme_name else None
    if phoneme_path is not None and phoneme_path.is_file():
        phonemes = _load_phonemes(phoneme_path)
    if not phonemes:
        phonemes = dict(root_phonemes)
        if phoneme_path is not None:
            warnings.append(f"{label}: {kind} phonemes file not found ({phoneme_name!r}); using root phonemes")

    languages = dict(root_languages)
    if config.use_lang_id and config.languages:
        language_path = config.resolve(config.languages)
        if language_path is not None and language_path.is_file():
            languages = _load_json_dict(language_path)
        else:
            warnings.append(f"{label}: {kind} languages file not found ({config.languages!r})")

    linguistic_path = config.resolve(config.linguistic) if config.linguistic else None
    if linguistic_path is not None and not linguistic_path.is_file():
        warnings.append(f"{label}: {kind} linguistic model not found ({config.linguistic!r})")
        linguistic_path = None

    raw_model = getattr(config, model_field, None)
    model_path = config.resolve(raw_model) if raw_model else None
    if model_path is None:
        warnings.append(f"{label}: {kind} dsconfig has no model path ('{model_field}')")
    elif not model_path.is_file():
        candidate = _find_model_candidate(directory, kind)
        if candidate is not None:
            warnings.append(f"{label}: {kind} model {raw_model!r} not found; using {candidate.name!r}")
            model_path = candidate
        else:
            warnings.append(f"{label}: {kind} model not found ({raw_model!r})")
            model_path = None

    return types.ModelGroup(
        kind=kind,
        dir=directory,
        config=config,
        phonemes=phonemes,
        languages=languages,
        linguistic_path=linguistic_path,
        model_path=model_path,
        emb_dir=directory,
    )


def _load_model_group(
    kind: str,
    directory: Path,
    root_config: types.DsConfig,
    root_phonemes: dict[str, int],
    root_languages: dict[str, int],
    warnings: list[str],
    label: str,
) -> types.ModelGroup | None:
    config_path = directory / "dsconfig.yaml"
    try:
        config = config_parser.parse_dsconfig(config_path)
    except Exception as error:  # noqa: BLE001 - 坏配置只警告
        warnings.append(f"{label}: failed to parse {kind} dsconfig.yaml: {error}")
        return None
    return _build_model_group(
        kind,
        directory,
        config,
        root_config=root_config,
        root_phonemes=root_phonemes,
        root_languages=root_languages,
        warnings=warnings,
        label=label,
    )


def _load_root_model_group(
    kind: str,
    directory: Path,
    root_config: types.DsConfig,
    phonemes: dict[str, int],
    languages: dict[str, int],
    warnings: list[str],
    label: str,
) -> types.ModelGroup | None:
    """子模型目录不存在但模型直接放在声库根目录时（少数打包方式）。"""
    model_field = _MODEL_FIELDS[kind]
    raw_model = getattr(root_config, model_field, None)
    if not raw_model or not root_config.linguistic:
        return None
    model_path = root_config.resolve(raw_model)
    linguistic_path = root_config.resolve(root_config.linguistic)
    if model_path is None or not model_path.is_file():
        warnings.append(f"{label}: root-level {kind} model not found ({raw_model!r})")
        model_path = None
    if linguistic_path is None or not linguistic_path.is_file():
        warnings.append(f"{label}: root-level {kind} linguistic model not found")
        linguistic_path = None
    return types.ModelGroup(
        kind=kind,
        dir=directory,
        config=root_config,
        phonemes=dict(phonemes),
        languages=dict(languages),
        linguistic_path=linguistic_path,
        model_path=model_path,
        emb_dir=directory,
    )


def _copy_group_for_speaker(
    group: types.ModelGroup | None,
    root_speakers: Sequence[str],
    speaker: str,
) -> types.ModelGroup | None:
    """复制 ModelGroup 并解析该 group 自己的 speaker 名。

    不同子模型的 .emb 命名可能不同（例如 JiHeHong 声学是
    '231111_hdl_ds_acoustic.jihehong'，dsdur 是 '231111_hdl_ds_variance.jihehong'），
    这里按 dsconfig.speakers 的下标做映射，映射结果放在动态属性 selected_speaker 上。
    """
    if group is None:
        return None
    copied = dataclasses.replace(group)
    mapped = speaker
    group_speakers = list(group.config.speakers or [])
    if group_speakers and speaker:
        try:
            index = list(root_speakers).index(speaker)
        except ValueError:
            index = -1
        if 0 <= index < len(group_speakers):
            mapped = group_speakers[index]
        directory = group.emb_dir if group.emb_dir is not None else group.dir
        if speaker and not (Path(directory) / (mapped + ".emb")).is_file():
            if (Path(directory) / (speaker + ".emb")).is_file():
                mapped = speaker
    copied.selected_speaker = mapped  # type: ignore[attr-defined]
    return copied


def speaker_for_group(bank: types.Voicebank, group: types.ModelGroup | None = None) -> str:
    """某个 ModelGroup（dsdur/dsvariance/dspitch）应使用的 speaker 名。"""
    if group is None:
        return bank.selected_speaker
    return getattr(group, "selected_speaker", "") or bank.selected_speaker


def emb_path_for_group(bank: types.Voicebank, group: types.ModelGroup | None = None) -> Path | None:
    """读取某个 ModelGroup 的 speaker embedding 文件路径；不存在返回 None。"""
    speaker = speaker_for_group(bank, group)
    if not speaker:
        return None
    directory = Path(group.emb_dir if group.emb_dir is not None else group.dir) if group else bank.path
    path = directory / (speaker + ".emb")
    return path if path.is_file() else None


# --------------------------------------------------------------------------------------
# 声码器
# --------------------------------------------------------------------------------------
def _warn_missing_vocoder_model(
    config: types.VocoderConfig,
    label: str,
    warnings: list[str],
) -> None:
    if config.model_path is None:
        warnings.append(f"{label}: vocoder model path is empty")
    elif not Path(config.model_path).is_file():
        warnings.append(f"{label}: vocoder model not found ({config.model_path})")


def _find_builtin_vocoder(
    name: str,
    vocoder_dirs: Sequence[Path],
    warnings: list[str],
    label: str,
) -> types.VocoderConfig | None:
    for directory in vocoder_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        candidates = [
            directory / name / "vocoder.yaml",
            directory / name,
        ]
        if directory.name == name:
            candidates.append(directory / "vocoder.yaml")
        for candidate in candidates:
            try:
                if candidate.is_file() and candidate.suffix.lower() in (".yaml", ".yml"):
                    return config_parser.parse_vocoder_yaml(candidate)
                if candidate.is_file() and candidate.suffix.lower() == ".onnx":
                    return config_parser.vocoder_from_model_file(candidate)
                if candidate.is_dir() and (candidate / "vocoder.yaml").is_file():
                    return config_parser.parse_vocoder_yaml(candidate / "vocoder.yaml")
            except Exception as error:  # noqa: BLE001 - 继续查下一个目录
                warnings.append(f"{label}: failed to parse vocoder config {candidate}: {error}")
                continue
    return None


def _resolve_vocoder(
    directory: Path,
    root_config: types.DsConfig,
    vocoder_dirs: Sequence[Path],
    warnings: list[str],
    label: str,
) -> types.VocoderConfig | None:
    # 1) 声库自带 dsvocoder 优先级最高
    dsvocoder = directory / "dsvocoder" / "vocoder.yaml"
    if dsvocoder.is_file():
        try:
            config = config_parser.parse_vocoder_yaml(dsvocoder)
        except Exception as error:  # noqa: BLE001
            warnings.append(f"{label}: failed to parse dsvocoder/vocoder.yaml: {error}")
        else:
            _warn_missing_vocoder_model(config, label, warnings)
            return config

    name = (root_config.vocoder or "").strip()
    if name:
        # 2) root dsconfig.vocoder 指向声库目录内的本地路径/文件
        local = Path(name).expanduser()
        if not local.is_absolute():
            local = directory / local
        try:
            if local.is_file():
                if local.suffix.lower() in (".yaml", ".yml"):
                    config = config_parser.parse_vocoder_yaml(local)
                    _warn_missing_vocoder_model(config, label, warnings)
                    return config
                config = config_parser.vocoder_from_model_file(local, dsconfig=root_config)
                _warn_missing_vocoder_model(config, label, warnings)
                return config
            if local.is_dir() and (local / "vocoder.yaml").is_file():
                config = config_parser.parse_vocoder_yaml(local / "vocoder.yaml")
                _warn_missing_vocoder_model(config, label, warnings)
                return config
        except Exception as error:  # noqa: BLE001
            warnings.append(f"{label}: failed to load local vocoder {name!r}: {error}")

        # 3) 内建声码器名：在 vocoder_dirs 里找 <name>/vocoder.yaml
        builtin = _find_builtin_vocoder(name, vocoder_dirs, warnings, label)
        if builtin is not None:
            _warn_missing_vocoder_model(builtin, label, warnings)
            return builtin

    warnings.append(f"{label}: vocoder not found ({name!r}); is a vocoder directory configured?")
    return None


# --------------------------------------------------------------------------------------
# 声库解析
# --------------------------------------------------------------------------------------
def _parse_bank(
    directory: Path,
    bank_id: str,
    vocoder_dirs: Sequence[Path],
    warnings: list[str],
) -> list[types.Voicebank]:
    label = bank_id
    dsconfig_path = directory / "dsconfig.yaml"
    if not dsconfig_path.is_file():
        warnings.append(f"{label}: skipped (dsconfig.yaml not found)")
        return []
    try:
        root_config = config_parser.parse_dsconfig(dsconfig_path)
    except Exception as error:  # noqa: BLE001
        warnings.append(f"{label}: skipped (failed to parse dsconfig.yaml: {error})")
        return []

    issues: list[str] = []
    missing: list[str] = []
    if not root_config.acoustic:
        missing.append("acoustic")
    if not root_config.phonemes:
        missing.append("phonemes")
    has_dsvocoder = (directory / "dsvocoder" / "vocoder.yaml").is_file()
    if not root_config.vocoder and not has_dsvocoder:
        missing.append("vocoder")
    if missing:
        warnings.append(f"{label}: skipped (dsconfig.yaml missing required field(s): {', '.join(missing)})")
        return []

    acoustic_path = root_config.resolve(root_config.acoustic)
    if acoustic_path is None or not acoustic_path.is_file():
        warnings.append(f"{label}: skipped (acoustic model not found: {root_config.acoustic!r})")
        return []

    phonemes_path = root_config.resolve(root_config.phonemes)
    phonemes = _load_phonemes(phonemes_path)
    if not phonemes:
        warnings.append(f"{label}: skipped (phonemes file missing/empty: {root_config.phonemes!r})")
        return []

    languages: dict[str, int] = {}
    if root_config.use_lang_id:
        if not root_config.languages:
            issues.append("use_lang_id is true but dsconfig.yaml has no 'languages' file")
        else:
            language_path = root_config.resolve(root_config.languages)
            languages = _load_json_dict(language_path)
            if not languages:
                issues.append(f"languages file missing/empty: {root_config.languages!r}")

    character = config_parser.load_character(directory)
    display_name = character.name or directory.name

    # 子模型
    groups: dict[str, types.ModelGroup | None] = {}
    for kind, subdir_name in _MODEL_SUBDIRS:
        subdir = directory / subdir_name
        group: types.ModelGroup | None = None
        if (subdir / "dsconfig.yaml").is_file():
            group = _load_model_group(
                kind, subdir, root_config, phonemes, languages, issues, label
            )
        else:
            group = _load_root_model_group(
                kind, directory, root_config, phonemes, languages, issues, label
            )
        groups[kind] = group

    vocoder = _resolve_vocoder(directory, root_config, vocoder_dirs, issues, label)

    aux_dirs = [
        directory / subdir_name
        for _, subdir_name in _MODEL_SUBDIRS
        if (directory / subdir_name).is_dir()
    ]
    dictionary = dictionary_module.build_dictionary(
        directory,
        aux_dirs,
        use_lang_id=root_config.use_lang_id,
        default_language=config_parser.language_from_phonemizer(character.default_phonemizer),
        source_name=bank_id,
    )

    speakers = list(root_config.speakers or [])
    subbanks = list(character.subbanks)

    def make_bank(voice_id: str, voice_name: str, selected_speaker: str) -> types.Voicebank:
        bank = types.Voicebank(
            id=voice_id,
            bank_id=bank_id,
            name=voice_name,
            path=directory,
            config=root_config,
            phonemes=phonemes,
            languages=languages,
            speakers=speakers,
            subbanks=subbanks,
            selected_speaker=selected_speaker,
            description=character.description,
            author=character.author,
            version=character.version,
            dur=_copy_group_for_speaker(groups["dur"], speakers, selected_speaker),
            pitch=_copy_group_for_speaker(groups["pitch"], speakers, selected_speaker),
            variance=_copy_group_for_speaker(groups["variance"], speakers, selected_speaker),
            vocoder=vocoder,
            dictionary=dictionary,
            warnings=[],
        )
        bank.character = character  # type: ignore[attr-defined] - 额外元数据（非契约字段）
        bank_issues = list(issues)
        aux_groups = (("dur", bank.dur), ("pitch", bank.pitch), ("variance", bank.variance))
        for kind, group in aux_groups:
            if group is not None and not group.available:
                bank_issues.append(f"{kind} model incomplete (missing model/linguistic onnx)")
        if root_config.speakers is not None and selected_speaker:
            if not (directory / (selected_speaker + ".emb")).is_file():
                bank_issues.append(f"speaker embedding missing: {selected_speaker}.emb")
        for kind, group in aux_groups:
            if group is None or not group.config.speakers:
                continue
            mapped = speaker_for_group(bank, group)
            group_dir = Path(group.emb_dir if group.emb_dir is not None else group.dir)
            if mapped and not (group_dir / (mapped + ".emb")).is_file():
                bank_issues.append(f"{kind} speaker embedding missing: {mapped}.emb")
        bank.warnings = bank_issues
        return bank

    result: list[types.Voicebank] = []
    if len(speakers) > 1:
        for speaker in speakers:
            result.append(make_bank(f"{bank_id}::{speaker}", f"{display_name} ({speaker})", speaker))
    else:
        selected = speakers[0] if speakers else ""
        result.append(make_bank(bank_id, display_name, selected))

    warnings.extend(issues)
    return result


# --------------------------------------------------------------------------------------
# 对外 API
# --------------------------------------------------------------------------------------
def scan_voicebanks(
    voice_dirs: Iterable[str | Path],
    vocoder_dirs: Iterable[str | Path],
    *,
    max_depth: int = 3,
) -> list[types.Voicebank]:
    """扫描 voice_dirs，返回全部可用声库（多说话人按 speaker 展开）。"""
    voices, warnings = _scan(voice_dirs, vocoder_dirs, max_depth=max_depth)
    for message in warnings:
        LOG.warning("%s", message)
    return voices


def _scan(
    voice_dirs: Iterable[str | Path],
    vocoder_dirs: Iterable[str | Path],
    *,
    max_depth: int = 3,
) -> tuple[list[types.Voicebank], list[str]]:
    warnings: list[str] = []
    voice_roots = _as_path_list(voice_dirs)
    vocoder_roots = _as_path_list(vocoder_dirs)
    seen_banks: set[str] = set()
    used_ids: set[str] = set()
    voices: list[types.Voicebank] = []

    for root in voice_roots:
        if not root.is_dir():
            warnings.append(f"voice dir not found, skipped: {root}")
            continue
        for directory, bank_id in _iter_bank_dirs(root, max_depth):
            try:
                real = os.path.realpath(directory)
            except OSError:  # pragma: no cover
                real = str(directory)
            if real in seen_banks:
                continue  # 同一个声库通过多个 voice_dir/符号链接被扫到
            seen_banks.add(real)
            try:
                banks = _parse_bank(directory, bank_id, vocoder_roots, warnings)
            except Exception as error:  # noqa: BLE001 - 单个坏声库不影响整体
                warnings.append(f"{bank_id}: skipped ({type(error).__name__}: {error})")
                LOG.exception("failed to scan voicebank %s", directory)
                continue
            for bank in banks:
                voice_id = bank.id
                if voice_id in used_ids:
                    suffix = 2
                    while f"{voice_id}#{suffix}" in used_ids:
                        suffix += 1
                    warnings.append(
                        f"{bank.bank_id}: duplicate voice id {voice_id!r}, renamed to {voice_id}#{suffix}"
                    )
                    bank.id = f"{voice_id}#{suffix}"
                used_ids.add(bank.id)
                voices.append(bank)

    LOG.info(
        "voicebank scan: %d voice(s) from %d dir(s), %d warning(s)",
        len(voices),
        len(voice_roots),
        len(warnings),
    )
    return voices, warnings


class VoiceCatalog:
    """扫描结果缓存；get() 按 voice id 查找，找不到抛 KeyError。"""

    def __init__(
        self,
        voice_dirs: Iterable[str | Path],
        vocoder_dirs: Iterable[str | Path] = (),
        *,
        max_depth: int = 3,
    ) -> None:
        self.voice_dirs: list[Path] = _as_path_list(voice_dirs)
        self.vocoder_dirs: list[Path] = _as_path_list(vocoder_dirs)
        self.max_depth = max(0, int(max_depth))
        self.warnings: list[str] = []
        self._voices: list[types.Voicebank] = []
        self._by_id: dict[str, types.Voicebank] = {}
        self._scanned = False

    def refresh(self) -> None:
        voices, warnings = _scan(self.voice_dirs, self.vocoder_dirs, max_depth=self.max_depth)
        self._voices = voices
        self._by_id = {bank.id: bank for bank in voices}
        self.warnings = warnings
        self._scanned = True
        for message in warnings:
            LOG.warning("%s", message)

    def get(self, voice_id: str) -> types.Voicebank:
        if not self._scanned:
            self.refresh()
        try:
            return self._by_id[voice_id]
        except KeyError:
            available = ", ".join(self._by_id) or "(none)"
            raise KeyError(f"unknown voice {voice_id!r}; available: {available}") from None

    def voices(self) -> list[types.Voicebank]:
        if not self._scanned:
            self.refresh()
        return list(self._voices)

    def ids(self) -> list[str]:
        return [bank.id for bank in self.voices()]

    def __len__(self) -> int:
        return len(self.voices())

    def __iter__(self) -> Iterator[types.Voicebank]:
        return iter(self.voices())

    def __contains__(self, voice_id: object) -> bool:
        if not self._scanned:
            self.refresh()
        return voice_id in self._by_id
