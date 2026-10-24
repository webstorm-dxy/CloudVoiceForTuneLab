"""dsdict 字典：grapheme -> phoneme 与音素类型表（types.SymbolSet 实现）。

支持:
  * 新版 dsdict.yaml（symbols + entries + 可选 replacements）；
  * 旧版 tab 分隔 dsdict.txt（只有 entries，没有类型表）；
  * 语言字典 dsdict-zh.yaml / dsdict-ja.yaml / dsdict-en.yaml（优先于基础字典）。

字典按需求懒加载：扫描声库时只记录路径，第一次 query/is_valid 时才解析。
文件可能是 25MB+ / 十几万条，因此 YAML 用针对该固定结构的流式解析器，
而不是 pyyaml 全量加载；遇到非常规结构的小文件再退回 pyyaml。
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from .config import read_text
from . import types

LOG = logging.getLogger(__name__)

_YAML_SUFFIXES = (".yaml", ".yml")
_TXT_SUFFIXES = (".txt", ".tsv", ".csv")
_LANGUAGE_DICT_RE = re.compile(r"^dsdict-([a-z]{2,8})\.(?:yaml|yml)$", re.IGNORECASE)
_FLOW_FALLBACK_MAX_BYTES = 8 * 1024 * 1024


# --------------------------------------------------------------------------------------
# 文件解析
# --------------------------------------------------------------------------------------
@dataclass
class DictionaryData:
    """一份字典文件解析后的原始内容。"""

    path: Path
    symbols: list[tuple[str, str]] = field(default_factory=list)
    entries: dict[str, list[str]] = field(default_factory=dict)
    replacements: dict[str, str] = field(default_factory=dict)


class _UnsupportedDictionaryYaml(ValueError):
    """fast parser 不认识的 YAML 结构（调用方按文件大小决定是否退回 pyyaml）。"""


def _iter_lines(path: Path) -> Iterator[str]:
    # read_text 已经处理 BOM / 编码；StringIO 迭代避免一次性 splitlines 拷贝
    stream = io.StringIO(read_text(path))
    for line in stream:
        yield line


def _scalar(value: str) -> str:
    """解析单个 YAML 标量（普通字符串原样返回，处理单/双引号）。"""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        quote = value[0]
        inner = value[1:-1]
        if quote == "'":
            return inner.replace("''", "'")
        try:
            loaded = yaml.safe_load(value)
        except yaml.YAMLError:
            return inner
        return loaded if isinstance(loaded, str) else inner
    return value


def _flow_list(value: str) -> list[str] | None:
    text = value.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    inner = text[1:-1].strip()
    if not inner:
        return []
    items: list[str] = []
    buffer: list[str] = []
    quote: str | None = None
    for char in inner:
        if quote is not None:
            buffer.append(char)
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            buffer.append(char)
        elif char == ",":
            items.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
    items.append("".join(buffer))
    return [_scalar(item) for item in items if item.strip()]


def _parse_legacy_dictionary(path: Path) -> DictionaryData:
    """旧版 dsdict.txt：每行 grapheme<TAB>phone1 phone2 ...。"""
    data = DictionaryData(path=path)
    for raw_line in _iter_lines(path):
        line = raw_line.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" in line:
            grapheme, _, phones = line.partition("\t")
        else:
            parts = line.split(None, 1)
            grapheme = parts[0]
            phones = parts[1] if len(parts) > 1 else parts[0]
        grapheme = grapheme.strip()
        if not grapheme:
            continue
        data.entries[grapheme] = [item for item in phones.replace("\t", " ").split(" ") if item]
    return data


def _parse_yaml_dictionary(path: Path) -> DictionaryData:
    """针对 OpenUTAU dsdict.yaml 固定结构的流式解析。"""
    data = DictionaryData(path=path)
    section: str | None = None
    current_symbol: list[str] | None = None
    current_phones: list[str] | None = None
    current_replacement: list[str] | None = None

    for raw_line in _iter_lines(path):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        is_top_level = (not line[:1].isspace()) and not stripped.startswith("-")
        if is_top_level:
            key, _, inline_value = stripped.partition(":")
            key = key.strip().lower()
            inline_value = inline_value.strip()
            if inline_value[:1] in ("[", "{"):
                raise _UnsupportedDictionaryYaml(f"flow style section: {line!r}")
            if key in ("symbols", "entries", "replacements"):
                section = key
            else:
                section = None
            current_symbol = None
            current_phones = None
            current_replacement = None
            continue

        if section == "symbols":
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if item.startswith("symbol:"):
                    current_symbol = [_scalar(item[len("symbol:"):].strip()), ""]
                    data.symbols.append((current_symbol[0], ""))
                elif item.startswith("type:") and current_symbol is not None:
                    current_symbol[1] = _scalar(item[len("type:"):].strip())
                    data.symbols[-1] = (current_symbol[0], current_symbol[1])
                else:
                    raise _UnsupportedDictionaryYaml(f"unsupported symbols line: {line!r}")
            else:
                key, _, value = stripped.partition(":")
                key = key.strip().lower()
                if key == "symbol":
                    name = _scalar(value)
                    current_symbol = [name, ""]
                    data.symbols.append((name, ""))
                elif key == "type" and current_symbol is not None:
                    current_symbol[1] = _scalar(value)
                    data.symbols[-1] = (current_symbol[0], current_symbol[1])
                else:
                    raise _UnsupportedDictionaryYaml(f"unsupported symbols line: {line!r}")

        elif section == "entries":
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if item.startswith("grapheme:"):
                    grapheme = _scalar(item[len("grapheme:"):].strip())
                    current_phones = []
                    data.entries[grapheme] = current_phones
                elif item.startswith("phonemes:"):
                    value = item[len("phonemes:"):].strip()
                    parsed = _flow_list(value)
                    if parsed is not None:
                        current_phones = parsed
                    elif current_phones is None:
                        raise _UnsupportedDictionaryYaml(f"unsupported entries line: {line!r}")
                elif item.startswith("{") or item.startswith("["):
                    raise _UnsupportedDictionaryYaml(f"flow style entry: {line!r}")
                elif current_phones is not None:
                    current_phones.append(_scalar(item))
                else:
                    raise _UnsupportedDictionaryYaml(f"unsupported entries line: {line!r}")
            else:
                key, _, value = stripped.partition(":")
                key = key.strip().lower()
                if key == "grapheme":
                    grapheme = _scalar(value)
                    current_phones = []
                    data.entries[grapheme] = current_phones
                elif key == "phonemes":
                    parsed = _flow_list(value)
                    if parsed is not None:
                        current_phones = parsed
                        if data.entries:
                            latest = next(reversed(data.entries))
                            data.entries[latest] = parsed
                    elif current_phones is None:
                        raise _UnsupportedDictionaryYaml(f"unsupported entries line: {line!r}")
                else:
                    raise _UnsupportedDictionaryYaml(f"unsupported entries line: {line!r}")

        elif section == "replacements":
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if item.startswith("from:"):
                    current_replacement = [_scalar(item[len("from:"):].strip()), ""]
                elif item.startswith("to:") and current_replacement is not None:
                    current_replacement[1] = _scalar(item[len("to:"):].strip())
                    data.replacements[current_replacement[0]] = current_replacement[1]
                elif item.startswith("{"):
                    raise _UnsupportedDictionaryYaml(f"flow style replacement: {line!r}")
                else:
                    raise _UnsupportedDictionaryYaml(f"unsupported replacements line: {line!r}")
            else:
                key, _, value = stripped.partition(":")
                key = key.strip().lower()
                if key == "from":
                    current_replacement = [_scalar(value), ""]
                elif key == "to" and current_replacement is not None:
                    current_replacement[1] = _scalar(value)
                    data.replacements[current_replacement[0]] = current_replacement[1]
                else:
                    raise _UnsupportedDictionaryYaml(f"unsupported replacements line: {line!r}")
        else:
            raise _UnsupportedDictionaryYaml(f"line outside sections: {line!r}")
    return data


def _dictionary_from_mapping(mapping: Mapping[str, Any], path: Path) -> DictionaryData:
    data = DictionaryData(path=path)
    symbols = mapping.get("symbols") or []
    if isinstance(symbols, (list, tuple)):
        for item in symbols:
            if isinstance(item, Mapping):
                data.symbols.append((str(item.get("symbol", "")), str(item.get("type", ""))))
            elif isinstance(item, str):
                data.symbols.append((item, ""))
    entries = mapping.get("entries") or []
    if isinstance(entries, (list, tuple)):
        for item in entries:
            if not isinstance(item, Mapping):
                continue
            grapheme = str(item.get("grapheme", ""))
            phones = item.get("phonemes") or []
            if isinstance(phones, str):
                phones = phones.split()
            data.entries[grapheme] = [str(phone) for phone in phones]
    replacements = mapping.get("replacements") or []
    if isinstance(replacements, (list, tuple)):
        for item in replacements:
            if isinstance(item, Mapping):
                source = str(item.get("from", ""))
                target = str(item.get("to", ""))
                if source:
                    data.replacements[source] = target
    return data


def parse_dictionary_file(path: str | Path) -> DictionaryData:
    """解析 dsdict.yaml / dsdict-*.yaml / 旧版 dsdict.txt。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in _TXT_SUFFIXES:
        return _parse_legacy_dictionary(path)
    try:
        return _parse_yaml_dictionary(path)
    except _UnsupportedDictionaryYaml as error:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        if size > _FLOW_FALLBACK_MAX_BYTES:
            raise
        LOG.debug("falling back to pyyaml for %s: %s", path, error)
        mapping = yaml.safe_load(read_text(path)) or {}
        if not isinstance(mapping, Mapping):
            raise ValueError(f"expected YAML mapping in {path}")
        return _dictionary_from_mapping(mapping, path)


def scan_dictionary_symbols(path: str | Path) -> list[tuple[str, str]]:
    """只扫描 symbols 区段（多数字典写在文件开头，可以提前结束）。"""
    path = Path(path)
    if path.suffix.lower() in _TXT_SUFFIXES:
        return []
    result: list[tuple[str, str]] = []
    current: list[str] | None = None
    section: str | None = None
    lines = _iter_lines(path)
    try:
        for raw_line in lines:
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            is_top_level = (not line[:1].isspace()) and not stripped.startswith("-")
            if is_top_level:
                key = stripped.split(":", 1)[0].strip().lower()
                if key == "symbols":
                    section = "symbols"
                    continue
                if section == "symbols":
                    break  # symbols 区段结束
                section = key
                continue
            if section != "symbols":
                continue
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if item.startswith("symbol:"):
                    current = [_scalar(item[len("symbol:"):].strip()), ""]
                    result.append((current[0], ""))
                elif item.startswith("type:") and current is not None:
                    current[1] = _scalar(item[len("type:"):].strip())
                    result[-1] = (current[0], current[1])
            else:
                key, _, value = stripped.partition(":")
                key = key.strip().lower()
                if key == "symbol":
                    current = [_scalar(value), ""]
                    result.append((current[0], ""))
                elif key == "type" and current is not None:
                    current[1] = _scalar(value)
                    result[-1] = (current[0], current[1])
    finally:
        close = getattr(lines, "close", None)
        if callable(close):
            close()
    return result


# --------------------------------------------------------------------------------------
# 路径发现
# --------------------------------------------------------------------------------------
def find_dictionary_paths(
    bank_dir: str | Path,
    aux_dirs: Iterable[str | Path] = (),
) -> tuple[list[Path], dict[str, Path], list[Path]]:
    """定位一个声库的字典文件。

    返回 (base_paths, language_paths, symbol_paths):
      * base_paths: 基础字典候选（dsdict.yaml 优先于旧版 dsdict.txt）；
      * language_paths: {"zh": dsdict-zh.yaml, ...}；
      * symbol_paths: 子模型目录里的字典（基础字典没有类型表时补 symbols，
        例如 JiHeHong 根目录只有旧版 dsdict.txt，类型表在 dsdur/dsdict.yaml）。
    """
    bank_dir = Path(bank_dir)
    base_paths: list[Path] = []
    for name in ("dsdict.yaml", "dsdict.yml", "dsdict.txt"):
        path = bank_dir / name
        if path.is_file():
            base_paths.append(path)

    language_paths: dict[str, Path] = {}
    try:
        children = sorted(bank_dir.iterdir())
    except OSError:
        children = []
    for path in children:
        if not path.is_file():
            continue
        match = _LANGUAGE_DICT_RE.match(path.name)
        if match:
            language_paths.setdefault(match.group(1).lower(), path)

    symbol_paths: list[Path] = []
    for directory in aux_dirs:
        directory = Path(directory)
        for name in ("dsdict.yaml", "dsdict.yml", "dsdict.txt"):
            path = directory / name
            if path.is_file() and path not in base_paths and path not in symbol_paths:
                symbol_paths.append(path)
    return base_paths, language_paths, symbol_paths


# --------------------------------------------------------------------------------------
# types.SymbolSet 实现
# --------------------------------------------------------------------------------------
class DiffSingerDictionary:
    """懒加载的 dsdict 字典（实现 types.SymbolSet）。

    query/query_lang 返回音素列表；is_valid/is_vowel/is_glide 依据 symbols
    区段的类型表。use_lang_id 声库的语言前缀由 g2p 层添加，这里只管字典本身。
    """

    def __init__(
        self,
        base_paths: Sequence[str | Path] = (),
        language_paths: Mapping[str, str | Path] | None = None,
        symbol_paths: Sequence[str | Path] = (),
        *,
        use_lang_id: bool = False,
        default_language: str = "auto",
        source_name: str = "",
    ) -> None:
        self.base_paths: list[Path] = [Path(path) for path in base_paths]
        self.language_paths: dict[str, Path] = {
            str(lang).lower(): Path(path) for lang, path in (language_paths or {}).items()
        }
        self.symbol_paths: list[Path] = [Path(path) for path in symbol_paths]
        self.use_lang_id = bool(use_lang_id)
        self.default_language = default_language or "auto"
        self.source_name = source_name
        self.warnings: list[str] = []

        self._symbol_types: dict[str, str] = {}
        self._glide: set[str] = set()
        self._symbols_loaded = False

        self._entries: dict[str, list[str]] = {}
        self._base_loaded = False
        self._replacements: dict[str, str] = {}

        self._language_entries: dict[str, dict[str, list[str]]] = {}
        self._language_loaded: set[str] = set()

    # ---- 状态 ---------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        """是否已经解析过任何字典文件（用于测试懒加载）。"""
        return self._symbols_loaded or self._base_loaded or bool(self._language_loaded)

    @property
    def entry_count(self) -> int:
        self._ensure_base()
        return len(self._entries)

    @property
    def replacements(self) -> dict[str, str]:
        self._ensure_base()
        return dict(self._replacements)

    @property
    def languages(self) -> list[str]:
        return sorted(self.language_paths)

    def has_language(self, language: str) -> bool:
        return (language or "").lower() in self.language_paths

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"DiffSingerDictionary(name={self.source_name!r}, "
            f"base={[p.name for p in self.base_paths]}, langs={self.languages})"
        )

    # ---- 懒加载 -------------------------------------------------------------------
    def _merge_symbols(self, symbols: Iterable[tuple[str, str]]) -> None:
        for name, symbol_type in symbols:
            name = (name or "").strip()
            if not name:
                continue
            symbol_type = (symbol_type or "").strip().lower()
            self._symbol_types[name] = symbol_type
            if symbol_type in ("semivowel", "liquid"):
                self._glide.add(name)
            else:
                self._glide.discard(name)

    def _add_symbol(self, name: str, symbol_type: str = "") -> None:
        self._symbol_types.setdefault(name, symbol_type)
        if symbol_type in ("semivowel", "liquid"):
            self._glide.add(name)

    def _ensure_symbols(self) -> None:
        if self._symbols_loaded:
            return
        found = False

        # 1) 语言字典：symbols 一般写在文件开头，scan 几行就能结束
        for language in sorted(self.language_paths):
            symbols = scan_dictionary_symbols(self.language_paths[language])
            if symbols:
                self._merge_symbols(symbols)
                found = True

        # 2) 基础 YAML 字典（symbols 在前时也很快）
        if not found:
            for path in self.base_paths:
                if path.suffix.lower() not in _YAML_SUFFIXES:
                    continue
                symbols = scan_dictionary_symbols(path)
                if symbols:
                    self._merge_symbols(symbols)
                    found = True
                    break

        # 3) 子模型目录的字典（旧版声库 root 的 dsdict.txt 没有类型表）
        if not found:
            for path in self.symbol_paths:
                try:
                    data = parse_dictionary_file(path)
                except Exception as error:  # noqa: BLE001 - 坏字典不致命
                    self.warnings.append(f"failed to read {path}: {error}")
                    continue
                if data.symbols:
                    self._merge_symbols(data.symbols)
                    found = True
                    break

        if not found:
            self.warnings.append("no dsdict symbol table found (is_vowel/is_glide unavailable)")

        # OpenUTAU 约定：SP / AP 永远是元音
        self._add_symbol("SP", "vowel")
        self._add_symbol("AP", "vowel")
        self._symbols_loaded = True

    def _ensure_base(self) -> None:
        if self._base_loaded:
            return
        self._base_loaded = True
        loaded = False
        for path in self.base_paths:
            try:
                data = parse_dictionary_file(path)
            except Exception as error:  # noqa: BLE001
                self.warnings.append(f"failed to read {path}: {error}")
                LOG.warning("failed to read dictionary %s: %s", path, error)
                continue
            if data.symbols:
                self._merge_symbols(data.symbols)
            if data.entries:
                self._entries.update(data.entries)
                loaded = True
            for source, target in data.replacements.items():
                self._replacements.setdefault(source, target)
            break  # 基础字典只取第一个有效文件
        if not loaded:
            # 没有基础字典时（少见），用子模型目录的字典兜底
            for path in self.symbol_paths:
                try:
                    data = parse_dictionary_file(path)
                except Exception:  # noqa: BLE001
                    continue
                if data.symbols:
                    self._merge_symbols(data.symbols)
                if data.entries:
                    self._entries.update(data.entries)
                    for source, target in data.replacements.items():
                        self._replacements.setdefault(source, target)
                    loaded = True
                    break
        if not loaded:
            self.warnings.append("no dsdict.yaml/dsdict.txt found")

    def _ensure_language(self, language: str) -> None:
        language = (language or "").lower()
        if not language or language in self._language_loaded:
            return
        self._language_loaded.add(language)
        path = self.language_paths.get(language)
        if path is None:
            return
        try:
            data = parse_dictionary_file(path)
        except Exception as error:  # noqa: BLE001
            self.warnings.append(f"failed to read {path}: {error}")
            LOG.warning("failed to read language dictionary %s: %s", path, error)
            return
        if data.symbols:
            self._merge_symbols(data.symbols)
        self._language_entries[language] = data.entries
        for source, target in data.replacements.items():
            self._replacements[source] = target

    def _remap(self, phones: Sequence[str]) -> list[str]:
        if not self._replacements:
            return list(phones)
        return [self._replacements.get(phone, phone) for phone in phones]

    # ---- types.SymbolSet ----------------------------------------------------------
    def query(self, grapheme: str) -> list[str] | None:
        self._ensure_base()
        phones = self._entries.get(grapheme)
        if phones is None:
            return None
        return self._remap(phones)

    def query_lang(self, lang: str, grapheme: str) -> list[str] | None:
        language = (lang or "").lower()
        if language in self.language_paths:
            self._ensure_language(language)
            entries = self._language_entries.get(language)
            if entries is not None:
                phones = entries.get(grapheme)
                if phones is not None:
                    return self._remap(phones)
        return self.query(grapheme)

    def is_valid(self, symbol: str) -> bool:
        self._ensure_symbols()
        return symbol in self._symbol_types

    def is_vowel(self, symbol: str) -> bool:
        self._ensure_symbols()
        return self._symbol_types.get(symbol) == "vowel"

    def is_glide(self, symbol: str) -> bool:
        self._ensure_symbols()
        return symbol in self._glide

    def symbol_type(self, symbol: str) -> str:
        self._ensure_symbols()
        return self._symbol_types.get(symbol, "")

    def symbols(self) -> Iterable[str]:
        self._ensure_symbols()
        return tuple(self._symbol_types)


def build_dictionary(
    bank_dir: str | Path,
    aux_dirs: Iterable[str | Path] = (),
    *,
    use_lang_id: bool = False,
    default_language: str = "auto",
    source_name: str = "",
) -> DiffSingerDictionary:
    """为声库目录构造懒加载字典对象（不解析文件）。"""
    base_paths, language_paths, symbol_paths = find_dictionary_paths(bank_dir, aux_dirs)
    return DiffSingerDictionary(
        base_paths,
        language_paths,
        symbol_paths,
        use_lang_id=use_lang_id,
        default_language=default_language,
        source_name=source_name or Path(bank_dir).name,
    )
