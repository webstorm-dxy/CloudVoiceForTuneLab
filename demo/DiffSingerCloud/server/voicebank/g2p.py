"""歌词 -> DiffSinger 音素符号（G2P）。

覆盖:
  * 钉死音素（NoteInput.pinned，由调用方直接给出符号序列）的校验/补语言前缀；
  * 空格分隔的合法音素串（例如 "l a" / "zh/b zh/a"）；
  * 汉字：pypinyin（Style.NORMAL，去声调）-> 拼音 -> 查 dsdict(-zh)；
    也支持用户直接输入带声调数字的拼音（ba / ba3 / nv3）；
  * 假名：内置 kana -> romaji 表 -> 查 dsdict-ja；
  * 拉丁词：查 dsdict-en / 基础 dsdict，整词优先；
  * use_lang_id 多语言声库：给字典返回的裸音素补 "zh/" / "ja/" / "en/" 前缀
    （字典里已经是前缀符号时保持原样）；
  * 未命中兜底：拼音声母/韵母拆分查表；最后记录 warning 返回空列表。

只做"文本 -> 符号"，音素到时间轴（含 dsdur）由 phonemizer.py 负责。
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Sequence

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - 环境已装 pypinyin，缺失时只影响汉字
    from pypinyin import Style, lazy_pinyin
except ImportError:  # pragma: no cover
    Style = None  # type: ignore[assignment]
    lazy_pinyin = None  # type: ignore[assignment]

LANGUAGES: tuple[str, ...] = ("zh", "ja", "en")

_HANZI_CLASS = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_KANA_CLASS = "\u3040-\u30ff\u31f0-\u31ff"
_LATIN_CLASS = "A-Za-z\u00c0-\u024f"

_RUN_RE = re.compile(
    f"(?P<kana>[{_KANA_CLASS}]+)"
    f"|(?P<hanzi>[{_HANZI_CLASS}]+)"
    f"|(?P<latin>[{_LATIN_CLASS}][{_LATIN_CLASS}0-9'\u2019-]*)"
)
_HANZI_RE = re.compile(f"[{_HANZI_CLASS}]")
_KANA_RE = re.compile(f"[{_KANA_CLASS}]")
_LATIN_RE = re.compile(f"[{_LATIN_CLASS}]")
_PINYIN_RE = re.compile(r"^[a-zv]+$")
_TONE_NUMBER_RE = re.compile(r"[0-5]$")

# 常见拼音声母（拆分兜底用）
_PINYIN_INITIALS: tuple[str, ...] = (
    "zh",
    "ch",
    "sh",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "r",
    "z",
    "c",
    "s",
    "y",
    "w",
)

# --------------------------------------------------------------------------------------
# 假名 -> 罗马音
# --------------------------------------------------------------------------------------
_KANA_BASE: dict[str, str] = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "を": "o", "ん": "n",
    "ゔ": "vu",
    "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゎ": "wa",
}

_KANA_DIGRAPHS: dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "ぢゃ": "ja", "ぢゅ": "ju", "ぢょ": "jo",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du",
    "ゔぁ": "va", "ゔぃ": "vi", "ゔぇ": "ve", "ゔぉ": "vo",
}
_SMALL_KANA = frozenset("ゃゅょぁぃぅぇぉゎ")
_SOKUON = "っ"
_LONG_VOWEL = "ー"


def katakana_to_hiragana(text: str) -> str:
    """片假名转平假名（保留长音符/促音等）。"""
    result: list[str] = []
    for char in text:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:
            result.append(chr(code - 0x60))
        else:
            result.append(char)
    return "".join(result)


def kana_to_romaji(text: str) -> list[str]:
    """假名串 -> 罗马音莫拉列表（促音 -> cl，长音符复制前一个元音）。"""
    kana = katakana_to_hiragana(text)
    tokens: list[str] = []
    index = 0
    length = len(kana)
    while index < length:
        char = kana[index]
        following = kana[index + 1] if index + 1 < length else ""
        pair = char + following
        if pair in _KANA_DIGRAPHS:
            tokens.append(_KANA_DIGRAPHS[pair])
            index += 2
            continue
        if char == _SOKUON:
            tokens.append("cl")
            index += 1
            continue
        if char == _LONG_VOWEL:
            if tokens:
                previous = tokens[-1]
                if previous:
                    tokens.append(previous[-1])
            index += 1
            continue
        romaji = _KANA_BASE.get(char)
        if romaji is not None:
            tokens.append(romaji)
        index += 1
    return tokens


# --------------------------------------------------------------------------------------
# 语言判断 / 拼音归一化
# --------------------------------------------------------------------------------------
def detect_language(text: str) -> str:
    """按字符集猜测歌词语言：zh（含汉字）/ ja（含假名）/ en（拉丁）/ auto。"""
    if not text:
        return "auto"
    has_hanzi = bool(_HANZI_RE.search(text))
    has_kana = bool(_KANA_RE.search(text))
    if has_hanzi and not has_kana:
        return "zh"
    if has_kana and not has_hanzi:
        return "ja"
    if has_hanzi and has_kana:
        return "zh"
    if _LATIN_RE.search(text):
        return "en"
    return "auto"


def normalize_pinyin(text: str) -> str:
    """拼音归一化：小写、ü/u: -> v、去声调数字。"""
    value = (text or "").strip().lower()
    value = value.replace("ü", "v").replace("u:", "v").replace("'", "")
    value = _TONE_NUMBER_RE.sub("", value)
    return value


def _split_pinyin(syllable: str) -> tuple[str, str] | None:
    for initial in _PINYIN_INITIALS:
        if syllable.startswith(initial) and len(syllable) > len(initial):
            return initial, syllable[len(initial):]
    return None


def _split_runs(text: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    for match in _RUN_RE.finditer(text):
        kind = match.lastgroup or ""
        if kind == "latin":
            runs.append(("latin", match.group(0)))
        elif kind == "hanzi":
            runs.append(("hanzi", match.group(0)))
        elif kind == "kana":
            runs.append(("kana", match.group(0)))
    return runs


# --------------------------------------------------------------------------------------
# G2P
# --------------------------------------------------------------------------------------
class G2P:
    """把一条歌词转换成 DiffSinger 音素符号列表。"""

    def __init__(
        self,
        dictionary: object,
        *,
        use_lang_id: bool = False,
        default_language: str = "auto",
        warnings: list[str] | None = None,
    ) -> None:
        self.dictionary = dictionary
        self.use_lang_id = bool(use_lang_id)
        default = (default_language or "auto").lower()
        self.default_language = default if default in LANGUAGES else "auto"
        self.warnings: list[str] = warnings if warnings is not None else []

    # ---- 小工具 -------------------------------------------------------------------
    def _warn(self, message: str) -> None:
        self.warnings.append(message)
        LOG.warning("g2p: %s", message)

    def _valid(self, symbol: str) -> bool:
        try:
            return bool(self.dictionary.is_valid(symbol))
        except Exception:  # noqa: BLE001 - 自定义 SymbolSet 可能没有实现全
            return False

    def _has_language(self, language: str) -> bool:
        has = getattr(self.dictionary, "has_language", None)
        if callable(has):
            try:
                return bool(has(language))
            except Exception:  # noqa: BLE001
                return False
        return False

    def _query(self, language: str, grapheme: str) -> list[str] | None:
        try:
            phones = self.dictionary.query_lang(language, grapheme)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            phones = None
        if phones:
            return list(phones)
        try:
            phones = self.dictionary.query(grapheme)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            phones = None
        if phones:
            return list(phones)
        return None

    def _prefix(self, symbols: Sequence[str], language: str) -> list[str]:
        """use_lang_id 声库：给裸音素补语言前缀（字典里已带前缀的保持不变）。"""
        if not self.use_lang_id or language not in LANGUAGES:
            return list(symbols)
        result: list[str] = []
        for symbol in symbols:
            if "/" in symbol:
                result.append(symbol)
                continue
            candidate = language + "/" + symbol
            result.append(candidate if self._valid(candidate) else symbol)
        return result

    def _language_order(self, hint: str) -> list[str]:
        if hint in LANGUAGES:
            return [hint] + [language for language in LANGUAGES if language != hint]
        if self.default_language in LANGUAGES:
            default = self.default_language
            return [default] + [language for language in LANGUAGES if language != default]
        if self._has_language("en"):
            return ["en", "zh", "ja"]
        return ["zh", "en", "ja"]

    # ---- 入口 ---------------------------------------------------------------------
    def phonemize(self, lyric: str, language: str = "auto") -> list[str]:
        """歌词 -> 音素符号。language: auto / zh / ja / en（part 级覆盖）。"""
        text = (lyric or "").strip()
        if not text:
            return []
        hint = language if language in LANGUAGES else "auto"

        # 1) 空格分隔的合法音素序列（也兼容 "zh/b zh/a" 这种显式前缀）
        tokens = text.split()
        if len(tokens) > 1:
            resolved = self._resolve_symbols(tokens, hint)
            if resolved is not None:
                return resolved

        # 2) 按字符类型分段处理（汉字 / 假名 / 拉丁）
        output: list[str] = []
        for kind, run in _split_runs(text):
            if kind == "hanzi":
                output.extend(self._hanzi(run, hint))
            elif kind == "kana":
                output.extend(self._kana(run))
            elif kind == "latin":
                output.extend(self._latin(run, hint))
        if output:
            return output

        # 3) 单段整串兜底（例如单个音素符号）
        resolved = self._resolve_symbols([text], hint)
        if resolved is not None:
            return resolved
        self._warn(f"no phonemes for lyric {lyric!r} (language={hint})")
        return []

    def resolve_pinned(self, pinned: Iterable[str] | None, language: str = "auto") -> list[str] | None:
        """校验钉死的音素列表；无效项跳过，全部无效返回 None。"""
        if not pinned:
            return None
        hint = language if language in LANGUAGES else "auto"
        tokens = [str(token).strip() for token in pinned if str(token).strip()]
        if not tokens:
            return None
        resolved = self._resolve_symbols(tokens, hint, strict=False)
        if not resolved:
            self._warn(f"pinned phonemes are all invalid: {tokens}")
            return None
        return resolved

    # ---- 解析辅助 -----------------------------------------------------------------
    def _resolve_symbols(
        self,
        tokens: Sequence[str],
        hint: str,
        *,
        strict: bool = True,
    ) -> list[str] | None:
        result: list[str] = []
        prefix_order = self._language_order(hint)
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            if self._valid(token):
                result.append(token)
                continue
            if "/" in token:
                if strict:
                    return None
                continue
            resolved = None
            if self.use_lang_id:
                for language in prefix_order:
                    candidate = language + "/" + token
                    if self._valid(candidate):
                        resolved = candidate
                        break
            if resolved is None:
                if strict:
                    return None
                continue
            result.append(resolved)
        return result or None

    def _pinyin_symbols(self, syllable: str) -> list[str] | None:
        normalized = normalize_pinyin(syllable)
        if not normalized or not _PINYIN_RE.match(normalized):
            return None
        symbols = self._query("zh", normalized)
        if symbols:
            return symbols
        split = _split_pinyin(normalized)
        if split is not None:
            head, tail = split
            head_symbols = self._query("zh", head)
            tail_symbols = self._query("zh", tail)
            if head_symbols and tail_symbols:
                return head_symbols + tail_symbols
        return None

    def _char_pinyin(self, char: str) -> str | None:
        if lazy_pinyin is None:  # pragma: no cover - pypinyin 已安装
            return None
        try:
            syllables = lazy_pinyin(char, style=Style.NORMAL, errors="default")
        except Exception:  # noqa: BLE001 - pypinyin 对奇怪字符可能抛错
            return None
        if not syllables:
            return None
        return syllables[0] or None

    def _hanzi(self, run: str, hint: str) -> list[str]:
        output: list[str] = []
        for char in run:
            if char.isspace():
                continue
            syllable = self._char_pinyin(char)
            if not syllable:
                self._warn(f"failed to romanize hanzi {char!r}")
                continue
            symbols = self._pinyin_symbols(syllable)
            if not symbols:
                self._warn(f"pinyin {syllable!r} of hanzi {char!r} not found in dictionary")
                continue
            output.extend(self._prefix(symbols, "zh"))
        return output

    def _kana(self, run: str) -> list[str]:
        output: list[str] = []
        for token in kana_to_romaji(run):
            if not token:
                continue
            symbols = self._query("ja", token)
            if not symbols:
                split = _split_romaji(token)
                if split is not None:
                    head_symbols = self._query("ja", split[0])
                    tail_symbols = self._query("ja", split[1])
                    if head_symbols and tail_symbols:
                        symbols = head_symbols + tail_symbols
            if not symbols:
                self._warn(f"kana romaji {token!r} not found in dictionary")
                continue
            output.extend(self._prefix(symbols, "ja"))
        return output

    def _latin(self, run: str, hint: str) -> list[str]:
        word = run.strip()
        if not word:
            return []
        language, symbols = self._lookup_word(word, hint)
        if not symbols:
            self._warn(f"lyric word {word!r} not found in dictionary (language={hint})")
            return []
        return self._prefix(symbols, language)

    def _lookup_word(self, word: str, hint: str) -> tuple[str, list[str] | None]:
        raw = word.strip()
        if not raw:
            return "", None
        if self._valid(raw):
            return "", [raw]
        lowered = raw.lower().replace("\u2019", "'")
        for language in self._language_order(hint):
            if language == "zh":
                pinyin = normalize_pinyin(lowered)
                if _PINYIN_RE.match(pinyin):
                    symbols = self._pinyin_symbols(pinyin)
                    if symbols:
                        return "zh", symbols
                continue
            symbols = self._query(language, lowered)
            if symbols:
                return language, symbols
            if lowered != raw:
                symbols = self._query(language, raw)
                if symbols:
                    return language, symbols
        # 最后用基础字典查整词（大小写变体）
        for candidate in (lowered, raw):
            symbols = self._query("", candidate) if candidate else None
            if symbols:
                return "", symbols
        return "", None


def _split_romaji(token: str) -> tuple[str, str] | None:
    """罗马音兜底拆分：kya -> ky + a；ka -> k + a。"""
    vowels = "aiueo"
    index = max(token.rfind(vowel) for vowel in vowels) if token else -1
    if index <= 0:
        return None
    return token[:index], token[index:]
