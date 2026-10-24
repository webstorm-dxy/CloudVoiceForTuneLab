"""歌词 -> 带绝对毫秒时间的音素序列（DiffSingerPhonemizer）。

流程:
  1. NoteInput -> ResolvedNote（保留顺序与休止音符，绝对毫秒）；
  2. 每个 note 的歌词经 pinned / G2P 得到符号，并过滤到声学/时长模型的音素表；
  3. 连续（前 note end == 后 note start）且非休止的 note 组成一个 phrase；
  4. 有可用 dsdur（vb.dur.available && config.predict_dur && options.use_duration_model）
     时按 OpenUTAU DiffSingerBasePhonemizer.ProcessPart 的算法组句、跑 linguistic+duration
     模型、再用 stretch() 对齐到各 note 锚点（junction-anchored：body 从 note 头正向铺，
     前导辅音 / 下一 note 的辅音落在 note 头之前）。否则 / 失败时用启发式
     （前导辅音 ~70ms，元音均分，总长严格等于 note duration）；
  5. 句与句之间的空隙用 SP 填充，保证 Phrase.phones 沿时间轴连续
     （engine 渲染器按 token 顺序渲染，不连续会导致时间轴漂移）；
  6. 输出 types.Phrase：phones 绝对毫秒、note_index/tone/vowel/glide/speaker 齐全，
     Phrase.notes 与输入 notes 等长同序（含休止）。
"""
from __future__ import annotations

import logging
import math
from typing import Iterable, Sequence

from . import durations, g2p as g2p_module, types

LOG = logging.getLogger(__name__)

#: note 首尾相接判定容差（毫秒）
_CONTIGUOUS_EPS_MS = 0.5
#: 音素时间轴连续性判定容差（毫秒）
_GAP_EPS_MS = 1e-6
#: 默认 pause 符号
DEFAULT_PAUSE = "SP"


class DiffSingerPhonemizer:
    """OpenUTAU DiffSinger 音素定时器（G2P + 启发式 / dsdur 两种时长来源）。"""

    def __init__(
        self,
        runner: types.OnnxRunner | None,
        options: types.PhonemizerOptions | None = None,
    ) -> None:
        self.runner = runner
        self.options = options or types.PhonemizerOptions()
        self.warnings: list[str] = []
        self._g2p_cache: dict[int, g2p_module.G2P] = {}

    # ------------------------------------------------------------------ 公开入口
    def phonemize(
        self,
        vb: types.Voicebank,
        notes: Sequence[types.NoteInput],
        *,
        language: str = "auto",
        options: types.PhonemizerOptions | None = None,
    ) -> types.Phrase:
        options = options or self.options or types.PhonemizerOptions()
        note_list = list(notes)
        resolved = self._resolve_notes(note_list)
        if not note_list:
            return types.Phrase(phones=[], notes=resolved)

        g2p = self._g2p_for(vb)
        use_model = self._can_use_duration_model(vb, options) and self.runner is not None
        valid = self._valid_symbols(vb, use_model)

        note_symbols: dict[int, list[str]] = {}
        for index, note in enumerate(note_list):
            if resolved[index].rest:
                continue
            note_symbols[index] = self._symbols_for_note(note, g2p, language, valid)

        phones: list[types.Phone] = []
        for phrase in self._group_phrases(resolved):
            if use_model:
                try:
                    phrase_phones = self._phones_with_duration_model(
                        vb, resolved, phrase, note_symbols, options
                    )
                except Exception as error:  # noqa: BLE001 - 任何 dsdur 问题都回退启发式
                    self._warn(
                        f"voicebank {vb.id!r}: dsdur timing failed, using heuristic fallback "
                        f"({type(error).__name__}: {error})"
                    )
                    phrase_phones = self._phones_heuristic(vb, resolved, phrase, note_symbols, options)
            else:
                phrase_phones = self._phones_heuristic(vb, resolved, phrase, note_symbols, options)
            phones.extend(phrase_phones)

        phones = self._fill_gaps(phones)
        return types.Phrase(phones=phones, notes=resolved)

    # ------------------------------------------------------------------ G2P / 符号
    def _warn(self, message: str) -> None:
        if message not in self.warnings:
            if len(self.warnings) < 200:
                self.warnings.append(message)
            LOG.warning("%s", message)

    def _g2p_for(self, vb: types.Voicebank) -> g2p_module.G2P | None:
        dictionary = getattr(vb, "dictionary", None)
        if dictionary is None:
            return None
        key = id(dictionary)
        cached = self._g2p_cache.get(key)
        if cached is None or cached.dictionary is not dictionary:
            cached = g2p_module.G2P(
                dictionary,
                use_lang_id=bool(vb.config.use_lang_id),
                default_language=getattr(dictionary, "default_language", "auto"),
                warnings=self.warnings,
            )
            self._g2p_cache[key] = cached
        return cached

    def _can_use_duration_model(
        self, vb: types.Voicebank, options: types.PhonemizerOptions
    ) -> bool:
        if not options.use_duration_model:
            return False
        group = vb.dur
        if group is None or not group.available:
            return False
        return bool(group.config.predict_dur)

    def _valid_symbols(self, vb: types.Voicebank, use_model: bool) -> set[str]:
        symbols = set(vb.phonemes or ())
        if use_model and vb.dur is not None and vb.dur.phonemes:
            symbols &= set(vb.dur.phonemes)
        if not symbols:
            symbols = set(vb.phonemes or ())
        symbols.update(("SP", "AP"))
        return symbols

    def _symbols_for_note(
        self,
        note: types.NoteInput,
        g2p: g2p_module.G2P | None,
        language: str,
        valid: set[str],
    ) -> list[str]:
        pinned = getattr(note, "pinned", None)
        if pinned:
            if g2p is not None:
                resolved = g2p.resolve_pinned(pinned, language)
                filtered = [symbol for symbol in (resolved or ()) if symbol in valid]
            else:
                filtered = [str(symbol) for symbol in pinned if str(symbol) in valid]
            if filtered:
                return filtered
            self._warn(f"pinned phonemes of note {note.key!r} are not supported by the voicebank")
        lyric = (note.lyric or "").strip()
        if not lyric:
            return [DEFAULT_PAUSE]
        symbols: list[str] = []
        if g2p is not None:
            symbols = list(g2p.phonemize(lyric, language))
        else:  # 没有字典时只能接受空格分隔的合法音素
            tokens = lyric.split()
            if tokens and all(token in valid for token in tokens):
                symbols = tokens
        filtered = [symbol for symbol in symbols if symbol in valid]
        if filtered:
            return filtered
        if symbols:
            self._warn(
                f"lyric {lyric!r} produced phonemes {symbols!r} not supported by voicebank "
                f"{note.key!r}; using {DEFAULT_PAUSE}"
            )
        else:
            self._warn(f"lyric {lyric!r} could not be converted to phonemes; using {DEFAULT_PAUSE}")
        return [DEFAULT_PAUSE]

    # ------------------------------------------------------------------ 音符/乐句
    @staticmethod
    def _resolve_notes(notes: Sequence[types.NoteInput]) -> list[types.ResolvedNote]:
        resolved: list[types.ResolvedNote] = []
        for index, note in enumerate(notes):
            start = float(note.start_ms)
            end = float(note.end_ms)
            if end < start:
                end = start
            resolved.append(
                types.ResolvedNote(
                    key=str(note.key),
                    index=index,
                    start_ms=start,
                    end_ms=end,
                    pitch=int(note.pitch),
                    rest=bool(note.rest),
                    lyric=note.lyric or "",
                )
            )
        return resolved

    @staticmethod
    def _group_phrases(notes: Sequence[types.ResolvedNote]) -> list[list[int]]:
        phrases: list[list[int]] = []
        current: list[int] = []
        for index, note in enumerate(notes):
            if note.rest:
                if current:
                    phrases.append(current)
                    current = []
                continue
            if current:
                previous = notes[current[-1]]
                if abs(note.start_ms - previous.end_ms) <= _CONTIGUOUS_EPS_MS:
                    current.append(index)
                    continue
                phrases.append(current)
            current = [index]
        if current:
            phrases.append(current)
        return phrases

    # ------------------------------------------------------------------ 符号分类
    @staticmethod
    def _is_vowel(vb: types.Voicebank, symbol: str) -> bool:
        dictionary = getattr(vb, "dictionary", None)
        if dictionary is not None:
            try:
                return bool(dictionary.is_vowel(symbol))
            except Exception:  # noqa: BLE001
                pass
        return symbol in ("SP", "AP")

    @staticmethod
    def _is_glide(vb: types.Voicebank, symbol: str) -> bool:
        dictionary = getattr(vb, "dictionary", None)
        if dictionary is not None:
            try:
                return bool(dictionary.is_glide(symbol))
            except Exception:  # noqa: BLE001
                pass
        return False

    def _make_phone(
        self,
        vb: types.Voicebank,
        symbol: str,
        start_ms: float,
        end_ms: float,
        note: types.ResolvedNote,
        speaker: str,
        *,
        vowel: bool | None = None,
        glide: bool | None = None,
    ) -> types.Phone:
        return types.Phone(
            symbol=symbol,
            start_ms=float(start_ms),
            end_ms=float(max(start_ms, end_ms)),
            note_index=note.index,
            tone=int(note.pitch),
            vowel=self._is_vowel(vb, symbol) if vowel is None else bool(vowel),
            glide=self._is_glide(vb, symbol) if glide is None else bool(glide),
            speaker=speaker,
        )

    # ------------------------------------------------------------------ 启发式
    def _phones_heuristic(
        self,
        vb: types.Voicebank,
        notes: Sequence[types.ResolvedNote],
        phrase: Sequence[int],
        note_symbols: dict[int, list[str]],
        options: types.PhonemizerOptions,
    ) -> list[types.Phone]:
        speaker = vb.selected_speaker or ""
        phones: list[types.Phone] = []
        for index in phrase:
            note = notes[index]
            symbols = note_symbols.get(index) or [DEFAULT_PAUSE]
            vowel_flags = [self._is_vowel(vb, symbol) for symbol in symbols]
            duration_ms = max(0.0, note.end_ms - note.start_ms)
            phone_durations = durations.heuristic_note_durations(
                symbols,
                vowel_flags,
                duration_ms,
                min_phone_ms=float(options.min_phone_ms),
                max_leading_ms=float(options.max_leading_ms),
            )
            cursor = note.start_ms
            for position, (symbol, phone_duration) in enumerate(zip(symbols, phone_durations)):
                is_last = position == len(symbols) - 1
                end = note.end_ms if is_last else cursor + float(phone_duration)
                end = min(end, note.end_ms)
                phones.append(self._make_phone(vb, symbol, cursor, end, note, speaker))
                cursor = end
        return phones

    # ------------------------------------------------------------------ dsdur
    def _process_word(
        self,
        vb: types.Voicebank,
        symbols: Sequence[str],
        speaker: str,
    ) -> tuple[list[durations.GroupPhoneme], list[durations.GroupPhoneme]]:
        """OpenUTAU ProcessWord 的简化版（word 内只有一个 note）。

        返回 (leading, note_phones)：leading 会并入前一个 phonemesPerNote，
        note_phones 从 note 起点开始。
        """
        vowel_flags = [self._is_vowel(vb, symbol) for symbol in symbols]
        glide_flags = [self._is_glide(vb, symbol) for symbol in symbols]
        starts = [False] * len(symbols)
        if not any(vowel_flags):
            starts[0] = True
        for index, is_vowel in enumerate(vowel_flags):
            if not is_vowel:
                continue
            if index >= 2 and glide_flags[index - 1] and not vowel_flags[index - 2]:
                starts[index - 1] = True
            else:
                starts[index] = True
        leading: list[durations.GroupPhoneme] = []
        note_phones: list[durations.GroupPhoneme] = []
        started = False
        for index, symbol in enumerate(symbols):
            item = durations.GroupPhoneme(symbol, speaker)
            if starts[index] and not started:
                started = True
            if started:
                note_phones.append(item)
            else:
                leading.append(item)
        if not note_phones:  # 理论上不会发生（starts[0] 兜底），保险起见
            note_phones = leading
            leading = []
        return leading, note_phones

    def _phones_with_duration_model(
        self,
        vb: types.Voicebank,
        notes: Sequence[types.ResolvedNote],
        phrase: Sequence[int],
        note_symbols: dict[int, list[str]],
        options: types.PhonemizerOptions,
    ) -> list[types.Phone]:
        speaker = vb.selected_speaker or ""
        first = notes[phrase[0]]
        last = notes[phrase[-1]]
        groups: list[durations.PhonemeGroup] = [
            durations.PhonemeGroup(
                position_ms=first.start_ms - durations.SENTENCE_PADDING_MS,
                tone=first.pitch,
                phonemes=[durations.GroupPhoneme(DEFAULT_PAUSE, speaker)],
            )
        ]
        note_ranges: dict[int, tuple[int, int]] = {}
        flat_count = len(groups[0].phonemes)
        for index in phrase:
            note = notes[index]
            symbols = note_symbols.get(index) or [DEFAULT_PAUSE]
            leading, note_phones = self._process_word(vb, symbols, speaker)
            start_index = flat_count
            groups[-1].phonemes.extend(leading)
            flat_count += len(leading)
            groups.append(durations.PhonemeGroup(note.start_ms, note.pitch, note_phones))
            flat_count += len(note_phones)
            note_ranges[index] = (start_index, flat_count)
        end_ms = last.end_ms
        groups.append(durations.PhonemeGroup(position_ms=end_ms, tone=last.pitch, phonemes=[]))

        frame_ms = float(vb.dur.config.frame_ms) if vb.dur is not None else float(vb.frame_ms)
        if not math.isfinite(frame_ms) or frame_ms <= 0:
            raise ValueError(f"invalid dsdur frame_ms: {frame_ms}")
        duration_frames = [
            float(value)
            for value in durations.predict_duration_frames(
                self.runner, vb, groups, frame_ms=frame_ms
            )
        ]
        flat_phones = [phoneme for group in groups for phoneme in group.phonemes]
        if len(duration_frames) != len(flat_phones):
            raise ValueError(
                f"duration frames {len(duration_frames)} != phonemes {len(flat_phones)}"
            )

        # OpenUTAU junction-anchored 布局（TuneLab 音素契约）：
        #   * 每个 note 的正文（note 自己的音素组）从 note.start 正向铺到下一个锚点；
        #   * 前导辅音属于前一个 group，会随 stretch 落到 note 头之前
        #     （第一个 word 的前导辅音落在 note.start 之前、句首 500ms padding 区间内）；
        #   * 下一 note 的前导辅音落在上一 note 的尾段；
        #   * 句首 SP 不参与输出，它的时长只用于 word_dur 输入。
        positions = durations.align_positions(
            duration_frames, groups, frame_ms=frame_ms, end_pos_ms=end_ms
        )
        if len(positions) != len(flat_phones) - 1:
            raise ValueError(
                f"alignment positions {len(positions)} != phonemes-1 {len(flat_phones) - 1}"
            )

        phones: list[types.Phone] = []
        total = len(flat_phones)
        for index in phrase:
            note = notes[index]
            start_index, end_index = note_ranges[index]
            if end_index <= start_index:
                continue
            for flat_index in range(start_index, end_index):
                if flat_index <= 0:
                    continue
                start_ms = positions[flat_index - 1]
                end_phone = end_ms if flat_index >= total - 1 else positions[flat_index]
                item = flat_phones[flat_index]
                phones.append(
                    self._make_phone(vb, item.symbol, start_ms, end_phone, note, item.speaker)
                )
        return phones

    # ------------------------------------------------------------------ 空隙填充
    @staticmethod
    def _fill_gaps(phones: Sequence[types.Phone]) -> list[types.Phone]:
        """用 SP 填满内部空隙（engine 渲染要求 token 时间轴连续）。"""
        result: list[types.Phone] = []
        for phone in phones:
            if not result:
                result.append(phone)
                continue
            previous = result[-1]
            if phone.start_ms < previous.end_ms - _GAP_EPS_MS:
                # 轻微重叠/倒序：钳到前一个结束；完全错位时保留原样
                if phone.end_ms > previous.end_ms:
                    phone.start_ms = previous.end_ms
                else:
                    continue
            gap = phone.start_ms - previous.end_ms
            if gap > _GAP_EPS_MS:
                result.append(
                    types.Phone(
                        symbol=DEFAULT_PAUSE,
                        start_ms=float(previous.end_ms),
                        end_ms=float(phone.start_ms),
                        note_index=phone.note_index,
                        tone=int(phone.tone),
                        vowel=False,
                        glide=False,
                        speaker=phone.speaker,
                    )
                )
            result.append(phone)
        return result
