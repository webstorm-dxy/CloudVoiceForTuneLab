"""DiffSinger dspitch 自动音高预测（OpenUTAU DsPitch.Process 的 Python 实现）。

参考：/Users/daxingyi/RiderProjects/OpenUtau/OpenUtau.Core/DiffSinger/DiffSingerPitch.cs

一次预测 = linguistic 编码器 + pitch 预测器两段 ONNX：

1. linguistic 输入：
   - tokens = SP + phones + SP（用 vb.pitch.phonemes 的 token 表）；
   - ph_dur = [headFrames] + 每个 phone 的帧数 + [tailFrames]（headMs/tailMs = 100）；
   - predict_dur=true 时用 word_div / word_dur（按元音分组，与 DsVariance 相同），
     否则用 ph_dur；
   - use_lang_id=true 时 languages（音素前缀 "zh/" -> vb.pitch.languages[vb.languages]，SP=0）。
   输出 encoder_out。

2. pitch 输入：
   - encoder_out / ph_dur / note_midi / note_dur / pitch（全 60.0）/ retake（全 true）；
   - use_note_rest 时 note_rest（滑音 note 沿用前一个 note 的 rest 状态；
     note 时间范围内全是 SP/AP/无元音音素 -> rest）；
   - use_expr 时 expr（本协议没有 PEXP 曲线，恒 1.0）；
   - use_continuous_acceleration 时 steps，否则 speedup；
   - 多 speaker 时 spk_embed（group=vb.pitch 的 .emb，按帧）。
   输出 pitch_pred（半音/帧）。

note_midi/note_dur 的构造与 OpenUTAU 一致：
- note_midi = [首 note tone] + 每个 note 的 tone，然后把连续 rest note 组的 tone 反向填充成
  与相邻非 rest note 相同（首组取右侧、尾组取左侧、中间组按中点分给左右）；
- note_dur[0] = round(首 note 起点/帧) - round(startMs/帧)，startMs = min(首 note, 首 phone) - headMs；
  中间是相邻 note 起点的帧差，最后一格 = totalFrames - 前面之和。

帧长约定：predict() 的 frame_ms 默认取 dspitch 自己的 dsconfig（OpenUTAU 行为）；渲染器为了与
声学帧对齐会显式传声学 frame_ms（真实声库两者相同：44100/512）。模型输出帧数与计算出的
totalFrames 不一致时抛清晰 RuntimeError。

本模块只依赖注入的 types.OnnxRunner、voicebank 的 types 数据类和 engine/speaker.py，
不直接 import onnxruntime；模型路径由 voicebank discovery 填好的 vb.pitch（ModelGroup）提供。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..voicebank import types
from . import speaker

LOG = logging.getLogger(__name__)

HEAD_MS = float(types.HEAD_MS)
TAIL_MS = float(types.TAIL_MS)

#: OpenUTAU 认为一定是 rest 的音素。
_REST_SYMBOLS = ("AP", "SP")


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
def _round_frames(ms: float, frame_ms: float) -> int:
    """毫秒 -> 帧号（np.round = C# Math.Round 的银行家舍入）。"""
    return int(np.round(float(ms) / float(frame_ms)))


def _phoneme_language(symbol: str) -> str:
    if "/" in symbol:
        return symbol.split("/", 1)[0]
    return ""


def _speedup_for_steps(steps: int) -> int:
    speedup = max(1, 1000 // int(steps))
    while speedup > 1 and 1000 % speedup != 0:
        speedup -= 1
    return speedup


def _first_output(outputs: Mapping[str, np.ndarray], preferred: str | None = None) -> np.ndarray:
    if preferred is not None and preferred in outputs:
        return np.asarray(outputs[preferred])
    if not outputs:
        raise RuntimeError("ONNX model returned no outputs")
    return np.asarray(next(iter(outputs.values())))


def _vowel_state(vb: types.Voicebank, symbol: str) -> bool | None:
    """True/False = dsdict 明确分类；None = 没有字典/查询失败（调用方走兜底）。"""
    dictionary = getattr(vb, "dictionary", None)
    if dictionary is None:
        return None
    try:
        return bool(dictionary.is_vowel(symbol))
    except Exception:  # noqa: BLE001 - 字典实现异常时按未知处理
        return None


def _vowel_indices(vb: types.Voicebank, phrase: types.Phrase) -> list[int]:
    """phone 序列里的元音下标（word_div 用）；无字典时兜底为每 note 最后一个 phone。"""
    phones = list(phrase.phones)
    if not phones:
        return []
    indices: list[int] = []
    for index, phone in enumerate(phones):
        state = _vowel_state(vb, phone.symbol)
        if state is None:
            indices = []
            break
        if state:
            indices.append(index)
    if indices:
        return indices
    ranges = phrase.phone_index_ranges()
    fallback = [end - 1 for begin, end in ranges if end > begin]
    return fallback or [len(phones) - 1]


def _word_layout(
    vb: types.Voicebank,
    phrase: types.Phrase,
    frame_ms: float,
    head_frames: int,
    tail_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """OpenUTAU 的 word_div / word_dur（predict_dur=True 的 linguistic 输入）。"""
    phones = list(phrase.phones)
    count = len(phones)
    vowel_ids = _vowel_indices(vb, phrase)
    if not vowel_ids:
        vowel_ids = [count - 1]

    def trunc(ms: float) -> int:
        # OpenUTAU: (Int64)(x / frameMs) —— 截断取整。
        return int(float(ms) / float(frame_ms))

    word_div: list[int] = [vowel_ids[0] + 1]
    word_div.extend(b - a for a, b in zip(vowel_ids, vowel_ids[1:]))
    word_div.append(count - vowel_ids[-1] + 1)

    word_dur: list[int] = [
        trunc(phones[vowel_ids[0]].start_ms) - trunc(phones[0].start_ms) + int(head_frames)
    ]
    for a, b in zip(vowel_ids, vowel_ids[1:]):
        word_dur.append(trunc(phones[b - 1].end_ms) - trunc(phones[a].start_ms))
    notes = list(getattr(phrase, "notes", ()) or ())
    end_ms = float(notes[-1].end_ms) if notes else float(phones[-1].end_ms)
    word_dur.append(trunc(end_ms) - trunc(phones[vowel_ids[-1]].start_ms) + int(tail_frames))
    return np.asarray(word_div, dtype=np.int64), np.asarray(word_dur, dtype=np.int64)


def _language_ids(vb: types.Voicebank, phrase: types.Phrase, group: types.ModelGroup | None) -> np.ndarray:
    lang_map: dict[str, int] = {}
    if group is not None and group.languages:
        lang_map = dict(group.languages)
    elif vb.languages:
        lang_map = dict(vb.languages)
    ids = [0]
    for phone in phrase.phones:
        ids.append(int(lang_map.get(_phoneme_language(phone.symbol), 0)))
    ids.append(0)
    return np.asarray(ids, dtype=np.int64)


def _note_rest_flags(vb: types.Voicebank, phrase: types.Phrase) -> list[bool]:
    """OpenUTAU DsPitch.Process 的 note_rest 计算（长度 = len(notes)+1）。

    - 第一个元素固定 True（对应 prepend 的"前一个 note 是 rest"）；
    - 以 + 开头的延音 note 沿用前一个 note 的 rest 状态（本协议延音已合并，保留兼容）；
    - note 时间范围内音素为空 / 全是 SP、AP / 没有元音 -> rest。
    """
    phones = list(phrase.phones)
    notes = list(getattr(phrase, "notes", ()) or ())
    flags: list[bool] = [True]
    previous = True
    for note in notes:
        lyric = str(getattr(note, "lyric", "") or "")
        if lyric.startswith("+"):
            flags.append(previous)
            continue
        selected: list[types.Phone] = []
        started = False
        for phone in phones:
            if not started:
                if phone.end_ms <= note.start_ms + 1.0:
                    continue
                started = True
            if phone.start_ms < note.end_ms - 1.0:
                selected.append(phone)
            else:
                break
        if _vowel_state(vb, "SP") is None:
            # 没有 dsdict 分类：保守用 note.rest 标记。
            is_rest = bool(getattr(note, "rest", False)) or not selected
        else:
            is_rest = (not selected) or all(
                phone.symbol in _REST_SYMBOLS or not bool(_vowel_state(vb, phone.symbol))
                for phone in selected
            )
        flags.append(is_rest)
        previous = is_rest
    return flags


def _note_midi_layout(notes: Sequence[types.ResolvedNote], rest_flags: Sequence[bool]) -> np.ndarray:
    """note_midi = [首 note tone] + 各 note tone，并把连续 rest note 组的 tone 填充成相邻非 rest tone。"""
    if not notes:
        return np.zeros(1, dtype=np.float32)
    midi = np.asarray([float(notes[0].pitch)] + [float(note.pitch) for note in notes], dtype=np.float32)
    count = int(midi.size)
    flags = list(rest_flags)
    if len(flags) < count:
        flags = flags + [False] * (count - len(flags))
    index = 0
    while index < count:
        if not flags[index]:
            index += 1
            continue
        end = index + 1
        while end < count and flags[end]:
            end += 1
        if index == 0 and end == count:
            break  # 全是 rest：不改 tone
        if index == 0:
            midi[0:end] = midi[end]
        elif end == count:
            midi[index:count] = midi[index - 1]
        else:
            middle = (index + end + 1) // 2
            midi[index:middle] = midi[index - 1]
            midi[middle:end] = midi[end]
        index = end
    return midi


def _note_durations(
    notes: Sequence[types.ResolvedNote],
    phones: Sequence[types.Phone],
    frame_ms: float,
    total_frames: int,
) -> np.ndarray:
    """OpenUTAU 的 note_dur（长度 = len(notes)+1，总和 = totalFrames）。"""
    if not notes:
        return np.zeros(1, dtype=np.int64)
    start_ms = min(float(notes[0].start_ms), float(phones[0].start_ms)) - HEAD_MS
    values: list[int] = [
        _round_frames(notes[0].start_ms, frame_ms) - _round_frames(start_ms, frame_ms)
    ]
    for current, following in zip(notes, notes[1:]):
        values.append(
            _round_frames(following.start_ms, frame_ms) - _round_frames(current.start_ms, frame_ms)
        )
    values.append(0)
    values[-1] = int(total_frames) - sum(values)
    return np.asarray(values, dtype=np.int64)


# --------------------------------------------------------------------------------------
# 预测器
# --------------------------------------------------------------------------------------
class DiffSingerPitchPredictor:
    """dspitch 自动音高预测器（构造只注入 OnnxRunner，模型在 predict 时按需加载）。"""

    def __init__(self, runner: types.OnnxRunner) -> None:
        self.runner = runner

    # ---- 可用性 ----------------------------------------------------------------------
    @staticmethod
    def available(vb: types.Voicebank) -> bool:
        group = getattr(vb, "pitch", None)
        return bool(group is not None and group.available)

    # ---- 主入口 ----------------------------------------------------------------------
    def predict(
        self,
        vb: types.Voicebank,
        phrase: types.Phrase,
        *,
        frame_ms: float | None = None,
        steps: int = 20,
    ) -> np.ndarray:
        """返回预测音高（MIDI 半音，float32，长度 = token 帧数之和）。

        frame_ms 默认用 dspitch 自己的 dsconfig；渲染器传声学 frame_ms 以严格对齐声学帧。
        """
        group = getattr(vb, "pitch", None)
        if group is None or not group.available:
            raise RuntimeError(
                "voicebank '" + str(vb.id) + "' has no usable dspitch model "
                "(need linguistic + pitch onnx in dspitch/dsconfig.yaml)"
            )
        config = group.config
        phones = list(getattr(phrase, "phones", ()) or ())
        notes = list(getattr(phrase, "notes", ()) or ())
        if not phones:
            raise RuntimeError("pitch prediction of voicebank '" + str(vb.id) + "' requires at least one phone")
        if not notes:
            raise RuntimeError("pitch prediction of voicebank '" + str(vb.id) + "' requires at least one note")

        effective_frame_ms = float(frame_ms if frame_ms is not None else config.frame_ms)
        if effective_frame_ms <= 0:
            raise RuntimeError("dspitch of voicebank '" + str(vb.id) + "' has invalid frame_ms")
        head_frames = _round_frames(HEAD_MS, effective_frame_ms)
        tail_frames = _round_frames(TAIL_MS, effective_frame_ms)

        symbols = ["SP"] + [phone.symbol for phone in phones] + ["SP"]
        tokens = self._tokenize(group.phonemes, symbols, vb=vb)
        durations: list[int] = [int(head_frames)]
        for phone in phones:
            start = _round_frames(phone.start_ms, effective_frame_ms)
            end = _round_frames(phone.end_ms, effective_frame_ms)
            durations.append(max(0, end - start))
        durations.append(int(tail_frames))
        ph_dur = np.asarray(durations, dtype=np.int64)
        total_frames = int(ph_dur.sum())
        if total_frames <= 0:
            raise RuntimeError("pitch prediction of voicebank '" + str(vb.id) + "' got 0 frames")

        # ---- linguistic 编码器 ------------------------------------------------------
        linguistic_path = self._model_path(group, "linguistic", vb)
        linguistic_inputs: dict[str, Any] = {"tokens": tokens[np.newaxis, :]}
        if bool(config.predict_dur):
            word_div, word_dur = _word_layout(vb, phrase, effective_frame_ms, head_frames, tail_frames)
            linguistic_inputs["word_div"] = word_div[np.newaxis, :]
            linguistic_inputs["word_dur"] = word_dur[np.newaxis, :]
        else:
            linguistic_inputs["ph_dur"] = ph_dur[np.newaxis, :]
        if bool(config.use_lang_id):
            linguistic_inputs["languages"] = _language_ids(vb, phrase, group)[np.newaxis, :]
        linguistic_outputs = self._run_model(
            linguistic_path,
            linguistic_inputs,
            context="dspitch linguistic model of voicebank '" + str(vb.id) + "'",
        )
        encoder_out = _first_output(linguistic_outputs, "encoder_out").astype(np.float32)

        # ---- pitch 预测器 -----------------------------------------------------------
        pitch_path = self._model_path(group, "pitch", vb)
        pitch_names = set(self.runner.input_names(pitch_path))
        rest_flags = _note_rest_flags(vb, phrase)
        note_midi = _note_midi_layout(notes, rest_flags)
        note_dur = _note_durations(notes, phones, effective_frame_ms, total_frames)

        pitch_inputs: dict[str, Any] = {
            "encoder_out": encoder_out,
            "ph_dur": ph_dur[np.newaxis, :],
            "note_midi": note_midi[np.newaxis, :],
            "note_dur": note_dur[np.newaxis, :],
            "pitch": np.full((1, total_frames), 60.0, dtype=np.float32),
            "retake": np.ones((1, total_frames), dtype=np.bool_),
        }
        if "note_rest" in pitch_names:
            pitch_inputs["note_rest"] = np.asarray(rest_flags, dtype=np.bool_)[np.newaxis, :]
        if "expr" in pitch_names:
            pitch_inputs["expr"] = np.ones((1, total_frames), dtype=np.float32)
        steps_value = max(1, int(steps or 20))
        if bool(config.use_continuous_acceleration):
            pitch_inputs["steps"] = np.asarray([steps_value], dtype=np.int64)
        else:
            pitch_inputs["speedup"] = np.asarray([_speedup_for_steps(steps_value)], dtype=np.int64)
        if "spk_embed" in pitch_names:
            pitch_inputs["spk_embed"] = speaker.embed_by_frame(
                vb, phrase, ph_dur, total_frames, head_frames, tail_frames, group=group
            )
        outputs = self._run_model(
            pitch_path,
            pitch_inputs,
            context="dspitch pitch model of voicebank '" + str(vb.id) + "'",
        )
        predicted = _first_output(outputs, "pitch_pred").astype(np.float32).reshape(-1)
        if predicted.size != total_frames:
            raise RuntimeError(
                "dspitch pitch model of voicebank '" + str(vb.id) + "' returned "
                + str(predicted.size) + " frames, expected " + str(total_frames)
                + " (phonemes=" + str(len(phones)) + ", frame_ms=" + str(effective_frame_ms) + ")"
            )
        if not np.isfinite(predicted).all():
            raise RuntimeError("dspitch pitch model of voicebank '" + str(vb.id) + "' returned non-finite values")
        return predicted

    # ---- 内部 ------------------------------------------------------------------------
    def _tokenize(self, phonemes: Mapping[str, int], symbols: Sequence[str], *, vb: types.Voicebank) -> np.ndarray:
        tokens = np.empty(len(symbols), dtype=np.int64)
        for index, symbol in enumerate(symbols):
            try:
                tokens[index] = int(phonemes[symbol])
            except KeyError as error:
                raise RuntimeError(
                    "dspitch of voicebank '" + str(vb.id) + "' does not support phoneme '"
                    + str(symbol) + "' (" + str(len(phonemes)) + " symbols in its phoneme table)"
                ) from error
        return tokens

    @staticmethod
    def _model_path(group: types.ModelGroup, kind: str, vb: types.Voicebank) -> Path:
        path = group.linguistic_path if kind == "linguistic" else group.model_path
        if path is None:
            raise RuntimeError(
                "dspitch " + kind + " model path is not configured for voicebank '" + str(vb.id) + "'"
            )
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(
                "dspitch " + kind + " model of voicebank '" + str(vb.id) + "' not found: " + str(path)
            )
        return path

    def _run_model(self, path: Path, inputs: Mapping[str, Any], *, context: str) -> dict[str, np.ndarray]:
        names = self.runner.input_names(path)
        expected = set(names)
        missing = sorted(expected - set(inputs))
        if missing:
            raise RuntimeError(
                context + ": model requires input(s) that were not built: " + ", ".join(missing)
            )
        filtered = {name: inputs[name] for name in names}
        return self.runner.run(path, filtered)
