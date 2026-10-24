"""dsdur 时长预测与音素对齐（严格对齐 OpenUTAU DiffSingerBasePhonemizer）。

OpenUTAU 的算法（DiffSingerBasePhonemizer.ProcessPart）:
  1. 组乐句：连续（前 note end == 后 note start）的无休止音符为一个 phrase；
  2. 句首补 SP（Position = 首音符 - 500ms，只用于 word 位置/word_dur 输入）、句尾补 SP；
  3. linguistic 模型: tokens + word_div + word_dur（use_lang_id 时再加 languages）
     -> encoder_out / x_masks；
  4. duration 模型: encoder_out + x_masks + ph_midi（+ spk_embed）-> ph_dur_pred（帧）；
  5. 对齐：以每个 phonemesPerNote 的 Position 为锚点，按预测帧时长的比例 stretch
     到锚点之间（stretch 规则与 OpenUTAU 完全一致），得到每个音素的绝对毫秒位置。

无 dsdur / 推理失败时用启发式：note 内前导辅音约 70ms，元音均分剩余，
总时长严格等于 note duration，单音素不小于 min_phone_ms。

所有重活都走注入的 types.OnnxRunner（不直接 import onnxruntime）。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import types

LOG = logging.getLogger(__name__)

#: OpenUTAU 句首/句尾额外时间（毫秒）
HEAD_MS = 100.0
TAIL_MS = 100.0

#: OpenUTAU DiffSingerBasePhonemizer.ProcessPart 的 padding（ms）
SENTENCE_PADDING_MS = 500.0

#: 启发式：每个前导辅音的基准时长（毫秒）
HEURISTIC_LEADING_MS = 70.0


@dataclass
class GroupPhoneme:
    """乐句内一个待定时的音素。"""

    symbol: str
    speaker: str = ""


@dataclass
class PhonemeGroup:
    """OpenUTAU 的 phonemesPerNote：一组共享位置/音高的音素。"""

    position_ms: float
    tone: int
    phonemes: list[GroupPhoneme] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# 基础数学（与 OpenUTAU stretch() 一致）
# --------------------------------------------------------------------------------------
def stretch(source: Sequence[float], ratio: float, end_pos: float) -> list[float]:
    """OpenUTAU DiffSingerBasePhonemizer.stretch()：返回各音素起始位置（ms）。"""
    if not source:
        return []
    start_pos = end_pos - float(sum(source)) * float(ratio)
    result: list[float] = []
    cursor = start_pos
    for value in source:
        result.append(cursor)
        cursor += float(value) * float(ratio)
    return result


def frames_between(position_ms_1: float, position_ms_2: float, frame_ms: float) -> int:
    """OpenUTAU framesBetweenTickPos：int(ms2/frame) - int(ms1/frame)（向零截断）。"""
    if frame_ms <= 0:
        return 0
    return math.trunc(float(position_ms_2) / frame_ms) - math.trunc(float(position_ms_1) / frame_ms)


def _stretch_group(group: Sequence[float], span_ms: float, end_pos: float) -> list[float]:
    """把一组预测帧时长缩放到 [end-span, end]，返回组内各音素起点。"""
    if not group:
        return []
    total = float(sum(group))
    if total <= 1e-9 or not math.isfinite(total) or span_ms <= 0:
        # 退化情况：等分该区间
        count = len(group)
        start = end_pos - span_ms
        return [start + span_ms * index / count for index in range(count)]
    ratio = float(span_ms) / total
    return stretch(group, ratio, end_pos)


def align_positions(
    duration_frames: Sequence[float],
    groups: Sequence[PhonemeGroup],
    *,
    frame_ms: float,
    end_pos_ms: float,
) -> list[float]:
    """把 duration 模型输出的帧时长对齐到各 phonemesPerNote 锚点。

    返回长度为 (总音素数 - 1) 的列表：positions[k] 是第 k+1 个音素的绝对起点（ms）。
    （第 0 个音素是句首 SP，不参与输出，只用它的时长参与第一个锚点前的对齐组。）
    """
    counts = [len(group.phonemes) for group in groups]
    total_phones = int(sum(counts))
    if total_phones <= 1:
        return []
    frames = [float(value) for value in duration_frames]
    if len(frames) < total_phones:
        raise ValueError(
            f"duration model returned {len(frames)} frames for {total_phones} phonemes"
        )

    cumulative: list[int] = []
    running = 0
    for count in counts:
        running += count
        cumulative.append(running)

    align_points: list[tuple[int, float]] = [
        (cumulative[index], float(groups[index + 1].position_ms))
        for index in range(len(groups) - 1)
    ]
    if not align_points:
        # 只有一个 group（理论不会出现）：整段等分
        start = float(groups[0].position_ms)
        span = end_pos_ms - start
        count = total_phones
        return [start + span * index / count for index in range(1, count)]

    positions: list[float] = []
    first_index, first_pos = align_points[0]
    # 句首 SP 之后、第一个锚点之前的音素：保持预测时长不变（ratio = frameMs）
    positions.extend(stretch(frames[1:first_index], frame_ms, first_pos))
    for (a_index, a_pos), (b_index, b_pos) in zip(align_points, align_points[1:]):
        group = frames[a_index:b_index]
        positions.extend(_stretch_group(group, b_pos - a_pos, b_pos))

    if len(positions) < total_phones - 1:
        raise ValueError(
            f"alignment produced {len(positions)} positions for {total_phones - 1} phonemes"
        )
    result = positions[: total_phones - 1]
    for value in result:
        if not math.isfinite(value):
            raise ValueError("alignment produced non-finite position")
    return result


# --------------------------------------------------------------------------------------
# speaker embedding（只给 dsdur 时长模型用；裸 float32 LE）
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _read_embedding(path: str, hidden_size: int) -> np.ndarray:
    data = Path(path).read_bytes()
    needed = int(hidden_size) * 4
    if len(data) < needed:
        raise ValueError(f"speaker embedding too small: {path} ({len(data)} < {needed} bytes)")
    array = np.frombuffer(data[:needed], dtype="<f4").astype(np.float32)
    array.setflags(write=False)
    return array


def speaker_embed_by_phone(
    bank: types.Voicebank,
    group: types.ModelGroup,
    phone_speakers: Sequence[str],
) -> np.ndarray:
    """dsdur spk_embed 输入：[1, n_phones, hidden_size]。"""
    names = list(group.config.speakers or [])
    if not names:
        names = list(bank.speakers or [])
    if not names:
        raise ValueError("duration model expects spk_embed but dsconfig has no speakers")
    hidden = int(group.config.hidden_size or 256)
    directory = Path(group.emb_dir if group.emb_dir is not None else group.dir)
    matrix = np.stack(
        [_read_embedding(str(directory / (name + ".emb")), hidden) for name in names]
    ).astype(np.float32)
    indices = []
    for speaker in phone_speakers:
        try:
            indices.append(names.index(speaker))
        except ValueError:
            indices.append(0)
    return matrix[indices][np.newaxis, :, :]


# --------------------------------------------------------------------------------------
# duration 模型推理
# --------------------------------------------------------------------------------------
def _phoneme_language(symbol: str) -> str:
    return symbol.split("/", 1)[0] if "/" in symbol else ""


def _first_output(outputs: dict[str, Any], preferred: str | None = None) -> np.ndarray:
    if preferred and preferred in outputs:
        return np.asarray(outputs[preferred])
    for value in outputs.values():
        return np.asarray(value)
    raise ValueError("model returned no outputs")


def predict_duration_frames(
    runner: types.OnnxRunner,
    bank: types.Voicebank,
    groups: Sequence[PhonemeGroup],
    *,
    frame_ms: float,
) -> np.ndarray:
    """跑 dsdur linguistic + duration 模型，返回每个音素的预测帧时长（float 序列）。"""
    group = bank.dur
    if group is None:
        raise ValueError("voicebank has no dsdur model")
    if not group.available:
        raise ValueError("voicebank dsdur model is incomplete (model/linguistic onnx missing)")
    if group.linguistic_path is None or group.model_path is None:
        raise ValueError("voicebank dsdur model paths are not configured")
    config = group.config
    if not bool(config.predict_dur):
        raise ValueError("dsdur dsconfig has predict_dur: false")

    phoneme_tokens = group.phonemes
    symbols = [phoneme.symbol for phoneme_group in groups for phoneme in phoneme_group.phonemes]
    missing = [symbol for symbol in symbols if symbol not in phoneme_tokens]
    if missing:
        raise ValueError(
            "phonemes not supported by duration model: " + ", ".join(sorted(set(missing))[:8])
        )
    tokens = np.asarray([[int(phoneme_tokens[symbol]) for symbol in symbols]], dtype=np.int64)
    word_div = np.asarray([[len(item.phonemes) for item in groups[:-1]]], dtype=np.int64)
    word_dur = np.asarray(
        [[
            frames_between(groups[index].position_ms, groups[index + 1].position_ms, frame_ms)
            for index in range(len(groups) - 1)
        ]],
        dtype=np.int64,
    )

    linguistic_inputs: dict[str, Any] = {
        "tokens": tokens,
        "word_div": word_div,
        "word_dur": word_dur,
    }
    if bool(config.use_lang_id):
        language_ids = group.languages or bank.languages
        if not language_ids:
            raise ValueError("duration model uses use_lang_id but has no languages table")
        linguistic_inputs["languages"] = np.asarray(
            [[int(language_ids.get(_phoneme_language(symbol), 0)) for symbol in symbols]],
            dtype=np.int64,
        )

    linguistic_outputs = runner.run(group.linguistic_path, linguistic_inputs)
    encoder_out = np.asarray(linguistic_outputs.get("encoder_out", _first_output(linguistic_outputs)))
    if "x_masks" in linguistic_outputs:
        x_masks = np.asarray(linguistic_outputs["x_masks"])
    else:
        x_masks = np.ones((1, len(symbols)), dtype=bool)

    ph_midi = np.asarray(
        [[int(item.tone) for item in groups for _ in item.phonemes]], dtype=np.int64
    )
    duration_inputs: dict[str, Any] = {
        "encoder_out": encoder_out,
        "x_masks": x_masks,
        "ph_midi": ph_midi,
    }
    if config.speakers is not None:
        duration_inputs["spk_embed"] = speaker_embed_by_phone(
            bank, group, [item.speaker for item in groups for item in item.phonemes]
        )

    duration_outputs = runner.run(group.model_path, duration_inputs)
    predicted = _first_output(duration_outputs, "ph_dur_pred").astype(np.float64).reshape(-1)
    if predicted.size != len(symbols):
        raise ValueError(
            f"duration model returned {predicted.size} durations for {len(symbols)} phonemes"
        )
    if not np.all(np.isfinite(predicted)):
        raise ValueError("duration model returned non-finite durations")
    return predicted


# --------------------------------------------------------------------------------------
# 启发式时长
# --------------------------------------------------------------------------------------
def _apply_minimum(durations: Sequence[float], total_ms: float, min_ms: float) -> list[float]:
    """保证每个音素 >= min_ms 且总和精确等于 total_ms。"""
    count = len(durations)
    if count == 0:
        return []
    total_ms = max(0.0, float(total_ms))
    min_ms = max(0.0, float(min_ms))
    if total_ms < count * min_ms:
        return [total_ms / count] * count
    weights = [max(0.0, float(value) - min_ms) for value in durations]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [total_ms / count] * count
    base = min_ms
    remaining = total_ms - count * min_ms
    return [base + remaining * weight / weight_sum for weight in weights]


def heuristic_note_durations(
    symbols: Sequence[str],
    vowel_flags: Sequence[bool],
    duration_ms: float,
    *,
    min_phone_ms: float = 15.0,
    max_leading_ms: float = 120.0,
) -> list[float]:
    """无 dsdur 时的 note 内时长分配：前导辅音 ~70ms，元音均分剩余。"""
    count = len(symbols)
    if count == 0:
        return []
    duration_ms = max(0.0, float(duration_ms))
    first_vowel = next((index for index, flag in enumerate(vowel_flags) if flag), count)
    leading = first_vowel
    body = count - leading
    raw = [0.0] * count
    if leading:
        budget = min(
            HEURISTIC_LEADING_MS * leading,
            duration_ms * 0.5,
            max(0.0, float(max_leading_ms)),
        )
        if body:
            budget = min(budget, max(0.0, duration_ms - body * float(min_phone_ms)))
        budget = max(0.0, min(budget, duration_ms))
        per = budget / leading
        for index in range(leading):
            raw[index] = per
    remaining = duration_ms - sum(raw)
    if body:
        per = remaining / body
        for index in range(leading, count):
            raw[index] = per
    else:
        per = remaining / count
        for index in range(count):
            raw[index] += per
    durations = _apply_minimum(raw, duration_ms, min_phone_ms)
    # 消除浮点累计误差，保证总和严格等于 duration
    durations[-1] += duration_ms - sum(durations)
    return durations
