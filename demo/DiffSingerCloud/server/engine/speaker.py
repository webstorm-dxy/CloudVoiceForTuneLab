"""speaker embedding（OpenUTAU .emb 文件 -> 帧级 spk_embed 张量）。

OpenUTAU 的 .emb 是裸 float32 little-endian：hidden_size 个 float（通常 256 = 1024 字节），
文件名是 dsconfig speakers 列表里的 speaker 名（例如
"231111_hdl_ds_acoustic.jihehong.emb"）。单 speaker 声库等价于"常量帧"，多 speaker 声库
（如 yousa 的 Bright/Cute/Normal/...）按每个 phone 的 speaker 逐帧填充。

与 OpenUTAU DiffSingerSpeakerEmbedManager.PhraseSpeakerEmbedByFrame 的对应关系：

- 帧布局：[headFrames 个首个 phone 的 speaker][每个 phone 的 duration 帧][tailFrames 个末 phone
  的 speaker]，正好 totalFrames 帧；
- speaker 名查不到时回落到 index 0（与 OpenUTAU getSpeakerIndexBySuffix 一致）；
- OpenUTAU 还会把 clXX voice color 曲线叠加成 speaker 混合；本协议没有 voice color 曲线，
  这里只做"每帧一个 speaker 的 one-hot"（若以后要加，在 embed_by_frame 里按帧权重混合即可）。

公开函数（inference.py 使用）：

- load_embedding(path, hidden_size)：读一个 .emb；
- speaker_embedding_matrix(vb, group=None, hidden_size=None)：读整份 speaker 表；
- embed_by_phone(...)：[1, n_phones, hidden]（dsdur 声库层用）；
- embed_by_frame(...)：[1, total_frames, hidden]（声学 / 方差模型用）。
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from ..voicebank import types

LOG = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def _read_embedding_cached(path: str, hidden_size: int) -> np.ndarray:
    data = Path(path).read_bytes()
    needed = int(hidden_size) * 4
    if len(data) < needed:
        raise ValueError(
            "speaker embedding file is too small: " + path + " has " + str(len(data))
            + " bytes, need " + str(needed) + " (hidden_size=" + str(hidden_size) + " float32)"
        )
    array = np.frombuffer(data[:needed], dtype="<f4").astype(np.float32)
    array.setflags(write=False)
    return array


def load_embedding(path: Path | str, hidden_size: int) -> np.ndarray:
    """读取 .emb（float32 LE，hidden_size 个 float），返回只读 float32 一维数组。"""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("speaker embedding file not found: " + str(resolved))
    return _read_embedding_cached(str(resolved), int(hidden_size))


def clear_cache() -> None:
    """释放 .emb 缓存（测试用；正常运行时 .emb 只有几 KB，不需要清理）。"""
    _read_embedding_cached.cache_clear()
    _speaker_matrix_cached.cache_clear()


def _config_of(vb: types.Voicebank, group: types.ModelGroup | None) -> types.DsConfig:
    return group.config if group is not None else vb.config


def speaker_names(vb: types.Voicebank, group: types.ModelGroup | None = None) -> list[str]:
    """该声库（或辅助模型组）的 speaker 列表；单 speaker 声库返回 []。"""
    config = _config_of(vb, group)
    names = list(config.speakers or [])
    if not names:
        names = list(vb.speakers or [])
    return names


def hidden_size_of(vb: types.Voicebank, group: types.ModelGroup | None = None, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    return int(_config_of(vb, group).hidden_size or 256)


@lru_cache(maxsize=16)
def _speaker_matrix_cached(paths: tuple[str, ...], hidden_size: int) -> np.ndarray:
    return np.stack([_read_embedding_cached(path, hidden_size) for path in paths]).astype(np.float32)


def speaker_embedding_matrix(
    vb: types.Voicebank,
    *,
    group: types.ModelGroup | None = None,
    hidden_size: int | None = None,
) -> tuple[list[str], np.ndarray]:
    """读取整份 speaker 表，返回 (speaker 名列表, [n_speakers, hidden] float32 矩阵)。"""
    names = speaker_names(vb, group)
    if not names:
        raise ValueError(
            "voicebank '" + str(vb.id) + "' has no speakers configured; cannot build spk_embed"
        )
    hidden = hidden_size_of(vb, group, hidden_size)
    paths: list[Path] = []
    missing: list[str] = []
    for name in names:
        path = vb.emb_path(name, group=group)
        if path is None:
            missing.append(name)
        else:
            paths.append(path.expanduser().resolve())
    if missing:
        where = (group.dir if group is not None else vb.path)
        raise FileNotFoundError(
            "speaker embedding .emb missing for voicebank '" + str(vb.id) + "' in " + str(where)
            + ": " + ", ".join(missing) + " (expected files like '<speaker>.emb', hidden_size=" + str(hidden) + ")"
        )
    matrix = _speaker_matrix_cached(tuple(str(path) for path in paths), hidden)
    return names, matrix


def speaker_index(names: Sequence[str], speaker: str | None) -> int:
    """speaker 名 -> 索引；未知/空名回落到 0（OpenUTAU getSpeakerIndexBySuffix 语义）。"""
    if speaker:
        try:
            return list(names).index(speaker)
        except ValueError:
            LOG.debug("speaker %r is not in dsconfig speakers %s; falling back to index 0", speaker, list(names))
    return 0


def _phone_speaker(phone: types.Phone, vb: types.Voicebank) -> str:
    return getattr(phone, "speaker", "") or vb.selected_speaker


def embed_by_phone(
    vb: types.Voicebank,
    phrase: types.Phrase,
    *,
    group: types.ModelGroup | None = None,
    hidden_size: int | None = None,
) -> np.ndarray:
    """[1, n_phones, hidden]，供 dsdur 时长模型的 spk_embed 输入使用。"""
    names, matrix = speaker_embedding_matrix(vb, group=group, hidden_size=hidden_size)
    phones = list(phrase.phones)
    if not phones:
        return np.zeros((1, 0, matrix.shape[1]), dtype=np.float32)
    indices = np.asarray([speaker_index(names, _phone_speaker(phone, vb)) for phone in phones], dtype=np.int64)
    return matrix[indices][np.newaxis, :, :].astype(np.float32)


def embed_by_frame(
    vb: types.Voicebank,
    phrase: types.Phrase,
    durations: Sequence[int],
    total_frames: int,
    head_frames: int,
    tail_frames: int,
    *,
    group: types.ModelGroup | None = None,
    hidden_size: int | None = None,
) -> np.ndarray:
    """[1, total_frames, hidden]，供声学 / 方差模型的 spk_embed 输入使用。

    durations 是 OpenUTAU 风格的 token 帧时长列表（含首尾 SP）：
    [headFrames, phone1, phone2, ..., tailFrames]，总帧数 = sum(durations)。
    """
    names, matrix = speaker_embedding_matrix(vb, group=group, hidden_size=hidden_size)
    phones = list(phrase.phones)
    hidden = matrix.shape[1]
    total = int(total_frames)
    head = max(0, int(head_frames))
    tail = max(0, int(tail_frames))
    if total <= 0:
        return np.zeros((1, 0, hidden), dtype=np.float32)
    indices = np.zeros(total, dtype=np.int64)
    if not phones:
        return np.zeros((1, total, hidden), dtype=np.float32)
    head_index = speaker_index(names, _phone_speaker(phones[0], vb))
    tail_index = speaker_index(names, _phone_speaker(phones[-1], vb))
    cursor = min(head, total)
    indices[:cursor] = head_index
    body_counts = list(durations[1:1 + len(phones)]) if len(durations) > 1 else []
    if len(body_counts) < len(phones):
        body_counts = body_counts + [0] * (len(phones) - len(body_counts))
    for phone, count in zip(phones, body_counts):
        count = int(count)
        if count <= 0 or cursor >= total:
            continue
        end = min(total, cursor + count)
        indices[cursor:end] = speaker_index(names, _phone_speaker(phone, vb))
        cursor = end
    if cursor < total:
        indices[cursor:] = tail_index
    return matrix[indices][np.newaxis, :, :].astype(np.float32)
