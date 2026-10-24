"""音频 / 曲线工具（HTTP 响应编码、块裁剪、重采样、能量包络）。"""
from __future__ import annotations

import base64
import io
import math
import wave
from typing import Sequence

import numpy as np


def semitone_to_hz(semitone: float | np.ndarray) -> float | np.ndarray:
    """MIDI 半音（60 = C4）转频率（Hz）。与 OpenUTAU MusicMath.ToneToFreq 一致。"""
    return 440.0 * np.power(2.0, (np.asarray(semitone, dtype=np.float64) - 69.0) / 12.0)


def hz_to_semitone(freq: float | np.ndarray) -> float | np.ndarray:
    return 12.0 * np.log2(np.maximum(np.asarray(freq, dtype=np.float64), 1e-6) / 440.0) + 69.0


def sample_points(
    points: Sequence[tuple[float, float]] | None,
    times: np.ndarray,
    default: float,
) -> np.ndarray:
    """把 (t, v) 折线按 times 采样（线性插值，两端保持端点值）。

    points 为 None/空时返回全 default。与协议里 pitch / parameters 的 100Hz、50Hz 采样点对应。
    """
    times = np.asarray(times, dtype=np.float64)
    if not points:
        return np.full(times.shape, float(default), dtype=np.float32)
    arr = np.asarray([(float(t), float(v)) for t, v in points], dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    if arr.shape[0] == 1:
        return np.full(times.shape, float(arr[0, 1]), dtype=np.float32)
    values = np.interp(times, arr[:, 0], arr[:, 1])
    return values.astype(np.float32)


def resample_linear(values: Sequence[float] | np.ndarray | None, length: int) -> np.ndarray | None:
    """把一条曲线线性重采样到 length 点（对应 OpenUTAU DiffSingerUtils.ResampleCurve）。"""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    if arr.size == length:
        return arr.copy()
    if arr.size == 1:
        return np.full(length, arr[0], dtype=np.float32)
    src = np.linspace(0.0, 1.0, arr.size)
    dst = np.linspace(0.0, 1.0, length)
    return np.interp(dst, src, arr).astype(np.float32)


def pcm_f32le(samples: np.ndarray) -> bytes:
    """float32 单声道 little-endian PCM 字节。"""
    return np.ascontiguousarray(np.asarray(samples, dtype="<f4").reshape(-1)).tobytes()


def encode_wav16(samples: np.ndarray, sample_rate: int) -> bytes:
    """16-bit PCM WAV（调试 / 导出用）。"""
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def encode_wav_f32(samples: np.ndarray, sample_rate: int) -> bytes:
    """32-bit float WAV（导出音质无损）。"""
    data = pcm_f32le(samples)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(4)
        wav.setframerate(int(sample_rate))
        wav.writeframes(data)
    return buffer.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def block_slice(
    samples: np.ndarray,
    sample_rate: int,
    source_start_s: float,
    block_start_s: float,
    block_end_s: float,
) -> np.ndarray:
    """从整段渲染结果里裁出请求块 [block_start_s, block_end_s)（越界补零）。

    samples 的第 0 个样本对应 source_start_s。返回长度 = round((end-start)*sr) 的 float32。
    """
    total = max(0, int(round((block_end_s - block_start_s) * sample_rate)))
    out = np.zeros(total, dtype=np.float32)
    if total == 0:
        return out
    offset = int(round((block_start_s - source_start_s) * sample_rate))
    src = np.asarray(samples, dtype=np.float32).reshape(-1)
    dst_begin = max(0, -offset)
    src_begin = max(0, offset)
    count = min(total - dst_begin, src.size - src_begin)
    if count > 0:
        out[dst_begin:dst_begin + count] = src[src_begin:src_begin + count]
    return out


def rms_energy_curve(
    samples: np.ndarray,
    sample_rate: int,
    start_s: float,
    points: int = 100,
) -> list[tuple[float, float]]:
    """从音频算 0..100 的 RMS 能量包络（响应 parameters.energy 的兜底值）。"""
    src = np.asarray(samples, dtype=np.float32).reshape(-1)
    if src.size == 0 or points <= 0:
        return []
    points = min(points, max(1, src.size))
    bounds = np.linspace(0, src.size, points + 1, dtype=np.int64)
    windows = src[bounds[:-1]]
    length = max(1, src.size // points)
    values: list[float] = []
    for index in range(points):
        chunk = src[bounds[index]:bounds[index + 1]]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
        db = 20.0 * math.log10(max(rms, 1e-7))
        # -60dB..0dB 映射到 0..100
        values.append(max(0.0, min(100.0, (db + 60.0) / 60.0 * 100.0)))
    duration = src.size / float(sample_rate)
    return [
        (start_s + duration * index / max(1, points - 1), values[index])
        for index in range(points)
    ]


def db_to_linear(db: float | np.ndarray) -> float | np.ndarray:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)
