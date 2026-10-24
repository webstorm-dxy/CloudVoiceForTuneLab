"""自动音高（dspitch）测试：预测器 + 渲染器 pitch_mode 集成。

覆盖：

- 纯逻辑：note_midi / note_dur / note_rest 的 OpenUTAU 布局算法；
- 合成测试声库（无 dspitch）：auto 模式 warning + 回退请求曲线，pitch_echo 与请求曲线一致；
  tunelab 模式回显 == 请求曲线；
- pitch_mode 别名解析（auto/predict/dspitch -> auto；tunelab/manual -> tunelab）；
- 真实模型（yousa dspitch / JiHeHong 无 dspitch）：默认 skip，设置
  TUNELAB_TEST_REAL_MODELS=1（或 true/yes/on）后运行：
    * yousa Bright 单音 zh/a A4：预测曲线中位数≈69、std>0.01、帧数正确、耗时；
    * 用 auto 渲染的音频自相关基频中位数与预测中位数偏差 < 3 半音；
    * auto 的 pitch_echo 中位数≈预测中位数（合成音高线与音频一致）；
    * JiHeHong auto：无 dspitch -> 回退请求曲线、无异常。

运行（在 demo/DiffSingerCloud 下）::

    .venv/bin/python -m pytest tests/test_pitch.py -q
    TUNELAB_TEST_REAL_MODELS=1 .venv/bin/python -m pytest tests/test_pitch.py -q -s
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from server import audio  # noqa: E402
from server.engine.backends import OnnxRunnerImpl, resolve_backend  # noqa: E402
from server.engine.inference import DiffSingerRenderer  # noqa: E402
from server.engine.pitch import (  # noqa: E402
    DiffSingerPitchPredictor,
    _note_durations,
    _note_midi_layout,
    _note_rest_flags,
)
from server.voicebank import types  # noqa: E402

REAL_MODELS = os.environ.get("TUNELAB_TEST_REAL_MODELS", "").strip().lower() in ("1", "true", "yes", "on")
needs_real = pytest.mark.skipif(
    not REAL_MODELS,
    reason="real-model test; set TUNELAB_TEST_REAL_MODELS=1 to run",
)

OPENUTAU = DEMO_DIR / "OpenUTAU"
SYNTHETIC_DIR = Path("/tmp/TestVoice")
CPU_BACKEND = resolve_backend("cpu", threads=4)


def make_phrase(start_s: float, duration_s: float, pitch: int, symbols: list[str],
                *, speaker: str = "") -> types.Phrase:
    """构造单 note phrase（与 tests/test_renderer.py 同款，避免跨文件 import）。"""
    start_ms = float(start_s) * 1000.0
    end_ms = (float(start_s) + float(duration_s)) * 1000.0
    phones: list[types.Phone] = []
    count = len(symbols)
    for index, symbol in enumerate(symbols):
        begin = start_ms + (end_ms - start_ms) * index / count
        end = start_ms + (end_ms - start_ms) * (index + 1) / count
        phones.append(types.Phone(
            symbol=symbol, start_ms=begin, end_ms=end, note_index=0, tone=pitch,
            vowel=index == count - 1, speaker=speaker,
        ))
    notes = [types.ResolvedNote(key="n1", index=0, start_ms=start_ms, end_ms=end_ms, pitch=pitch)]
    return types.Phrase(phones=phones, notes=notes)


def new_runner() -> OnnxRunnerImpl:
    return OnnxRunnerImpl(CPU_BACKEND, max_cached_sessions=8, threads=4)


def synthetic_bank() -> types.Voicebank:
    if not (SYNTHETIC_DIR / "dsconfig.yaml").is_file():
        pytest.skip("synthetic test voicebank /tmp/TestVoice not available")
    from tests.test_renderer import bank_from_directory
    return bank_from_directory(SYNTHETIC_DIR, bank_id="TestVoice")


def catalog_voice(voice_id: str) -> types.Voicebank:
    from server.voicebank.discovery import VoiceCatalog
    catalog = VoiceCatalog([OPENUTAU], [OPENUTAU / "Vocoders"])
    catalog.refresh()
    try:
        return catalog.get(voice_id)
    except KeyError:
        pytest.skip("voicebank " + voice_id + " not available")


def autocorr_f0(samples: np.ndarray, sample_rate: int, fmin: float = 80.0, fmax: float = 1200.0) -> float:
    """自相关基频估计（Hz）；失败返回 0。"""
    data = np.asarray(samples, dtype=np.float64).reshape(-1)
    if data.size < 512:
        return 0.0
    data = data - float(np.mean(data))
    data = data * np.hanning(data.size)
    corr = np.correlate(data, data, mode="full")[data.size - 1:]
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = min(data.size - 1, int(sample_rate / fmin))
    if max_lag <= min_lag:
        return 0.0
    lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    return float(sample_rate) / float(lag)


class _FakeDictionary:
    """最小 SymbolSet 替身：只有元音分类/合法性查询。"""

    def __init__(self, vowels: set[str]) -> None:
        self.vowels = vowels

    def is_vowel(self, symbol: str) -> bool:
        return symbol in self.vowels

    def is_valid(self, symbol: str) -> bool:
        return True

    def query(self, grapheme: str):  # pragma: no cover - 本测试不用
        return None

    def symbol_type(self, symbol: str) -> str:
        return "vowel" if symbol in self.vowels else "consonant"


# --------------------------------------------------------------------------------------
# 纯逻辑：OpenUTAU note_midi / note_dur / note_rest 布局
# --------------------------------------------------------------------------------------
def test_note_midi_layout_leading_rest_group() -> None:
    notes = [
        types.ResolvedNote(key="a", index=0, start_ms=0.0, end_ms=1000.0, pitch=60, rest=True),
        types.ResolvedNote(key="b", index=1, start_ms=1000.0, end_ms=2000.0, pitch=72),
    ]
    flags = [True, True, False]
    midi = _note_midi_layout(notes, flags)
    # 前置 [notes[0].tone] + 两个 note tone -> [60, 60, 72]；首组 rest 用右侧 tone 填充
    assert np.allclose(midi, [72.0, 72.0, 72.0])


def test_note_midi_layout_trailing_rest_group() -> None:
    notes = [
        types.ResolvedNote(key="a", index=0, start_ms=0.0, end_ms=1000.0, pitch=60),
        types.ResolvedNote(key="b", index=1, start_ms=2000.0, end_ms=3000.0, pitch=72, rest=True),
    ]
    flags = [True, False, True]
    midi = _note_midi_layout(notes, flags)
    assert np.allclose(midi, [60.0, 60.0, 60.0])


def test_note_midi_layout_all_rest_keeps_tones() -> None:
    notes = [types.ResolvedNote(key="a", index=0, start_ms=0.0, end_ms=1000.0, pitch=60)]
    midi = _note_midi_layout(notes, [True, True])
    assert np.allclose(midi, [60.0, 60.0])


def test_note_durations_sum_to_total() -> None:
    frame_ms = 1000.0 * 512 / 44100
    notes = [
        types.ResolvedNote(key="a", index=0, start_ms=1000.0, end_ms=2000.0, pitch=69),
        types.ResolvedNote(key="b", index=1, start_ms=2000.0, end_ms=3000.0, pitch=69),
    ]
    phones = [types.Phone(symbol="a", start_ms=1000.0, end_ms=2000.0, note_index=0, tone=69)]
    total_frames = 190
    durations = _note_durations(notes, phones, frame_ms, total_frames)
    assert durations.size == len(notes) + 1
    assert int(durations.sum()) == total_frames


def test_note_rest_flags_with_fake_dictionary() -> None:
    dictionary = _FakeDictionary({"a", "i", "zh/a"})
    bank = types.Voicebank(
        id="fake", bank_id="fake", name="fake", path=Path("."),
        config=types.DsConfig(hidden_size=8), dictionary=dictionary,
    )
    # note0 有元音 -> 非 rest；note1 只有辅音 -> rest
    phrase = types.Phrase(
        phones=[
            types.Phone(symbol="zh/a", start_ms=0.0, end_ms=500.0, note_index=0, tone=60),
            types.Phone(symbol="zh/b", start_ms=500.0, end_ms=1000.0, note_index=1, tone=62),
        ],
        notes=[
            types.ResolvedNote(key="n0", index=0, start_ms=0.0, end_ms=500.0, pitch=60),
            types.ResolvedNote(key="n1", index=1, start_ms=500.0, end_ms=1000.0, pitch=62),
        ],
    )
    flags = _note_rest_flags(bank, phrase)
    assert flags == [True, False, True]

    # 延音 note（lyric 以 + 开头）沿用前一个 note 的 rest 状态
    slur = types.Phrase(
        phones=[types.Phone(symbol="zh/a", start_ms=0.0, end_ms=1000.0, note_index=0, tone=60)],
        notes=[
            types.ResolvedNote(key="n0", index=0, start_ms=0.0, end_ms=1000.0, pitch=60, lyric="a"),
            types.ResolvedNote(key="s", index=1, start_ms=1000.0, end_ms=2000.0, pitch=60, lyric="+"),
        ],
    )
    assert _note_rest_flags(bank, slur) == [True, False, False]

    # 没有字典时保守用 note.rest 标记
    bare = types.Voicebank(
        id="bare", bank_id="bare", name="bare", path=Path("."),
        config=types.DsConfig(hidden_size=8),
    )
    rest_phrase = types.Phrase(
        phones=[],
        notes=[types.ResolvedNote(key="r", index=0, start_ms=0.0, end_ms=1000.0, pitch=60, rest=True)],
    )
    assert _note_rest_flags(bare, rest_phrase) == [True, True]


# --------------------------------------------------------------------------------------
# 合成声库（无 dspitch）：auto 回退 + tunelab 回显
# --------------------------------------------------------------------------------------
def test_pitch_mode_aliases() -> None:
    assert DiffSingerRenderer._resolve_pitch_mode("auto") == "auto"
    assert DiffSingerRenderer._resolve_pitch_mode("predict") == "auto"
    assert DiffSingerRenderer._resolve_pitch_mode("dspitch") == "auto"
    assert DiffSingerRenderer._resolve_pitch_mode("tunelab") == "tunelab"
    assert DiffSingerRenderer._resolve_pitch_mode("manual") == "tunelab"
    assert DiffSingerRenderer._resolve_pitch_mode(None) == "auto"
    assert DiffSingerRenderer._resolve_pitch_mode("bogus") == "auto"


def test_predictor_unavailable_without_dspitch() -> None:
    bank = synthetic_bank()
    predictor = DiffSingerPitchPredictor(None)  # type: ignore[arg-type] - available() 不需要 runner
    assert predictor.available(bank) is False
    with pytest.raises(RuntimeError):
        predictor.predict(bank, make_phrase(0.0, 1.0, 69, ["a"]))


def test_auto_mode_falls_back_to_request_curve(caplog) -> None:
    bank = synthetic_bank()
    phrase = make_phrase(1.0, 2.0, 69, ["a"])
    request_pitch = [(0.0, 69.0), (4.0, 69.0)]
    runner = new_runner()
    try:
        renderer = DiffSingerRenderer(runner, types.RendererOptions(threads=4, pitch_mode="auto"))
        with caplog.at_level(logging.WARNING, logger="server.engine.inference"):
            block = renderer.render(
                bank, phrase, pitch_points=request_pitch, parameters={},
                block_start_s=0.0, block_end_s=3.0,
            )
        messages = " ".join(record.getMessage() for record in caplog.records).lower()
        assert "dspitch" in messages and "fall" in messages, messages
        assert block.samples.size == 3 * bank.sample_rate
        times = np.asarray([point[0] for point in block.pitch_echo], dtype=np.float64)
        values = np.asarray([point[1] for point in block.pitch_echo], dtype=np.float64)
        assert times.size > 0
        expected = audio.sample_points(request_pitch, times, 69.0)
        assert np.allclose(values, expected, atol=1e-3)
    finally:
        runner.close()


def test_tunelab_mode_echo_matches_request() -> None:
    bank = synthetic_bank()
    phrase = make_phrase(1.0, 2.0, 69, ["a"])
    request_pitch = [(0.0, 67.0), (1.5, 69.0), (3.0, 72.0)]
    runner = new_runner()
    try:
        renderer = DiffSingerRenderer(runner, types.RendererOptions(threads=4, pitch_mode="tunelab"))
        # 不显式传 pitch_mode：用 RendererOptions.pitch_mode="tunelab"
        block = renderer.render(
            bank, phrase, pitch_points=request_pitch, parameters={},
            block_start_s=0.0, block_end_s=3.0,
        )
        times = np.asarray([point[0] for point in block.pitch_echo], dtype=np.float64)
        values = np.asarray([point[1] for point in block.pitch_echo], dtype=np.float64)
        expected = audio.sample_points(request_pitch, times, 69.0)
        assert np.allclose(values, expected, atol=1e-3)
        # 显式 pitch_mode 覆盖 options 里的值也生效
        block2 = renderer.render(
            bank, phrase, pitch_points=request_pitch, parameters={},
            block_start_s=0.0, block_end_s=3.0, pitch_mode="tunelab",
        )
        assert np.allclose(
            [point[1] for point in block2.pitch_echo], expected, atol=1e-3
        )
    finally:
        runner.close()


# --------------------------------------------------------------------------------------
# 真实模型：yousa dspitch 预测 + 渲染音频基频对照；JiHeHong 无 dspitch 回退
# --------------------------------------------------------------------------------------
@needs_real
def test_yousa_dspitch_prediction_stats() -> None:
    vb = catalog_voice("yousaV1.51_rope::Bright")
    assert vb.pitch is not None and vb.pitch.available
    phrase = make_phrase(1.0, 2.0, 69, ["zh/a"], speaker="Bright")
    runner = new_runner()
    try:
        predictor = DiffSingerPitchPredictor(runner)
        assert predictor.available(vb)
        t0 = time.perf_counter()
        predicted = predictor.predict(vb, phrase, frame_ms=vb.frame_ms, steps=20)
        elapsed = time.perf_counter() - t0
        assert predicted.size == 190, "unexpected dspitch frame count " + str(predicted.size)
        assert np.isfinite(predicted).all()
        median = float(np.median(predicted))
        std = float(np.std(predicted))
        assert abs(median - 69.0) < 2.0, "predicted median " + str(median)
        assert std > 0.01, "predicted curve is constant (std=" + str(std) + ")"
        print("\n[yousa dspitch] median=%.3f std=%.3f min=%.2f max=%.2f frames=%d elapsed=%.2fs"
              % (median, std, float(predicted.min()), float(predicted.max()),
                 predicted.size, elapsed))
    finally:
        runner.close()


@needs_real
def test_yousa_auto_render_audio_matches_prediction() -> None:
    vb = catalog_voice("yousaV1.51_rope::Bright")
    phrase = make_phrase(1.0, 2.0, 69, ["zh/a"], speaker="Bright")
    request_pitch = [(0.0, 69.0), (4.0, 69.0)]
    runner = new_runner()
    try:
        predictor = DiffSingerPitchPredictor(runner)
        predicted = predictor.predict(vb, phrase, frame_ms=vb.frame_ms, steps=20)
        renderer = DiffSingerRenderer(runner, types.RendererOptions(threads=4, pitch_mode="auto"))
        t0 = time.perf_counter()
        block = renderer.render(
            vb, phrase, pitch_points=request_pitch, parameters={},
            block_start_s=0.5, block_end_s=3.5, pitch_mode="auto",
        )
        render_s = time.perf_counter() - t0
        sr = vb.sample_rate
        assert block.samples.size == 3 * sr
        assert float(np.sqrt(np.mean(np.square(block.samples)))) > 1e-3

        # 预测帧 k 的全局时刻 = 首音素 - HEAD_MS + k*frame_ms；取音频 1.7~2.7s 对照。
        frame_ms = vb.frame_ms
        frame_times = (1000.0 - 100.0) / 1000.0 + np.arange(predicted.size) * frame_ms / 1000.0
        window = (frame_times >= 1.7) & (frame_times < 2.7)
        predicted_midi = float(np.median(predicted[window]))
        segment = block.samples[int(round((1.7 - 0.5) * sr)):int(round((2.7 - 0.5) * sr))]
        f0_hz = autocorr_f0(segment, sr)
        assert f0_hz > 0
        f0_midi = float(audio.hz_to_semitone(np.asarray([f0_hz]))[0])
        diff = abs(f0_midi - predicted_midi)
        assert diff < 3.0, (
            "rendered audio F0 %.1fHz (%.2f) vs predicted %.2f, diff %.2f semitones"
            % (f0_hz, f0_midi, predicted_midi, diff)
        )
        echo_values = np.asarray([point[1] for point in block.pitch_echo], dtype=np.float64)
        assert abs(float(np.median(echo_values)) - predicted_midi) < 2.0
        assert float(np.std(echo_values)) > 0.01
        print("\n[yousa auto render] predicted_median=%.2f audio_f0=%.1fHz(%.2f) diff=%.2f "
              "echo_median=%.2f render=%.2fs"
              % (predicted_midi, f0_hz, f0_midi, diff, float(np.median(echo_values)), render_s))
    finally:
        runner.close()


@needs_real
def test_jihehong_auto_falls_back_without_dspitch(caplog) -> None:
    vb = catalog_voice("JiHeHong-diffsinger-openutau-V2")
    assert vb.pitch is None or not vb.pitch.available
    phrase = make_phrase(1.0, 2.0, 69, ["a"], speaker=vb.selected_speaker)
    request_pitch = [(0.0, 69.0), (4.0, 69.0)]
    runner = new_runner()
    try:
        renderer = DiffSingerRenderer(runner, types.RendererOptions(threads=4, pitch_mode="auto"))
        with caplog.at_level(logging.WARNING, logger="server.engine.inference"):
            block = renderer.render(
                vb, phrase, pitch_points=request_pitch, parameters={},
                block_start_s=0.5, block_end_s=3.5, pitch_mode="auto",
            )
        messages = " ".join(record.getMessage() for record in caplog.records).lower()
        assert "dspitch" in messages, messages
        assert block.samples.size == 3 * vb.sample_rate
        assert float(np.sqrt(np.mean(np.square(block.samples)))) > 1e-3
        echo_values = np.asarray([point[1] for point in block.pitch_echo], dtype=np.float64)
        assert np.allclose(echo_values, 69.0, atol=1e-3)
    finally:
        runner.close()
