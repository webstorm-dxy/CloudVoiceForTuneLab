"""engine 层渲染管线测试：DiffSingerRenderer（方差/声学/声码器）+ speaker embedding。

覆盖：

- 合成测试声库（scripts/make_test_voicebank.py 生成的 /tmp/TestVoice）：acoustic 只有
  tokens/durations/f0 三个输入、vocoder 在 dsvocoder/ 本地 -> 验证"可选输入不存在就不传"
  以及"无 dsdur/dsvariance 声库"的路径；块裁剪/补零；全休止块。
- 真实 JiHeHong（老格式：depth int64 + speedup int64、energy/breathiness 方差、内置 nsf_hifigan）：
  单音 "a" 2s -> 非静音、长度正确、FFT 主频接近请求音高 A4=440Hz，并记录首块/热块耗时。
- 真实 ChiYu（新格式：depth float + steps int64、breathiness/voicing/tension 方差、
  mel_base=e -> 2024 声码器 mel_base=10 换算）：同时验证 dsvariance 模型损坏时的
  运行期兜底（本机 dsvariance/ChiYuAI.variance.onnx 是坏 protobuf，引擎会回退到
  dsdur/ 下的有效副本）。
- speaker.py 单测（真实 .emb 读取 + 单/多 speaker 帧级布局 + 缺失 .emb 报错）。
- 纯逻辑单测：帧舍入 / SampleCurve 头尾填充 / word_div/word_dur / 用户曲线公式。

运行（在 demo/DiffSingerCloud 下）::

    .venv/bin/python -m pytest tests/test_renderer.py -q -s

内存：一次只加载一个声库（每个测试自己建 runner、结束 close）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from server import audio  # noqa: E402
from server.engine import backends, inference, speaker  # noqa: E402
from server.engine.backends import OnnxRunnerImpl, resolve_backend  # noqa: E402
from server.engine.inference import (  # noqa: E402
    DiffSingerRenderer,
    _apply_variance_formula,
    _round_frames,
    _sample_curve_frames,
)
from server.voicebank import types  # noqa: E402

OPENUTAU = DEMO_DIR / "OpenUTAU"
_TEST_VOICE = Path("/tmp/TestVoice")
if not _TEST_VOICE.is_dir():
    _TEST_VOICE = Path("/tmp/TestVoice".replace("TestVoice", "TestVoice"))  # 占位，保持一致


# --------------------------------------------------------------------------------------
# 手工构造 Voicebank（等 voicebank 层 discovery 稳定后这些测试可直接换成 catalog.get）
# --------------------------------------------------------------------------------------
_DSCONFIG_FIELDS = (
    "phonemes", "languages", "acoustic", "vocoder", "speakers", "hidden_size",
    "use_key_shift_embed", "use_speed_embed", "use_energy_embed", "use_breathiness_embed",
    "use_voicing_embed", "use_tension_embed", "use_continuous_acceleration",
    "use_variable_depth", "use_lang_id", "use_expr", "use_note_rest",
    "dur", "linguistic", "pitch", "variance", "predict_dur", "predict_energy",
    "predict_breathiness", "predict_voicing", "predict_tension",
    "sample_rate", "hop_size", "win_size", "fft_size", "num_mel_bins",
    "mel_fmin", "mel_fmax", "mel_base", "mel_scale",
)


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_dsconfig(path: Path) -> types.DsConfig:
    raw = _read_yaml(path)
    config = types.DsConfig(raw=raw, source_path=path)
    for field in _DSCONFIG_FIELDS:
        if field in raw and raw[field] is not None:
            setattr(config, field, raw[field])
    config.max_depth = float(raw.get("max_depth", config.max_depth))
    if "use_variable_depth" not in raw and "use_shallow_diffusion" in raw:
        config.use_variable_depth = bool(raw["use_shallow_diffusion"])
    augmentation = raw.get("augmentation_args") or {}
    pitch_range = (augmentation.get("random_pitch_shifting") or {}).get("range")
    if pitch_range and len(pitch_range) == 2:
        config.pitch_shifting_range = (float(pitch_range[0]), float(pitch_range[1]))
    return config


def _load_phonemes(path: Path) -> dict[str, int]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(key): int(value) for key, value in raw.items()}
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip(): index for index, line in enumerate(lines)}


def _parse_vocoder(yaml_path: Path) -> types.VocoderConfig:
    raw = _read_yaml(yaml_path)
    config = types.VocoderConfig(raw=raw, source_path=yaml_path)
    for field in ("name", "model", "sample_rate", "hop_size", "win_size", "fft_size",
                  "num_mel_bins", "mel_fmin", "mel_fmax", "mel_base", "mel_scale",
                  "pitch_controllable"):
        if field in raw and raw[field] is not None:
            setattr(config, field, raw[field])
    config.model_path = (yaml_path.parent / str(config.model)).resolve()
    return config


def _model_group(kind: str, directory: Path) -> types.ModelGroup | None:
    config_path = directory / "dsconfig.yaml"
    if not config_path.is_file():
        return None
    config = _parse_dsconfig(config_path)
    phonemes: dict[str, int] = {}
    phoneme_path = config.resolve(config.phonemes)
    if phoneme_path is not None and Path(phoneme_path).is_file():
        phonemes = _load_phonemes(Path(phoneme_path))
    languages: dict[str, int] = {}
    if config.languages:
        language_path = config.resolve(config.languages)
        if language_path is not None and Path(language_path).is_file():
            languages = {str(k): int(v) for k, v in json.loads(Path(language_path).read_text(encoding="utf-8")).items()}
    model_key = {"dur": "dur", "variance": "variance", "pitch": "pitch"}[kind]
    return types.ModelGroup(
        kind=kind,
        dir=directory,
        config=config,
        phonemes=phonemes,
        languages=languages,
        linguistic_path=config.resolve(config.linguistic),
        model_path=config.resolve(getattr(config, model_key, None)),
        emb_dir=directory,
    )


def _resolve_vocoder(bank_dir: Path, config: types.DsConfig) -> types.VocoderConfig | None:
    # OpenUTAU 规则：声库自带 dsvocoder/vocoder.yaml 优先，其次 dsconfig 里的内建声码器名。
    local = bank_dir / "dsvocoder" / "vocoder.yaml"
    if local.is_file():
        return _parse_vocoder(local)
    name = config.vocoder
    if not name:
        return None
    for base in (OPENUTAU / "Vocoders", Path.home() / "Library/OpenUtau/Dependencies",
                 Path.home() / "Library/Application Support/TuneLab/Extensions/ChoristaDiffsinger/Vocoders"):
        candidate = base / str(name) / "vocoder.yaml"
        if candidate.is_file():
            return _parse_vocoder(candidate)
    return None


def bank_from_directory(directory: Path, *, bank_id: str | None = None, name: str | None = None,
                        selected_speaker: str = "", variance_model: Path | None = None) -> types.Voicebank:
    directory = Path(directory)
    config = _parse_dsconfig(directory / "dsconfig.yaml")
    phonemes: dict[str, int] = {}
    phoneme_path = config.resolve(config.phonemes)
    if phoneme_path is not None and Path(phoneme_path).is_file():
        phonemes = _load_phonemes(Path(phoneme_path))
    languages: dict[str, int] = {}
    if config.use_lang_id and config.languages:
        language_path = config.resolve(config.languages)
        if language_path is not None and Path(language_path).is_file():
            languages = {str(k): int(v) for k, v in json.loads(Path(language_path).read_text(encoding="utf-8")).items()}
    bank = types.Voicebank(
        id=bank_id or directory.name,
        bank_id=bank_id or directory.name,
        name=name or directory.name,
        path=directory,
        config=config,
        phonemes=phonemes,
        languages=languages,
        speakers=list(config.speakers or []),
        selected_speaker=selected_speaker or (list(config.speakers)[0] if config.speakers else ""),
        dur=_model_group("dur", directory / "dsdur"),
        variance=_model_group("variance", directory / "dsvariance"),
        pitch=_model_group("pitch", directory / "dspitch"),
        vocoder=_resolve_vocoder(directory, config),
    )
    if variance_model is not None and bank.variance is not None:
        bank.variance.model_path = Path(variance_model)
    return bank


def jihehong_bank() -> types.Voicebank:
    directory = OPENUTAU / "JiHeHong-diffsinger-openutau-V2"
    if not (directory / "dsconfig.yaml").is_file():
        pytest.skip("JiHeHong voicebank not available")
    return bank_from_directory(directory, bank_id="JiHeHong")


def chiyu_bank() -> types.Voicebank:
    directory = OPENUTAU / "ChiYu"
    if not (directory / "dsconfig.yaml").is_file():
        pytest.skip("ChiYu voicebank not available")
    return bank_from_directory(directory, bank_id="ChiYu")


# --------------------------------------------------------------------------------------
# 渲染公用工具
# --------------------------------------------------------------------------------------
def make_phrase(start_s: float, duration_s: float, pitch: int, symbols: list[str],
                *, speaker: str = "", vowel_last: bool = True) -> types.Phrase:
    start_ms = float(start_s) * 1000.0
    end_ms = (float(start_s) + float(duration_s)) * 1000.0
    phones: list[types.Phone] = []
    count = len(symbols)
    for index, symbol in enumerate(symbols):
        begin = start_ms + (end_ms - start_ms) * index / count
        end = start_ms + (end_ms - start_ms) * (index + 1) / count
        phones.append(types.Phone(
            symbol=symbol,
            start_ms=begin,
            end_ms=end,
            note_index=0,
            tone=pitch,
            vowel=bool(vowel_last and index == count - 1),
            speaker=speaker,
        ))
    notes = [types.ResolvedNote(key="n1", index=0, start_ms=start_ms, end_ms=end_ms, pitch=pitch)]
    return types.Phrase(phones=phones, notes=notes)


def render_with_new_runner(
    bank: types.Voicebank,
    phrase: types.Phrase,
    *,
    pitch_points,
    parameters=None,
    block_start_s: float,
    block_end_s: float,
    backend: str = "cpu",
    depth: float = 1.0,
    steps: int = 20,
    include_leading: bool = True,
) -> tuple[types.RenderedBlock, DiffSingerRenderer, OnnxRunnerImpl, dict]:
    info = resolve_backend(backend, threads=4)
    runner = OnnxRunnerImpl(info, max_cached_sessions=8, threads=4)
    renderer = DiffSingerRenderer(
        runner,
        types.RendererOptions(depth=depth, steps=steps, threads=4, include_leading=include_leading),
    )
    timings: dict = {}
    try:
        t0 = time.perf_counter()
        block = renderer.render(
            bank,
            phrase,
            pitch_points=list(pitch_points),
            parameters=dict(parameters or {}),
            block_start_s=float(block_start_s),
            block_end_s=float(block_end_s),
        )
        timings["render_first_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        block = renderer.render(
            bank,
            phrase,
            pitch_points=list(pitch_points),
            parameters=dict(parameters or {}),
            block_start_s=float(block_start_s),
            block_end_s=float(block_end_s),
        )
        timings["render_warm_s"] = time.perf_counter() - t0
    except Exception:
        runner.close()
        raise
    return block, renderer, runner, timings


def dominant_frequency(samples: np.ndarray, sample_rate: int, low: float = 100.0, high: float = 2000.0) -> float:
    data = np.asarray(samples, dtype=np.float64).reshape(-1)
    if data.size < 64:
        return 0.0
    data = data - float(np.mean(data))
    window = np.hanning(data.size)
    spectrum = np.abs(np.fft.rfft(data * window))
    frequencies = np.fft.rfftfreq(data.size, 1.0 / float(sample_rate))
    mask = (frequencies >= low) & (frequencies <= high)
    if not np.any(mask):
        return 0.0
    return float(frequencies[mask][int(np.argmax(spectrum[mask]))])


def rms(samples: np.ndarray) -> float:
    data = np.asarray(samples, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0


def segment(block: types.RenderedBlock, block_start_s: float, start_s: float, end_s: float) -> np.ndarray:
    sr = block.sample_rate
    begin = max(0, int(round((start_s - block_start_s) * sr)))
    end = min(block.samples.size, int(round((end_s - block_start_s) * sr)))
    return block.samples[begin:end]


# --------------------------------------------------------------------------------------
# 纯逻辑单测（不加载模型）
# --------------------------------------------------------------------------------------
def test_round_frames_matches_bankers_rounding() -> None:
    frame_ms = 1000.0 * 512 / 44100
    assert _round_frames(0.0, frame_ms) == 0
    assert _round_frames(1000.0, frame_ms) == int(np.round(1000.0 / frame_ms))
    assert _round_frames(3000.0, frame_ms) == int(np.round(3000.0 / frame_ms))
    # 帧数差与直接对端点取整一致（OpenUTAU 防累积误差写法）
    assert _round_frames(3000.0, frame_ms) - _round_frames(1000.0, frame_ms) == 172


def test_sample_curve_frames_head_tail_fill() -> None:
    points = [(0.0, 60.0), (2.0, 72.0)]
    frame_ms = 10.0
    frames = _sample_curve_frames(points, 1000.0, 3000.0, frame_ms, 7, 2, 2, 0.0)
    assert frames.shape == (7,)
    # 头 2 帧 = 曲线在 1000ms 的值 = 66，尾 2 帧 = 曲线在 3000ms 的值 = 72
    assert np.allclose(frames[:2], 66.0, atol=1e-4)
    assert np.allclose(frames[-2:], 72.0, atol=1e-4)
    # body 帧时刻 = 1000ms + i*10ms -> 1.00/1.01/1.02s，对应曲线 66/66.06/66.12
    assert np.allclose(frames[2:5], [66.0, 66.06, 66.12], atol=1e-4)


def test_sample_curve_frames_without_points_uses_default() -> None:
    frames = _sample_curve_frames(None, 0.0, 1000.0, 10.0, 5, 1, 1, 64.0)
    assert np.allclose(frames, 64.0)


def test_diffusion_inputs_jilenon_style() -> None:
    directory = OPENUTAU / "JiHeHong-diffsinger-openutau-V2"
    if not (directory / "dsconfig.yaml").is_file():
        pytest.skip("JiHeHong voicebank not available")
    bank = bank_from_directory(directory, bank_id="JiHeHong")
    renderer = DiffSingerRenderer(None)  # depth/steps 计算不需要 runner
    inputs = renderer._diffusion_inputs(bank)
    assert set(inputs) == {"depth", "speedup"}
    assert int(inputs["depth"][0]) == 300        # min(1.0, 300/1000) * 1000
    assert int(inputs["speedup"][0]) == 15       # 300 // 20
    assert inputs["depth"].dtype == np.int64


def test_diffusion_inputs_continuous_style() -> None:
    directory = OPENUTAU / "ChiYu"
    if not (directory / "dsconfig.yaml").is_file():
        pytest.skip("ChiYu voicebank not available")
    bank = bank_from_directory(directory, bank_id="ChiYu")
    renderer = DiffSingerRenderer(None, types.RendererOptions(depth=1.0, steps=20))
    inputs = renderer._diffusion_inputs(bank)
    assert set(inputs) == {"depth", "steps"}
    assert inputs["depth"].dtype == np.float32
    assert abs(float(inputs["depth"][0]) - 0.6) < 1e-6    # min(1.0, max_depth=0.6)
    assert int(inputs["steps"][0]) == 20


def test_variance_formula_matches_openutau() -> None:
    predicted = np.asarray([-2.0, -5.0, 0.0], dtype=np.float32)
    user = np.asarray([0.5, 1.0, 2.0], dtype=np.float32)
    energy = _apply_variance_formula("energy", predicted, user * 100.0)
    assert np.allclose(energy, np.minimum(predicted + user * 12.0, 0.0))
    voicing = _apply_variance_formula("voicing", predicted, 100.0 + user * 100.0)
    assert np.allclose(voicing, np.minimum(predicted + user * 12.0, 0.0))
    tension = _apply_variance_formula("tension", predicted, user * 100.0)
    assert np.allclose(tension, predicted + user * 5.0)
    velocity = _apply_variance_formula("velocity", np.zeros(3, dtype=np.float32),
                                       np.asarray([50.0, 100.0, 200.0], dtype=np.float32))
    assert np.allclose(velocity, [2.0 ** -0.5, 1.0, 2.0])


def test_word_layout_single_note() -> None:
    bank = types.Voicebank(id="x", bank_id="x", name="x", path=Path("."), config=types.DsConfig(hidden_size=8))
    phrase = make_phrase(1.0, 2.0, 69, ["a", "i"])
    renderer = DiffSingerRenderer(None)
    word_div, word_dur = renderer._word_layout(
        bank, phrase, 1000.0 * 512 / 44100, 9, 9
    )
    # 无 dsdict 时用"每个 note 的最后一个 phone 是元音"兜底：["a","i"] 属于同一个 word。
    assert int(word_div.sum()) == len(phrase.phones) + 2     # 首尾 SP
    assert int(word_div[0]) == 2 and int(word_div[-1]) == 2
    # word_dur 总和应等于总帧数（首尾 SP + 两个音素）
    total = 9 + 9 + sum(int(round(p.end_ms / (1000.0 * 512 / 44100)))
                        - int(round(p.start_ms / (1000.0 * 512 / 44100))) for p in phrase.phones)
    assert int(word_dur.sum()) == total


# --------------------------------------------------------------------------------------
# speaker embedding
# --------------------------------------------------------------------------------------
def test_speaker_embedding_single_speaker_frames() -> None:
    directory = OPENUTAU / "JiHeHong-diffsinger-openutau-V2"
    if not (directory / "dsconfig.yaml").is_file():
        pytest.skip("JiHeHong voicebank not available")
    bank = bank_from_directory(directory, bank_id="JiHeHong")
    name = bank.speakers[0]
    path = bank.emb_path(name)
    assert path is not None, "JiHeHong .emb missing"
    vector = speaker.load_embedding(path, 256)
    assert vector.shape == (256,)
    assert np.isfinite(vector).all() and np.any(vector != 0)
    phrase = make_phrase(0.0, 1.0, 60, ["a"], speaker=name)
    durations = [9, 86, 9]
    embed = speaker.embed_by_frame(bank, phrase, durations, 104, 9, 9)
    assert embed.shape == (1, 104, 256)
    assert np.allclose(embed[0, 0], vector)
    assert np.allclose(embed[0, 50], vector)
    assert np.allclose(embed[0, -1], vector)


def test_speaker_embedding_multi_speaker_layout(tmp_path: Path) -> None:
    name_a, name_b = "SpkA", "SpkB"
    hidden = 8
    np.asarray([1.0] * hidden, dtype="<f4").tofile(tmp_path / (name_a + ".emb"))
    np.asarray([2.0] * hidden, dtype="<f4").tofile(tmp_path / (name_b + ".emb"))
    config = types.DsConfig(hidden_size=hidden, speakers=[name_a, name_b], sample_rate=44100)
    bank = types.Voicebank(
        id="multi", bank_id="multi", name="multi", path=tmp_path, config=config,
        speakers=[name_a, name_b], selected_speaker=name_b,
    )
    phrase = types.Phrase(
        phones=[
            types.Phone(symbol="a", start_ms=0.0, end_ms=100.0, note_index=0, tone=60, speaker=name_a),
            types.Phone(symbol="i", start_ms=100.0, end_ms=200.0, note_index=0, tone=62, speaker=name_b),
        ],
        notes=[types.ResolvedNote(key="n", index=0, start_ms=0.0, end_ms=200.0, pitch=60)],
    )
    durations = [2, 3, 4, 2]
    embed = speaker.embed_by_frame(bank, phrase, durations, 11, 2, 2)
    assert embed.shape == (1, 11, hidden)
    assert np.allclose(embed[0, :2], 1.0)      # head 用首个 phone 的 speaker
    assert np.allclose(embed[0, 2:5], 1.0)     # phone a
    assert np.allclose(embed[0, 5:9], 2.0)     # phone b
    assert np.allclose(embed[0, 9:], 2.0)      # tail 用末个 phone 的 speaker
    by_phone = speaker.embed_by_phone(bank, phrase)
    assert by_phone.shape == (1, 2, hidden)
    assert np.allclose(by_phone[0, 0], 1.0) and np.allclose(by_phone[0, 1], 2.0)


def test_speaker_embedding_missing_file(tmp_path: Path) -> None:
    config = types.DsConfig(hidden_size=4, speakers=["Ghost"])
    bank = types.Voicebank(id="x", bank_id="x", name="x", path=tmp_path, config=config, speakers=["Ghost"])
    phrase = make_phrase(0.0, 1.0, 60, ["a"], speaker="Ghost")
    with pytest.raises(FileNotFoundError):
        speaker.embed_by_frame(bank, phrase, [2, 3, 2], 7, 2, 2)


# --------------------------------------------------------------------------------------
# 合成声库：可选输入缺失 / 无 dsdur+dsvariance / 块裁剪
# --------------------------------------------------------------------------------------
def test_synthetic_voicebank_render_without_variance() -> None:
    if not (_TEST_VOICE / "dsconfig.yaml").is_file():
        pytest.skip("synthetic test voicebank /tmp/TestVoice not available")
    bank = bank_from_directory(_TEST_VOICE, bank_id="TestVoice")
    assert bank.variance is None
    assert bank.dur is None
    phrase = make_phrase(1.0, 2.0, 69, ["a"])
    block, renderer, runner, timings = render_with_new_runner(
        bank, phrase,
        pitch_points=[(0.0, 69.0), (4.0, 69.0)],
        block_start_s=0.0, block_end_s=4.0,
    )
    try:
        sr = bank.sample_rate
        assert block.sample_rate == sr
        assert block.samples.shape == (4 * sr,)
        assert block.samples.dtype == np.float32
        assert np.isfinite(block.samples).all()
        assert rms(segment(block, 0.0, 1.5, 2.5)) > 1e-4
        peak = dominant_frequency(segment(block, 0.0, 1.5, 2.5), sr)
        assert abs(peak - 440.0) / 440.0 < 0.05, "synthetic FFT peak " + str(peak)
        # 渲染起点 = 首音素 - 100ms = 0.9s；块 [0, 0.85) 应该是补的零
        assert np.max(np.abs(segment(block, 0.0, 0.0, 0.85))) < 1e-6
        # 末尾超出渲染长度的部分补零（渲染到 3.0s + 100ms 尾 = 3.106s）
        assert np.max(np.abs(segment(block, 0.0, 3.2, 4.0))) < 1e-6
        assert block.pitch_echo and abs(block.pitch_echo[-1][1] - 69.0) < 1e-3
        assert block.parameters_echo.get("energy")
        print("\n[synthetic] first=%.3fs warm=%.3fs rms=%.4f fft=%.1fHz"
              % (timings["render_first_s"], timings["render_warm_s"],
                 rms(segment(block, 0.0, 1.5, 2.5)), peak))
    finally:
        runner.close()


def test_rest_only_phrase_returns_silence() -> None:
    if not (_TEST_VOICE / "dsconfig.yaml").is_file():
        pytest.skip("synthetic test voicebank /tmp/TestVoice not available")
    bank = bank_from_directory(_TEST_VOICE, bank_id="TestVoice")
    phrase = types.Phrase(
        phones=[],
        notes=[types.ResolvedNote(key="r1", index=0, start_ms=2000.0, end_ms=3000.0, pitch=60, rest=True)],
    )
    block, renderer, runner, timings = render_with_new_runner(
        bank, phrase, pitch_points=[(0.0, 60.0), (5.0, 60.0)], block_start_s=1.5, block_end_s=3.5,
    )
    try:
        sr = bank.sample_rate
        assert block.samples.shape == (2 * sr,)
        assert np.max(np.abs(block.samples)) == 0.0
        assert block.pitch_echo and abs(block.pitch_echo[0][1] - 60.0) < 1e-6
        assert block.parameters_echo.get("energy")
        assert all(0.0 <= value <= 100.0 for _, value in block.parameters_echo["energy"])
    finally:
        runner.close()


# --------------------------------------------------------------------------------------
# 真实 JiHeHong：老格式签名（depth int64/speedup int64 + energy/breathiness 方差）
# --------------------------------------------------------------------------------------
def test_jihehong_render_single_note() -> None:
    bank = jihehong_bank()
    phrase = make_phrase(1.0, 2.0, 69, ["a"], speaker=bank.selected_speaker)
    pitch_points = [(0.0, 69.0), (5.0, 69.0)]
    block, renderer, runner, timings = render_with_new_runner(
        bank, phrase, pitch_points=pitch_points, block_start_s=0.5, block_end_s=3.5,
    )
    try:
        sr = bank.sample_rate
        assert block.sample_rate == sr
        assert block.samples.shape == (int(round(3.0 * sr)),)
        assert np.isfinite(block.samples).all()
        middle = segment(block, 0.5, 1.7, 2.7)
        assert rms(middle) > 1e-3, "JiHeHong render is silent (rms=%s)" % rms(middle)
        peak = dominant_frequency(middle, sr)
        assert abs(peak - 440.0) / 440.0 < 0.05, "JiHeHong FFT peak %s (expected ~440Hz)" % peak
        assert block.pitch_echo and abs(block.pitch_echo[-1][1] - 69.0) < 1e-3
        assert block.parameters_echo.get("energy")
        print("\n[jihehong A4 2s] first=%.2fs warm=%.2fs rms=%.4f fft=%.1fHz (backend=%s)"
              % (timings["render_first_s"], timings["render_warm_s"], rms(middle), peak,
                 runner.backend.resolved))
    finally:
        runner.close()


# --------------------------------------------------------------------------------------
# 真实 ChiYu：新格式签名（depth float/steps int64 + breathiness/voicing/tension）
#          以及 dsvariance 损坏模型的运行期兜底
# --------------------------------------------------------------------------------------
def test_chiyu_render_and_variance_fallback() -> None:
    bank = chiyu_bank()
    if bank.variance is None:
        pytest.skip("ChiYu has no dsvariance")
    assert bank.variance is not None and bank.variance.model_path is not None
    primary = Path(bank.variance.model_path)
    if not primary.is_file():
        pytest.skip("ChiYu variance model missing")
    phrase = make_phrase(1.0, 2.0, 69, ["a"])
    block, renderer, runner, timings = render_with_new_runner(
        bank, phrase,
        pitch_points=[(0.0, 69.0), (5.0, 69.0)],
        parameters={"DYN": [(0.0, 0.0), (5.0, 0.3)], "VOI": [(0.0, 0.0), (5.0, -0.2)]},
        block_start_s=0.5, block_end_s=3.5,
    )
    try:
        sr = bank.sample_rate
        assert block.samples.shape == (int(round(3.0 * sr)),)
        middle = segment(block, 0.5, 1.7, 2.7)
        assert rms(middle) > 1e-3, "ChiYu render is silent (rms=%s)" % rms(middle)
        peak = dominant_frequency(middle, sr)
        assert abs(peak - 440.0) / 440.0 < 0.05, "ChiYu FFT peak %s (expected ~440Hz)" % peak
        used = [str(path) for path in renderer._model_path_cache.values()]
        assert any(
            "dsdur" in path and "variance" in Path(path).name for path in used
        ), "expected runtime fallback to a loadable ChiYu variance model in dsdur/, used: " + repr(used)
        print("\n[chiyu A4 2s] first=%.2fs warm=%.2fs rms=%.4f fft=%.1fHz fallback=%s"
              % (timings["render_first_s"], timings["render_warm_s"], rms(middle), peak,
                 [Path(p).name for p in used]))
    finally:
        runner.close()


# --------------------------------------------------------------------------------------
# 多 speaker / use_lang_id 声库（yousa）：只验证 speaker 布局与 tokenize，不做完整渲染
# --------------------------------------------------------------------------------------
def test_yousa_speaker_embedding_and_language() -> None:
    directory = OPENUTAU / "yousaV1.51_rope"
    if not (directory / "dsconfig.yaml").is_file():
        pytest.skip("yousa voicebank not available")
    bank = bank_from_directory(directory, bank_id="yousa", selected_speaker="Cute")
    if not bank.speakers:
        pytest.skip("yousa dsconfig has no speakers")
    phrase = make_phrase(0.0, 1.0, 60, ["zh/a"], speaker="Cute")
    durations = [9, 86, 9]
    embed = speaker.embed_by_frame(bank, phrase, durations, 104, 9, 9)
    assert embed.shape == (1, 104, int(bank.config.hidden_size))
    cute_path = bank.emb_path("Cute")
    assert cute_path is not None
    cute = speaker.load_embedding(cute_path, int(bank.config.hidden_size))
    assert np.allclose(embed[0, 10], cute)
    # languages.json / tokenize 只验证映射函数（不加载 300MB 声学模型）
    renderer = DiffSingerRenderer(None)
    languages = renderer._language_ids(bank, phrase)
    assert languages.shape == (3,)
    assert int(languages[0]) == 0 and int(languages[-1]) == 0
    assert "zh/a" in bank.phonemes


# --------------------------------------------------------------------------------------
# audioStart 前置（include_leading）：首音节辅音在首音符之前时不要把辅音裁掉
# --------------------------------------------------------------------------------------
def test_jihehong_include_leading_window() -> None:
    bank = jihehong_bank()
    speaker = bank.selected_speaker
    # "la"：leading 辅音 l 在 note 头之前（junction-anchored 语义），a 从 94ms 起铺。
    phones = [
        types.Phone(symbol="l", start_ms=-50.0, end_ms=94.0, note_index=0, tone=69,
                    vowel=False, speaker=speaker),
        types.Phone(symbol="a", start_ms=94.0, end_ms=1500.0, note_index=0, tone=69,
                    vowel=True, speaker=speaker),
    ]
    notes = [types.ResolvedNote(key="n1", index=0, start_ms=0.0, end_ms=1500.0, pitch=69)]
    phrase = types.Phrase(phones=phones, notes=notes)
    pitch = [(0.0, 69.0), (2.0, 69.0)]
    info = resolve_backend("cpu", threads=4)
    runner = OnnxRunnerImpl(info, max_cached_sessions=8, threads=4)
    renderer = DiffSingerRenderer(runner, types.RendererOptions(threads=4, include_leading=True))
    try:
        block = renderer.render(
            bank, phrase, pitch_points=pitch, parameters={}, block_start_s=0.0, block_end_s=1.5
        )
        sr = bank.sample_rate
        # 渲染窗口起点 = 首音素 start - HEAD_MS = -0.05 - 0.1 = -0.15s
        assert block.audio_start_s == pytest.approx(-0.15, abs=1e-6)
        assert block.samples.size == int(round((1.5 - block.audio_start_s) * sr))
        head = block.samples[: int(round(0.4 * sr))]
        assert np.abs(head).max() > 1e-3, "include_leading 窗口开头应包含辅音音频（非静音）"

        # 同一个 runner 上切回 include_leading=False：保持旧行为（正好覆盖请求块）。
        renderer.options = types.RendererOptions(threads=4, include_leading=False)
        block2 = renderer.render(
            bank, phrase, pitch_points=pitch, parameters={}, block_start_s=0.0, block_end_s=1.5
        )
        assert block2.audio_start_s == 0.0
        assert block2.samples.size == int(round(1.5 * sr))
        # 注意：DiffSinger 声学模型内部含随机扩散噪声（ONNX 里带 random op），同一请求两次
        # 渲染的波形并不逐样本一致，所以这里只校验窗口/长度语义，不做样本级 allclose。
        offset = int(round((0.0 - block.audio_start_s) * sr))
        assert block.samples[offset:].size == block2.samples.size
        assert rms(block.samples) > 1e-4 and rms(block2.samples) > 1e-4
        print("\n[jihehong la include_leading] audio_start=%.3fs len=%d head_peak=%.3f"
              % (block.audio_start_s, block.samples.size, float(np.abs(head).max())))
    finally:
        runner.close()
