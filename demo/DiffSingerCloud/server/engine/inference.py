"""DiffSinger 渲染管线：方差预测 -> 声学 mel -> 声码器 waveform，并按请求块裁剪。

实现目标是与 OpenUTAU 的 DiffSinger 渲染数值行为一致
（参考 /Users/daxingyi/RiderProjects/OpenUtau/OpenUtau.Core/DiffSinger/）：

- DiffSingerRenderer.InvokeDiffsinger：
  tokens = SP + phones + SP；durations 是每个 token 的帧数，首尾各 round(headMs/frameMs)、
  round(tailMs/frameMs) 帧（headMs = tailMs = 100，见 types.HEAD_MS/TAIL_MS）；
  每帧时刻 = 首个音素起点 + i * frameMs，头/尾帧用曲线端点值填充（DiffSingerUtils.SampleCurve）；
  f0 = 半音 -> Hz；gender/velocity 按 dsconfig 的 use_key_shift_embed/use_speed_embed；
  depth/steps/speedup 按 use_continuous_acceleration / use_variable_depth 计算；
  mel 是声学模型第一个输出（本仓库模型叫 "mel"），按 mel_base 换算后连同 f0 送声码器。
- DiffSingerVariance.Process：linguistic 编码器（predict_dur 时 word_div/word_dur，
  否则 ph_dur；use_lang_id 时 languages）-> encoder_out；方差模型输入
  encoder_out/ph_dur/pitch/[energy]/[breathiness]/[voicing]/[tension]/retake/steps|speedup/[spk_embed]；
  预测曲线 resample 到声学帧数（DiffSingerUtils.ResampleCurve）。
- 用户可画线参数的 OpenUTAU 公式（schema 轨 id 见 server/schema.py）：
    DYN -> energy       u = DYN*100            energy_in = min(pred + u*0.12, 0)
    BRE -> breathiness  u = BRE*100            breath_in = min(pred + u*0.12, 0)
    TEN -> tension      u = TEN*100            tension_in = pred + u*0.05
    VOI -> voicing      u = 100 + VOI*100      voicing_in = min(pred + (u-100)*0.12, 0)
    VEL -> velocity     u = VEL (0..200)       velocity_in = 2 ** ((u-100)/100)
  （DYN/BRE/TEN/VOI 的 schema 坐标是 [-1,1] 默认 0；VEL 是 [0,200] 默认 100。）
- 自动音高（dspitch，见 engine/pitch.py）：RendererOptions.pitch_mode / render(pitch_mode=...)。
  "auto"（默认）= 声库 dspitch 预测的半音 + 请求曲线相对 note tone 的偏差（保留用户画线）；
  无 dspitch 或预测失败 -> warning + 回退请求曲线。"tunelab"/"manual" = 请求曲线直接当绝对 F0。
  auto 模式下 pitch_echo 回显最终使用的 F0（预测+偏差），让 TuneLab 合成音高线与音频一致。
  variance 阶段仍用请求 pitch（与 OpenUTAU DsVariance 语义一致）。
- speaker embedding 见 engine/speaker.py。
- audioStart 前置（RendererOptions.include_leading，默认 True）：DiffSinger 首音节的 leading
  辅音在首音符之前（TuneLab 的音素契约是 junction-anchored，辅音向左铺）。因此输出窗口起点取
  min(block_start, 首个 Phone.start_ms/1000 - HEAD_MS/1000)，RenderedBlock.audio_start_s 返回
  该起点，samples 覆盖 [audio_start_s, block_end)；include_leading=False 时保持旧行为
  （正好覆盖请求块，audio_start_s = block_start）。无音素/全休止块仍从 block_start 起给零音频。

模型可加载性兜底（运行期、按需，不在启动时扫描 ONNX）
------------------------------------------------------
真实声库目录可能损坏（例如本机 ChiYu 的 dsvariance/ChiYuAI.variance.onnx 是坏的
protobuf，而 dsdur/ChiYuAI.variance.onnx 是好的）。因此 variance / linguistic 模型第一次
建 session 失败时，会按下列顺序找替代文件（见 _fallback_candidates）：

1. 同一模型目录里的 *variance*.onnx / *linguistic*.onnx；
2. dsvariance -> dsdur -> dspitch 目录里的同名匹配文件（variance 先试 dsdur）；
3. 全部失败时抛错，错误信息包含声库 id 与所有尝试过的路径。

成功兜底时 LOG.warning 记录实际使用的路径。声学模型与声码器不做跨目录兜底（换模型会改变
音色/采样率，必须用 dsconfig 指定的那个）；找不到/打不开时错误信息同样包含声库 id。

声学模型需要的方差参数如果声库没有可用的 dsvariance（例如合成测试声库只有
tokens/durations/f0 三个输入），只有模型真正声明了该输入时才会报错；模型没有该输入时
直接跳过方差阶段（可选输入不存在就不传）。

内存约定：一次只渲染一个声库（由上层 max_concurrent 控制）；本模块不持有 session 所有权，
session 池由注入的 types.OnnxRunner 管理。
"""
from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .. import audio
from ..voicebank import types
from . import speaker
from .pitch import DiffSingerPitchPredictor

LOG = logging.getLogger(__name__)

HEAD_MS = float(types.HEAD_MS)
TAIL_MS = float(types.TAIL_MS)

#: 声学模型可能需要的方差参数（OpenUTAU 的 4 条方差曲线）。
_VARIANCE_PARAMS = ("energy", "breathiness", "voicing", "tension")

#: 协议 schema 自动化轨 id（及 OpenUTAU 缩略名/全名）-> 内部参数名。
_PARAM_ALIASES: dict[str, str] = {
    "DYN": "energy",
    "dyn": "energy",
    "energy": "energy",
    "ene": "energy",
    "BRE": "breathiness",
    "bre": "breathiness",
    "breathiness": "breathiness",
    "brec": "breathiness",
    "TEN": "tension",
    "ten": "tension",
    "tension": "tension",
    "tenc": "tension",
    "VOI": "voicing",
    "voi": "voicing",
    "voicing": "voicing",
    "voic": "voicing",
    "VEL": "velocity",
    "vel": "velocity",
    "velocity": "velocity",
    "velc": "velocity",
}

#: schema 坐标缺省值（DYN/BRE/TEN/VOI = 0，VEL = 100）。
_PARAM_DEFAULTS = {
    "energy": 0.0,
    "breathiness": 0.0,
    "tension": 0.0,
    "voicing": 0.0,
    "velocity": 100.0,
}

#: 方差曲线 -> 模型输入名（名字一致，保留映射表便于调试）。
_VARIANCE_INPUTS = {name: name for name in _VARIANCE_PARAMS}


# --------------------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------------------
def _clean_points(points: Sequence[tuple[float, float]] | None) -> list[tuple[float, float]] | None:
    """过滤 NaN/Inf 点（协议承诺不会出现，但稳妥处理）；全空返回 None。"""
    if not points:
        return None
    cleaned: list[tuple[float, float]] = []
    for point in points:
        try:
            t = float(point[0])
            v = float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(t) and math.isfinite(v):
            cleaned.append((t, v))
    return cleaned or None


def _round_frames(ms: float, frame_ms: float) -> int:
    """毫秒 -> 帧号。用 np.round（银行家舍入），与 OpenUTAU C# Math.Round(double) 一致。"""
    return int(np.round(float(ms) / float(frame_ms)))


def _uses_variable_depth(config: types.DsConfig) -> bool:
    """use_variable_depth；兼容旧声库只写 use_shallow_diffusion 的情况（OpenUTAU 语义）。"""
    if config.use_variable_depth:
        return True
    raw = config.raw or {}
    value = raw.get("use_shallow_diffusion")
    if value is None:
        # 部分解析器可能保留驼峰 key
        value = raw.get("useShallowDiffusion")
    return bool(value)


def _speedup_for_steps(steps: int) -> int:
    """OpenUTAU 的 speedup：<= 1000/steps 且能整除 1000 的最大整数。"""
    speedup = max(1, 1000 // int(steps))
    while speedup > 1 and 1000 % speedup != 0:
        speedup -= 1
    return speedup


def _phoneme_language(symbol: str) -> str:
    if "/" in symbol:
        return symbol.split("/", 1)[0]
    return ""


def _param_value_to_unit(name: str, values: np.ndarray) -> np.ndarray:
    """schema 坐标 -> OpenUTAU 曲线单位（DYN/BRE/TEN/VOI: [-100,100]，VEL 原样）。"""
    if name == "voicing":
        return 100.0 + values * 100.0
    if name == "velocity":
        return values
    # energy / breathiness / tension
    return values * 100.0


def _apply_variance_formula(name: str, predicted: np.ndarray, user_unit: np.ndarray) -> np.ndarray:
    """OpenUTAU DiffSingerRenderer 里的用户曲线叠加公式（见模块 docstring）。"""
    if name == "energy":
        return np.minimum(predicted + user_unit * 0.12, 0.0)
    if name == "breathiness":
        return np.minimum(predicted + user_unit * 0.12, 0.0)
    if name == "tension":
        return predicted + user_unit * 0.05
    if name == "voicing":
        return np.minimum(predicted + (user_unit - 100.0) * 0.12, 0.0)
    if name == "velocity":
        return np.power(2.0, (user_unit - 100.0) / 100.0)
    raise ValueError("unknown variance parameter " + repr(name))


def _sample_curve_frames(
    points: Sequence[tuple[float, float]] | None,
    first_ms: float,
    last_ms: float,
    frame_ms: float,
    total_frames: int,
    head_frames: int,
    tail_frames: int,
    default: float,
) -> np.ndarray:
    """按 OpenUTAU DiffSingerUtils.SampleCurve 语义把曲线采样到 total_frames 帧。

    - 帧 i（含 head 填充）的时刻: 首音素起点 - leading + i*frameMs；DiffSinger 的
      phoneme preutter/leading 恒为 0，因此 body 第 i 帧 = first_ms + i*frameMs；
    - 前 head_frames 帧填充曲线在 first_ms 的值，后 tail_frames 帧填充 last_ms 的值
      （等价于 OpenUTAU 的 curve[0] / curve[^1]）。
    """
    total = max(0, int(total_frames))
    result = np.full(total, float(default), dtype=np.float32)
    if total == 0:
        return result
    head = min(total, max(0, int(head_frames)))
    tail = min(total - head, max(0, int(tail_frames)))
    body = total - head - tail
    # 注意：points 的 t 是全局秒，first_ms/last_ms/frame_ms 是毫秒，内部统一换算成秒。
    if body > 0:
        times = (float(first_ms) + np.arange(body, dtype=np.float64) * float(frame_ms)) / 1000.0
        result[head:head + body] = audio.sample_points(points, times, default)
    if head > 0:
        head_value = float(audio.sample_points(points, np.asarray([float(first_ms) / 1000.0]), default)[0])
        result[:head] = head_value
    if tail > 0:
        tail_value = float(audio.sample_points(points, np.asarray([float(last_ms) / 1000.0]), default)[0])
        result[total - tail:] = tail_value
    return result


def _first_output(outputs: Mapping[str, np.ndarray], preferred: str | None = None) -> np.ndarray:
    if preferred is not None and preferred in outputs:
        return np.asarray(outputs[preferred])
    if not outputs:
        raise RuntimeError("ONNX model returned no outputs")
    return np.asarray(next(iter(outputs.values())))


# --------------------------------------------------------------------------------------
# 渲染器
# --------------------------------------------------------------------------------------
class DiffSingerRenderer:
    """DiffSinger 渲染器（方差 -> 声学 -> 声码器）。

    参数：
        runner:  types.OnnxRunner（backend 层的会话池；声库层与渲染层可共用同一个实例）。
        options: types.RendererOptions（depth / steps / max_cached_sessions / threads）。

    线程安全：render() 本身可并发调用（session 池线程安全；speaker embedding 有 lru_cache），
    但同一时刻加载多个声库的 acoustic 模型会占用大量内存，建议上层 max_concurrent=1。
    """

    def __init__(self, runner: types.OnnxRunner, options: types.RendererOptions | None = None) -> None:
        self.runner = runner
        self.options = options or types.RendererOptions()
        self._lock = threading.RLock()
        self._model_path_cache: dict[tuple[str, str, str], Path] = {}
        self._pitch_predictor = DiffSingerPitchPredictor(runner)
        self._pitch_warned: set[str] = set()

    # ------------------------------------------------------------------ 公开入口
    def render(
        self,
        vb: types.Voicebank,
        phrase: types.Phrase,
        *,
        pitch_points: Sequence[tuple[float, float]],
        parameters: Mapping[str, Sequence[tuple[float, float]]],
        block_start_s: float,
        block_end_s: float,
        pitch_mode: str | None = None,
    ) -> types.RenderedBlock:
        """渲染 phrase 并返回覆盖 [audio_start_s, block_end_s) 的样本。

        pitch_points: (全局秒, MIDI 半音) 折线（TuneLab 100Hz 曲线，已含 vibrato）；
        parameters:   schema 自动化轨 id -> (全局秒, 值) 折线（DYN/BRE/TEN/VOI/VEL）；
        pitch_mode:   "auto"（默认，与 RendererOptions.pitch_mode 一致）= dspitch 预测 +
                      用户相对 note tone 的偏差，失败回退请求曲线；"tunelab"/"manual" = 请求曲线。
        """
        block_start = float(block_start_s)
        block_end = float(block_end_s)
        sample_rate = int(vb.sample_rate)
        if not (block_end > block_start):
            return self._empty_block(vb, pitch_points, block_start, block_end)
        if sample_rate <= 0:
            raise RuntimeError("voicebank '" + str(vb.id) + "' has invalid sample_rate " + str(sample_rate))

        phones = list(getattr(phrase, "phones", ()) or ())
        pitch_curve = _clean_points(pitch_points)
        param_curves: dict[str, list[tuple[float, float]]] = {}
        for key, points in (parameters or {}).items():
            kind = _PARAM_ALIASES.get(str(key), _PARAM_ALIASES.get(str(key).lower()))
            cleaned = _clean_points(points)
            if kind is not None and cleaned is not None and kind not in param_curves:
                param_curves[kind] = cleaned
        default_pitch = self._default_pitch(phrase)

        if not phones:
            return self._empty_block(vb, pitch_curve, block_start, block_end, default_pitch)

        first_ms = float(phones[0].start_ms)
        last_ms = float(max(phone.end_ms for phone in phones))
        if last_ms <= first_ms:
            return self._empty_block(vb, pitch_curve, block_start, block_end, default_pitch)
        frame_ms = float(vb.frame_ms)
        if frame_ms <= 0:
            raise RuntimeError("voicebank '" + str(vb.id) + "' has invalid frame_ms " + str(frame_ms))
        head_frames = _round_frames(HEAD_MS, frame_ms)
        tail_frames = _round_frames(TAIL_MS, frame_ms)

        # 1) tokens / durations（SP + phones + SP）
        tokens, durations = self._token_durations(vb, phrase, frame_ms, head_frames, tail_frames)
        total_frames = int(durations.sum())
        if total_frames <= 0:
            return self._empty_block(vb, pitch_curve, block_start, block_end, default_pitch)

        # 2) 音高：auto = dspitch 预测 + 用户相对 note tone 的偏差；tunelab = 请求曲线
        pitch_frames = _sample_curve_frames(
            pitch_curve, first_ms, last_ms, frame_ms, total_frames, head_frames, tail_frames, default_pitch
        )
        # render(pitch_mode=...) 优先；未显式传时用 RendererOptions.pitch_mode（默认 auto）。
        resolved_pitch_mode = self._resolve_pitch_mode(
            pitch_mode if pitch_mode is not None else getattr(self.options, "pitch_mode", "auto")
        )
        used_pitch = pitch_frames
        pitch_source = "request"
        if resolved_pitch_mode == "auto":
            if self._pitch_predictor.available(vb):
                try:
                    predicted = self._pitch_predictor.predict(
                        vb, phrase, frame_ms=frame_ms, steps=int(self.options.steps or 20)
                    )
                    tone_frames = self._note_tone_frames(
                        phrase, first_ms, last_ms, frame_ms, total_frames,
                        head_frames, tail_frames, default_pitch,
                    )
                    # 保留用户画线：最终音高 = 预测绝对音高 + (请求曲线 - 该帧 note tone)。
                    used_pitch = (predicted + (pitch_frames - tone_frames)).astype(np.float32)
                    pitch_source = "dspitch"
                except Exception as error:  # noqa: BLE001 - 预测失败回退请求曲线
                    self._warn_pitch_fallback(vb, "dspitch prediction failed: " + str(error))
            else:
                self._warn_pitch_fallback(vb, "voicebank has no usable dspitch model")
        f0 = np.asarray(audio.semitone_to_hz(used_pitch), dtype=np.float32)

        acoustic_path = self._acoustic_path(vb)
        try:
            acoustic_names = set(self.runner.input_names(acoustic_path))
        except Exception as error:  # noqa: BLE001 - 补声库上下文
            raise RuntimeError(
                "failed to load acoustic model for voicebank '" + str(vb.id) + "' ("
                + str(acoustic_path) + "): " + str(error)
            ) from error

        inputs: dict[str, Any] = {
            "tokens": tokens[np.newaxis, :],
            "durations": durations[np.newaxis, :],
            "f0": f0[np.newaxis, :],
        }

        # 3) 扩散加速参数（depth / steps / speedup）
        for key, value in self._diffusion_inputs(vb).items():
            inputs[key] = value

        # 4) velocity（VEL）、gender（本协议没有 gender 曲线 -> 0）
        if "velocity" in acoustic_names:
            user_velocity = _sample_curve_frames(
                param_curves.get("velocity"), first_ms, last_ms, frame_ms, total_frames,
                head_frames, tail_frames, _PARAM_DEFAULTS["velocity"],
            )
            velocity = _apply_variance_formula(
                "velocity", np.zeros(total_frames, dtype=np.float32), user_velocity.astype(np.float32)
            )
            inputs["velocity"] = np.asarray(velocity, dtype=np.float32)[np.newaxis, :]
        if "gender" in acoustic_names:
            inputs["gender"] = np.zeros((1, int(total_frames)), dtype=np.float32)

        # 5) languages（use_lang_id：音素前缀 "xx/" -> 语言 id；头尾 0）
        if "languages" in acoustic_names:
            inputs["languages"] = self._language_ids(vb, phrase)[np.newaxis, :]

        # 6) speaker embedding（按帧；单 speaker 即常量帧）
        if "spk_embed" in acoustic_names:
            inputs["spk_embed"] = speaker.embed_by_frame(
                vb, phrase, durations, total_frames, head_frames, tail_frames
            )

        # 7) 方差参数：只有声学模型真正声明了该输入时才跑方差模型
        variance_needed = [name for name in _VARIANCE_PARAMS if name in acoustic_names]
        if variance_needed:
            try:
                predicted = self._run_variance(vb, phrase, pitch_curve, frame_ms, total_frames, variance_needed)
            except Exception as error:  # noqa: BLE001 - 补声库上下文
                raise RuntimeError(
                    "variance stage failed for voicebank '" + str(vb.id) + "': " + str(error)
                ) from error
            for name in variance_needed:
                if name not in predicted:
                    raise RuntimeError(
                        "acoustic model of voicebank '" + str(vb.id) + "' requires variance parameter '"
                        + name + "', but the variance model did not predict it"
                    )
                user_unit = _sample_curve_frames(
                    param_curves.get(name), first_ms, last_ms, frame_ms, total_frames,
                    head_frames, tail_frames, _PARAM_DEFAULTS[name],
                )
                unit = _param_value_to_unit(name, user_unit.astype(np.float32))
                inputs[name] = np.asarray(
                    _apply_variance_formula(name, predicted[name], unit), dtype=np.float32
                )[np.newaxis, :]

        # 8) 声学模型 -> mel
        try:
            outputs = self._run_model(acoustic_path, inputs, context="acoustic model of voicebank '" + str(vb.id) + "'")
        except RuntimeError:
            raise
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                "acoustic model of voicebank '" + str(vb.id) + "' failed: " + str(error)
            ) from error
        mel = _first_output(outputs, "mel").astype(np.float32)
        if mel.ndim == 3:
            if mel.shape[0] != 1:
                raise RuntimeError("acoustic model returned batch mel shape " + str(mel.shape))
            mel = mel[0]
        if mel.ndim != 2:
            raise RuntimeError("acoustic model returned unexpected mel shape " + str(mel.shape))
        mel = self._convert_mel_base(vb, mel)

        # 9) 声码器 -> waveform
        audio_start_ms = first_ms - HEAD_MS
        try:
            waveform = self._run_vocoder(vb, mel, f0)
        except RuntimeError:
            raise
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                "vocoder failed for voicebank '" + str(vb.id) + "': " + str(error)
            ) from error

        # 10) 统一采样率 + 按请求块裁剪/补零
        vocoder_rate = int(vb.vocoder.sample_rate) if vb.vocoder is not None else sample_rate
        if vocoder_rate > 0 and vocoder_rate != sample_rate:
            target_length = int(round(waveform.size * float(sample_rate) / float(vocoder_rate)))
            LOG.warning(
                "voicebank '%s': vocoder sample rate %d != voicebank sample rate %d; resampling linearly",
                vb.id, vocoder_rate, sample_rate,
            )
            waveform = audio.resample_linear(waveform, target_length)
        audio_start_s = audio_start_ms / 1000.0
        output_start_s = float(block_start)
        if bool(getattr(self.options, "include_leading", True)):
            # 首个音素的 leading 辅音在首音符之前：把输出窗口向前扩到渲染起点，
            # 让音频里包含完整辅音（可能早于请求块，audio_start_s 会相应变小/为负）。
            output_start_s = min(float(block_start), audio_start_s)
        samples = audio.block_slice(waveform, sample_rate, audio_start_s, output_start_s, block_end)
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if pitch_source == "dspitch":
            # auto 模式：回显实际使用的 F0（预测+偏差），让 TuneLab 合成音高线和音频一致。
            echo = self._pitch_echo_from_frames(used_pitch, first_ms, frame_ms, output_start_s, block_end)
        else:
            echo = self._pitch_echo(pitch_curve, output_start_s, block_end, default_pitch)
        return types.RenderedBlock(
            samples=samples,
            sample_rate=sample_rate,
            audio_start_s=output_start_s,
            pitch_echo=echo,
            parameters_echo={"energy": self._energy_echo(samples, sample_rate, output_start_s)},
        )

    # ------------------------------------------------------------------ 空块 / 回显
    def _empty_block(
        self,
        vb: types.Voicebank,
        pitch_points: Sequence[tuple[float, float]] | None,
        block_start: float,
        block_end: float,
        default_pitch: float = 60.0,
    ) -> types.RenderedBlock:
        sample_rate = int(vb.sample_rate)
        total = max(0, int(round((float(block_end) - float(block_start)) * sample_rate)))
        samples = np.zeros(total, dtype=np.float32)
        return types.RenderedBlock(
            samples=samples,
            sample_rate=sample_rate,
            audio_start_s=float(block_start),
            pitch_echo=self._pitch_echo(pitch_points, block_start, block_end, default_pitch),
            parameters_echo={"energy": self._energy_echo(samples, sample_rate, block_start)},
        )

    def _pitch_echo(
        self,
        pitch_points: Sequence[tuple[float, float]] | None,
        block_start: float,
        block_end: float,
        default_pitch: float,
    ) -> list[tuple[float, float]]:
        duration = float(block_end) - float(block_start)
        if duration <= 0 or not math.isfinite(duration):
            return []
        # 10ms 采样（协议建议 <= 10ms）；超长块自适应抽稀，最多 2 万点。
        step = max(0.01, duration / 20000.0)
        count = int(math.floor(duration / step)) + 1
        times = float(block_start) + np.arange(count, dtype=np.float64) * step
        values = audio.sample_points(pitch_points, times, default_pitch)
        return [(float(t), float(v)) for t, v in zip(times, values)]

    def _energy_echo(self, samples: np.ndarray, sample_rate: int, block_start: float) -> list[tuple[float, float]]:
        duration = samples.size / float(sample_rate) if sample_rate > 0 else 0.0
        points = max(2, min(2000, int(round(duration * 20.0))))
        return audio.rms_energy_curve(samples, sample_rate, float(block_start), points)

    @staticmethod
    def _default_pitch(phrase: types.Phrase) -> float:
        notes = list(getattr(phrase, "notes", ()) or ())
        if notes:
            return float(notes[0].pitch)
        phones = list(getattr(phrase, "phones", ()) or ())
        if phones:
            return float(phones[0].tone)
        return 60.0


    # ------------------------------------------------------------------ 自动音高辅助
    @staticmethod
    def _resolve_pitch_mode(pitch_mode: str | None) -> str:
        """统一 pitch_mode 取值：auto/predict/dspitch -> "auto"；tunelab/manual/curve -> "tunelab"。"""
        mode = str(pitch_mode).strip().lower() if pitch_mode is not None else ""
        if mode in ("auto", "predict", "predicted", "dspitch"):
            return "auto"
        if mode in ("tunelab", "manual", "curve", "request"):
            return "tunelab"
        LOG.warning("unknown pitch_mode %r; falling back to 'auto'", pitch_mode)
        return "auto"

    def _warn_pitch_fallback(self, vb: types.Voicebank, reason: str) -> None:
        """无 dspitch / 预测失败时警告并回退请求曲线（每个声库只 warning 一次，避免刷日志）。"""
        key = str(vb.id)
        if key in self._pitch_warned:
            LOG.debug("voicebank '%s': pitch fallback (%s)", vb.id, reason)
            return
        self._pitch_warned.add(key)
        LOG.warning(
            "voicebank '%s': %s; falling back to the requested pitch curve",
            vb.id, reason,
        )

    @staticmethod
    def _note_tone_frames(
        phrase: types.Phrase,
        first_ms: float,
        last_ms: float,
        frame_ms: float,
        total_frames: int,
        head_frames: int,
        tail_frames: int,
        default_pitch: float,
    ) -> np.ndarray:
        """每帧对应的 note tone（MIDI 半音），用于 auto 模式保留用户相对 note 的偏差。

        body 帧时刻 = 首音素起点 + i*frameMs（与 SampleCurve 一致）；note 间隙取前一个
        note 的 tone；头/尾帧分别取首/末时刻所在 note 的 tone。
        """
        total = max(0, int(total_frames))
        result = np.full(total, float(default_pitch), dtype=np.float32)
        if total == 0:
            return result
        notes = list(getattr(phrase, "notes", ()) or ())
        if not notes:
            return result
        starts = np.asarray([float(note.start_ms) for note in notes], dtype=np.float64)
        tones = np.asarray([float(note.pitch) for note in notes], dtype=np.float32)
        head = min(total, max(0, int(head_frames)))
        tail = min(total - head, max(0, int(tail_frames)))
        body = total - head - tail

        def tone_at(times_ms: np.ndarray) -> np.ndarray:
            indices = np.searchsorted(starts, times_ms, side="right") - 1
            indices = np.clip(indices, 0, len(notes) - 1)
            return tones[indices]

        if body > 0:
            times = float(first_ms) + np.arange(body, dtype=np.float64) * float(frame_ms)
            result[head:head + body] = tone_at(times)
        if head > 0:
            result[:head] = float(tone_at(np.asarray([float(first_ms)]))[0])
        if tail > 0:
            result[total - tail:] = float(tone_at(np.asarray([float(last_ms)]))[0])
        return result

    @staticmethod
    def _pitch_echo_from_frames(
        pitch_frames: np.ndarray,
        first_ms: float,
        frame_ms: float,
        block_start: float,
        block_end: float,
    ) -> list[tuple[float, float]]:
        """把逐帧半音曲线按 ~10ms 采样成 (全局秒, 半音) 回显（auto 模式的合成音高）。"""
        values = np.asarray(pitch_frames, dtype=np.float64).reshape(-1)
        duration = float(block_end) - float(block_start)
        if values.size == 0 or duration <= 0 or not math.isfinite(duration):
            return []
        frame_times = (float(first_ms) - HEAD_MS + np.arange(values.size, dtype=np.float64) * float(frame_ms)) / 1000.0
        step = max(0.01, duration / 20000.0)
        count = int(math.floor(duration / step)) + 1
        times = float(block_start) + np.arange(count, dtype=np.float64) * step
        if frame_times.size >= 2:
            sampled = np.interp(times, frame_times, values)
        else:
            sampled = np.full(times.shape, float(values[0]))
        return [(float(t), float(v)) for t, v in zip(times, sampled)]

    # ------------------------------------------------------------------ 模型路径
    def _acoustic_path(self, vb: types.Voicebank) -> Path:
        path = vb.config.resolve(vb.config.acoustic) if vb.config.acoustic else None
        if path is None:
            raise RuntimeError(
                "voicebank '" + str(vb.id) + "' has no acoustic model configured (dsconfig.yaml 'acoustic')"
            )
        if not Path(path).is_file():
            raise RuntimeError(
                "acoustic model of voicebank '" + str(vb.id) + "' not found: " + str(path)
            )
        return Path(path)

    def _ensure_model(self, vb: types.Voicebank, primary: Path | None, kind: str) -> Path:
        """返回可加载的模型路径；variance/linguistic 打不开时按兜底规则换文件。"""
        if primary is None:
            raise RuntimeError(
                kind + " model path is not configured for voicebank '" + str(vb.id) + "'"
            )
        primary = Path(primary)
        key = (str(vb.bank_id), kind, str(primary))
        with self._lock:
            cached = self._model_path_cache.get(key)
        if cached is not None:
            return cached
        candidates = [primary] + self._fallback_candidates(vb, primary, kind)
        errors: list[tuple[Path, Exception]] = []
        for candidate in candidates:
            try:
                self.runner.input_names(candidate)
            except Exception as error:  # noqa: BLE001 - 损坏文件 / 缺文件
                errors.append((candidate, error))
                continue
            if candidate != primary:
                LOG.warning(
                    "voicebank '%s': %s model %s failed to load, using fallback %s",
                    vb.id, kind, primary, candidate,
                )
            with self._lock:
                self._model_path_cache[key] = candidate
            return candidate
        detail = "; ".join(str(path) + " -> " + str(error) for path, error in errors)
        raise RuntimeError(
            "failed to load a usable " + kind + " model for voicebank '" + str(vb.id)
            + "'; tried: " + detail
        )

    def _fallback_candidates(self, vb: types.Voicebank, primary: Path, kind: str) -> list[Path]:
        """损坏模型兜底候选（逻辑见模块 docstring 的"模型可加载性兜底"）。"""
        pattern = {"variance": "*variance*.onnx", "linguistic": "*linguistic*.onnx"}.get(kind)
        if pattern is None:
            return []
        directories: list[Path] = []
        if kind == "variance":
            if vb.variance is not None:
                directories.append(Path(vb.variance.dir))
            if vb.dur is not None:
                directories.append(Path(vb.dur.dir))
        else:
            for group in (vb.variance, vb.dur, vb.pitch):
                if group is not None:
                    directories.append(Path(group.dir))
        directories.append(primary.parent)
        seen = {str(primary.resolve())}
        result: list[Path] = []
        for directory in directories:
            if not directory.is_dir():
                continue
            try:
                matches = sorted(directory.glob(pattern))
            except OSError:
                continue
            for path in matches:
                try:
                    key = str(path.resolve())
                except OSError:
                    continue
                if key in seen or not path.is_file():
                    continue
                seen.add(key)
                result.append(path)
        return result

    # ------------------------------------------------------------------ token 化
    def _tokenize(self, phonemes: Mapping[str, int], symbols: Sequence[str], *, which: str, vb: types.Voicebank) -> np.ndarray:
        tokens = np.empty(len(symbols), dtype=np.int64)
        for index, symbol in enumerate(symbols):
            try:
                tokens[index] = int(phonemes[symbol])
            except KeyError as error:
                raise RuntimeError(
                    which + " model of voicebank '" + str(vb.id) + "' does not support phoneme '"
                    + str(symbol) + "' (" + str(len(phonemes)) + " symbols in its phoneme table)"
                ) from error
        return tokens

    def _token_durations(
        self,
        vb: types.Voicebank,
        phrase: types.Phrase,
        frame_ms: float,
        head_frames: int,
        tail_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        phones = list(phrase.phones)
        symbols = ["SP"] + [phone.symbol for phone in phones] + ["SP"]
        tokens = self._tokenize(vb.phonemes, symbols, which="acoustic", vb=vb)
        durations: list[int] = [int(head_frames)]
        for phone in phones:
            start = _round_frames(phone.start_ms, frame_ms)
            end = _round_frames(phone.end_ms, frame_ms)
            durations.append(max(0, end - start))
        durations.append(int(tail_frames))
        return tokens, np.asarray(durations, dtype=np.int64)

    def _language_ids(
        self,
        vb: types.Voicebank,
        phrase: types.Phrase,
        group: types.ModelGroup | None = None,
    ) -> np.ndarray:
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

    # ------------------------------------------------------------------ 扩散参数
    def _effective_depth(self, vb: types.Voicebank, config: types.DsConfig) -> float:
        max_depth = float(vb.max_depth_for(config))
        if max_depth < 0:
            raise RuntimeError(
                "voicebank '" + str(vb.id) + "': max_depth is unset or negative with use_variable_depth"
            )
        options_depth = float(self.options.depth if self.options.depth is not None else 1.0)
        return min(options_depth, max_depth)

    def _diffusion_inputs(self, vb: types.Voicebank) -> dict[str, np.ndarray]:
        """声学模型的 depth / steps / speedup 输入（OpenUTAU 语义）。"""
        config = vb.config
        steps = max(1, int(self.options.steps or 20))
        if config.use_continuous_acceleration:
            inputs: dict[str, np.ndarray] = {"steps": np.asarray([steps], dtype=np.int64)}
            if _uses_variable_depth(config):
                depth = self._effective_depth(vb, config)
                inputs["depth"] = np.asarray([depth], dtype=np.float32)
            return inputs
        if _uses_variable_depth(config):
            max_depth = float(vb.max_depth_for(config))
            if max_depth <= 0:
                raise RuntimeError(
                    "voicebank '" + str(vb.id) + "': use_variable_depth but max_depth <= 0 ("
                    + str(max_depth) + ")"
                )
            depth = min(float(self.options.depth if self.options.depth is not None else 1.0), max_depth)
            int64_depth = int(np.round(depth * 1000.0))
            speedup = max(1, int64_depth // steps)
            int64_depth = (int64_depth // speedup) * speedup
            return {
                "depth": np.asarray([int64_depth], dtype=np.int64),
                "speedup": np.asarray([max(1, speedup)], dtype=np.int64),
            }
        return {"speedup": np.asarray([_speedup_for_steps(steps)], dtype=np.int64)}

    def _variance_diffusion_inputs(self, config: types.DsConfig) -> dict[str, np.ndarray]:
        steps = max(1, int(self.options.steps or 20))
        if config.use_continuous_acceleration:
            return {"steps": np.asarray([steps], dtype=np.int64)}
        return {"speedup": np.asarray([_speedup_for_steps(steps)], dtype=np.int64)}

    # ------------------------------------------------------------------ 方差
    def _run_variance(
        self,
        vb: types.Voicebank,
        phrase: types.Phrase,
        pitch_curve: Sequence[tuple[float, float]] | None,
        acoustic_frame_ms: float,
        acoustic_total_frames: int,
        needed: Sequence[str],
    ) -> dict[str, np.ndarray]:
        """跑 dsvariance（linguistic + variance），返回 resample 到声学帧数的预测曲线。"""
        group = vb.variance
        if group is None or not group.available:
            raise RuntimeError(
                "acoustic model requires variance parameter(s) " + ", ".join(needed)
                + " but voicebank '" + str(vb.id) + "' has no usable dsvariance model "
                + "(need linguistic + variance onnx in dsconfig)"
            )
        config = group.config
        var_frame_ms = float(config.frame_ms)
        if var_frame_ms <= 0:
            raise RuntimeError("dsvariance of voicebank '" + str(vb.id) + "' has invalid frame_ms")
        head_frames = _round_frames(HEAD_MS, var_frame_ms)
        tail_frames = _round_frames(TAIL_MS, var_frame_ms)

        phones = list(phrase.phones)
        symbols = ["SP"] + [phone.symbol for phone in phones] + ["SP"]
        tokens = self._tokenize(group.phonemes, symbols, which="variance", vb=vb)
        durations: list[int] = [int(head_frames)]
        for phone in phones:
            start = _round_frames(phone.start_ms, var_frame_ms)
            end = _round_frames(phone.end_ms, var_frame_ms)
            durations.append(max(0, end - start))
        durations.append(int(tail_frames))
        ph_dur = np.asarray(durations, dtype=np.int64)
        total_frames = int(ph_dur.sum())
        if total_frames <= 0:
            raise RuntimeError("variance stage of voicebank '" + str(vb.id) + "' got 0 frames")

        first_ms = float(phones[0].start_ms)
        last_ms = float(max(phone.end_ms for phone in phones))

        # --- linguistic 编码器 ---
        linguistic_path = self._ensure_model(vb, group.linguistic_path, "linguistic")
        linguistic_inputs: dict[str, Any] = {"tokens": tokens[np.newaxis, :]}
        if bool(config.predict_dur):
            word_div, word_dur = self._word_layout(vb, phrase, var_frame_ms, head_frames, tail_frames)
            linguistic_inputs["word_div"] = word_div[np.newaxis, :]
            linguistic_inputs["word_dur"] = word_dur[np.newaxis, :]
        else:
            linguistic_inputs["ph_dur"] = ph_dur[np.newaxis, :]
        if bool(config.use_lang_id):
            linguistic_inputs["languages"] = self._language_ids(vb, phrase, group)[np.newaxis, :]
        linguistic_outputs = self._run_model(
            linguistic_path,
            linguistic_inputs,
            context="variance linguistic model of voicebank '" + str(vb.id) + "'",
        )
        encoder_out = _first_output(linguistic_outputs, "encoder_out").astype(np.float32)

        # --- variance 预测器 ---
        variance_path = self._ensure_model(vb, group.model_path, "variance")
        variance_names = set(self.runner.input_names(variance_path))
        present_params = [name for name in _VARIANCE_PARAMS if name in variance_names]
        num_variances = len(present_params)
        if num_variances == 0:
            flags = [
                bool(config.predict_energy),
                bool(config.predict_breathiness),
                bool(config.predict_voicing),
                bool(config.predict_tension),
            ]
            num_variances = max(1, sum(1 for flag in flags if flag))
        pitch_frames = _sample_curve_frames(
            pitch_curve, first_ms, last_ms, var_frame_ms, total_frames, head_frames, tail_frames, 60.0
        )
        variance_inputs: dict[str, Any] = {
            "encoder_out": encoder_out,
            "ph_dur": ph_dur[np.newaxis, :],
            "pitch": pitch_frames.astype(np.float32)[np.newaxis, :],
            "retake": np.ones((1, total_frames, num_variances), dtype=np.bool_),
        }
        for name in present_params:
            variance_inputs[name] = np.zeros((1, total_frames), dtype=np.float32)
        variance_inputs.update(self._variance_diffusion_inputs(config))
        if "spk_embed" in variance_names:
            variance_inputs["spk_embed"] = speaker.embed_by_frame(
                vb, phrase, ph_dur, total_frames, head_frames, tail_frames, group=group
            )
        variance_outputs = self._run_model(
            variance_path,
            variance_inputs,
            context="variance model of voicebank '" + str(vb.id) + "'",
        )
        predicted: dict[str, np.ndarray] = {}
        for name in needed:
            key = name + "_pred"
            curve = variance_outputs.get(key)
            if curve is None:
                continue
            values = np.asarray(curve, dtype=np.float32).reshape(-1)
            if values.size == 0:
                continue
            resampled = audio.resample_linear(values, int(acoustic_total_frames))
            if resampled is not None:
                predicted[name] = resampled
        # 帧长不一致时（方差 hop 与声学 hop 不同）用 ResampleCurve 语义线性重采样；
        # 上面的 acoustic_total_frames 已是声学帧数，声学侧无需再处理。
        if float(config.frame_ms) <= 0:  # pragma: no cover - 已在上面校验
            raise RuntimeError("invalid variance frame_ms")
        _ = acoustic_frame_ms
        return predicted

    def _vowel_indices(self, vb: types.Voicebank, phrase: types.Phrase) -> list[int]:
        """phone 序列里的元音下标（OpenUTAU 用 dsdict 的 IsVowel；引擎层用注入的 SymbolSet）。"""
        phones = list(phrase.phones)
        if not phones:
            return []
        dictionary = getattr(vb, "dictionary", None)
        if dictionary is not None:
            indices: list[int] = []
            for index, phone in enumerate(phones):
                try:
                    if dictionary.is_vowel(phone.symbol):
                        indices.append(index)
                except Exception:  # noqa: BLE001 - 字典实现异常时走兜底
                    indices = []
                    break
            if indices:
                return indices
        # 兜底：每个 note 的最后一个 phone 视为元音（G2P 正常时与 dsdict 结果接近）。
        ranges = phrase.phone_index_ranges()
        fallback = [end - 1 for begin, end in ranges if end > begin]
        if fallback:
            return fallback
        return [len(phones) - 1]

    def _word_layout(
        self,
        vb: types.Voicebank,
        phrase: types.Phrase,
        frame_ms: float,
        head_frames: int,
        tail_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """OpenUTAU DsVariance 的 word_div / word_dur（predict_dur=True 的 linguistic 输入）。"""
        phones = list(phrase.phones)
        count = len(phones)
        vowel_ids = self._vowel_indices(vb, phrase)
        if not vowel_ids:
            vowel_ids = [count - 1]

        def trunc(ms: float) -> int:
            # OpenUTAU: (Int64)(x / frameMs) —— 截断取整（不是 Math.Round）。
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

    # ------------------------------------------------------------------ 声学/声码器
    def _run_model(self, path: Path, inputs: Mapping[str, Any], *, context: str) -> dict[str, np.ndarray]:
        """按 session 实际输入签名过滤候选输入后运行；缺必需输入时给出清晰错误。"""
        names = self.runner.input_names(path)
        name_set = set(names)
        missing = sorted(name_set - set(inputs))
        if missing:
            raise RuntimeError(
                context + ": model requires input(s) that this pipeline did not build: "
                + ", ".join(missing) + (" (available inputs: " + ", ".join(sorted(inputs)) + ")" if inputs else "")
            )
        filtered = {name: inputs[name] for name in names}
        return self.runner.run(path, filtered)

    def _convert_mel_base(self, vb: types.Voicebank, mel: np.ndarray) -> np.ndarray:
        """声学 / 声码器 mel_base 不一致时换算（OpenUTAU DiffSingerRenderer 同款常数）。"""
        vocoder = vb.vocoder
        if vocoder is None:
            raise RuntimeError("voicebank '" + str(vb.id) + "' has no vocoder configured")
        acoustic_base = str(vb.config.mel_base or "10").lower()
        vocoder_base = str(vocoder.mel_base or "10").lower()
        for label, value in (("acoustic", acoustic_base), ("vocoder", vocoder_base)):
            if value not in ("10", "e"):
                raise RuntimeError(
                    "voicebank '" + str(vb.id) + "': " + label + " mel_base must be '10' or 'e' (got '"
                    + value + "')"
                )
        if acoustic_base == vocoder_base:
            return mel.astype(np.float32, copy=False)
        if vocoder_base == "e" and acoustic_base == "10":
            factor = np.float32(2.30259)
        else:
            factor = np.float32(0.434294)
        LOG.debug(
            "voicebank %s: converting mel from base %s to %s (x%.6f)",
            vb.id, acoustic_base, vocoder_base, float(factor),
        )
        return (mel * factor).astype(np.float32)

    def _vocoder_path(self, vb: types.Voicebank) -> Path:
        vocoder = vb.vocoder
        if vocoder is None:
            raise RuntimeError(
                "voicebank '" + str(vb.id) + "' has no vocoder configured (dsconfig 'vocoder' / dsvocoder)"
            )
        path = getattr(vocoder, "model_path", None)
        if path is None:
            source = getattr(vocoder, "source_path", None)
            model = getattr(vocoder, "model", None)
            if source is not None and model:
                path = Path(source).parent / str(model)
        if path is None:
            raise RuntimeError(
                "voicebank '" + str(vb.id) + "': vocoder '" + str(getattr(vocoder, "name", "vocoder"))
                + "' has no resolved model path"
            )
        path = Path(path)
        if not path.is_file():
            raise RuntimeError(
                "vocoder model of voicebank '" + str(vb.id) + "' not found: " + str(path)
                + " (dsconfig vocoder name '" + str(getattr(vocoder, "name", "")) + "')"
            )
        return path

    def _run_vocoder(self, vb: types.Voicebank, mel: np.ndarray, f0: np.ndarray) -> np.ndarray:
        path = self._vocoder_path(vb)
        vocoder = vb.vocoder
        mel_frames = int(mel.shape[0])
        f0_flat = np.asarray(f0, dtype=np.float32).reshape(-1)
        if f0_flat.size != mel_frames:
            resampled = audio.resample_linear(f0_flat, mel_frames)
            if resampled is None:
                raise RuntimeError(
                    "voicebank '" + str(vb.id) + "': cannot align f0 (" + str(f0_flat.size)
                    + " frames) with mel (" + str(mel_frames) + " frames)"
                )
            LOG.warning(
                "voicebank '%s': mel has %d frames but f0 has %d; resampling f0",
                vb.id, mel_frames, f0_flat.size,
            )
            f0_flat = resampled
        inputs = {
            "mel": np.ascontiguousarray(mel[np.newaxis, :, :], dtype=np.float32),
            "f0": f0_flat[np.newaxis, :],
        }
        outputs = self._run_model(path, inputs, context="vocoder of voicebank '" + str(vb.id) + "'")
        waveform = _first_output(outputs, "waveform")
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size == 0:
            raise RuntimeError("vocoder of voicebank '" + str(vb.id) + "' returned empty waveform")
        return np.ascontiguousarray(waveform, dtype=np.float32)
