"""跨层共享的数据契约。

**这个文件由 lead 维护**：HTTP 层、声库层（voicebank/）与推理层（engine/）都依赖它。
实现时允许"只增不改"地补默认字段；重命名/删除已有字段前必须先和 lead 协调，
否则另一层会直接 import 失败。

数据流::

    HTTP 请求 --> NoteInput[]
                     |  DiffSingerPhonemizer（voicebank 层）
                     v
                   Phrase（Phone[] + ResolvedNote[]，绝对毫秒时间轴）
                     |  DiffSingerRenderer（engine 层）
                     v
                   RenderedBlock（按请求块裁剪好的 float32 单声道样本）

模型 / 配置元数据由 voicebank 层扫描后填进 Voicebank；engine 层只负责"按这些路径和
dsconfig 字段跑 ONNX"，不再自己解析 YAML。

参考实现（只读）：/Users/daxingyi/RiderProjects/OpenUtau/OpenUtau.Core/DiffSinger/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

# DiffSinger 渲染在句子前后各补的时间（毫秒），与 OpenUTAU DiffSingerUtils.headMs/tailMs 一致。
HEAD_MS = 100.0
TAIL_MS = 100.0

# 协议默认采样率（TuneLab 请求；声库原生多为 44100）。
DEFAULT_SAMPLE_RATE = 48000


# --------------------------------------------------------------------------------------
# 模型配置（对应 dsconfig.yaml / vocoder.yaml；字段名与 OpenUTAU DsConfig 对齐）
# --------------------------------------------------------------------------------------
@dataclass
class DsConfig:
    """一份 DiffSinger 配置（声学 / dsdur / dsvariance / dspitch 各有一份）。

    source_path 是这份 yaml 文件本身，用于解析同目录/相对路径里的模型与音素表。
    """

    raw: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    phonemes: str = "phonemes.txt"
    languages: str | None = None
    acoustic: str | None = None
    vocoder: str | None = None
    speakers: list[str] | None = None
    hidden_size: int = 256

    use_key_shift_embed: bool = False
    use_speed_embed: bool = False
    use_energy_embed: bool = False
    use_breathiness_embed: bool = False
    use_voicing_embed: bool = False
    use_tension_embed: bool = False
    use_continuous_acceleration: bool = False
    use_variable_depth: bool = False
    max_depth: float = 1.0
    use_lang_id: bool = False
    use_expr: bool = False
    use_note_rest: bool = False

    dur: str | None = None
    linguistic: str | None = None
    pitch: str | None = None
    variance: str | None = None
    predict_dur: bool = True
    predict_energy: bool = True
    predict_breathiness: bool = True
    predict_voicing: bool = False
    predict_tension: bool = False

    # key shift 增强参数（gender 输入换算用）；默认 (-5,5)，对应 OpenUTAU augmentation_args
    pitch_shifting_range: tuple[float, float] = (-5.0, 5.0)

    sample_rate: int = 44100
    hop_size: int = 512
    win_size: int = 2048
    fft_size: int = 2048
    num_mel_bins: int = 128
    mel_fmin: float = 40.0
    mel_fmax: float = 16000.0
    mel_base: str = "10"       # "10" | "e"
    mel_scale: str = "slaney"  # "slaney" | "htk"

    @property
    def frame_ms(self) -> float:
        """一帧时长（毫秒）。"""
        return 1000.0 * self.hop_size / self.sample_rate

    def resolve(self, name: str | None) -> Path | None:
        """把 dsconfig 里的相对路径解析成绝对路径（相对 yaml 所在目录）。"""
        if not name:
            return None
        path = Path(name)
        if path.is_absolute():
            return path
        assert self.source_path is not None, "DsConfig.source_path is required"
        return (self.source_path.parent / path).resolve()


@dataclass
class VocoderConfig:
    """vocoder.yaml 内容（内建声码器名或声库自带的 dsvocoder）。"""

    raw: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    name: str = "vocoder"
    model: str = "model.onnx"
    model_path: Path | None = None
    sample_rate: int = 44100
    hop_size: int = 512
    win_size: int = 2048
    fft_size: int = 2048
    num_mel_bins: int = 128
    mel_fmin: float = 40.0
    mel_fmax: float = 16000.0
    mel_base: str = "10"
    mel_scale: str = "slaney"
    pitch_controllable: bool = False

    @property
    def frame_ms(self) -> float:
        return 1000.0 * self.hop_size / self.sample_rate


@dataclass
class ModelGroup:
    """一个辅助模型目录（dsdur / dsvariance / dspitch），含自己的 dsconfig 与音素表。"""

    kind: str                     # "dur" | "variance" | "pitch"
    dir: Path
    config: DsConfig
    phonemes: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    linguistic_path: Path | None = None
    model_path: Path | None = None
    emb_dir: Path | None = None    # .emb 所在目录（通常是 dir 本身）

    @property
    def available(self) -> bool:
        return bool(self.linguistic_path and self.model_path)


@dataclass
class Subbank:
    """character.yaml 里的 subbank（音色 / speaker 后缀）。"""

    color: str = ""
    suffix: str = ""
    prefix: str = ""


# --------------------------------------------------------------------------------------
# 声库（目录中的一个条目 = 一个 voice id；多说话人声库按 speaker 展开成多个条目）
# --------------------------------------------------------------------------------------
@dataclass
class Voicebank:
    """扫描后可直接用于推理的声库描述。

    id 是对外（GET /voices 与请求 voice 字段）唯一的 id；多说话人声库按 speaker
    展开，格式为 "<bank_id>::<speaker>"。
    """

    id: str
    bank_id: str
    name: str
    path: Path
    config: DsConfig
    phonemes: dict[str, int] = field(default_factory=dict)   # 声学模型音素表
    languages: dict[str, int] = field(default_factory=dict)  # use_lang_id 用
    speakers: list[str] = field(default_factory=list)
    subbanks: list[Subbank] = field(default_factory=list)
    selected_speaker: str = ""
    description: str = ""
    author: str = ""
    version: str = ""
    dur: ModelGroup | None = None
    pitch: ModelGroup | None = None
    variance: ModelGroup | None = None
    vocoder: VocoderConfig | None = None
    dictionary: "SymbolSet | None" = None
    warnings: list[str] = field(default_factory=list)

    # ---- 便捷属性（engine / app 层用） -------------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def frame_ms(self) -> float:
        return self.config.frame_ms

    def emb_path(self, speaker: str | None = None, *, group: ModelGroup | None = None) -> Path | None:
        """speaker 的 .emb 文件路径；不存在返回 None。"""
        speaker = self.selected_speaker if speaker is None else speaker
        if not speaker:
            return None
        if group is not None:
            directory = group.emb_dir if group.emb_dir is not None else group.dir
        else:
            directory = self.path
        path = Path(directory) / (speaker + ".emb")
        return path if path.is_file() else None

    def max_depth_for(self, config: DsConfig) -> float:
        """OpenUTAU 的 useVariableDepth 语义：max_depth<0 表示非法。"""
        if config.use_continuous_acceleration:
            return config.max_depth
        return config.max_depth / 1000.0


# --------------------------------------------------------------------------------------
# 歌词 / 音素 / 乐句
# --------------------------------------------------------------------------------------
@dataclass
class NoteInput:
    """请求里的一个音符（延音已由客户端合并到链头 duration）。"""

    key: str
    start_s: float
    duration_s: float
    pitch: int
    lyric: str = ""
    rest: bool = False
    properties: dict[str, Any] = field(default_factory=dict)
    pinned: list[str] | None = None   # 用户钉死音素（leading+body 展开后的符号序列）

    @property
    def start_ms(self) -> float:
        return self.start_s * 1000.0

    @property
    def end_ms(self) -> float:
        return (self.start_s + self.duration_s) * 1000.0


@dataclass
class ResolvedNote:
    """经过休止/延音归一化后的音符（绝对毫秒）。"""

    key: str
    index: int
    start_ms: float
    end_ms: float
    pitch: int
    rest: bool = False
    lyric: str = ""


@dataclass
class Phone:
    """一个音素及其在工程时间轴上的绝对起止时间（毫秒）。"""

    symbol: str
    start_ms: float
    end_ms: float
    note_index: int
    tone: int
    vowel: bool = False
    glide: bool = False
    speaker: str = ""

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass
class Phrase:
    """一个待渲染的乐句。"""

    phones: list[Phone]
    notes: list[ResolvedNote]

    def phone_index_ranges(self) -> list[tuple[int, int]]:
        """每个 note 在 phones 里对应的 [start, end) 下标（没有音素的 note 为空区间）。"""
        ranges: list[tuple[int, int]] = []
        cursor = 0
        for note in self.notes:
            begin = cursor
            while cursor < len(self.phones) and self.phones[cursor].note_index == note.index:
                cursor += 1
            ranges.append((begin, cursor))
        return ranges


@dataclass
class PhonemizerOptions:
    language: str = "auto"          # auto | zh | ja | en | ...
    use_duration_model: bool = True  # 有 dsdur 时用模型预测时长
    min_phone_ms: float = 15.0
    max_leading_ms: float = 120.0


@dataclass
class RendererOptions:
    depth: float = 1.0
    steps: int = 20
    max_cached_sessions: int = 12
    threads: int = 0                 # 0 = ORT 默认
    max_concurrent: int = 1
    # 首音节辅音（leading）在首音符之前：True 时 RenderedBlock 会向前多覆盖
    # [首个 Phone.start_ms - HEAD_MS, block_end]，audio_start_s 返回该窗口起点，
    # 这样 TuneLab 的 junction-anchored 音素布局（辅音在拍前）能拿到完整辅音音频。
    # False 时保持旧行为（正好覆盖请求块）。
    include_leading: bool = True
    # 音高来源："auto" = 优先用声库 dspitch 预测音高（无 dspitch/预测失败时回退请求曲线）；
    # "tunelab" = 始终使用请求里的 pitch 曲线（旧行为）。
    pitch_mode: str = "auto"


@dataclass
class RenderedBlock:
    """engine 层输出：覆盖 [audio_start_s, block_end_s) 的样本。

    默认（RendererOptions.include_leading=True）audio_start_s 会早于请求块起点
    （首个音素的 leading 辅音在首音符之前）；否则 audio_start_s == 请求块起点。
    """

    samples: np.ndarray              # float32, mono, 长度 = round((block_end - audio_start_s) * sample_rate)
    sample_rate: int
    audio_start_s: float
    pitch_echo: list[tuple[float, float]] = field(default_factory=list)   # (全局秒, MIDI 半音)
    parameters_echo: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# 层间协议（voicebank 层使用 engine 层的 runner；engine 层使用 voicebank 层的数据类）
# --------------------------------------------------------------------------------------
class OnnxRunner(Protocol):
    """engine 层暴露给 voicebank 层的 ONNX 会话池（按模型路径缓存）。"""

    def run(self, model_path: Path, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """跑一次推理；inputs 的值可以是 list/np.ndarray，runner 按模型签名转换 dtype/shape。

        返回 {输出名: ndarray}。输入名不匹配（缺失/多余）时抛 ValueError。
        """

    def input_names(self, model_path: Path) -> list[str]:
        ...

    def output_names(self, model_path: Path) -> list[str]:
        ...


class SymbolSet(Protocol):
    """voicebank 层给 engine 层用的音素分类/查询接口（来自 dsdict 字典）。"""

    def query(self, grapheme: str) -> list[str] | None: ...
    def query_lang(self, lang: str, grapheme: str) -> list[str] | None: ...
    def is_valid(self, symbol: str) -> bool: ...
    def is_vowel(self, symbol: str) -> bool: ...
    def is_glide(self, symbol: str) -> bool: ...
    def symbol_type(self, symbol: str) -> str: ...
    def symbols(self) -> Iterable[str]: ...
