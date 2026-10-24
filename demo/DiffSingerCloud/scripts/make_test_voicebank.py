#!/usr/bin/env python3
"""生成一个微型但**合法**的 OpenUTAU / DiffSinger 格式测试声库（不依赖任何真实模型）。

用途：在没有真实声库的干净环境（CI / 新机器 / 交付验收）里验证服务端整条链路：
发现 -> dsconfig 解析 -> 字典/G2P -> 音素定时 -> ONNX 声学 -> ONNX 声码器 -> 协议响应。

产物布局（默认写到 demo/DiffSingerCloud/.test-voicebank）::

    <output>/
    |- character.txt            name=TestVoice
    |- character.yaml           singer_type: diffsinger
    |- dsconfig.yaml            acoustic / vocoder / phonemes / 44.1k hop512 128 bins
    |- phonemes.txt             SP AP a o i u e m n l s（行号 = token id）
    |- dsdict.yaml              symbols(vowel/fricative) + entries
    |- acoustic.onnx            tokens,durations,f0 -> mel[1,F,128]，F = sum(durations)
    |- dsvocoder/               OpenUTAU 声库自带声码器约定（优先于 root 的 vocoder 名字）
    |   |- vocoder.yaml
    |   '- model.onnx           mel,f0 -> waveform[1,F*hop]，正弦合成，主频 = mean(f0)
    '- .test-voicebank.json     生成标记（重复执行时用于判断目录能否安全覆盖）

两个 ONNX 模型都用 onnx 包手工构建，体积各约 2KB，推理结果完全确定：

* acoustic: 输出 mel 只是常量模板（形状/流程验证用，下游不依赖具体数值）；
* vocoder:  freq = mean(f0)、waveform[n] = 0.3 * sin(2*pi*freq*n/sample_rate)，
  因此 pitch=69（A4=440Hz）的请求做 FFT 能稳定测到 440Hz —— verify.py 据此断言音高。

用法::

    .venv/bin/python scripts/make_test_voicebank.py --output /tmp/test-voicebank --force
    .venv/bin/python scripts/make_test_voicebank.py            # 默认路径，幂等覆盖
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = DEMO_DIR / ".test-voicebank"
MARKER_NAME = ".test-voicebank.json"

# 音素表顺序 = acoustic 模型 token id（与真实声库 phonemes.txt 语义一致）。
# 任务要求的 10 个音素保持在前 10 个 token；末尾补 s 让 dsdict 里的 sa->s a 也合法。
PHONEMES: tuple[str, ...] = ("SP", "AP", "a", "o", "i", "u", "e", "m", "n", "l", "s")

# dsdict symbol 类型：与真实声库（ChiYu 等）一致，只区分 vowel / fricative。
# 注意 l 不要标成 liquid：OpenUTAU 把 liquid 当 glide，会改变前导音素定时。
SYMBOL_TYPES: dict[str, str] = {
    "SP": "vowel",
    "AP": "vowel",
    "a": "vowel",
    "o": "vowel",
    "i": "vowel",
    "u": "vowel",
    "e": "vowel",
    "m": "fricative",
    "n": "fricative",
    "l": "fricative",
    "s": "fricative",
}

# 歌词 -> 音素。G2P 命中这些 entry；"啊" 经 pypinyin 归一成 "a"，再查表。
DICT_ENTRIES: dict[str, list[str]] = {
    "SP": ["SP"],
    "AP": ["AP"],
    "a": ["a"],
    "o": ["o"],
    "i": ["i"],
    "u": ["u"],
    "e": ["e"],
    "m": ["m"],
    "n": ["n"],
    "l": ["l"],
    "s": ["s"],
    "la": ["l", "a"],
    "sa": ["s", "a"],
    "si": ["s", "i"],
    "su": ["s", "u"],
    "se": ["s", "e"],
    "so": ["s", "o"],
    "ma": ["m", "a"],
    "na": ["n", "a"],
    "ni": ["n", "i"],
    "lo": ["l", "o"],
    "mi": ["m", "i"],
    "nu": ["n", "u"],
    "mo": ["m", "o"],
    "li": ["l", "i"],
    "lu": ["l", "u"],
}

SAMPLE_RATE = 44100
HOP_SIZE = 512
WIN_SIZE = 2048
FFT_SIZE = 2048
NUM_MEL_BINS = 128
MEL_FMIN = 40
MEL_FMAX = 16000
MEL_BASE = "e"
MEL_SCALE = "slaney"

# 生成的条目（用于幂等覆盖时清理旧产物）。
_GENERATED_ENTRIES = {
    "character.txt",
    "character.yaml",
    "dsconfig.yaml",
    "phonemes.txt",
    "dsdict.yaml",
    "acoustic.onnx",
    "dsvocoder",
    MARKER_NAME,
}


# --------------------------------------------------------------------------------------
# ONNX 模型构建
# --------------------------------------------------------------------------------------
def _build_acoustic_onnx(path: Path) -> None:
    """tokens[int64,1,T], durations[int64,1,T], f0[float,1,F] -> mel[float,1,F,128]。

    帧数 F 由 durations 求和得到；mel 内容为常量模板（训练无关，只保证形状正确）。
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inputs = [
        helper.make_tensor_value_info("tokens", TensorProto.INT64, [1, "T"]),
        helper.make_tensor_value_info("durations", TensorProto.INT64, [1, "T"]),
        helper.make_tensor_value_info("f0", TensorProto.FLOAT, [1, "F"]),
    ]
    outputs = [helper.make_tensor_value_info("mel", TensorProto.FLOAT, [1, "F", NUM_MEL_BINS])]

    initializers = [
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "reduce_axes"),
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "shape_one"),
        numpy_helper.from_array(np.asarray([NUM_MEL_BINS], dtype=np.int64), "shape_bins"),
        # mel 模板：所有帧所有 bin 都是 1.0（保持 "e" 底数下非零、有限）。
        numpy_helper.from_array(
            np.ones((1, 1, NUM_MEL_BINS), dtype=np.float32), "mel_template"
        ),
    ]

    nodes = [
        # frames = sum(durations, axis=1)  -> shape [1]
        helper.make_node("ReduceSum", ["durations", "reduce_axes"], ["frames"], keepdims=0),
        # mel_shape = [1, frames, 128]
        helper.make_node("Concat", ["shape_one", "frames", "shape_bins"], ["mel_shape"], axis=0),
        # mel = Expand(template[1,1,128], mel_shape) -> [1, F, 128]
        helper.make_node("Expand", ["mel_template", "mel_shape"], ["mel"]),
    ]

    graph = helper.make_graph(nodes, "test_acoustic", inputs, outputs, initializer=initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="tunelab-diffsinger-cloud/make-test-voicebank",
    )
    model.ir_version = 8  # 兼容较老的 onnxruntime（>=1.14）
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _build_vocoder_onnx(path: Path) -> None:
    """mel[float,1,F,128], f0[float,1,F] -> waveform[float,1,F*hop]。

    波形 = 0.3 * sin(2*pi*mean(f0)*n/sample_rate)，n = 0 .. F*hop-1。
    mel 不参与计算（但必须是模型输入，保持 DiffSinger 声码器签名）。
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inputs = [
        helper.make_tensor_value_info("mel", TensorProto.FLOAT, [1, "F", NUM_MEL_BINS]),
        helper.make_tensor_value_info("f0", TensorProto.FLOAT, [1, "F"]),
    ]
    outputs = [helper.make_tensor_value_info("waveform", TensorProto.FLOAT, [1, "N"])]

    initializers = [
        numpy_helper.from_array(np.asarray([1], dtype=np.int64), "axis_frames"),
        numpy_helper.from_array(np.asarray([], dtype=np.int64), "scalar_shape"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int64), "zero_i64"),
        numpy_helper.from_array(np.asarray(1, dtype=np.int64), "one_i64"),
        numpy_helper.from_array(np.asarray([HOP_SIZE], dtype=np.int64), "hop_size"),
        numpy_helper.from_array(np.asarray(2.0 * np.pi / SAMPLE_RATE, dtype=np.float32), "omega"),
        numpy_helper.from_array(np.asarray(0.3, dtype=np.float32), "amplitude"),
    ]

    nodes = [
        # frames = f0.shape[1]
        helper.make_node("Shape", ["f0"], ["f0_shape"]),
        helper.make_node("Gather", ["f0_shape", "axis_frames"], ["frames"], axis=0),
        # samples = frames * hop_size
        helper.make_node("Mul", ["frames", "hop_size"], ["sample_count"]),
        helper.make_node("Reshape", ["sample_count", "scalar_shape"], ["sample_count_scalar"]),
        # n = 0 .. samples-1（int64 -> float）
        helper.make_node("Range", ["zero_i64", "sample_count_scalar", "one_i64"], ["sample_index"]),
        helper.make_node("Cast", ["sample_index"], ["sample_index_f32"], to=TensorProto.FLOAT),
        # freq = mean(f0)  -> [1, 1]
        helper.make_node("ReduceMean", ["f0"], ["mean_f0"], axes=[1], keepdims=1),
        # phase = mean(f0) * 2*pi/sr * n -> [1, N]
        helper.make_node("Mul", ["mean_f0", "omega"], ["omega_scaled"]),
        helper.make_node("Mul", ["omega_scaled", "sample_index_f32"], ["phase"]),
        helper.make_node("Sin", ["phase"], ["sine"]),
        helper.make_node("Mul", ["sine", "amplitude"], ["waveform"]),
    ]

    graph = helper.make_graph(nodes, "test_vocoder", inputs, outputs, initializer=initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="tunelab-diffsinger-cloud/make-test-voicebank",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


# --------------------------------------------------------------------------------------
# 文本产物
# --------------------------------------------------------------------------------------
def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _character_txt(name: str) -> str:
    return "name=" + name + "\n"


def _character_yaml() -> str:
    return (
        "singer_type: diffsinger\n"
        "default_phonemizer: OpenUtau.Core.DiffSinger.DiffSingerPhonemizer\n"
        "text_file_encoding: utf-8\n"
    )


def _dsconfig_yaml() -> str:
    return (
        'phonemes: "phonemes.txt"\n'
        'acoustic: "acoustic.onnx"\n'
        'vocoder: "test_vocoder"   # 实际使用 dsvocoder/vocoder.yaml（自带给声库的声码器优先）\n'
        "use_key_shift_embed: false\n"
        "use_speed_embed: false\n"
        "use_energy_embed: false\n"
        "use_breathiness_embed: false\n"
        "use_voicing_embed: false\n"
        "use_tension_embed: false\n"
        "use_continuous_acceleration: false\n"
        "use_variable_depth: false\n"
        "use_lang_id: false\n"
        "predict_dur: false\n"
        "predict_energy: false\n"
        "predict_breathiness: false\n"
        "predict_voicing: false\n"
        "predict_tension: false\n"
        "sample_rate: " + str(SAMPLE_RATE) + "\n"
        "hop_size: " + str(HOP_SIZE) + "\n"
        "win_size: " + str(WIN_SIZE) + "\n"
        "fft_size: " + str(FFT_SIZE) + "\n"
        "num_mel_bins: " + str(NUM_MEL_BINS) + "\n"
        "mel_fmin: " + str(MEL_FMIN) + "\n"
        "mel_fmax: " + str(MEL_FMAX) + "\n"
        "mel_base: " + MEL_BASE + "\n"
        "mel_scale: " + MEL_SCALE + "\n"
    )


def _vocoder_yaml() -> str:
    return (
        "name: test_vocoder\n"
        'model: "model.onnx"\n'
        "sample_rate: " + str(SAMPLE_RATE) + "\n"
        "hop_size: " + str(HOP_SIZE) + "\n"
        "win_size: " + str(WIN_SIZE) + "\n"
        "fft_size: " + str(FFT_SIZE) + "\n"
        "num_mel_bins: " + str(NUM_MEL_BINS) + "\n"
        "mel_fmin: " + str(MEL_FMIN) + "\n"
        "mel_fmax: " + str(MEL_FMAX) + "\n"
        "mel_base: " + MEL_BASE + "\n"
        "mel_scale: " + MEL_SCALE + "\n"
        "pitch_controllable: false\n"
    )


def _phonemes_txt() -> str:
    return "\n".join(PHONEMES) + "\n"


def _dsdict_yaml() -> str:
    lines = ["symbols:"]
    for symbol in PHONEMES:
        lines.append("- symbol: " + symbol)
        lines.append("  type: " + SYMBOL_TYPES.get(symbol, "fricative"))
    lines.append("entries:")
    for grapheme, symbols in DICT_ENTRIES.items():
        lines.append("- grapheme: " + grapheme)
        lines.append("  phonemes:")
        for symbol in symbols:
            lines.append("  - " + symbol)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# 覆盖 / 写入 / 自检
# --------------------------------------------------------------------------------------
def prepare_output(output: Path, force: bool) -> None:
    """准备输出目录：生成的旧目录可幂等覆盖，其他非空目录需要 --force。"""
    output = output.expanduser()
    if output.exists():
        if not output.is_dir():
            raise SystemExit("error: output path exists and is not a directory: " + str(output))
        existing = {entry.name for entry in output.iterdir()}
        generated = MARKER_NAME in existing
        safe = generated or not existing or existing <= _GENERATED_ENTRIES
        if force:
            shutil.rmtree(output)
        elif not safe:
            extras = sorted(existing - _GENERATED_ENTRIES)
            raise SystemExit(
                "error: refusing to overwrite a non-generated directory: " + str(output) + "\n"
                "       unexpected entries: " + ", ".join(extras[:8]) + "\n"
                "       pass --force to delete and recreate it."
            )
        else:
            for name in sorted(existing & _GENERATED_ENTRIES):
                target = output / name
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
    output.mkdir(parents=True, exist_ok=True)


def smoke_test(bank_dir: Path) -> dict[str, object]:
    """用 onnxruntime 跑一次两个模型（形状/音高/RMS 自检）；未安装 ORT 时跳过。"""
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - 正常 venv 里一定有
        print("warning: onnxruntime not installed; skipping ONNX smoke test", file=sys.stderr)
        return {"smoke": "skipped"}

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    acoustic = ort.InferenceSession(
        str(bank_dir / "acoustic.onnx"),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    vocoder = ort.InferenceSession(
        str(bank_dir / "dsvocoder" / "model.onnx"),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    durations = np.asarray([[9, 17, 9]], dtype=np.int64)  # SP(100ms) a(200ms) SP(100ms) 近似
    frames = int(durations.sum())
    f0 = np.full((1, frames), 440.0, dtype=np.float32)
    tokens = np.asarray([[0, 2, 0]], dtype=np.int64)  # SP a SP

    mel = acoustic.run(None, {"tokens": tokens, "durations": durations, "f0": f0})[0]
    if tuple(mel.shape) != (1, frames, NUM_MEL_BINS):
        raise SystemExit(
            "error: acoustic output shape " + str(tuple(mel.shape))
            + " != (1, " + str(frames) + ", " + str(NUM_MEL_BINS) + ")"
        )
    waveform = vocoder.run(None, {"mel": mel, "f0": f0})[0]
    expected = (1, frames * HOP_SIZE)
    if tuple(waveform.shape) != expected:
        raise SystemExit(
            "error: vocoder output shape " + str(tuple(waveform.shape)) + " != " + str(expected)
        )

    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    window = np.hanning(samples.size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(samples * window))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / SAMPLE_RATE)
    peak = float(freqs[1:][int(np.argmax(spectrum[1:]))])
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if abs(peak - 440.0) > 440.0 * 0.05:
        raise SystemExit("error: vocoder FFT peak %.2f Hz is not ~440 Hz" % peak)
    if not (0.15 < rms < 0.35):
        raise SystemExit("error: vocoder RMS %.4f is not ~0.21 (amplitude 0.3)" % rms)
    return {"smoke": "ok", "fftPeakHz": round(peak, 2), "rms": round(rms, 4)}


def build_voicebank(
    output: Path,
    *,
    name: str = "TestVoice",
    force: bool = False,
    smoke: bool = True,
) -> Path:
    """构建测试声库到 output 目录并返回它（会被 verify / pytest 直接调用）。"""
    output = Path(output).expanduser()
    prepare_output(output, force)

    _write_text(output / "character.txt", _character_txt(name))
    _write_text(output / "character.yaml", _character_yaml())
    _write_text(output / "dsconfig.yaml", _dsconfig_yaml())
    _write_text(output / "phonemes.txt", _phonemes_txt())
    _write_text(output / "dsdict.yaml", _dsdict_yaml())
    _build_acoustic_onnx(output / "acoustic.onnx")
    _write_text(output / "dsvocoder" / "vocoder.yaml", _vocoder_yaml())
    _build_vocoder_onnx(output / "dsvocoder" / "model.onnx")

    result = smoke_test(output) if smoke else {"smoke": "disabled"}
    marker = {
        "generator": "scripts/make_test_voicebank.py",
        "formatVersion": 1,
        "name": name,
        "phonemes": list(PHONEMES),
        "sampleRate": SAMPLE_RATE,
        "hopSize": HOP_SIZE,
        "numMelBins": NUM_MEL_BINS,
        "models": {"acoustic": "acoustic.onnx", "vocoder": "dsvocoder/model.onnx"},
        "selfTest": result,
    }
    _write_text(output / MARKER_NAME, json.dumps(marker, ensure_ascii=False, indent=2) + "\n")
    return output


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_test_voicebank.py",
        description="生成微型 OpenUTAU/DiffSinger 测试声库（onnx 手写模型，不依赖真实声库）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出目录（声库根）")
    parser.add_argument("--name", default="TestVoice", help="character.txt 里的 name")
    parser.add_argument("--force", action="store_true", help="输出目录非生成目录时也强制删除重建")
    parser.add_argument("--no-smoke", action="store_true", help="跳过 onnxruntime 模型自检")
    parser.add_argument("--quiet", "-q", action="store_true", help="只输出最终声库路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        bank = build_voicebank(args.output, name=args.name, force=args.force, smoke=not args.no_smoke)
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - CLI 顶层统一转成可读错误
        print("error: failed to build test voicebank: " + type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1

    marker = {}
    try:
        marker = json.loads((bank / MARKER_NAME).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    files = sorted(str(path.relative_to(bank)) for path in bank.rglob("*") if path.is_file())
    sizes = {path: (bank / path).stat().st_size for path in files}
    if args.quiet:
        print(str(bank))
    else:
        print("Test voicebank: " + str(bank))
        print("  name:      " + str(marker.get("name", args.name)))
        print("  phonemes:  " + " ".join(marker.get("phonemes", list(PHONEMES))))
        for path in files:
            if path.endswith(".onnx"):
                print("  model:     " + path + " (" + str(sizes[path]) + " bytes)")
        self_test = marker.get("selfTest") or {}
        print(
            "  self-test: " + str(self_test.get("smoke", "?"))
            + (" (FFT peak %.1f Hz, RMS %.3f)" % (self_test["fftPeakHz"], self_test["rms"])
               if self_test.get("smoke") == "ok" else "")
        )
        total = sum(sizes.values())
        if total > 1024 * 1024:
            print("error: generated voicebank is larger than 1MB", file=sys.stderr)
            return 1
        print("  total:     " + str(total) + " bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
