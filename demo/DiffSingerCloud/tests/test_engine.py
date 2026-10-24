"""engine 层：ONNX 后端（CPU / CUDA / MPS=CoreML EP）与会话池的单元测试。

运行（在 demo/DiffSingerCloud 下）::

    .venv/bin/python -m pytest tests/test_engine.py -q

后端解析用注入的 provider 列表做纯逻辑测试；真实模型测试只用 nsf_hifigan 声码器
（57MB，CPU + 可选 CoreML），不做硬性性能断言，只打印实测耗时。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from server.engine import backends  # noqa: E402
from server.engine.backends import (  # noqa: E402
    BACKENDS,
    BackendInfo,
    OnnxRunnerImpl,
    _convert_input,
    load_onnxruntime,
    resolve_backend,
)

VOC_DIR = DEMO_DIR / "OpenUTAU" / "Vocoders"
NEW_VOCODER = VOC_DIR / "nsf_hifigan_44.1k_hop512_128bin_2024.02" / "nsf_hifigan_44.1k_hop512_128bin_2024.02.onnx"
OLD_VOCODER = VOC_DIR / "nsf_hifigan" / "nsf_hifigan.onnx"


def _vocoder_model() -> Path:
    for path in (NEW_VOCODER, OLD_VOCODER):
        if path.is_file():
            return path
    pytest.skip("no nsf_hifigan vocoder model found under OpenUTAU/Vocoders")


def _vocoder_inputs(frames: int = 32, f0_hz: float = 440.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1234)
    mel = (rng.standard_normal((1, frames, 128)) * 2.0 - 6.0).astype(np.float32)
    f0 = np.full((1, frames), f0_hz, dtype=np.float32)
    return {"mel": mel, "f0": f0}


# --------------------------------------------------------------------------------------
# 后端解析（provider 列表注入，不依赖本机硬件）
# --------------------------------------------------------------------------------------
def test_resolve_auto_prefers_cuda() -> None:
    info = resolve_backend(
        "auto",
        available_providers=["CPUExecutionProvider", "CUDAExecutionProvider"],
        system="linux",
        machine="x86_64",
    )
    assert info.resolved == "cuda"
    assert info.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert info.requested == "auto"
    assert info.is_gpu


def test_resolve_auto_prefers_cpu_over_coreml_on_apple_silicon() -> None:
    # 实测真实 DiffSinger 模型 CoreML 可能比 CPU 慢 -> auto 不自动选 mps（显式 --backend mps 仍可用）。
    info = resolve_backend(
        "auto",
        available_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        system="darwin",
        machine="arm64",
    )
    assert info.resolved == "cpu"
    assert info.providers == ["CPUExecutionProvider"]
    assert "CoreML" in info.detail and "mps" in info.detail


def test_resolve_auto_cpu_on_intel_mac() -> None:
    # Intel Mac 上 CoreML EP 虽可用但通常比 CPU 慢，auto 不选它。
    info = resolve_backend(
        "auto",
        available_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        system="darwin",
        machine="x86_64",
    )
    assert info.resolved == "cpu"


def test_resolve_auto_cpu_when_no_gpu() -> None:
    info = resolve_backend(
        "auto", available_providers=["CPUExecutionProvider"], system="linux", machine="x86_64"
    )
    assert info.resolved == "cpu"
    assert info.providers == ["CPUExecutionProvider"]
    assert not info.is_gpu
    assert "auto->cpu" in info.detail


def test_resolve_explicit_cpu() -> None:
    info = resolve_backend("cpu", device_id=2, threads=4)
    assert info.resolved == "cpu"
    assert info.device_id == 2
    assert info.threads == 4


def test_resolve_explicit_cuda_unavailable_raises() -> None:
    with pytest.raises(RuntimeError) as error:
        resolve_backend(
            "cuda", available_providers=["CPUExecutionProvider"], system="darwin", machine="arm64"
        )
    message = str(error.value)
    assert "onnxruntime-gpu" in message
    assert "CUDAExecutionProvider" in message


def test_resolve_explicit_mps_on_linux_raises() -> None:
    with pytest.raises(RuntimeError) as error:
        resolve_backend(
            "mps", available_providers=["CPUExecutionProvider"], system="linux", machine="x86_64"
        )
    assert "macOS" in str(error.value)


def test_resolve_explicit_mps_without_coreml_raises() -> None:
    with pytest.raises(RuntimeError) as error:
        resolve_backend(
            "mps",
            available_providers=["CPUExecutionProvider"],
            system="darwin",
            machine="arm64",
        )
    assert "CoreML" in str(error.value)


def test_resolve_invalid_backend_raises() -> None:
    with pytest.raises(ValueError) as error:
        resolve_backend("directml", available_providers=["CPUExecutionProvider"])
    assert "auto" in str(error.value) and "cuda" in str(error.value)


def test_backend_alias_coreml_is_mps() -> None:
    info = resolve_backend(
        "coreml",
        available_providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        system="darwin",
        machine="arm64",
    )
    assert info.resolved == "mps"


def test_backend_info_to_dict() -> None:
    info = BackendInfo(requested="cpu", resolved="cpu", providers=["CPUExecutionProvider"], detail="d")
    payload = info.to_dict()
    assert payload == {
        "requested": "cpu",
        "resolved": "cpu",
        "providers": ["CPUExecutionProvider"],
        "detail": "d",
    }
    assert set(BACKENDS) == {"auto", "cpu", "cuda", "mps"}


# --------------------------------------------------------------------------------------
# runner 输入转换 / 校验（纯逻辑，不加载模型）
# --------------------------------------------------------------------------------------
def test_convert_input_adds_batch_dim() -> None:
    array = _convert_input(
        np.arange(5, dtype=np.float32),
        (1, "n_frames"),
        "tensor(float)",
        path=Path("x.onnx"),
        name="f0",
    )
    assert array.shape == (1, 5)
    assert array.dtype == np.float32


def test_convert_input_scalar() -> None:
    scalar = _convert_input(
        np.array([20], dtype=np.int32), (), "tensor(int64)", path=Path("x.onnx"), name="steps"
    )
    assert scalar.shape == ()
    assert scalar.dtype == np.int64
    assert int(scalar) == 20


def test_convert_input_scalar_from_python_int() -> None:
    scalar = _convert_input(300, (), "tensor(int64)", path=Path("x.onnx"), name="depth")
    assert scalar.shape == () and int(scalar) == 300


def test_convert_input_rejects_bad_rank() -> None:
    with pytest.raises(ValueError):
        _convert_input(np.zeros((2, 3, 4)), (1, "n"), "tensor(float)", path=Path("x.onnx"), name="f0")


# --------------------------------------------------------------------------------------
# 真实声码器模型（CPU；可选 CoreML）
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cpu_runner():
    info = resolve_backend("cpu", threads=2)
    runner = OnnxRunnerImpl(info, max_cached_sessions=4, threads=2)
    yield runner
    runner.close()


def test_runner_input_names(cpu_runner: OnnxRunnerImpl) -> None:
    model = _vocoder_model()
    assert cpu_runner.input_names(model) == ["mel", "f0"]
    assert cpu_runner.output_names(model) == ["waveform"]


def test_runner_verifies_input_names(cpu_runner: OnnxRunnerImpl) -> None:
    model = _vocoder_model()
    inputs = _vocoder_inputs(frames=8)
    with pytest.raises(ValueError) as error:
        cpu_runner.run(model, {"mel": inputs["mel"]})
    assert "Missing input(s)" in str(error.value)

    with pytest.raises(ValueError) as error:
        cpu_runner.run(model, {**inputs, "bogus": np.zeros(3, dtype=np.float32)})
    assert "Unexpected input(s)" in str(error.value)


def test_vocoder_cpu_mel_f0_to_waveform(cpu_runner: OnnxRunnerImpl) -> None:
    model = _vocoder_model()
    frames = 43
    t0 = time.perf_counter()
    outputs = cpu_runner.run(model, _vocoder_inputs(frames=frames))
    elapsed = time.perf_counter() - t0
    wave = outputs["waveform"]
    assert wave.ndim == 2 and wave.shape[0] == 1
    assert wave.dtype == np.float32
    # nsf_hifigan: 每个 mel 帧 -> hop_size=512 个采样点
    assert wave.shape[1] == frames * 512
    assert np.isfinite(wave).all()
    print(f"\n[vocoder cpu] {frames} frames -> {wave.shape[1]} samples in {elapsed * 1000:.0f}ms")


def test_runner_accepts_python_lists(cpu_runner: OnnxRunnerImpl) -> None:
    model = _vocoder_model()
    inputs = _vocoder_inputs(frames=8)
    outputs = cpu_runner.run(
        model, {"mel": inputs["mel"].tolist(), "f0": inputs["f0"].reshape(-1).tolist()}
    )
    assert outputs["waveform"].shape == (1, 8 * 512)


def test_runner_lru_eviction() -> None:
    models = [path for path in (NEW_VOCODER, OLD_VOCODER) if path.is_file()]
    if len(models) < 2:
        pytest.skip("need two vocoder models for LRU eviction test")
    info = resolve_backend("cpu", threads=1)
    runner = OnnxRunnerImpl(info, max_cached_sessions=1, threads=1)
    try:
        for model in models:
            runner.run(model, _vocoder_inputs(frames=4))
            assert len(runner.cached_model_paths()) == 1
        assert runner.cached_model_paths()[0] == str(models[-1].resolve())
    finally:
        runner.close()


def test_real_auto_backend_providers() -> None:
    info = resolve_backend("auto")
    payload = info.to_dict()
    assert payload.keys() == {"requested", "resolved", "providers", "detail"}
    assert info.resolved in BACKENDS
    assert info.providers
    ort = load_onnxruntime()
    assert set(info.providers) <= set(ort.get_available_providers())
    print(f"\n[auto backend] {info.resolved}: {info.detail}")


def test_coreml_vocoder_if_available() -> None:
    """macOS + CoreML EP：跑一次真实声码器，记录建 session / 首跑 / 热跑耗时。

    不做硬性性能断言（不同芯片/ORT 版本差异大），只验证 provider 真的被使用且输出有限。
    """
    ort = load_onnxruntime()
    if "CoreMLExecutionProvider" not in ort.get_available_providers():
        pytest.skip("CoreML execution provider not available")
    info = resolve_backend("mps", threads=2)
    assert info.resolved == "mps"
    model = _vocoder_model()
    runner = OnnxRunnerImpl(info, max_cached_sessions=2, threads=2)
    try:
        t0 = time.perf_counter()
        runner.input_names(model)  # 触发建 session
        create_s = time.perf_counter() - t0
        inputs = _vocoder_inputs(frames=43)
        t0 = time.perf_counter()
        first = runner.run(model, inputs)
        first_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        warm = runner.run(model, inputs)
        warm_s = time.perf_counter() - t0
        cpu = resolve_backend("cpu", threads=2)
        cpu_runner = OnnxRunnerImpl(cpu, max_cached_sessions=1, threads=2)
        try:
            cpu_wave = cpu_runner.run(model, inputs)["waveform"]
        finally:
            cpu_runner.close()
        wave = warm["waveform"]
        assert wave.shape == cpu_wave.shape
        assert np.isfinite(wave).all()
        diff = float(np.max(np.abs(wave - cpu_wave)))
        assert diff < 0.1, "CoreML 与 CPU 输出差异过大: " + str(diff)
        print(
            "\n[coreml vocoder] create=%.2fs first=%.2fs warm=%.2fs max|diff-vs-cpu|=%.2e"
            % (create_s, first_s, warm_s, diff)
        )
    finally:
        runner.close()


# --------------------------------------------------------------------------------------
# 运行时 CPU 回退（session.run 在加速 EP 上执行失败；与"建 session 失败回退"不同）
# --------------------------------------------------------------------------------------
class _ExplodingSession:
    """第一次 run 抛错的假 session（模拟 CoreML EP 动态 resize 失败）。"""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, output_names, inputs):  # noqa: ANN001 - 模拟 ort.InferenceSession.run
        self.calls += 1
        raise RuntimeError(
            "simulated CoreML EP failure: Error in dynamically resizing for sequence length"
        )


def test_runtime_cpu_fallback_after_session_run_error() -> None:
    if "CoreMLExecutionProvider" not in load_onnxruntime().get_available_providers():
        pytest.skip("needs CoreML execution provider for the accelerated-fallback path")
    model = _vocoder_model()
    inputs = _vocoder_inputs(frames=12)
    runner = OnnxRunnerImpl(resolve_backend("mps", threads=2), max_cached_sessions=2, threads=2)
    cpu_runner = OnnxRunnerImpl(resolve_backend("cpu", threads=2), max_cached_sessions=1, threads=2)
    try:
        reference = cpu_runner.run(model, inputs)["waveform"]
        runner.input_names(model)  # 建 CoreML session
        key = backends._model_key(model)
        entry = runner._entries[key]
        assert any(provider != "CPUExecutionProvider" for provider in entry.providers)
        exploding = _ExplodingSession()
        entry.session = exploding
        wave = runner.run(model, inputs)["waveform"]
        assert exploding.calls == 1, "session.run 应该只失败一次（之后用 CPU-only session）"
        replaced = runner._entries[key]
        assert replaced.cpu_only
        assert replaced.providers == ["CPUExecutionProvider"]
        assert key in runner.cpu_fallback_models
        assert "runtime-fell back to CPU" in runner.backend.detail
        assert np.allclose(wave, reference, atol=1e-5)
        # 同一个模型后续 run 直接用缓存里的 CPU-only session，不再失败。
        wave2 = runner.run(model, inputs)["waveform"]
        assert exploding.calls == 1
        assert np.allclose(wave2, reference, atol=1e-5)
    finally:
        runner.close()
        cpu_runner.close()


def test_cpu_backend_does_not_attempt_runtime_fallback() -> None:
    model = _vocoder_model()
    inputs = _vocoder_inputs(frames=8)
    runner = OnnxRunnerImpl(resolve_backend("cpu", threads=2), max_cached_sessions=1, threads=2)
    try:
        runner.input_names(model)
        key = backends._model_key(model)
        exploding = _ExplodingSession()
        runner._entries[key].session = exploding
        with pytest.raises(RuntimeError) as error:
            runner.run(model, inputs)
        assert "ONNX inference failed" in str(error.value)
        assert exploding.calls == 1
        assert not runner.cpu_fallback_models
    finally:
        runner.close()


def test_coreml_synthetic_vocoder_auto_falls_back() -> None:
    """lead 的复现场景：/tmp/TestVoice/dsvocoder/model.onnx 在 CoreML EP 上动态 resize 失败。

    引擎应自动回退 CPU-only 并成功返回 waveform（没有该合成声库/没有 CoreML 时跳过）。
    """
    if "CoreMLExecutionProvider" not in load_onnxruntime().get_available_providers():
        pytest.skip("needs CoreML execution provider")
    model = Path("/tmp/TestVoice/dsvocoder/model.onnx")
    if not model.is_file():
        pytest.skip("synthetic test voicebank /tmp/TestVoice not available")
    inputs = _vocoder_inputs(frames=43)
    runner = OnnxRunnerImpl(resolve_backend("mps", threads=2), max_cached_sessions=2, threads=2)
    cpu_runner = OnnxRunnerImpl(resolve_backend("cpu", threads=2), max_cached_sessions=1, threads=2)
    try:
        wave = runner.run(model, inputs)["waveform"]
        assert wave.size > 0 and np.isfinite(wave).all()
        reference = cpu_runner.run(model, inputs)["waveform"]
        assert wave.shape == reference.shape
        if runner.cpu_fallback_models:
            assert str(model.resolve()) in runner.cpu_fallback_models
            assert np.allclose(wave, reference, atol=1e-5)
            print("\n[synthetic coreml vocoder] runtime-fell back to CPU: "
                  + runner.cpu_fallback_models[str(model.resolve())][:120])
        else:
            print("\n[synthetic coreml vocoder] CoreML executed without fallback")
    finally:
        runner.close()
        cpu_runner.close()
