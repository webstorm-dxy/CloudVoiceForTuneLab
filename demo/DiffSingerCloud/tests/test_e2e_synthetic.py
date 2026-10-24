"""合成声库端到端测试：make_test_voicebank -> FastAPI（ASGI TestClient）-> POST /synthesize。

不依赖任何真实声库 / 真实模型，适合 CI 与干净环境；覆盖：
- GET /health、/voices、/schema 的协议字段；
- /schema 自动音高声明（pitch.auto=true、partProperties.PitchMode default=predict）；
  合成声库没有 dspitch 时 PitchMode=predict 回退请求曲线（回显平坦、音频非静音）；
- 单音 2s、pitch=69（A4）合成成功：音频非空、长度约等于块长、RMS 非静音、FFT 主频 440 +-5%；
- phonemes 含元音、pitch / parameters 回显结构合法；
- 错误 protocol / version / 未知 voice / endTime<=startTime 返回 4xx。

运行（在 demo/DiffSingerCloud 下）::

    .venv/bin/python -m pytest tests/test_e2e_synthetic.py -v

真实声库/真实推理的验证见 scripts/verify.py。
"""
from __future__ import annotations

import base64
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = DEMO_DIR / "scripts"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

PROTOCOL = "tunelab-cloudvoice"
VERSION = 1
PITCH = 69
HZ = 440.0
START = 0.0
END = 2.0
NOTE_KEY = "n1"
VOWELS = {"a", "e", "i", "o", "u", "v", "y", "ai", "ei", "ao", "ou", "an", "en", "in", "un"}

REQUIRED_SERVER_FILES = (
    DEMO_DIR / "server" / "voicebank" / "discovery.py",
    DEMO_DIR / "server" / "voicebank" / "phonemizer.py",
    DEMO_DIR / "server" / "engine" / "backends.py",
    DEMO_DIR / "server" / "engine" / "inference.py",
)
SERVER_READY = all(path.is_file() for path in REQUIRED_SERVER_FILES)
pytestmark = pytest.mark.skipif(
    not SERVER_READY,
    reason="服务端实现尚未就绪（缺少 voicebank/engine 模块）",
)


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def voicebank(tmp_path_factory: pytest.TempPathFactory) -> Path:
    builder = _load_script(SCRIPTS_DIR / "make_test_voicebank.py", "make_test_voicebank_for_pytest")
    root = tmp_path_factory.mktemp("voicebanks")
    return builder.build_voicebank(root / "TestVoice", name="TestVoice", force=True, smoke=False)


@pytest.fixture(scope="module")
def client(voicebank: Path):
    from fastapi.testclient import TestClient

    from server.app import create_app
    from server.config import ServerConfig

    config = ServerConfig(
        voice_dirs=[voicebank.parent],
        vocoder_dirs=[],
        backend="cpu",
        log_level="WARNING",
        max_concurrent=1,
    )
    with TestClient(create_app(config)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def voice_id(client) -> str:
    response = client.get("/voices")
    assert response.status_code == 200, response.text
    body = response.json()
    voices = body["voices"] if isinstance(body, dict) else body
    assert voices, "server reported no voices"
    return voices[0]["id"]


def build_request(voice: str, **overrides):
    points = [{"t": round(START + index * 0.01, 4), "v": float(PITCH)}
              for index in range(int(round((END - START) / 0.01)) + 1)]
    payload = {
        "protocol": PROTOCOL,
        "version": VERSION,
        "voice": voice,
        "sampleRate": 48000,
        "startTime": START,
        "endTime": END,
        "pitchMode": "manual",
        "partProperties": {},
        "notes": [{
            "key": NOTE_KEY,
            "start": START,
            "duration": END - START,
            "pitch": PITCH,
            "lyric": "啊",
            "rest": False,
            "properties": {},
        }],
        "pitch": points,
        "parameters": {"DYN": [{"t": START, "v": 0.0}, {"t": END, "v": 0.0}]},
    }
    payload.update(overrides)
    return payload


def decode_audio(response_json: dict) -> tuple[np.ndarray, int]:
    audio = response_json["audio"]
    assert audio["format"] in ("pcm_f32le", "pcm_s16le", "wav"), audio["format"]
    raw = base64.b64decode(audio["data"], validate=True)
    assert raw, "audio payload is empty"
    if audio["format"] == "pcm_f32le":
        samples = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif audio["format"] == "pcm_s16le":
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    else:  # wav：服务端测试里只会是 16-bit PCM
        assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE", "invalid wav header"
        import struct

        position = 12
        fmt = None
        data = None
        while position + 8 <= len(raw):
            chunk_id = raw[position:position + 4]
            size = struct.unpack("<I", raw[position + 4:position + 8])[0]
            body = raw[position + 8:position + 8 + size]
            if chunk_id == b"fmt ":
                fmt = body
            elif chunk_id == b"data":
                data = body
            position += 8 + size + (size & 1)
        audio_format, _channels, sample_rate, _rate, _align, bits = struct.unpack("<HHIIHH", fmt[:16])
        assert audio_format == 1, "unsupported wav format tag %d" % audio_format
        if bits == 16:
            pcm = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        elif bits == 32:
            pcm = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise AssertionError("unsupported wav bits %d" % bits)
        channels = int(response_json["audio"].get("channels") or 1)
        return pcm.reshape(-1, channels).mean(axis=1), int(sample_rate)
    channels = int(audio.get("channels") or 1)
    return samples.reshape(-1, channels).mean(axis=1), int(response_json["sampleRate"])


def fft_peak(samples: np.ndarray, sample_rate: int) -> float:
    segment = np.asarray(samples, dtype=np.float64)
    quarter = segment.size // 4
    segment = segment[quarter:segment.size - quarter] - np.mean(segment[quarter:segment.size - quarter])
    spectrum = np.abs(np.fft.rfft(segment * np.hanning(segment.size)))
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    valid = freqs >= 50.0
    return float(freqs[valid][int(np.argmax(spectrum[valid]))])


# --------------------------------------------------------------------------------------
# 测试
# --------------------------------------------------------------------------------------
def test_health_contract(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["protocol"] == PROTOCOL
    assert body["version"] == VERSION
    assert body["backend"], "health.backend missing"
    assert isinstance(body["voices"], int) and body["voices"] >= 1


def test_catalog_contract(client, voice_id: str) -> None:
    response = client.get("/voices")
    assert response.status_code == 200, response.text
    body = response.json()
    voices = body["voices"] if isinstance(body, dict) else body
    assert isinstance(voices, list) and voices
    for voice in voices:
        assert isinstance(voice.get("id"), str) and voice["id"]
    assert voice_id in [voice["id"] for voice in voices], "voice fixture not in catalog"

    schema = client.get("/schema")
    assert schema.status_code == 200, schema.text
    schema_body = schema.json()
    assert isinstance(schema_body, dict) and schema_body
    assert schema_body.get("pitch", {}).get("auto") is True, "schema.pitch.auto should be true (auto-pitch capability)"
    part_properties = schema_body.get("partProperties")
    assert isinstance(part_properties, list), "schema.partProperties should be a list"
    pitch_mode = next(
        (item for item in part_properties if isinstance(item, dict) and item.get("id") == "PitchMode"),
        None,
    )
    assert pitch_mode is not None, "schema.partProperties should declare PitchMode"
    assert pitch_mode.get("default") == "predict", "PitchMode default should be 'predict'"
    option_values = [
        str(option.get("value")).lower()
        for option in (pitch_mode.get("options") or [])
        if isinstance(option, dict)
    ]
    assert "predict" in option_values and "tunelab" in option_values, \
        "PitchMode options should include predict/tunelab: %r" % option_values


def test_synthesize_synthetic_voicebank(client, voice_id: str) -> None:
    response = client.post("/synthesize", json=build_request(voice_id))
    assert response.status_code == 200, response.text
    body = response.json()
    assert not body.get("error"), body.get("error")

    samples, sample_rate = decode_audio(body)
    assert sample_rate > 0
    # 协议允许 audioStart 覆盖默认对齐（服务端可能保留块首之前的辅音 pre-roll）：
    # 音频覆盖 [audioStart, endTime]，所以时长按 audioStart 起算，同时必须覆盖块首。
    audio_start = float(body.get("audioStart", START))
    assert audio_start <= START + 1e-3, "audioStart %.3fs is later than block start" % audio_start
    assert audio_start >= START - 0.5, "audioStart %.3fs has suspicious pre-roll" % audio_start
    duration = samples.size / float(sample_rate)
    expected = END - audio_start
    assert abs(duration - expected) <= 0.05, "audio %.3fs != endTime-audioStart %.3fs" % (duration, expected)
    assert np.isfinite(samples).all()
    quarter = samples.size // 4
    middle = samples[quarter:samples.size - quarter]
    rms = float(np.sqrt(np.mean(np.square(middle.astype(np.float64)))))
    assert rms > 0.005, "audio is silent (rms=%.5f)" % rms
    peak = fft_peak(samples, sample_rate)
    assert abs(peak - HZ) <= HZ * 0.05, "FFT peak %.1f Hz is not 440Hz +-5%%" % peak

    phonemes = body["phonemes"]
    assert NOTE_KEY in phonemes, phonemes
    entry = phonemes[NOTE_KEY]
    symbols = [item["symbol"] for item in entry["leading"] + entry["body"]]
    assert symbols, "phonemes empty"
    assert entry["body"], "body phonemes empty for a vowel note"
    vowels = [symbol for symbol in symbols
              if symbol.strip().split("/")[-1].strip().lower() in VOWELS]
    assert vowels, "no vowel phoneme in %r" % symbols
    assert isinstance(entry.get("bodyOffset", 0.0), (int, float))

    pitch = body.get("pitch")
    assert isinstance(pitch, list) and len(pitch) >= 2, "pitch echo missing"
    values = [float(point["v"]) for point in pitch]
    assert abs(float(np.median(values)) - PITCH) <= 2.0, "pitch echo median %.2f" % float(np.median(values))

    energy = body["parameters"]["energy"]
    assert isinstance(energy, list) and energy
    for point in energy:
        assert 0.0 <= float(point["v"]) <= 100.0


def test_predict_mode_falls_back_without_dspitch(client, voice_id: str) -> None:
    """合成声库没有 dspitch：PitchMode=predict 应回退到请求曲线（200、回显平坦、音频非静音）。"""
    payload = build_request(voice_id, partProperties={"PitchMode": "predict", "language": "zh"})
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert not body.get("error"), body.get("error")
    meta = body.get("meta") or {}
    assert meta.get("pitchSource") in ("auto", "tunelab"), "meta.pitchSource: %r" % (meta.get("pitchSource"),)
    pitch = body.get("pitch")
    assert isinstance(pitch, list) and len(pitch) >= 2, "pitch echo missing"
    values = [float(point["v"]) for point in pitch]
    assert max(values) - min(values) <= 0.05, "no dspitch -> predict should fall back to the flat request curve"
    samples, sample_rate = decode_audio(body)
    assert samples.size > 0 and sample_rate > 0
    quarter = samples.size // 4
    middle = samples[quarter:samples.size - quarter] if samples.size >= 8 else samples
    rms = float(np.sqrt(np.mean(np.square(middle.astype(np.float64)))))
    assert rms > 0.005, "fallback synthesis is silent (rms=%.5f)" % rms


def test_rest_note_produces_silence_structure(client, voice_id: str) -> None:
    payload = build_request(voice_id)
    payload["notes"] = [{
        "key": "rest-note",
        "start": START,
        "duration": END - START,
        "pitch": PITCH,
        "lyric": "",
        "rest": True,
        "properties": {},
    }]
    response = client.post("/synthesize", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    samples, _sample_rate = decode_audio(body)
    assert samples.size > 0
    assert float(np.max(np.abs(samples))) < 0.05, "rest block should be (near) silent"


def test_error_paths(client, voice_id: str) -> None:
    cases = []
    payload = build_request(voice_id)
    payload["protocol"] = "tunelab-cloudvoice-bogus"
    cases.append(("bad protocol", payload))

    payload = build_request(voice_id)
    payload["version"] = 99
    cases.append(("bad version", payload))

    payload = build_request(voice_id)
    payload["voice"] = "__no_such_voice__"
    cases.append(("unknown voice", payload))

    payload = build_request(voice_id)
    payload["endTime"] = payload["startTime"]
    cases.append(("empty block", payload))

    for label, payload in cases:
        response = client.post("/synthesize", json=payload)
        assert 400 <= response.status_code < 500, label + " -> HTTP %d: %s" % (response.status_code, response.text[:300])
        body = response.json()
        assert isinstance(body.get("error"), str) and body["error"].strip(), label + " has no error message"
