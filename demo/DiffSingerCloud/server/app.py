"""FastAPI 服务端装配：/health /voices /schema /synthesize。

设计参考 tools/server_example/app.py（协议校验、鉴权、请求体限制、并发信号量、错误语义），
后端换成 DiffSinger 推理管线（engine 层），声库来自 OpenUTAU 格式目录（voicebank 层）。

启动：
    demo/DiffSingerCloud/.venv/bin/python -m server --port 18791 --backend cpu
    demo/DiffSingerCloud/.venv/bin/python -m server --config config.yaml

TuneLab 侧（设置 -> 扩展 -> Cloud Voice）：
    Service URL = http://127.0.0.1:18791/synthesize
    API key     = 启动时 --api-key 的值（没有则不填）
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import math
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import __version__, audio
from .config import ServerConfig
from .protocol import (
    PROTOCOL,
    VERSION,
    CloudSynthesisRequest,
    parse_request,
    phoneme_payload,
    validate_request,
)
from .schema import get_schema
from .voicebank.types import NoteInput

LOG = logging.getLogger("diffsinger-server")


class ServiceState:
    """服务端运行期对象（启动时构造一次）。"""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.backend_info: Any = None
        self.runner: Any = None
        self.catalog: Any = None
        self.phonemizer: Any = None
        self.renderer: Any = None
        self.semaphore: asyncio.Semaphore | None = None
        self.started_at = time.time()
        self.synth_count = 0
        self.built = False

    def build(self) -> None:
        from .engine.backends import OnnxRunnerImpl, resolve_backend
        from .engine.inference import DiffSingerRenderer
        from .voicebank.discovery import VoiceCatalog
        from .voicebank.phonemizer import DiffSingerPhonemizer
        from .voicebank.types import PhonemizerOptions, RendererOptions

        config = self.config
        LOG.info("resolving ONNX backend (requested=%s)", config.backend)
        self.backend_info = resolve_backend(config.backend, device_id=config.device_id, threads=config.threads)
        LOG.info("ONNX backend resolved: %s (%s)", self.backend_info.resolved, ", ".join(self.backend_info.providers))
        self.runner = OnnxRunnerImpl(
            self.backend_info,
            max_cached_sessions=config.max_cached_sessions,
            threads=config.threads,
        )
        self.catalog = VoiceCatalog(config.voice_dirs, config.vocoder_dirs, max_depth=config.scan_depth)
        self.catalog.refresh()
        LOG.info("scanned %d voice(s) from %s", len(self.catalog.voices()), ", ".join(str(p) for p in config.voice_dirs))
        self.phonemizer = DiffSingerPhonemizer(
            self.runner,
            PhonemizerOptions(language=config.language, use_duration_model=True),
        )
        self.renderer = DiffSingerRenderer(
            self.runner,
            RendererOptions(
                depth=config.depth,
                steps=config.steps,
                max_cached_sessions=config.max_cached_sessions,
                threads=config.threads,
                max_concurrent=config.max_concurrent,
                include_leading=config.include_leading,
                pitch_mode=config.pitch_mode,
            ),
        )
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

    def close(self) -> None:
        runner = self.runner
        close = getattr(runner, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - 关闭失败不影响进程退出
                LOG.exception("failed to close ONNX runner")


# --------------------------------------------------------------------------------------
# 请求 -> 内部音符；内部音素 -> 协议回填
# --------------------------------------------------------------------------------------
def build_notes(request: CloudSynthesisRequest) -> list[NoteInput]:
    notes: list[NoteInput] = []
    for note in request.notes:
        pinned: list[str] | None = None
        if note.phonemes is not None:
            flattened = note.phonemes.flatten()
            if flattened:
                pinned = flattened
        if pinned is None:
            override = note.properties.get("phoneme") if note.properties else None
            if isinstance(override, str) and override.strip():
                pinned = override.split()
        notes.append(
            NoteInput(
                key=note.key,
                start_s=float(note.start),
                duration_s=float(note.duration),
                pitch=int(note.pitch),
                lyric=note.lyric or "",
                rest=bool(note.rest),
                properties=dict(note.properties or {}),
                pinned=pinned,
            )
        )
    return notes


def build_phonemes_payload(phrase: Any, request: CloudSynthesisRequest) -> dict[str, Any]:
    """把 Phrase 里的音素按 note key 组织成协议要求的结构。"""
    payload: dict[str, Any] = {}
    if not getattr(phrase, "notes", None):
        return payload
    ranges = phrase.phone_index_ranges()
    notes = phrase.notes
    for index, note in enumerate(notes):
        if index >= len(ranges):
            continue
        begin, end = ranges[index]
        phones = phrase.phones[begin:end]
        if not phones:
            payload[note.key] = {"leading": [], "body": [], "bodyOffset": 0.0}
            continue
        first_vowel = next((i for i, phone in enumerate(phones) if getattr(phone, "vowel", False)), len(phones))
        if first_vowel >= len(phones):
            leading = []
            body = phones
        else:
            leading = phones[:first_vowel]
            body = phones[first_vowel:]
        body_offset = 0.0
        if body:
            body_offset = (float(body[0].start_ms) - float(note.start_ms)) / 1000.0
        payload[note.key] = {
            "leading": [
                phoneme_payload(
                    phone.symbol,
                    (float(phone.end_ms) - float(phone.start_ms)) / 1000.0,
                    0.0,
                )
                for phone in leading
            ],
            "body": [
                phoneme_payload(
                    phone.symbol,
                    (float(phone.end_ms) - float(phone.start_ms)) / 1000.0,
                    1.0 if getattr(phone, "vowel", False) else 0.0,
                )
                for phone in body
            ],
            "bodyOffset": body_offset,
        }
    return payload


def build_parameter_echo(rendered: Any, request: CloudSynthesisRequest) -> dict[str, Any]:
    echo = getattr(rendered, "parameters_echo", None) or {}
    energy = echo.get("energy")
    if not energy:
        audio_start = float(getattr(rendered, "audio_start_s", request.startTime))
        energy = audio.rms_energy_curve(rendered.samples, rendered.sample_rate, audio_start)
    result: dict[str, Any] = {}
    if energy:
        result["energy"] = [{"t": float(t), "v": float(v)} for t, v in energy]
    return result


def build_pitch_echo(rendered: Any) -> list[dict[str, float]]:
    points = getattr(rendered, "pitch_echo", None) or []
    return [{"t": float(t), "v": float(v)} for t, v in points]


def voice_description(bank: Any) -> str:
    parts = ["DiffSinger"]
    try:
        parts.append(str(int(bank.sample_rate) / 1000.0) + "kHz")
    except Exception:  # noqa: BLE001
        pass
    if bank.selected_speaker:
        parts.append(bank.selected_speaker)
    if bank.dur is None:
        parts.append("no-dsdur")
    if bank.variance is None:
        parts.append("no-variance")
    if getattr(bank, "pitch", None) is not None:
        parts.append("auto-pitch")
    else:
        parts.append("manual-pitch")
    if bank.author:
        parts.append("by " + str(bank.author))
    return " · ".join(parts)


# --------------------------------------------------------------------------------------
# 同步合成（放进线程池，避免阻塞事件循环）
# --------------------------------------------------------------------------------------
def synthesize_sync(state: ServiceState, request: CloudSynthesisRequest, voice_id: str) -> dict[str, Any]:
    from .voicebank.types import PhonemizerOptions

    config = state.config
    catalog = state.catalog
    try:
        bank = catalog.get(voice_id)
    except KeyError:
        # 运行期新放入的声库：刷新一次再试
        catalog.refresh()
        bank = catalog.get(voice_id)

    notes = build_notes(request)
    language = request.partProperties.get("language") if request.partProperties else None
    if not isinstance(language, str) or language not in ("auto", "zh", "ja", "en"):
        language = config.language
    part_options = PhonemizerOptions(language=language, use_duration_model=True)
    if hasattr(state.phonemizer, "options"):
        state.phonemizer.options = part_options

    phrase = state.phonemizer.phonemize(bank, notes, language=language)

    # part 属性 PitchMode：predict（默认，有 dspitch 自动预测）/ tunelab（跟随画线）。
    part_pitch_mode = request.partProperties.get("PitchMode") if request.partProperties else None
    if isinstance(part_pitch_mode, str) and part_pitch_mode.strip().lower() in ("predict", "auto"):
        pitch_mode = "auto"
    elif isinstance(part_pitch_mode, str) and part_pitch_mode.strip().lower() in ("tunelab", "manual"):
        pitch_mode = "tunelab"
    else:
        pitch_mode = config.pitch_mode

    pitch_points = [(float(point.t), float(point.v)) for point in request.pitch]
    parameters: dict[str, list[tuple[float, float]]] = {
        key: [(float(point.t), float(point.v)) for point in points]
        for key, points in (request.parameters or {}).items()
    }
    rendered = state.renderer.render(
        bank,
        phrase,
        pitch_points=pitch_points,
        parameters=parameters,
        block_start_s=float(request.startTime),
        block_end_s=float(request.endTime),
        pitch_mode=pitch_mode,
    )

    samples = rendered.samples
    # 允许 engine 返回前置辅音 pre-roll：audioStart 必须与 samples 起点一致（协议 audioStart 覆盖）。
    audio_start = float(getattr(rendered, "audio_start_s", request.startTime))
    if not (math.isfinite(audio_start) and audio_start <= float(request.startTime) + 1e-6):
        audio_start = float(request.startTime)
    response = {
        "protocol": PROTOCOL,
        "version": VERSION,
        "sampleRate": int(rendered.sample_rate),
        "audioStart": audio_start,
        "audio": {
            "format": "pcm_f32le",
            "channels": 1,
            "data": audio.b64(audio.pcm_f32le(samples)),
        },
        "phonemes": build_phonemes_payload(phrase, request),
        "pitch": build_pitch_echo(rendered),
        "parameters": build_parameter_echo(rendered, request),
        "meta": {
            "engine": "diffsinger",
            "voice": bank.id,
            "voicebank": bank.name,
            "backend": state.backend_info.resolved if state.backend_info else "unknown",
            "sampleRate": int(rendered.sample_rate),
            "phones": len(getattr(phrase, "phones", [])),
            "pitchMode": request.pitchMode,
            "pitchSource": pitch_mode,
            "autoPitchModel": "dspitch" if getattr(bank, "pitch", None) is not None else None,
            "server": "tunelab-diffsinger-cloud/" + __version__,
        },
    }
    if not response["parameters"].get("energy"):
        # 协议里 energy 是只读回显轨；没有方差模型时也要给一条可用曲线。
        fallback = audio.rms_energy_curve(samples, rendered.sample_rate, audio_start)
        response["parameters"]["energy"] = [{"t": float(t), "v": float(v)} for t, v in fallback]
    return response


# --------------------------------------------------------------------------------------
# ASGI app
# --------------------------------------------------------------------------------------
def create_app(config: ServerConfig) -> FastAPI:
    state = ServiceState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ANN202 - FastAPI lifespan 签名
        if not state.built:
            state.build()
            state.built = True
        LOG.info(
            "%s ready: %d voice(s), backend=%s, protocol=%s/%d",
            "tunelab-diffsinger-cloud",
            len(state.catalog.voices()),
            state.backend_info.resolved,
            PROTOCOL,
            VERSION,
        )
        try:
            yield
        finally:
            state.close()

    app = FastAPI(title="tunelab-diffsinger-cloud", version=__version__, lifespan=lifespan)
    app.state.service = state
    app.state.config = config

    @app.get("/")
    async def index() -> dict[str, Any]:
        return {
            "name": "tunelab-diffsinger-cloud",
            "version": __version__,
            "protocol": PROTOCOL,
            "versionNumber": VERSION,
            "endpoints": ["/health", "/voices", "/schema", "/synthesize"],
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        service: ServiceState = app.state.service
        return {
            "status": "ok",
            "protocol": PROTOCOL,
            "version": VERSION,
            "backend": service.backend_info.to_dict() if service.backend_info else None,
            "voices": len(service.catalog.voices()) if service.catalog else 0,
            "synthesized": service.synth_count,
            "uptimeSeconds": round(time.time() - service.started_at, 3),
            "engine": "diffsinger",
        }

    @app.get("/voices")
    async def voices() -> dict[str, Any]:
        service: ServiceState = app.state.service
        catalog_voices = service.catalog.voices() if service.catalog else []
        return {
            "voices": [
                {
                    "id": bank.id,
                    "name": bank.name,
                    "description": voice_description(bank),
                    "voice": bank.id,
                }
                for bank in catalog_voices
            ],
            "schema": get_schema(),
        }

    @app.get("/schema")
    async def schema() -> dict[str, Any]:
        return get_schema()

    @app.post("/synthesize")
    async def synthesize(request: Request) -> JSONResponse:
        service: ServiceState = app.state.service
        config = service.config
        started = time.perf_counter()

        if config.api_key:
            header = request.headers.get("authorization", "")
            if not hmac.compare_digest(header, "Bearer " + config.api_key):
                return JSONResponse(status_code=401, content={"error": "invalid API key"})

        raw = await request.body()
        if len(raw) > config.max_request_bytes:
            return JSONResponse(status_code=413, content={"error": "request body too large"})

        try:
            payload = parse_request(raw)
        except ValidationError as error:
            message = "invalid request"
            try:
                details = error.errors()
                if details:
                    message = str(details[0].get("msg", "invalid request"))
            except Exception:  # noqa: BLE001
                pass
            return JSONResponse(status_code=400, content={"error": message})

        error = validate_request(payload, max_notes=config.max_notes, max_block_seconds=config.max_block_seconds)
        if error is not None:
            return JSONResponse(status_code=400, content={"error": error})

        if service.semaphore is None:
            return JSONResponse(status_code=503, content={"error": "server is starting"}, headers={"Retry-After": "1"})
        try:
            await asyncio.wait_for(service.semaphore.acquire(), timeout=config.queue_timeout_seconds)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=503,
                content={"error": "server busy, retry shortly"},
                headers={"Retry-After": "1"},
            )

        try:
            result = await asyncio.to_thread(synthesize_sync, service, payload, payload.voice)
        except KeyError:
            available = ", ".join(bank.id for bank in service.catalog.voices()[:12])
            return JSONResponse(
                status_code=400,
                content={"error": "unknown voice '" + str(payload.voice) + "'; available: " + (available or "(none)")},
            )
        except Exception as error:  # noqa: BLE001 - 协议错误，不返回 traceback
            LOG.exception("synthesis failed (voice=%s notes=%d)", payload.voice, len(payload.notes))
            return JSONResponse(
                status_code=500,
                content={"error": "synthesis failed: " + type(error).__name__ + ": " + str(error)},
            )
        finally:
            service.semaphore.release()

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            service.synth_count += 1
            LOG.info(
                "synthesized voice=%s notes=%d block=%.2fs phones=%s elapsed=%.0fms",
                payload.voice,
                len(payload.notes),
                payload.endTime - payload.startTime,
                result.get("meta", {}).get("phones"),
                elapsed_ms,
            )
        except Exception:  # noqa: BLE001
            pass
        return JSONResponse(content=result)

    return app
