#!/usr/bin/env python3
"""Reference cloud singing-synthesis service for the TuneLab Cloud Voice plugin.

This is the production-shaped companion to tools/mock_cloud_server.py:
  - FastAPI + Pydantic v2 request validation (unknown fields tolerated);
  - optional bearer-token auth, request size limit, bounded concurrency (503 + Retry-After);
  - one blocking synthesize_block() function that is the ONLY place to plug in a real
    singing-synthesis / vocoder backend;
  - protocol-compliant JSON response: WAV base64 audio + phonemes + pitch + energy echo.

Run it:

    python3 -m venv .venv
    .venv/bin/pip install -r tools/server_example/requirements.txt
    .venv/bin/python tools/server_example/app.py --port 18767
    # with auth:
    .venv/bin/python tools/server_example/app.py --port 18767 --api-key secret-token

Then in TuneLab: Settings -> Extensions -> Cloud Voice:
    service URL = http://127.0.0.1:18767/synthesize
    API key     = secret-token (only if --api-key was passed)
The plugin loads the voice catalog from GET /voices automatically; this example serves "demo".

The demo backend renders additive-sine audio so the whole round trip is testable without a
model. Replace synthesize_block() where marked with your engine; keep the request/response
handling and the HTTP layer untouched.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import io
import logging
import math
import struct
import time
import wave
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROTOCOL = "tunelab-cloudvoice"
VERSION = 1

# Backend capability schema: the plugin turns this into property panels, editable automation lanes,
# read-only echo lanes and the auto-pitch capability declaration. Replace with your engine's parameters.
SCHEMA = {
    "pitch": {"auto": False},
    "partProperties": [
        {
            "id": "style",
            "label": "Style",
            "type": "combo",
            "default": "natural",
            "options": [
                {"value": "natural", "label": "Natural"},
                {"value": "soft", "label": "Soft"},
            ],
        }
    ],
    "noteProperties": [
        {"id": "tone", "label": "Tone", "type": "slider", "min": 0, "max": 1, "default": 0.5}
    ],
    "automations": [
        {"id": "DYN", "label": "Dynamics", "min": -1, "max": 1, "default": 0, "color": "#73ACE5"}
    ],
    "synthesizedParameters": [
        {"id": "energy", "label": "Energy", "min": 0, "max": 100, "color": "#8ADF73"}
    ],
}
MAX_NOTES = 256
MAX_BLOCK_SECONDS = 600.0
ALLOWED_SAMPLE_RATES = (16000, 22050, 24000, 32000, 44100, 48000, 96000, 192000)

log = logging.getLogger("cloudvoice")


# --------------------------------------------------------------------------------------
# Protocol models (tunelab-cloudvoice/1). Extra fields are ignored on purpose: the protocol
# is additive, and a server must not break when the client starts sending a new field.
# --------------------------------------------------------------------------------------
class CloudPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")
    t: float
    v: float


class CloudPhoneme(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str = ""
    duration: float = 0.0
    stretchWeight: float = 0.0


class CloudPhonemes(BaseModel):
    model_config = ConfigDict(extra="ignore")
    leading: list[CloudPhoneme] = Field(default_factory=list)
    body: list[CloudPhoneme] = Field(default_factory=list)
    bodyOffset: float = 0.0


class CloudNote(BaseModel):
    model_config = ConfigDict(extra="ignore")
    key: str
    start: float
    duration: float
    pitch: int
    lyric: str = ""
    rest: bool = False
    phonemes: Optional[CloudPhonemes] = None
    properties: dict[str, Any] = Field(default_factory=dict)


class CloudSynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    protocol: str = PROTOCOL
    version: int = VERSION
    voice: str = "default"
    model: Optional[str] = None
    sampleRate: int = 48000
    startTime: float = 0.0
    endTime: float = 1.0
    pitchMode: Optional[str] = None
    partProperties: dict[str, Any] = Field(default_factory=dict)
    notes: list[CloudNote] = Field(default_factory=list)
    pitch: list[CloudPoint] = Field(default_factory=list)
    parameters: dict[str, list[CloudPoint]] = Field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Tiny G2P used by the demo backend. Real backends should use their own model phonology.
# --------------------------------------------------------------------------------------
VOWELS = "aeiou"


def predict_phonemes(lyric: str) -> Optional[CloudPhonemes]:
    text = (lyric or "").strip()
    if not text or text == "-" or text.upper() == "R":
        return None
    while text and text[-1].isdigit():
        text = text[:-1]
    if not text:
        return None
    if any(ord(ch) > 0x2E80 for ch in text):
        # CJK: keep the character as the nucleus; a real backend would run pinyin G2P here.
        return CloudPhonemes(body=[CloudPhoneme(symbol=text, duration=0.28, stretchWeight=1.0)])
    lower = text.lower()
    first = next((i for i, ch in enumerate(lower) if ch in VOWELS), -1)
    last = next((i for i in range(len(lower) - 1, -1, -1) if lower[i] in VOWELS), -1)
    if first < 0 or last < first:
        return CloudPhonemes(
            body=[CloudPhoneme(symbol=ch, duration=0.07, stretchWeight=0.0) for ch in lower]
        )
    leading = [
        CloudPhoneme(symbol=ch, duration=0.07, stretchWeight=0.0) for ch in lower[:first]
    ]
    body = [
        CloudPhoneme(symbol=ch, duration=0.28, stretchWeight=1.0)
        for ch in lower[first : last + 1]
    ]
    body += [
        CloudPhoneme(symbol=ch, duration=0.07, stretchWeight=0.0) for ch in lower[last + 1 :]
    ]
    return CloudPhonemes(leading=leading, body=body, bodyOffset=len(leading) * 0.07)


# --------------------------------------------------------------------------------------
# Demo backend: additive sine + vibrato. Replace the marked part with your model.
# --------------------------------------------------------------------------------------
def synthesize_block(request: CloudSynthesisRequest) -> dict[str, Any]:
    sample_rate = request.sampleRate
    duration = max(0.05, request.endTime - request.startTime)
    frames = max(1, int(round(duration * sample_rate)))

    # PLUG IN YOUR MODEL HERE -----------------------------------------------------------
    # Inputs available: request.notes (start/duration/pitch/lyric/pinned phonemes/
    # properties), request.pitch (100 Hz resolved F0 target in semitones, global seconds),
    # request.parameters (dynamics / breathiness at 50 Hz).
    # Expected output: mono float samples at sample_rate, plus the phonemes you actually
    # sang and, optionally, the F0 contour you actually produced (see the response below).
    samples = render_demo_sine(request, frames, sample_rate)
    # -----------------------------------------------------------------------------------

    wav = encode_wav16(samples, sample_rate)
    phonemes: dict[str, Any] = {}
    for note in request.notes:
        if note.rest:
            phonemes[note.key] = {"leading": [], "body": [], "bodyOffset": 0.0}
        elif note.phonemes is not None and (note.phonemes.leading or note.phonemes.body):
            phonemes[note.key] = note.phonemes.model_dump()
        else:
            predicted = predict_phonemes(note.lyric)
            phonemes[note.key] = (
                predicted.model_dump()
                if predicted is not None
                else {"leading": [], "body": [], "bodyOffset": 0.0}
            )

    energy = [
        {"t": request.startTime + duration * i / 19, "v": 60.0} for i in range(20)
    ]
    return {
        "protocol": PROTOCOL,
        "version": VERSION,
        "sampleRate": sample_rate,
        "audioStart": request.startTime,
        "audio": {
            "format": "wav",
            "channels": 1,
            # For long blocks prefer audio.url (see docs/server-guide.md): base64 grows ~4/3
            # and a 4-minute 48 kHz float block is tens of megabytes.
            "data": base64.b64encode(wav).decode("ascii"),
        },
        "phonemes": phonemes,
        # Echo the F0 target the client asked for. A real backend should return the contour
        # it actually produced - that is what the user sees on the pitch track.
        "pitch": [point.model_dump() for point in request.pitch],
        "parameters": {"energy": energy},
        "meta": {"server": "tunelab-cloudvoice-fastapi-example", "frames": frames},
    }


def render_demo_sine(
    request: CloudSynthesisRequest, frames: int, sample_rate: int
) -> list[float]:
    samples = [0.0] * frames
    pitch = [(point.t, point.v) for point in request.pitch]
    for note in request.notes:
        if note.rest or note.duration <= 0:
            continue
        first = max(0, int(round((note.start - request.startTime) * sample_rate)))
        last = min(
            frames,
            int(round((note.start + note.duration - request.startTime) * sample_rate)),
        )
        if last <= first:
            continue
        length = last - first
        attack = max(1, int(0.02 * sample_rate))
        release = max(1, int(0.05 * sample_rate))
        phase = 0.0
        for i in range(first, last):
            t = request.startTime + i / sample_rate
            semitone = sample_pitch(pitch, t) + 0.2 * math.sin(
                2 * math.pi * 5.4 * (t - note.start)
            )
            frequency = 440.0 * math.pow(2.0, (semitone - 69.0) / 12.0)
            phase += 2 * math.pi * frequency / sample_rate
            position = i - first
            envelope = (
                position / attack
                if position < attack
                else (length - position) / release
                if position >= length - release
                else 1.0
            )
            samples[i] += 0.2 * envelope * (
                math.sin(phase) + 0.42 * math.sin(2 * phase) + 0.2 * math.sin(3 * phase)
            )
    peak = max((abs(value) for value in samples), default=0.0)
    if peak > 1e-6:
        scale = 0.72 / peak
        samples = [value * scale for value in samples]
    return samples


def sample_pitch(points: list[tuple[float, float]], time_seconds: float) -> float:
    if not points:
        return 60.0
    if time_seconds <= points[0][0]:
        return points[0][1]
    if time_seconds >= points[-1][0]:
        return points[-1][1]
    low, high = 0, len(points) - 1
    while high - low > 1:
        middle = (low + high) // 2
        if points[middle][0] <= time_seconds:
            low = middle
        else:
            high = middle
    t0, v0 = points[low]
    t1, v1 = points[high]
    return v1 if t1 <= t0 else v0 + (v1 - v0) * (time_seconds - t0) / (t1 - t0)


def encode_wav16(samples: list[float], sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(
            b"".join(
                struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767))
                for value in samples
            )
        )
    return buffer.getvalue()


# --------------------------------------------------------------------------------------
# HTTP layer: validation, auth, limits, concurrency, error semantics.
# --------------------------------------------------------------------------------------
def create_app(
    api_key: Optional[str] = None,
    max_concurrent: int = 2,
    max_request_bytes: int = 8 * 1024 * 1024,
    queue_timeout_seconds: float = 10.0,
) -> FastAPI:
    app = FastAPI(title="tunelab-cloudvoice example server")
    limiter = asyncio.Semaphore(max_concurrent)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "protocol": PROTOCOL, "version": VERSION}

    @app.get("/voices")
    async def voices() -> dict[str, Any]:
        # The plugin pulls this automatically at load time to build its voice catalog.
        # Only "id" is required; name / description / voice are optional.
        return {
            "voices": [
                {"id": "demo", "name": "Demo Voice", "description": "Reference sine backend"}
            ],
            "schema": SCHEMA,
        }

    @app.get("/schema")
    async def schema() -> dict[str, Any]:
        return SCHEMA

    @app.post("/synthesize")
    async def synthesize(request: Request) -> JSONResponse:
        started = time.perf_counter()

        if api_key is not None:
            header = request.headers.get("authorization", "")
            if not hmac.compare_digest(header, f"Bearer {api_key}"):
                return JSONResponse(status_code=401, content={"error": "invalid API key"})

        raw = await request.body()
        if len(raw) > max_request_bytes:
            return JSONResponse(status_code=413, content={"error": "request body too large"})

        try:
            payload = CloudSynthesisRequest.model_validate_json(raw)
        except ValidationError as exc:
            message = exc.errors()[0].get("msg", "validation failed")
            return JSONResponse(status_code=400, content={"error": "invalid request: " + str(message)})

        error = validate(payload)
        if error is not None:
            return JSONResponse(status_code=400, content={"error": error})

        try:
            await asyncio.wait_for(limiter.acquire(), timeout=queue_timeout_seconds)
        except asyncio.TimeoutError:
            # 503 is retried by the plugin, and Retry-After tells a human what is happening.
            return JSONResponse(
                status_code=503,
                content={"error": "server busy, retry shortly"},
                headers={"Retry-After": "1"},
            )

        try:
            # CPU/GPU work must not block the event loop: asyncio.to_thread keeps the server
            # responsive (health checks, cancellation of queued requests) while a block runs.
            result = await asyncio.to_thread(synthesize_block, payload)
        except Exception:  # noqa: BLE001 - return a protocol error, never a traceback
            log.exception("synthesis failed (voice=%s notes=%d)", payload.voice, len(payload.notes))
            return JSONResponse(status_code=500, content={"error": "internal synthesis error"})
        finally:
            limiter.release()

        elapsed_ms = (time.perf_counter() - started) * 1000
        log.info(
            "synthesized voice=%s notes=%d audioSeconds=%.2f elapsedMs=%.0f",
            payload.voice,
            len(payload.notes),
            payload.endTime - payload.startTime,
            elapsed_ms,
        )
        return JSONResponse(content=result)

    return app


def validate(request: CloudSynthesisRequest) -> Optional[str]:
    if request.protocol != PROTOCOL:
        return "unsupported protocol '" + request.protocol + "'"
    if request.version != VERSION:
        return "unsupported protocol version " + str(request.version)
    if not (math.isfinite(request.startTime) and math.isfinite(request.endTime)):
        return "startTime/endTime must be finite"
    if request.endTime <= request.startTime:
        return "endTime must be after startTime"
    if request.endTime - request.startTime > MAX_BLOCK_SECONDS:
        return "block longer than " + str(int(MAX_BLOCK_SECONDS)) + "s"
    if len(request.notes) > MAX_NOTES:
        return "too many notes (max " + str(MAX_NOTES) + ")"
    if request.sampleRate not in ALLOWED_SAMPLE_RATES:
        return "unsupported sampleRate " + str(request.sampleRate)
    for note in request.notes:
        if not note.key:
            return "note.key is required"
        if not (math.isfinite(note.start) and math.isfinite(note.duration)):
            return "note " + note.key + ": start/duration must be finite"
        if note.duration <= 0:
            return "note " + note.key + ": duration must be positive"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="tunelab-cloudvoice example server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18767)
    parser.add_argument("--api-key", default=None, help="require 'Authorization: Bearer <key>' when set")
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import uvicorn

    app = create_app(api_key=args.api_key, max_concurrent=args.max_concurrent)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()