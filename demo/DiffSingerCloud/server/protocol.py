"""tunelab-cloudvoice/1 线上协议模型（请求校验 + 响应构造辅助）。

字段字典见 docs/cloud-protocol.md；参考实现见 tools/server_example/app.py。
"""
from __future__ import annotations

import math
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PROTOCOL = "tunelab-cloudvoice"
VERSION = 1

# 服务端可接受的上限（可用 ServerConfig 覆盖后传给 validate_request）
MAX_NOTES = 512
MAX_BLOCK_SECONDS = 600.0
ALLOWED_SAMPLE_RATES = (16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000)


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

    def flatten(self) -> list[str]:
        """展开成符号序列（钉死音素时用）。"""
        return [item.symbol for item in self.leading] + [item.symbol for item in self.body if item.symbol]


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


def parse_request(raw: bytes) -> CloudSynthesisRequest:
    """解析请求体；失败抛 pydantic.ValidationError（HTTP 层转 400）。"""
    return CloudSynthesisRequest.model_validate_json(raw)


def validate_request(
    request: CloudSynthesisRequest,
    *,
    max_notes: int = MAX_NOTES,
    max_block_seconds: float = MAX_BLOCK_SECONDS,
) -> Optional[str]:
    """返回错误文案；None 表示通过。"""
    if request.protocol != PROTOCOL:
        return "unsupported protocol '" + str(request.protocol) + "'"
    if request.version != VERSION:
        return "unsupported protocol version " + str(request.version)
    if not (math.isfinite(request.startTime) and math.isfinite(request.endTime)):
        return "startTime/endTime must be finite"
    if request.endTime <= request.startTime:
        return "endTime must be after startTime"
    if request.endTime - request.startTime > max_block_seconds:
        return "block longer than " + str(int(max_block_seconds)) + "s"
    if len(request.notes) > max_notes:
        return "too many notes (max " + str(max_notes) + ")"
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


def error_response(message: str) -> dict[str, Any]:
    return {"error": message}


def phoneme_payload(
    symbol: str,
    duration: float,
    stretch_weight: float,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "duration": float(duration),
        "stretchWeight": float(stretch_weight),
    }
