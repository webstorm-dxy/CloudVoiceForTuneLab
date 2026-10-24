#!/usr/bin/env python3
"""Mock cloud singing-synthesis service for the TuneLab Cloud Voice plugin.

Speaks the tunelab-cloudvoice/1 HTTP protocol (see docs/cloud-protocol.md):

    POST /synthesize   request JSON -> response JSON with base64 WAV audio + phonemes/pitch/parameters
    GET  /voices       a sample voice catalog (plus the capability schema)
    GET  /schema       the backend capability schema (properties / automations / echo tracks / pitch)

It is deliberately small and dependency-free: use it to verify the plugin's cloud path end to end
without any real service. Run it, set the plugin's Service URL to
http://127.0.0.1:18765/synthesize, then synthesize - the plugin pulls GET /voices automatically.

    python3 tools/mock_cloud_server.py --port 18765 --log .sandbox/server-requests.jsonl
"""
import argparse
import base64
import io
import json
import math
import struct
import sys
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 18765

# Minimal backend capability schema: the plugin turns this into part/note property panels,
# editable automation tracks, echo tracks and the auto-pitch capability declaration.
SCHEMA = {
    "protocol": "tunelab-cloudvoice",
    "version": 1,
    "engine": "mock-cloud",
    "pitch": {"auto": True},
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


def render_wav(request, sample_rate, duration):
    """Render one sine per note (rests are silence) into a mono 16-bit WAV."""
    total = max(1, int(round(duration * sample_rate)))
    samples = [0.0] * total
    start_time = float(request.get("startTime", 0.0))
    for note in request.get("notes", []):
        if note.get("rest"):
            continue
        note_start = float(note.get("start", start_time))
        note_duration = max(0.02, float(note.get("duration", 0.1)))
        pitch = int(note.get("pitch", 60))
        frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
        first = max(0, int(round((note_start - start_time) * sample_rate)))
        last = min(total, int(round((note_start - start_time + note_duration) * sample_rate)))
        length = max(1, last - first)
        attack = max(1, int(0.02 * sample_rate))
        release = max(1, int(0.05 * sample_rate))
        for i in range(first, last):
            position = i - first
            envelope = 1.0
            if position < attack:
                envelope = position / attack
            elif position >= length - release:
                envelope = (length - position) / release
            phase = 2.0 * math.pi * frequency * (i / sample_rate)
            samples[i] += 0.25 * envelope * math.sin(phase)

    peak = max((abs(value) for value in samples), default=0.0)
    scale = (0.72 / peak) if peak > 1e-9 else 0.0
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, value * scale)) * 32767)) for value in samples
        ))
    return buffer.getvalue(), samples, scale


def build_response(request):
    sample_rate = int(request.get("sampleRate", 48000) or 48000)
    start_time = float(request.get("startTime", 0.0))
    end_time = float(request.get("endTime", start_time + 1.0))
    duration = max(0.05, end_time - start_time)
    wav_bytes, samples, scale = render_wav(request, sample_rate, duration)

    # The server answers with its OWN phonemes on purpose (symbols prefixed srv-) so a client
    # that blindly falls back to local G2P cannot hide a broken response path.
    phonemes = {}
    for index, note in enumerate(request.get("notes", [])):
        key = note.get("key") or f"note-{index}"
        if note.get("rest"):
            phonemes[key] = {"leading": [], "body": [], "bodyOffset": 0.0}
        else:
            phonemes[key] = {
                "leading": [],
                "body": [{"symbol": "srv-" + str(note.get("lyric", "a")), "duration": 0.3, "stretchWeight": 1.0}],
                "bodyOffset": 0.0,
            }

    # Echo the requested pitch curve (already resolved by the client) so the pitch echo track is real.
    pitch = request.get("pitch") or []
    energy = []
    points = max(2, int(duration * 20))
    for i in range(points):
        energy.append({"t": start_time + duration * i / (points - 1), "v": 66.0})

    return {
        "protocol": "tunelab-cloudvoice",
        "version": 1,
        "sampleRate": sample_rate,
        "audioStart": start_time,
        "audio": {
            "format": "wav",
            "channels": 1,
            "data": base64.b64encode(wav_bytes).decode("ascii"),
        },
        "phonemes": phonemes,
        "pitch": pitch,
        "parameters": {"energy": energy},
        "meta": {"server": "tunelab-mock-cloud", "notes": len(request.get("notes", [])), "peakScale": scale},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "TuneLabMockCloud/1.0"
    request_log = None

    def log_message(self, fmt, *args):  # keep stdout clean for the test harness
        sys.stderr.write("[mock-cloud] " + (fmt % args) + "\n")

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/voices", "/v1/voices"):
            self._send_json(200, {
                "voices": [
                    {"id": "test-voice", "name": "Test Voice", "description": "Mock cloud voice from tools/mock_cloud_server.py"},
                    {"id": "test-voice-b", "name": "Test Voice B", "description": "Another mock cloud voice"},
                ],
                "schema": SCHEMA,
            })
            return
        if self.path.rstrip("/") in ("/schema", "/v1/schema"):
            self._send_json(200, SCHEMA)
            return
        if self.path.rstrip("/") in ("/health", "/"):
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/synthesize", "/v1/synthesize"):
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw.decode("utf-8"))
        except Exception as ex:  # noqa: BLE001 - mock server: report any parse failure
            self._send_json(400, {"error": f"invalid JSON: {ex}"})
            return

        if Handler.request_log is not None:
            with open(Handler.request_log, "a", encoding="utf-8") as log:
                log.write(json.dumps({"path": self.path, "headers": dict(self.headers), "request": request}) + "\n")

        try:
            response = build_response(request)
        except Exception as ex:  # noqa: BLE001 - mock server: report any render failure
            self._send_json(500, {"error": f"mock render failed: {ex}"})
            return
        self._send_json(200, response)


def main():
    parser = argparse.ArgumentParser(description="Mock tunelab-cloudvoice/1 service")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--log", default="", help="append each request as one JSON line to this file")
    args = parser.parse_args()
    if args.log:
        Handler.request_log = args.log
        open(args.log, "w", encoding="utf-8").close()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock cloud listening on http://{args.host}:{args.port} (POST /synthesize, GET /voices)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()