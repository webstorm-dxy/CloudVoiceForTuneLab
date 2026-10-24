"""命令行入口：python -m server [--config ...] [--backend ...] ..."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .config import BACKENDS, DEFAULT_VOICE_DIR, load_config

LOG = logging.getLogger("diffsinger-server")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m server",
        description="TuneLab CloudVoice 服务端：DiffSinger 推理（OpenUTAU 格式声库）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML 配置文件路径")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--api-key", default=None, help="要求 Authorization: Bearer <key>；不传表示不鉴权")
    parser.add_argument("--backend", choices=list(BACKENDS), default=None, help="推理后端（mps = Apple Silicon GPU）")
    parser.add_argument("--device-id", type=int, default=None, help="CUDA 设备号")
    parser.add_argument("--threads", type=int, default=None, help="ONNX Runtime 线程数，0 = 默认")
    parser.add_argument("--max-cached-sessions", type=int, default=None)
    parser.add_argument("--voice-dir", action="append", default=None, metavar="DIR", help="声库目录；可重复，指定后覆盖默认 OpenUTAU/ 目录")
    parser.add_argument("--vocoder-dir", action="append", default=None, metavar="DIR", help="内建声码器目录；可重复，追加到自动探测目录之后")
    parser.add_argument("--scan-depth", type=int, default=None)
    parser.add_argument("--language", choices=["auto", "zh", "ja", "en"], default=None)
    parser.add_argument("--pitch-mode", choices=["auto", "tunelab"], default=None,
                        help="默认音高来源：auto=有 dspitch 自动预测，tunelab=始终跟随 TuneLab 画线")
    parser.add_argument("--depth", type=float, default=None, help="浅扩散深度")
    parser.add_argument("--steps", type=int, default=None, help="扩散步数")
    parser.add_argument(
        "--audio-leading",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="响应音频向前多覆盖首音节 leading 辅音（默认开启；--no-audio-leading 严格裁到请求块）",
    )
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--list-voices", action="store_true", help="只扫描声库并打印目录 JSON，然后退出")
    parser.add_argument("--dump-config", action="store_true", help="打印生效配置后退出")
    return parser


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "host": args.host,
        "port": args.port,
        "api_key": args.api_key,
        "backend": args.backend,
        "device_id": args.device_id,
        "threads": args.threads,
        "max_cached_sessions": args.max_cached_sessions,
        "scan_depth": args.scan_depth,
        "language": args.language,
        "pitch_mode": args.pitch_mode,
        "depth": args.depth,
        "steps": args.steps,
        "include_leading": args.audio_leading,
        "max_concurrent": args.max_concurrent,
        "log_level": args.log_level,
    }
    if args.voice_dir:
        overrides["voice_dirs"] = [Path(item).expanduser() for item in args.voice_dir]
    return overrides


def _apply_vocoder_dirs(config: Any, extra: list[str] | None) -> None:
    if not extra:
        return
    for item in extra:
        path = Path(item).expanduser()
        if path not in config.vocoder_dirs:
            config.vocoder_dirs.append(path)


def list_voices_json(config: Any) -> dict[str, Any]:
    from .voicebank.discovery import VoiceCatalog

    catalog = VoiceCatalog(config.voice_dirs, config.vocoder_dirs, max_depth=config.scan_depth)
    catalog.refresh()
    return {
        "voices": [
            {
                "id": bank.id,
                "name": bank.name,
                "bank": bank.bank_id,
                "speaker": bank.selected_speaker,
                "sampleRate": bank.sample_rate,
                "phonemes": len(bank.phonemes),
                "dsdur": bank.dur is not None,
                "variance": bank.variance is not None,
                "vocoder": bank.vocoder.name if bank.vocoder else None,
                "warnings": bank.warnings,
            }
            for bank in catalog.voices()
        ],
        "voiceDirs": [str(path) for path in config.voice_dirs],
        "vocoderDirs": [str(path) for path in config.vocoder_dirs],
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, _overrides_from_args(args))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
        return 2  # parser.error 会 SystemExit，这里只是类型收窄
    _apply_vocoder_dirs(config, args.vocoder_dir)

    logging.basicConfig(
        level=getattr(logging, (config.log_level or "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.dump_config:
        print(json.dumps(config.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.list_voices:
        print(json.dumps(list_voices_json(config), ensure_ascii=False, indent=2))
        return 0

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required: " + sys.executable + " -m pip install uvicorn", file=sys.stderr)
        return 1

    from .app import create_app

    app = create_app(config)
    LOG.info("starting DiffSinger CloudVoice server on http://%s:%d (backend=%s)", config.host, config.port, config.backend)
    uvicorn.run(app, host=config.host, port=config.port, log_level=(config.log_level or "info").lower())
    return 0
