"""Writers for the supported output formats: txt, srt, vtt and json."""

from __future__ import annotations

import json
from pathlib import Path

from .transcriber import TranscriptionResult

OUTPUT_FORMATS = ("txt", "srt", "vtt", "json")


def format_timestamp(seconds: float, decimal_marker: str = ",") -> str:
    """0 -> "00:00:00,000". SRT uses a comma before milliseconds, VTT a dot."""
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{ms:03d}"


def to_txt(result: TranscriptionResult) -> str:
    return result.text + "\n"


def to_srt(result: TranscriptionResult) -> str:
    blocks = []
    for index, seg in enumerate(result.segments, start=1):
        blocks.append(
            f"{index}\n"
            f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n"
            f"{seg.text}\n"
        )
    return "\n".join(blocks)


def to_vtt(result: TranscriptionResult) -> str:
    lines = ["WEBVTT", ""]
    for seg in result.segments:
        lines.append(f"{format_timestamp(seg.start, '.')} --> {format_timestamp(seg.end, '.')}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def to_json(result: TranscriptionResult) -> str:
    return json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n"


_WRITERS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json}


def render(result: TranscriptionResult, fmt: str) -> str:
    try:
        return _WRITERS[fmt](result)
    except KeyError:
        raise ValueError(f"Unknown format '{fmt}'. Choose from {OUTPUT_FORMATS}") from None


def save(result: TranscriptionResult, audio_path: str | Path, output_dir: str | Path, fmt: str) -> Path:
    """Write the transcript next to the others as <audio name>.<fmt>."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{Path(audio_path).stem}.{fmt}"
    out_path.write_text(render(result, fmt), encoding="utf-8")
    return out_path
