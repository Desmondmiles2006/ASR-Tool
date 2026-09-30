"""Command-line interface.

    python -m asr_tool transcribe <files or folders> [options]
    python -m asr_tool record --seconds 5 [options]
    python -m asr_tool evaluate <folder with .wav + .txt pairs> [options]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from . import __version__
from .audio import collect_audio_files, record_microphone, save_wav
from .formats import OUTPUT_FORMATS, save
from .metrics import character_error_rate, word_error_rate
from .transcriber import AVAILABLE_MODELS, Transcriber, TranscriptionResult


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-m", "--model", default="base",
                        help=f"Whisper model size: {', '.join(AVAILABLE_MODELS)} (default: base)")
    parser.add_argument("-l", "--language", default=None,
                        help="Language code, e.g. en, ta, hi. Omit to auto-detect.")
    parser.add_argument("--task", choices=("transcribe", "translate"), default="transcribe",
                        help="'translate' converts any spoken language to English text")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable voice-activity detection (silence removal)")


def _add_output_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-f", "--format", nargs="+", default=["txt"], choices=OUTPUT_FORMATS,
                        help="One or more output formats (default: txt)")
    parser.add_argument("-o", "--output-dir", default="outputs")
    parser.add_argument("--timestamps", action="store_true",
                        help="Print per-segment timestamps to the console")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asr_tool",
        description="Automatic Speech Recognition tool built on OpenAI Whisper (faster-whisper).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_file = sub.add_parser("transcribe", help="Transcribe audio/video files or folders")
    p_file.add_argument("inputs", nargs="+", help="Audio files and/or folders containing audio")
    _add_model_options(p_file)
    _add_output_options(p_file)

    p_mic = sub.add_parser("record", help="Record from the microphone and transcribe")
    p_mic.add_argument("-s", "--seconds", type=float, default=5.0)
    _add_model_options(p_mic)
    _add_output_options(p_mic)

    p_eval = sub.add_parser("evaluate",
                            help="Measure WER/CER on audio files that have a matching .txt reference")
    p_eval.add_argument("folder", help="Folder containing e.g. clip.wav + clip.txt pairs")
    _add_model_options(p_eval)

    return parser


def _load_model(args: argparse.Namespace) -> Transcriber:
    print(f"Loading Whisper '{args.model}' model (first run downloads it)...")
    return Transcriber(args.model, device=args.device)


def _run(model: Transcriber, audio, args: argparse.Namespace) -> TranscriptionResult:
    return model.transcribe(
        audio,
        language=args.language,
        task=args.task,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
    )


def _report(name: str, result: TranscriptionResult, show_timestamps: bool) -> None:
    print(f"\n=== {name} ===")
    print(f"Language: {result.language} ({result.language_probability:.0%} confidence) | "
          f"Audio: {result.duration:.1f}s | Processed in {result.processing_time:.1f}s "
          f"(RTF {result.real_time_factor:.2f})")
    if show_timestamps:
        for seg in result.segments:
            print(f"[{seg.start:7.2f} -> {seg.end:7.2f}] {seg.text}")
    else:
        print(result.text or "(no speech detected)")


def cmd_transcribe(args: argparse.Namespace) -> int:
    files = collect_audio_files(args.inputs)
    missing = [f for f in files if not f.is_file()]
    if missing:
        for f in missing:
            print(f"error: file not found: {f}", file=sys.stderr)
        return 1
    if not files:
        print("error: no audio files found", file=sys.stderr)
        return 1

    model = _load_model(args)
    for path in files:
        result = _run(model, path, args)
        _report(path.name, result, args.timestamps)
        for fmt in args.format:
            print(f"  saved -> {save(result, path, args.output_dir, fmt)}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    model = _load_model(args)
    audio = record_microphone(args.seconds)
    wav_path = Path(args.output_dir) / f"recording_{datetime.now():%Y%m%d_%H%M%S}.wav"
    save_wav(audio, wav_path)
    print(f"  audio saved -> {wav_path}")

    result = _run(model, audio, args)
    _report(wav_path.name, result, args.timestamps)
    for fmt in args.format:
        print(f"  saved -> {save(result, wav_path, args.output_dir, fmt)}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    pairs = [(a, a.with_suffix(".txt")) for a in collect_audio_files([args.folder])
             if a.with_suffix(".txt").is_file()]
    if not pairs:
        print(f"error: no audio files with matching .txt references in {args.folder}", file=sys.stderr)
        return 1

    model = _load_model(args)
    total_wer = total_cer = 0.0
    print(f"\n{'File':<20} {'WER':>7} {'CER':>7}  Hypothesis")
    print("-" * 78)
    for audio_path, ref_path in pairs:
        reference = ref_path.read_text(encoding="utf-8-sig").strip()
        result = _run(model, audio_path, args)
        wer = word_error_rate(reference, result.text)
        cer = character_error_rate(reference, result.text)
        total_wer += wer
        total_cer += cer
        print(f"{audio_path.name:<20} {wer:>7.2%} {cer:>7.2%}  {result.text}")
    print("-" * 78)
    n = len(pairs)
    print(f"{'Average (' + str(n) + ' files)':<20} {total_wer / n:>7.2%} {total_cer / n:>7.2%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"transcribe": cmd_transcribe, "record": cmd_record, "evaluate": cmd_evaluate}
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
