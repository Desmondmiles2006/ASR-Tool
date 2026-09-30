"""Microphone recording and WAV helpers."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000  # Whisper models expect 16 kHz mono audio

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".webm", ".mp4", ".mkv"}


def record_microphone(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record from the default input device and return float32 samples in [-1, 1]."""
    import sounddevice as sd

    print(f"Recording for {seconds:g} seconds... speak now.")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording finished.")
    return audio.flatten()


def save_wav(audio: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Save float32 samples as a 16-bit PCM mono WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


def collect_audio_files(inputs: list[str]) -> list[Path]:
    """Expand the CLI inputs: files are kept as-is, folders are searched for audio files."""
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS))
        else:
            files.append(path)
    return files
