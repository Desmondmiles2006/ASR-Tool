"""Core speech recognition engine.

Wraps faster-whisper (a CTranslate2 re-implementation of OpenAI Whisper) so the
rest of the project only deals with plain Python objects.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

AVAILABLE_MODELS = ("tiny", "base", "small", "medium", "large-v3", "turbo")


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float
    processing_time: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def real_time_factor(self) -> float:
        """Processing time divided by audio length (< 1 means faster than real time)."""
        return self.processing_time / self.duration if self.duration else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["real_time_factor"] = round(self.real_time_factor, 3)
        return data


class Transcriber:
    """Loads a Whisper model once and transcribes any number of audio inputs."""

    # Set once a CUDA inference fails. A second CUDA attempt in the same process
    # can hang, so later "auto" instances go straight to CPU.
    _cuda_unusable = False

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
    ):
        if device == "auto" and Transcriber._cuda_unusable:
            device = "cpu"
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = self._load(device)

    def _load(self, device: str):
        # Imported here so that `--help` and the unit tests stay fast.
        from faster_whisper import WhisperModel

        compute_type = self.compute_type
        if compute_type == "default" and device == "cpu":
            compute_type = "int8"  # much faster than float32 on CPU with negligible accuracy loss
        return WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def _fall_back_to_cpu(self, error: RuntimeError) -> bool:
        """With device="auto", a GPU may be detected even though the CUDA runtime
        libraries (cuBLAS/cuDNN) are not installed. The error only surfaces on the
        first inference, so switch to CPU at that point instead of crashing."""
        message = str(error).lower()
        if self.device != "auto" or not any(k in message for k in ("cublas", "cudnn", "cuda")):
            return False
        print("CUDA libraries unavailable, falling back to CPU.")
        Transcriber._cuda_unusable = True
        self.device = "cpu"
        self.model = self._load("cpu")
        return True

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        vad_filter: bool = True,
        word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe a file path or a 16 kHz mono float32 numpy array.

        language: ISO code such as "en" or "ta"; None lets Whisper detect it.
        task: "transcribe" keeps the spoken language, "translate" outputs English.
        """
        if isinstance(audio, Path):
            audio = str(audio)
        if isinstance(audio, str) and not Path(audio).is_file():
            raise FileNotFoundError(f"Audio file not found: {audio}")

        options = dict(
            language=language,
            task=task,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )
        started = time.perf_counter()
        try:
            segments, info = self._decode(audio, options)
        except RuntimeError as error:
            if not self._fall_back_to_cpu(error):
                raise
            started = time.perf_counter()
            segments, info = self._decode(audio, options)
        elapsed = time.perf_counter() - started

        return TranscriptionResult(
            text=" ".join(s.text for s in segments).strip(),
            language=info.language,
            language_probability=round(info.language_probability, 3),
            duration=round(info.duration, 2),
            processing_time=round(elapsed, 2),
            segments=segments,
        )

    def _decode(self, audio, options: dict):
        raw_segments, info = self.model.transcribe(audio, **options)
        # faster-whisper returns a lazy generator; decoding happens while we iterate.
        segments = [
            Segment(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
            for s in raw_segments
        ]
        return segments, info
