"""ASR Tool - speech-to-text using OpenAI Whisper models via faster-whisper."""

from .transcriber import Segment, Transcriber, TranscriptionResult
from .metrics import word_error_rate, character_error_rate

__all__ = [
    "Segment",
    "Transcriber",
    "TranscriptionResult",
    "word_error_rate",
    "character_error_rate",
]

__version__ = "1.0.0"
