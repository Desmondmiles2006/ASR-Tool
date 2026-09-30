"""End-to-end tests that run a real Whisper model on the sample clips.

The 'tiny' model (~75 MB) is downloaded on first run. Skip these with:
    pytest -m "not integration"
"""

from pathlib import Path

import numpy as np
import pytest

from asr_tool import Transcriber, word_error_rate
from asr_tool.cli import main

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
CLIPS = sorted(SAMPLES.glob("*.wav"))

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def model():
    return Transcriber("tiny")


@pytest.mark.parametrize("clip", CLIPS, ids=lambda p: p.name)
def test_sample_accuracy(model, clip):
    reference = clip.with_suffix(".txt").read_text(encoding="utf-8-sig")
    result = model.transcribe(clip, language="en")
    # Even the smallest model should get most words right on clean speech.
    assert word_error_rate(reference, result.text) <= 0.25, result.text


def test_language_detection_and_metadata(model):
    result = model.transcribe(SAMPLES / "sample1.wav")
    assert result.language == "en"
    assert 3.0 < result.duration < 6.0
    assert result.segments and result.segments[0].start >= 0
    assert result.segments[-1].end <= result.duration + 0.5


def test_numpy_input(model):
    result = model.transcribe(np.zeros(16_000 * 2, dtype=np.float32), language="en")
    assert result.duration == pytest.approx(2.0)


def test_missing_file(model):
    with pytest.raises(FileNotFoundError):
        model.transcribe("does_not_exist.wav")


def test_cli_transcribe_writes_outputs(tmp_path):
    code = main(["transcribe", str(SAMPLES / "sample2.wav"), "-m", "tiny",
                 "-f", "txt", "srt", "-o", str(tmp_path)])
    assert code == 0
    assert "fox" in (tmp_path / "sample2.txt").read_text(encoding="utf-8").lower()
    assert (tmp_path / "sample2.srt").exists()


def test_cli_missing_file_returns_error(tmp_path):
    assert main(["transcribe", str(tmp_path / "nope.wav")]) == 1
