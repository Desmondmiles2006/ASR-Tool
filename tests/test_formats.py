import json

import numpy as np
import pytest

from asr_tool.audio import collect_audio_files, save_wav
from asr_tool.formats import format_timestamp, render, save
from asr_tool.transcriber import Segment, TranscriptionResult


@pytest.fixture
def result():
    return TranscriptionResult(
        text="Hello there. General Kenobi.",
        language="en",
        language_probability=0.98,
        duration=4.0,
        processing_time=1.0,
        segments=[Segment(0.0, 1.5, "Hello there."), Segment(1.5, 3725.25, "General Kenobi.")],
    )


def test_format_timestamp():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3725.25) == "01:02:05,250"
    assert format_timestamp(61.5, ".") == "00:01:01.500"


def test_txt(result):
    assert render(result, "txt") == "Hello there. General Kenobi.\n"


def test_srt(result):
    assert render(result, "srt") == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n"
        "2\n00:00:01,500 --> 01:02:05,250\nGeneral Kenobi.\n"
    )


def test_vtt(result):
    out = render(result, "vtt")
    assert out.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello there.\n")


def test_json(result):
    data = json.loads(render(result, "json"))
    assert data["language"] == "en"
    assert data["real_time_factor"] == 0.25
    assert data["segments"][1] == {"start": 1.5, "end": 3725.25, "text": "General Kenobi."}


def test_unknown_format(result):
    with pytest.raises(ValueError):
        render(result, "docx")


def test_save_uses_audio_name(result, tmp_path):
    path = save(result, "some/dir/lecture.mp3", tmp_path / "out", "srt")
    assert path == tmp_path / "out" / "lecture.srt"
    assert path.read_text(encoding="utf-8").startswith("1\n")


def test_save_wav_and_collect(tmp_path):
    save_wav(np.zeros(1600, dtype=np.float32), tmp_path / "a.wav")
    (tmp_path / "notes.txt").write_text("not audio")
    assert collect_audio_files([str(tmp_path)]) == [tmp_path / "a.wav"]
