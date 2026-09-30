# ASR Tool: Automatic Speech Recognition

A speech-to-text tool written in Python. It takes an audio file (or live microphone input) and produces a transcript, with timestamps, subtitle files and an accuracy score if you give it the correct text to compare against. It runs fully offline once the model has been downloaded.

| | |
|---|---|
| Name | Aswin Kannaa R |
| Register No. | RA2311003050048 |
| Class | B.Tech IV CSE-A |
| Assignment | Task 2: ASR Tool Implementation |

## What it does

- Transcribes WAV, MP3, FLAC, OGG, M4A and other audio files, and the audio track of MP4/MKV videos. You can pass a whole folder at once.
- Records from the microphone and transcribes the recording.
- Detects the spoken language automatically (Whisper supports 99 languages, including Tamil, Hindi, Telugu and Malayalam), or you can fix the language with `-l`.
- Can translate speech in any supported language straight into English text (`--task translate`).
- Saves output as plain text, SRT and VTT subtitles, or JSON with per-segment timestamps.
- Measures accuracy with Word Error Rate (WER) and Character Error Rate (CER). I wrote the WER code myself using Levenshtein edit distance, so there is no extra library for it.
- Has a browser interface (Gradio) as well as the command line.
- Skips silent parts using voice activity detection, which stops Whisper from inventing text during long pauses.

## How it works

The recogniser is OpenAI's Whisper, an encoder-decoder transformer trained on 680,000 hours of labelled audio. The audio is resampled to 16 kHz mono and converted into a log-Mel spectrogram. The encoder turns the spectrogram into a sequence of hidden states, and the decoder generates text tokens from them one at a time using beam search (beam size 5 by default).

I run Whisper through [faster-whisper](https://github.com/SYSTRAN/faster-whisper), which reimplements it on the CTranslate2 inference engine. It gives the same output as the original model, runs up to four times faster and uses less memory. On CPU the tool loads the model with 8-bit integer weights, which is what makes it usable on a normal laptop without a GPU.

```
audio file / microphone
        |
        v
decode + resample to 16 kHz mono (PyAV)
        |
        v
voice activity detection, drop silence (Silero VAD)
        |
        v
log-Mel spectrogram -> Whisper encoder -> Whisper decoder (beam search)
        |
        v
segments with start/end times -> txt / srt / vtt / json, WER/CER
```

## Tools and technologies used

| Tool | Used for |
|---|---|
| Python 3.10+ (tested on 3.14) | Programming language |
| OpenAI Whisper | The speech recognition model |
| faster-whisper + CTranslate2 | Running Whisper quickly on CPU/GPU |
| PyAV (FFmpeg bindings) | Reading audio and video formats, no separate FFmpeg install needed |
| Silero VAD (bundled with faster-whisper) | Removing silence |
| NumPy | Audio sample arrays |
| sounddevice | Microphone recording |
| Gradio | Web interface |
| pytest | Unit and integration tests |
| argparse | Command-line interface |
| Git and GitHub | Version control and hosting |

## Project structure

```
.
├── asr_tool/
│   ├── __init__.py
│   ├── __main__.py       # lets you run "python -m asr_tool"
│   ├── cli.py            # transcribe / record / evaluate commands
│   ├── transcriber.py    # wraps the Whisper model, CPU fallback
│   ├── formats.py        # txt, srt, vtt and json writers
│   ├── metrics.py        # WER and CER (edit distance)
│   └── audio.py          # microphone recording, WAV saving, folder scanning
├── app.py                # Gradio web interface
├── samples/              # test clips (.wav) with their correct transcripts (.txt)
├── tests/                # pytest test suite
├── requirements.txt
└── pytest.ini
```

## Installation

You need Python 3.10 or newer and Git. FFmpeg is not required.

```bash
git clone https://github.com/Desmondmiles2006/ASR-Tool.git
cd ASR-Tool
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The first time you use a model size, it is downloaded from Hugging Face and cached (`tiny` is about 75 MB, `base` about 145 MB, `small` about 480 MB). After that everything works offline.

## How to run

### Transcribe files

```bash
python -m asr_tool transcribe samples/sample1.wav
```

Output:

```
Loading Whisper 'base' model (first run downloads it)...

=== sample1.wav ===
Language: en (99% confidence) | Audio: 4.4s | Processed in 1.2s (RTF 0.28)
Speech recognition converts spoken language into written text.
  saved -> outputs\sample1.txt
```

More examples:

```bash
# every audio file in a folder, saved as text and SRT subtitles, with timestamps printed
python -m asr_tool transcribe samples -f txt srt --timestamps

# a Tamil recording with the larger "small" model
python -m asr_tool transcribe lecture.mp3 -m small -l ta

# speech in any language, output in English
python -m asr_tool transcribe interview.m4a --task translate

# all output formats into a folder of your choice
python -m asr_tool transcribe talk.mp4 -f txt srt vtt json -o results
```

### Record from the microphone

```bash
python -m asr_tool record --seconds 10
```

The recording is saved as a WAV file in `outputs/` next to its transcript.

### Measure accuracy

Put each audio file next to a `.txt` file with the same name holding the correct transcript, then:

```bash
python -m asr_tool evaluate samples
```

```
File                     WER     CER  Hypothesis
------------------------------------------------------------------------------
sample1.wav            0.00%   0.00%  Speech recognition converts spoken language into written text.
sample2.wav            0.00%   0.00%  The quick brown fox jumps over the lazy dog near the river bank.
sample3.wav            8.33%   2.22%  Automatic speech recognition is used in voice assistance, captioning and dictation software.
------------------------------------------------------------------------------
Average (3 files)      2.78%   0.74%
```

### Web interface

```bash
python app.py
```

Open http://127.0.0.1:7860 in a browser. You can upload a file or record with the microphone, pick the model and language, and paste a reference sentence to get the WER. The transcript can be downloaded as txt, srt or json.

### Command options

| Option | Meaning | Default |
|---|---|---|
| `-m`, `--model` | `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo` | `base` |
| `-l`, `--language` | Language code such as `en`, `ta`, `hi` | auto-detect |
| `--task` | `transcribe` or `translate` (to English) | `transcribe` |
| `-f`, `--format` | One or more of `txt`, `srt`, `vtt`, `json` | `txt` |
| `-o`, `--output-dir` | Where output files go | `outputs` |
| `--timestamps` | Print each segment with its start and end time | off |
| `--device` | `auto`, `cpu` or `cuda` | `auto` |
| `--beam-size` | Beam width for decoding | `5` |
| `--no-vad` | Keep silent parts instead of skipping them | off |
| `-s`, `--seconds` | Recording length (record command only) | `5` |

Run `python -m asr_tool <command> --help` for the full list.

## Testing

```bash
pytest -v
```

There are 29 tests. The unit tests check the edit distance, WER and CER calculations against hand-worked examples, text normalisation, subtitle timestamp formatting and each output format. The integration tests load the real `tiny` model and check that:

- each sample clip is transcribed with a WER of 25% or less,
- English is detected correctly and segment times fall inside the clip,
- NumPy arrays work as input as well as files,
- a missing file raises an error instead of failing silently,
- the CLI writes the requested output files.

To run only the fast unit tests (no model download):

```bash
pytest -m "not integration"
```

All 29 tests pass on Windows 11 with Python 3.14 in about 7 seconds once the model is cached.

## Results

The three sample clips in `samples/` were made with the Windows text-to-speech engine, so the exact spoken words are known. Both the `tiny` and `base` models got two of the three clips word-perfect. On the third they wrote "voice assistance" instead of "voice assistants", which sound almost identical, giving an average WER of 2.78%.

Speed on a CPU (int8, laptop):

| Clip | Length | `base` processing time | Real-time factor |
|---|---|---|---|
| sample1.wav | 4.4 s | 1.2 s | 0.28 |
| sample3.wav | 6.3 s | 1.2 s | 0.20 |

A real-time factor below 1 means the audio is transcribed faster than it plays. Synthetic speech is clean, so expect higher error rates on noisy recordings, strong accents or overlapping speakers. The `small` or `turbo` models help in those cases, at the cost of speed.

## Notes and known issues

- If an NVIDIA GPU is present but the CUDA 12 libraries (cuBLAS, cuDNN) are not installed, the GPU fails on first use. The tool notices this, prints `CUDA libraries unavailable, falling back to CPU.` and carries on with the CPU.
- `requirements.txt` pins PyAV below version 17 because faster-whisper 1.2 passes an argument that PyAV 17 removed.
- On Windows, Hugging Face may warn about symlinks when it caches the model. The warning is harmless; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to hide it.

## References

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (the Whisper paper), 2022. https://arxiv.org/abs/2212.04356
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- CTranslate2: https://github.com/OpenNMT/CTranslate2
