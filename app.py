"""Browser interface for the ASR tool.

Run:  python app.py   then open http://127.0.0.1:7860
Upload an audio file or record from the microphone and get the transcript,
subtitles and (optionally) the WER against a reference sentence.
"""

from __future__ import annotations

import tempfile

import gradio as gr

from asr_tool import Transcriber, character_error_rate, word_error_rate
from asr_tool.formats import save
from asr_tool.transcriber import AVAILABLE_MODELS

LANGUAGES = {
    "Auto-detect": None, "English": "en", "Tamil": "ta", "Hindi": "hi", "Telugu": "te",
    "Malayalam": "ml", "Kannada": "kn", "French": "fr", "German": "de", "Spanish": "es",
}


_models: dict[str, Transcriber] = {}


def get_model(size: str) -> Transcriber:
    # Loading a model takes a few seconds, so keep each size once it is loaded.
    if size not in _models:
        _models[size] = Transcriber(size)
    return _models[size]


def transcribe(audio_path, model_size, language_name, translate, reference):
    if not audio_path:
        raise gr.Error("Upload an audio file or record something first.")

    result = get_model(model_size).transcribe(
        audio_path,
        language=LANGUAGES[language_name],
        task="translate" if translate else "transcribe",
    )

    timeline = "\n".join(f"[{s.start:6.2f}s - {s.end:6.2f}s]  {s.text}" for s in result.segments)
    stats = (
        f"**Language:** {result.language} ({result.language_probability:.0%})  |  "
        f"**Audio:** {result.duration:.1f}s  |  **Processing:** {result.processing_time:.1f}s  |  "
        f"**Real-time factor:** {result.real_time_factor:.2f}"
    )
    if reference and reference.strip():
        stats += (
            f"\n\n**WER:** {word_error_rate(reference, result.text):.2%}  |  "
            f"**CER:** {character_error_rate(reference, result.text):.2%}"
        )

    out_dir = tempfile.mkdtemp(prefix="asr_")
    files = [str(save(result, audio_path, out_dir, fmt)) for fmt in ("txt", "srt", "json")]
    return result.text, timeline, stats, files


with gr.Blocks(title="ASR Tool") as demo:
    gr.Markdown("# Automatic Speech Recognition\nWhisper-based speech-to-text. "
                "Upload a file or use your microphone.")
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
            model = gr.Dropdown(list(AVAILABLE_MODELS), value="base", label="Model size")
            language = gr.Dropdown(list(LANGUAGES), value="Auto-detect", label="Spoken language")
            translate = gr.Checkbox(label="Translate to English")
            reference = gr.Textbox(label="Reference text (optional, for WER)", lines=2)
            button = gr.Button("Transcribe", variant="primary")
        with gr.Column():
            text = gr.Textbox(label="Transcript", lines=6)
            stats = gr.Markdown()
            timeline = gr.Textbox(label="Segments with timestamps", lines=8)
            files = gr.File(label="Downloads (txt / srt / json)", file_count="multiple")

    button.click(transcribe, [audio, model, language, translate, reference],
                 [text, timeline, stats, files])
    gr.Examples([["samples/sample1.wav"], ["samples/sample2.wav"], ["samples/sample3.wav"]],
                inputs=[audio])


if __name__ == "__main__":
    demo.launch()
