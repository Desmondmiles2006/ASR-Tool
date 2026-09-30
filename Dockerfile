# Container for the web interface. Hugging Face Spaces builds this automatically;
# it also works locally:  docker build -t asr-tool .  &&  docker run -p 7860:7860 asr-tool
FROM python:3.12-slim

# Spaces run containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the default model during the build so the first visitor doesn't wait for it
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
