FROM astral/uv:python3.10-bookworm-slim

# Install system dependencies required for PyAudio and ALSA sound drivers, removing bloat from /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    gcc \
    g++ \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/* 


COPY . /app

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

RUN uv sync --locked

RUN ["uv", "run", "python", "-u", "-c", "import openwakeword; openwakeword.utils.download_models()"]

CMD ["uv", "run", "main.py"]