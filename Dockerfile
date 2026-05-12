# Use official Ubuntu base image
FROM ubuntu:22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, venv, GUI libs for matplotlib/tkinter,
# OpenGL/X11 libs often needed by plotting or cv-related packages,
# and ffmpeg which is required by Whisper for audio processing
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-tk \
        tk \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first to avoid old installer issues
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install Python packages:
# - numpy, pandas, matplotlib: your scientific stack
# - torch: needed by Whisper
# - openai-whisper: speech-to-text package imported as "whisper"
RUN python3 -m pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    torch \
    openai-whisper

# Set working directory inside container
WORKDIR /work

# Default command
# CMD ["python3"]