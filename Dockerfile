FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for OpenCV, ffmpeg, and builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    curl \
    ca-certificates \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU wheels compatible with Python 3.9
RUN pip install --upgrade pip && \
    pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml poetry.lock uv.lock README.md /app/
COPY sn_gamestate /app/sn_gamestate
COPY plugins /app/plugins
COPY pretrained_models /app/pretrained_models

RUN pip install -e ./plugins/calibration
RUN pip install -e .
RUN pip install openmim==0.3.9
RUN mim install mmcv==2.0.1

# Patch TrackLab downloader to use gamestate-2024 (public dataset name)
RUN python - <<'PY'
import pathlib
path = pathlib.Path('/usr/local/lib/python3.9/site-packages/tracklab/wrappers/dataset/soccernet/soccernet_game_state.py')
text = path.read_text()
text = text.replace('gamestate-2025', 'gamestate-2024')
path.write_text(text)
print('Patched', path)
PY

# Patch TrackLab search path plugin for newer importlib.metadata EntryPoint
RUN python - <<'PY'
import pathlib
path = pathlib.Path('/usr/local/lib/python3.9/site-packages/hydra_plugins/tracklab_searchpath_plugin/tracklab_searchpath_plugin.py')
text = path.read_text()
text = text.replace("            m = tracklab_plugin.dist\n", "")
path.write_text(text)
print('Patched', path)
PY

ENV PYTHONPATH=/app

ENTRYPOINT ["tracklab"]
