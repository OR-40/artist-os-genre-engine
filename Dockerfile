FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    FIREREDVAD_MODEL_DIR=/app/weights/FireRedVAD/AED \
    ARTIST_DNA_INSTRUMENTS_ENABLED=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-dna.txt requirements-api.txt requirements-vocal.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-dna.txt

COPY . .

EXPOSE 7860

CMD ["bash", "-lc", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
