"""Adapter for the existing ARTIST OS instruments HTTP service.

Contract grounded in DNA3's modules/instrumentation.py:
multipart field "audio", WAV PCM16, JSON response with "predictions".
"""
from __future__ import annotations

import io
import json
import os
import urllib.request
from typing import Any

import numpy as np
import soundfile as sf


DEFAULT_INSTRUMENTS_URL = ""


def build_montage(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return up to three 6-second excerpts: beginning, middle, and end."""
    if audio is None or len(audio) == 0 or sample_rate <= 0:
        return np.array([], dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    segment_samples = int(6 * sample_rate)
    if len(audio) <= segment_samples:
        return audio
    max_start = max(0, len(audio) - segment_samples)
    starts = (0, max_start // 2, max_start)
    return np.concatenate(
        [audio[start:start + segment_samples] for start in starts]
    ).astype(np.float32, copy=False)


def _multipart_body(field_name: str, filename: str, payload: bytes, boundary: str) -> bytes:
    return b"".join((
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8"),
        payload,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ))


class InstrumentsService:
    """Fail-soft client for the existing instrument-classifier endpoint."""

    def __init__(self, url: str | None = None, timeout: int | None = None) -> None:
        configured_url = url if url is not None else os.getenv("INSTRUMENTS_URL", DEFAULT_INSTRUMENTS_URL)
        self.url = configured_url.strip()
        self.timeout = timeout if timeout is not None else int(os.getenv("INSTRUMENTS_TIMEOUT", "90"))

    def analyze(self, audio: np.ndarray, sample_rate: int, top_k: int = 8) -> list[dict[str, Any]]:
        if not self.url or audio is None or len(audio) == 0:
            return []
        try:
            montage = build_montage(audio, sample_rate)
            if len(montage) == 0:
                return []
            buffer = io.BytesIO()
            sf.write(buffer, montage, sample_rate, format="WAV", subtype="PCM_16")
            boundary = "----ARTISTOSINSTRUMENTBoundary7MA4YWxkTrZu0gW"
            body = _multipart_body("audio", "artist-dna-montage.wav", buffer.getvalue(), boundary)
            request = urllib.request.Request(self.url, data=body, method="POST")
            request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
            request.add_header("Accept", "application/json")
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            if not isinstance(data, dict) or data.get("ok") is False:
                return []
            predictions = data.get("predictions", [])
            if not isinstance(predictions, list):
                return []
            results = []
            for item in predictions:
                if not isinstance(item, dict):
                    continue
                label = item.get("label") or item.get("name")
                if not label:
                    continue
                try:
                    score = float(item.get("score", item.get("confidence", 0)))
                except (TypeError, ValueError):
                    score = 0.0
                if not np.isfinite(score):
                    score = 0.0
                results.append({
                    "name": str(label),
                    "confidence": round(max(0.0, min(1.0, score)), 3),
                    "role": "",
                })
                if len(results) >= max(0, top_k):
                    break
            return results
        except Exception as exc:
            print(f"[ARTIST_DNA][INSTRUMENTS] {exc}", flush=True)
            return []
