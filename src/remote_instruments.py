"""Adapter for the existing ARTIST OS INSTRUMENTS HTTP service.

The service contract is POST /analyze with multipart field "audio" and a JSON
response shaped as {"ok": true, "model": "...", "predictions": [...]}.
No request is sent unless INSTRUMENTS_URL is configured.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import httpx


def _endpoint(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        return ""
    return value if value.endswith("/analyze") else f"{value}/analyze"


class RemoteInstrumentAnalyzer:
    """Call the existing full-song instrument service; leave it disabled by default."""

    def __init__(
        self,
        url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        configured_url = os.getenv("INSTRUMENTS_URL", "") if url is None else url
        self.url = _endpoint(configured_url)
        self.enabled = bool(self.url)
        self.model_id = "ARTIST OS INSTRUMENTS"
        configured_timeout = (
            os.getenv("INSTRUMENTS_TIMEOUT_SECONDS", "240")
            if timeout_seconds is None
            else timeout_seconds
        )
        self.timeout_seconds = float(configured_timeout)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("INSTRUMENTS_TIMEOUT_SECONDS doit être positif.")

    def analyze_file(self, audio_path: str | Path) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        path = Path(audio_path)
        with path.open("rb") as audio:
            response = httpx.post(
                self.url,
                files={"audio": (path.name, audio, "audio/wav")},
                timeout=httpx.Timeout(self.timeout_seconds, connect=min(5.0, self.timeout_seconds)),
            )

        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise RuntimeError("Le service instruments a renvoyé une réponse invalide.")
        predictions = payload.get("predictions")
        if not isinstance(predictions, list):
            raise RuntimeError("Le service instruments n'a pas renvoyé de liste predictions.")

        results: list[dict[str, Any]] = []
        for item in predictions:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            try:
                score = float(item.get("score"))
                max_score = float(item.get("maxScore", score))
                occurrences = int(item.get("occurrences", 0))
            except (TypeError, ValueError, OverflowError):
                continue
            if not label or not math.isfinite(score) or not math.isfinite(max_score):
                continue
            results.append({
                "name": label,
                "confidence": score,
                "score": score,
                "max_score": max_score,
                "occurrences": max(0, occurrences),
                "role": "",
            })
        return results
