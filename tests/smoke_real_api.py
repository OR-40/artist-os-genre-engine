"""Real CPU end-to-end smoke test for the isolated ARTIST DNA API.

Uses a short synthetic signal, a real MP3 encode/decode cycle, the private AST
ONNX genre model, the public AST AudioSet tagger, and real FireRedVAD weights.
This checks integration/runtime, not genre accuracy.
"""
from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

# Set before importing app: the app constructs its default engine at import time.
os.environ["ARTIST_DNA_ONNX_GENRE_ENABLED"] = "true"
os.environ["ARTIST_DNA_ONNX_GENRE_REPO_ID"] = os.getenv(
    "ARTIST_DNA_ONNX_GENRE_REPO_ID", "OR-40/artist-os-ast-onnx"
)
os.environ["ARTIST_DNA_INSTRUMENTS_ENABLED"] = "false"

from app import create_app


def main() -> int:
    if not os.getenv("HF_TOKEN", "").strip():
        raise RuntimeError("HF_TOKEN is required for the private AST ONNX model.")

    sample_rate = 16_000
    duration = 12
    time = np.arange(sample_rate * duration, dtype=np.float32) / sample_rate
    audio = (
        0.18 * np.sin(2 * np.pi * 110.0 * time)
        + 0.12 * np.sin(2 * np.pi * 220.0 * time)
        + 0.06 * np.sin(2 * np.pi * 440.0 * time)
    ).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="artist-dna-e2e-") as temp:
        root = Path(temp)
        wav_path = root / "synthetic.wav"
        mp3_path = root / "synthetic.mp3"
        sf.write(str(wav_path), audio, sample_rate)
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "128k",
                str(mp3_path),
            ],
            check=True,
            timeout=60,
        )

        app = create_app()
        with TestClient(app) as client, mp3_path.open("rb") as handle:
            response = client.post(
                "/dna/analyze",
                files={"audio": ("synthetic.mp3", handle, "audio/mpeg")},
            )

        if response.status_code != 200:
            raise AssertionError(
                f"/dna/analyze returned HTTP {response.status_code}: {response.text}"
            )
        payload = response.json()
        if payload.get("ok") is not True:
            raise AssertionError("API response does not contain ok=true.")
        dna = payload.get("dna") or {}
        models = (dna.get("genre_analysis") or {}).get("models") or {}
        if "genre_candidate_onnx" not in models:
            raise AssertionError(
                f"Private ONNX genre model missing from response; models={list(models)}"
            )
        if not models["genre_candidate_onnx"].get("aggregated_predictions"):
            raise AssertionError("Private ONNX genre model returned no aggregated predictions.")
        tag_analysis = dna.get("audio_tag_analysis") or {}
        if tag_analysis.get("status") != "ok" or not tag_analysis.get("predictions"):
            raise AssertionError(
                f"AST AudioSet tagger did not return predictions: {tag_analysis.get('status')}"
            )
        vocal = dna.get("vocal_analysis") or {}
        if vocal.get("model") != "FireRedVAD":
            raise AssertionError(f"Unexpected vocal model in response: {vocal.get('model')!r}")

        print("ARTIST DNA END-TO-END API: OK")
        print(f"Duration: {dna.get('duration_seconds')} seconds")
        print(f"Genre model windows: {models['genre_candidate_onnx'].get('window_count')}")
        print(f"AST audio tags: {len(tag_analysis['predictions'])}")
        print(f"FireRedVAD singing ratio: {(vocal.get('singing') or {}).get('ratio')}")
        print("Note: synthetic audio validates integration, not classification accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
