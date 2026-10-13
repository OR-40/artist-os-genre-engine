"""Real CPU end-to-end smoke test for the isolated ARTIST DNA API.

Uses a short synthetic signal, a real MP3 encode/decode cycle, the private AST
ONNX genre model, the public AST AudioSet tagger, and real FireRedVAD weights.
This checks integration/runtime, not genre accuracy.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

# When executed as `python tests/smoke_real_api.py`, Python puts tests/ on sys.path.
# Add the repository root so `app.py` and `src/` can be imported in CI.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

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

    external_audio = os.getenv("ARTIST_DNA_TEST_AUDIO_PATH", "").strip()
    expected_genre = os.getenv("ARTIST_DNA_EXPECTED_GENRE", "").strip().casefold()

    with tempfile.TemporaryDirectory(prefix="artist-dna-e2e-") as temp:
        root = Path(temp)
        if external_audio:
            mp3_path = Path(external_audio).expanduser().resolve()
            if not mp3_path.is_file() or mp3_path.suffix.lower() != ".mp3":
                raise RuntimeError("ARTIST_DNA_TEST_AUDIO_PATH must point to an existing MP3 file.")
        else:
            sample_rate = 16_000
            duration = 12
            time = np.arange(sample_rate * duration, dtype=np.float32) / sample_rate
            audio = (
                0.18 * np.sin(2 * np.pi * 110.0 * time)
                + 0.12 * np.sin(2 * np.pi * 220.0 * time)
                + 0.06 * np.sin(2 * np.pi * 440.0 * time)
            ).astype(np.float32)
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

        genre_predictions = models["genre_candidate_onnx"].get("aggregated_predictions") or []
        top_genre = str(genre_predictions[0].get("label", "")).casefold() if genre_predictions else ""
        print("ARTIST DNA END-TO-END API: OK")
        print(f"Input: {'real audio file' if external_audio else 'synthetic audio'}")
        print(f"Duration: {dna.get('duration_seconds')} seconds")
        print(f"Genre model windows: {models['genre_candidate_onnx'].get('window_count')}")
        print("Genre ranking:", [(item.get("label"), item.get("mean_score")) for item in genre_predictions[:5]])
        print(f"AST audio tags: {len(tag_analysis['predictions'])}")
        print("Top AST tags:", [(item.get("label"), item.get("mean_score")) for item in tag_analysis["predictions"][:5]])
        print(f"FireRedVAD singing ratio: {(vocal.get('singing') or {}).get('ratio')}")
        if expected_genre:
            if top_genre != expected_genre:
                raise AssertionError(f"Expected top genre {expected_genre!r}, got {top_genre!r}; ranking={genre_predictions[:5]}")
            print(f"Expected genre check: PASS ({expected_genre})")
        elif not external_audio:
            print("Note: synthetic audio validates integration, not classification accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
