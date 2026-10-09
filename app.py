"""Local FastAPI wrapper for the isolated FireRedVAD prototype.

This is not production-ready: no authentication, rate limiting, or deployment
configuration is included. Keep it behind a trusted boundary during testing.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.artist_dna import ArtistDNAEngine
from src.vocal_detection import FireRedVADDetector

logger = logging.getLogger("artist_os_genre_engine")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = {".mp3"}


def decode_mp3_to_wav(mp3_path: str | Path, wav_path: str | Path) -> None:
    """Decode an uploaded MP3 to compact mono 16 kHz PCM WAV using FFmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(mp3_path), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(wav_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg est requis pour décoder les MP3.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Le décodage MP3 a dépassé 180 secondes.") from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError("Fichier MP3 illisible ou invalide.") from exc

    try:
        info = sf.info(str(wav_path))
    except Exception as exc:
        raise ValueError("Le MP3 n'a pas produit un audio valide.") from exc
    if info.frames <= 0 or info.samplerate != 16000 or info.channels != 1:
        raise ValueError("Le MP3 décodé ne contient pas d'audio mono 16 kHz valide.")


def create_app(detector: Any | None = None, dna_engine: Any | None = None) -> FastAPI:
    app = FastAPI(title="ARTIST OS Genre Engine", version="0.1.0")
    model_dir = Path(os.environ.get("FIREREDVAD_MODEL_DIR", "weights/FireRedVAD/AED"))
    app.state.detector = detector or FireRedVADDetector(
        model_dir=model_dir,
        use_gpu=os.environ.get("FIREREDVAD_USE_GPU", "false").lower() == "true",
    )
    app.state.model_dir = model_dir
    app.state.dna_engine = dna_engine or ArtistDNAEngine(vocal_detector=app.state.detector)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "artist-os-genre-engine", "status": "prototype"}

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        # This checks configuration only; it deliberately does not load large weights.
        return {
            "status": "ok",
            "model": "FireRedVAD",
            "model_files_directory_exists": request.app.state.model_dir.is_dir(),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "supported_formats": sorted(ALLOWED_SUFFIXES),
            "decoded_audio": "mono PCM WAV 16 kHz, temporary only",
        }

    @app.post("/analyze")
    async def analyze(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        filename = Path(file.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Format non pris en charge. Fournir un MP3.")

        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if not payload:
            raise HTTPException(status_code=400, detail="Fichier audio vide.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Fichier supérieur à la limite de 50 Mio.")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                mp3_path = Path(temp_dir) / "upload.mp3"
                wav_path = Path(temp_dir) / "decoded.wav"
                mp3_path.write_bytes(payload)
                decode_mp3_to_wav(mp3_path, wav_path)
                try:
                    analysis = request.app.state.detector.analyze_file(str(wav_path))
                except FileNotFoundError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="Poids FireRedVAD introuvables : configurer FIREREDVAD_MODEL_DIR.",
                    ) from exc
                except RuntimeError as exc:
                    raise HTTPException(status_code=503, detail="Dépendance FireRedVAD indisponible.") from exc
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Fichier MP3 illisible ou invalide.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Impossible de traiter le fichier audio temporaire.") from exc

        return JSONResponse(content={"analysis": analysis})

    @app.post("/dna/analyze")
    async def analyze_artist_dna(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        filename = Path(file.filename or "").name
        if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Format non pris en charge. Fournir un MP3.")
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if not payload:
            raise HTTPException(status_code=400, detail="Fichier audio vide.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Fichier supérieur à la limite de 50 Mio.")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                mp3_path = Path(temp_dir) / "upload.mp3"
                wav_path = Path(temp_dir) / "decoded.wav"
                mp3_path.write_bytes(payload)
                decode_mp3_to_wav(mp3_path, wav_path)
                analysis = request.app.state.dna_engine.analyze_file(str(wav_path))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Fichier MP3 illisible ou invalide.") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="Poids du modèle ou fichier audio introuvables.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Impossible de traiter le fichier audio temporaire.") from exc
        except Exception as exc:
            # Keep client-facing errors generic, but retain the real traceback in
            # server logs so real-model smoke tests can identify the failing component.
            logger.exception("Échec d'un composant d'analyse ARTIST DNA (%s)", type(exc).__name__)
            raise HTTPException(status_code=502, detail="Échec d'un composant d'analyse ARTIST DNA.") from exc
        return JSONResponse(content={"analysis": analysis})

    return app


app = create_app()
