"""Local FastAPI wrapper for the isolated FireRedVAD prototype.

This is not production-ready: no authentication, rate limiting, or deployment
configuration is included. Keep it behind a trusted boundary during testing.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from src.vocal_detection import FireRedVADDetector

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg"}


def create_app(detector: Any | None = None) -> FastAPI:
    app = FastAPI(title="ARTIST OS Genre Engine", version="0.1.0")
    model_dir = Path(os.environ.get("FIREREDVAD_MODEL_DIR", "weights/FireRedVAD/AED"))
    app.state.detector = detector or FireRedVADDetector(
        model_dir=model_dir,
        use_gpu=os.environ.get("FIREREDVAD_USE_GPU", "false").lower() == "true",
    )
    app.state.model_dir = model_dir

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
        }

    @app.post("/analyze")
    async def analyze(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        filename = Path(file.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=415,
                detail="Format non pris en charge pour ce prototype. Utiliser WAV, FLAC ou OGG.",
            )

        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if not payload:
            raise HTTPException(status_code=400, detail="Fichier audio vide.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Fichier supérieur à la limite de 50 Mio.")

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp:
                temp.write(payload)
                temp.flush()
                try:
                    info = sf.info(temp.name)
                    if info.frames <= 0 or info.samplerate <= 0:
                        raise ValueError("empty audio")
                except Exception as exc:
                    raise HTTPException(status_code=400, detail="Fichier audio illisible ou invalide.") from exc

                try:
                    analysis = request.app.state.detector.analyze_file(temp.name)
                except FileNotFoundError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="Poids FireRedVAD introuvables : configurer FIREREDVAD_MODEL_DIR.",
                    ) from exc
                except RuntimeError as exc:
                    raise HTTPException(status_code=503, detail="Dépendance FireRedVAD indisponible.") from exc
        except HTTPException:
            raise
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Impossible de traiter le fichier temporaire.") from exc

        return JSONResponse(content={"analysis": analysis})

    return app


app = create_app()
