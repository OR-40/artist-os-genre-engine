"""ARTIST DNA CPU API. Protect inference routes with a server-side bearer token."""
from __future__ import annotations

import hmac
import logging
import os
import shutil
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


def decode_audio_to_wav(audio_path: str | Path, wav_path: str | Path) -> None:
    """Decode an uploaded supported audio file to mono 16 kHz PCM WAV using FFmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(audio_path), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(wav_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg est requis pour décoder les fichiers audio.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Le décodage audio a dépassé 180 secondes.") from exc
    except subprocess.CalledProcessError as exc:
        # Keep the client response generic, but retain FFmpeg's diagnostic in server logs.
        detail = (exc.stderr or "").strip().replace("\n", " ")[:600]
        logger.warning("FFmpeg audio decode failed (exit=%s): %s", exc.returncode, detail or "no stderr")
        raise ValueError("Fichier audio illisible ou invalide.") from exc

    try:
        info = sf.info(str(wav_path))
    except Exception as exc:
        raise ValueError("Le fichier audio n'a pas produit un audio valide.") from exc
    if info.frames <= 0 or info.samplerate != 16000 or info.channels != 1:
        raise ValueError("Le fichier audio décodé ne contient pas d'audio mono 16 kHz valide.")


def create_app(detector: Any | None = None, dna_engine: Any | None = None, api_token: str | None = None) -> FastAPI:
    app = FastAPI(title="ARTIST OS Genre Engine", version="0.1.0")
    model_dir = Path(os.environ.get("FIREREDVAD_MODEL_DIR", "weights/FireRedVAD/AED"))
    app.state.detector = detector or FireRedVADDetector(
        model_dir=model_dir,
        use_gpu=False,
    )
    app.state.model_dir = model_dir
    app.state.dna_engine = dna_engine or ArtistDNAEngine(vocal_detector=app.state.detector)
    app.state.api_token = (os.getenv('ARTIST_DNA_API_TOKEN', '') if api_token is None else api_token).strip()

    def require_api_token(request: Request) -> None:
        expected = request.app.state.api_token
        if not expected:
            raise HTTPException(status_code=503, detail='Jeton API ARTIST DNA non configuré côté serveur.')
        authorization = request.headers.get('authorization', '')
        expected_header = f'Bearer {expected}'
        if not hmac.compare_digest(authorization, expected_header):
            raise HTTPException(status_code=401, detail='Authentification du moteur requise.')

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "artist-os-genre-engine", "status": "prototype"}

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        # This checks configuration only; it deliberately does not load large weights.
        return {
            "status": "ok",
            "model": "FireRedVAD",
            "execution_device": "CPU",
            "model_files_directory_exists": request.app.state.model_dir.is_dir(),
            "ffmpeg_available": shutil.which("ffmpeg") is not None,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "supported_formats": sorted(ALLOWED_SUFFIXES),
            "decoded_audio": "mono PCM WAV 16 kHz, temporary only",
        }

    @app.post("/analyze")
    async def analyze(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        require_api_token(request)
        filename = Path(file.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Format non pris en charge. Fournir un fichier MP3.")

        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        await file.close()
        if not payload:
            raise HTTPException(status_code=400, detail="Fichier audio vide.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Fichier supérieur à la limite de 50 Mio.")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = Path(temp_dir) / ("upload" + suffix)
                wav_path = Path(temp_dir) / "decoded.wav"
                audio_path.write_bytes(payload)
                decode_audio_to_wav(audio_path, wav_path)
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
            raise HTTPException(status_code=400, detail="Fichier audio illisible ou invalide.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail="Impossible de traiter le fichier audio temporaire.") from exc

        return JSONResponse(content={"analysis": analysis})

    @app.post("/dna/analyze")
    async def analyze_artist_dna(
        request: Request,
        audio: UploadFile | None = File(default=None),
        file: UploadFile | None = File(default=None),
    ) -> JSONResponse:
        require_api_token(request)
        # The production /api/dna3 gateway sends the multipart field "audio".
        # Keep "file" as a compatibility alias for direct prototype callers.
        upload = audio or file
        if upload is None:
            raise HTTPException(status_code=400, detail="Fichier audio manquant.")
        filename = Path(upload.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Format non pris en charge. Fournir un MP3, WAV, FLAC, OGG ou M4A.")
        payload = await upload.read(MAX_UPLOAD_BYTES + 1)
        await upload.close()
        if not payload:
            raise HTTPException(status_code=400, detail="Fichier audio vide.")
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Fichier supérieur à la limite de 50 Mio.")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = Path(temp_dir) / ("upload" + suffix)
                wav_path = Path(temp_dir) / "decoded.wav"
                audio_path.write_bytes(payload)
                decode_audio_to_wav(audio_path, wav_path)
                analysis = request.app.state.dna_engine.analyze_file(str(wav_path))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Fichier audio illisible ou invalide.") from exc
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
        # Match the existing ARTIST OS /api/dna3 gateway contract.
        return JSONResponse(
            content={"ok": True, "dna": analysis},
            headers={"Cache-Control": "no-store"},
        )

    return app


app = create_app()
