"""Local FastAPI wrapper for the isolated FireRedVAD prototype.

This is not production-ready: no authentication, rate limiting, or deployment
configuration is included. Keep it behind a trusted boundary during testing.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

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
        # Keep the client response generic, but retain FFmpeg's diagnostic in server logs.
        detail = (exc.stderr or "").strip().replace("\n", " ")[:600]
        logger.warning("FFmpeg MP3 decode failed (exit=%s): %s", exc.returncode, detail or "no stderr")
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
        use_gpu=False,
    )
    app.state.model_dir = model_dir
    app.state.dna_engine = dna_engine or ArtistDNAEngine(vocal_detector=app.state.detector)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "artist-os-genre-engine", "status": "prototype"}

    @app.get("/lab", response_class=HTMLResponse)
    def lab() -> str:
        # Same-origin diagnostic UI for this isolated API only. It is not wired to ARTIST OS.
        return """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>ARTIST DNA — laboratoire FireRedVAD + AST</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#101114;color:#f3f3f5;font:16px/1.5 system-ui,sans-serif}
main{max-width:900px;margin:0 auto;padding:28px 18px}h1{font-size:clamp(1.8rem,5vw,2.8rem);margin:.3rem 0}
.panel{background:#1b1d22;border:1px solid #34363e;border-radius:14px;padding:18px;margin:16px 0}
.muted{color:#b6b8c2;font-size:.92rem}.eyebrow{letter-spacing:.12em;font-size:.75rem;color:#aeb1c0}
button{background:#e9e9ed;color:#17181b;border:0;border-radius:8px;padding:12px 16px;font-weight:700}
button:disabled{opacity:.5}input{max-width:100%;margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#111216;padding:14px;border-radius:8px;max-height:70vh;overflow:auto}
#status{margin-top:12px}strong{color:#fff}
</style></head>
<body><main>
<p class="eyebrow">ARTIST OS · PROTOTYPE ISOLÉ</p>
<h1>ARTIST DNA <span style="color:#aeb1c0">Local Lab</span></h1>
<p>Test du pipeline CPU : genre baseline + tags AST AudioSet + événements vocaux FireRedVAD + instrumentation ONNX optionnelle.</p>
<section class="panel"><strong>Confidentialité et périmètre</strong>
<p class="muted">Le fichier est envoyé uniquement à l’API de laboratoire sur laquelle cette page est servie, pour l’analyse en mémoire et dans un répertoire temporaire supprimé à la fin de la requête. Ce laboratoire n’est pas connecté aux comptes, paiements ou tableaux de bord ARTIST OS. Ne l’utilise pas sur un serveur public non protégé.</p></section>
<section class="panel"><label for="audio"><strong>Choisir un fichier MP3 (50 Mio maximum)</strong></label><br>
<input id="audio" type="file" accept=".mp3,audio/mpeg"><br>
<button id="run" disabled>ANALYSER AVEC LE PIPELINE</button><p id="status" role="status">Choisis un MP3 pour commencer.</p></section>
<section class="panel"><h2>Résultats structurés</h2><p class="muted">AST fournit des tags audio et n’est pas mélangé au score de genre. Les scores ne sont pas des probabilités calibrées. FireRedVAD détecte des événements/segments vocaux, pas l’identité ni le timbre du chanteur.</p><pre id="output">Aucun résultat.</pre></section>
</main>
<script>
const file=document.getElementById('audio'),run=document.getElementById('run'),status=document.getElementById('status'),output=document.getElementById('output');
file.addEventListener('change',()=>{const f=file.files&&file.files[0];run.disabled=!f||f.size>50*1024*1024;status.textContent=!f?'Choisis un MP3 pour commencer.':f.size>50*1024*1024?'Fichier supérieur à 50 Mio.':f.name+' · '+(f.size/1048576).toFixed(1)+' Mio';});
run.addEventListener('click',async()=>{const f=file.files&&file.files[0];if(!f||run.disabled)return;run.disabled=true;status.textContent='Analyse en cours. Le premier lancement peut télécharger les poids des modèles.';output.textContent='Chargement…';try{const form=new FormData();form.append('audio',f,f.name);const response=await fetch('/dna/analyze',{method:'POST',body:form,cache:'no-store'});const body=await response.json();output.textContent=JSON.stringify(body,null,2);status.textContent=response.ok?'Analyse terminée. Consulte les statuts de chaque composant dans le résultat.':'Échec HTTP '+response.status+' : '+(body.detail||'erreur inconnue');}catch(e){status.textContent='Erreur de requête : '+(e.message||String(e));output.textContent='Aucun résultat exploitable.';}finally{run.disabled=!file.files||!file.files[0]||file.files[0].size>50*1024*1024;}});
</script></body></html>"""

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
    async def analyze_artist_dna(
        request: Request,
        audio: UploadFile | None = File(default=None),
        file: UploadFile | None = File(default=None),
    ) -> JSONResponse:
        # The production /api/dna3 gateway sends the multipart field "audio".
        # Keep "file" as a compatibility alias for direct prototype callers.
        upload = audio or file
        if upload is None:
            raise HTTPException(status_code=400, detail="Fichier audio manquant.")
        filename = Path(upload.filename or "").name
        if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Format non pris en charge. Fournir un MP3.")
        payload = await upload.read(MAX_UPLOAD_BYTES + 1)
        await upload.close()
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
        # Match the existing ARTIST OS /api/dna3 gateway contract.
        return JSONResponse(
            content={"ok": True, "dna": analysis},
            headers={"Cache-Control": "no-store"},
        )

    return app


app = create_app()
