#!/usr/bin/env python3
"""Profile real CPU/RAM usage of the isolated ARTIST DNA engine on a local MP3.

Run on Linux with the full requirements installed:
  python scripts/profile_dna.py /path/to/authorized-test.mp3
  ARTIST_DNA_INSTRUMENTS_ENABLED=false python scripts/profile_dna.py /path/to/authorized-test.mp3

This is an opt-in diagnostic. It does not alter the API, download audio, or change
production configuration. First run may download model weights from Hugging Face.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import tempfile
import time
from pathlib import Path


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return value / (1024 if platform.system() == "Linux" else 1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mp3", type=Path, help="Local MP3 file (never uploaded by this script)")
    args = parser.parse_args()
    source = args.mp3.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".mp3":
        parser.error("Provide an existing .mp3 file.")
    if source.stat().st_size > 50 * 1024 * 1024:
        parser.error("MP3 must be at most 50 MiB.")

    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "audio_file_bytes": source.stat().st_size,
        "instruments_enabled": os.getenv("ARTIST_DNA_INSTRUMENTS_ENABLED", "true").lower() == "true",
        "stages_seconds": {},
        "peak_rss_mib": {},
    }
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="artist-dna-profile-") as temp:
        wav = Path(temp) / "profile.wav"
        decode_started = time.perf_counter()
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(wav)],
            check=True,
            timeout=180,
        )
        result["stages_seconds"]["decode_ffmpeg"] = round(time.perf_counter() - decode_started, 3)
        result["peak_rss_mib"]["after_decode"] = round(peak_rss_mib(), 1)

        # Import only after input validation and audio decode.
        from src.artist_dna import ArtistDNAEngine
        from src.vocal_detection import FireRedVADDetector

        model_dir = Path(os.environ.get("FIREREDVAD_MODEL_DIR", "weights/FireRedVAD/AED"))
        detector = FireRedVADDetector(model_dir=model_dir, use_gpu=False)
        engine = ArtistDNAEngine(vocal_detector=detector)
        analysis_started = time.perf_counter()
        output = engine.analyze_file(str(wav))
        result["stages_seconds"]["engine_total_including_model_load"] = round(time.perf_counter() - analysis_started, 3)
        result["peak_rss_mib"]["after_analysis"] = round(peak_rss_mib(), 1)
        result["audio_duration_seconds"] = output.get("duration_seconds")
        result["selected_audio_seconds"] = output.get("analysis_sampling", {}).get("selected_audio_seconds")
        result["classifiers"] = list(output.get("genre_analysis", {}).get("models", {}).keys()) if isinstance(output.get("genre_analysis", {}).get("models"), dict) else None
        result["instrument_status"] = output.get("instrument_analysis", {}).get("status")

    result["stages_seconds"]["total_wall_time"] = round(time.perf_counter() - started, 3)
    result["peak_rss_mib"]["process_high_water_mark"] = round(peak_rss_mib(), 1)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
