"""Benchmark the local instrument candidate on one real WAV file.

Run from the repository root:
    ARTIST_DNA_INSTRUMENTS_ENABLED=true python -m src.benchmark_instruments /path/to/song.wav

The reported timing is a measurement on the current machine, not a performance promise.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.instrument_classifier import LocalONNXInstrumentClassifier


def _peak_rss_mb() -> float | None:
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB; macOS reports bytes.
        import sys as _sys
        if _sys.platform == "darwin":
            return round(value / (1024 * 1024), 1)
        return round(value / 1024, 1)
    except (ImportError, OSError, ValueError):
        return None


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m src.benchmark_instruments /path/to/song.wav", file=sys.stderr)
        return 2

    audio_path = Path(sys.argv[1]).expanduser()
    if not audio_path.is_file():
        print(f"Fichier introuvable : {audio_path}", file=sys.stderr)
        return 2

    try:
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except Exception as exc:
        print(f"Lecture audio impossible : {exc}", file=sys.stderr)
        return 2
    if sample_rate <= 0 or len(audio) == 0:
        print("Fichier audio vide ou invalide.", file=sys.stderr)
        return 2

    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    duration = len(mono) / sample_rate
    classifier = LocalONNXInstrumentClassifier(enabled=True)

    load_started = time.perf_counter()
    try:
        classifier._load()
        load_seconds = time.perf_counter() - load_started
        analysis_started = time.perf_counter()
        predictions = classifier.analyze(mono, sample_rate, top_k=9)
        analysis_seconds = time.perf_counter() - analysis_started
    except Exception as exc:
        print(json.dumps({
            "status": "error",
            "error": str(exc),
            "model_id": classifier.model_id,
            "audio_duration_seconds": round(duration, 3),
        }, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps({
        "status": "completed",
        "model_id": classifier.model_id,
        "audio_file": audio_path.name,
        "audio_duration_seconds": round(duration, 3),
        "sample_rate_hz": int(sample_rate),
        "load_and_cache_seconds": round(load_seconds, 3),
        "analysis_seconds": round(analysis_seconds, 3),
        "real_time_factor": round(analysis_seconds / duration, 4) if duration else None,
        "peak_process_rss_mb": _peak_rss_mb(),
        "predictions": predictions,
        "warning": "Experimental classifier trained on isolated-instrument clips; outputs on a mixed full song are not validated.",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
