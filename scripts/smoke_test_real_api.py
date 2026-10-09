"""Real MP3 smoke test for FireRedVAD or the combined ARTIST DNA engine.

Examples:
  python scripts/smoke_test_real_api.py --audio "song.mp3"
  python scripts/smoke_test_real_api.py --audio "song.mp3" --dna
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from app import create_app


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a real MP3 file")
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "weights" / "FireRedVAD" / "AED"),
        help="FireRedVAD AED model directory; missing official weights are downloaded automatically",
    )
    parser.add_argument("--gpu", action="store_true", help="Enable GPU inference if configured")
    parser.add_argument(
        "--dna",
        action="store_true",
        help="Run the full ARTIST DNA pipeline (two genre models plus FireRedVAD)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        parser.error(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".mp3":
        parser.error("Only MP3 is accepted by this API.")

    os.environ["FIREREDVAD_MODEL_DIR"] = str(Path(args.model_dir))
    os.environ["FIREREDVAD_USE_GPU"] = "true" if args.gpu else "false"
    app = create_app()
    client = TestClient(app)
    endpoint = "/dna/analyze" if args.dna else "/analyze"

    started = time.perf_counter()
    try:
        with audio_path.open("rb") as audio:
            response = client.post(
                endpoint,
                files={"file": (audio_path.name, audio, "audio/mpeg")},
            )
    finally:
        client.close()
    elapsed = round(time.perf_counter() - started, 3)

    print(f"Endpoint: {endpoint}")
    print(f"HTTP status: {response.status_code}")
    print(f"End-to-end seconds: {elapsed}")
    if response.status_code != 200:
        print(response.text)
        return 1

    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("analysis"), dict):
        print("FAIL: response does not match {'analysis': {...}} contract")
        return 1
    analysis = payload["analysis"]

    if args.dna:
        required = ("genre_analysis", "vocal_analysis", "instrument_analysis", "analysis_sampling")
        missing = [key for key in required if key not in analysis]
        if analysis.get("engine") != "ARTIST DNA" or missing:
            print(f"FAIL: unexpected ARTIST DNA result; missing={missing}")
            return 1
        vocal = analysis["vocal_analysis"]
        singing = vocal.get("singing", {})
        summary = {
            "engine": analysis.get("engine"),
            "duration_seconds": analysis.get("duration_seconds"),
            "analysis_sampling": analysis.get("analysis_sampling"),
            "genre_models": list(analysis["genre_analysis"].get("models", {}).keys()),
            "singing_total_duration_seconds": singing.get("total_duration_seconds"),
            "singing_segment_count": len(singing.get("segments", [])),
            "speech_segment_count": len(vocal.get("speech", {}).get("segments", [])),
            "music_segment_count": len(vocal.get("music", {}).get("segments", [])),
            "instrument_status": analysis["instrument_analysis"].get("status"),
            "end_to_end_seconds": elapsed,
        }
    else:
        required = ("singing", "speech", "music")
        missing = [key for key in required if key not in analysis]
        if analysis.get("model") != "FireRedVAD" or missing:
            print(f"FAIL: unexpected FireRedVAD result; missing={missing}")
            return 1
        summary = {
            "model": analysis.get("model"),
            "duration_seconds": analysis.get("duration_seconds"),
            "singing_total_duration_seconds": analysis["singing"].get("total_duration_seconds"),
            "singing_segment_count": len(analysis["singing"].get("segments", [])),
            "speech_segment_count": len(analysis["speech"].get("segments", [])),
            "music_segment_count": len(analysis["music"].get("segments", [])),
            "end_to_end_seconds": elapsed,
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
