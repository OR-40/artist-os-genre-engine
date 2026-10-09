"""End-to-end smoke test using real FireRedVAD weights and a local WAV file.

Usage:
  python scripts/smoke_test_real_api.py --audio /path/to/song.wav --model-dir /path/to/FireRedVAD/AED
"""
from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path

# Make the repository root importable when this script is launched by absolute path.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from app import create_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to a WAV file accessible to this machine")
    parser.add_argument("--model-dir", required=True, help="Path to FireRedVAD AED model directory")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU inference")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    model_dir = Path(args.model_dir)
    if not audio_path.is_file():
        parser.error(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        parser.error("Only WAV is accepted by this smoke test until other decoders are validated.")
    if not model_dir.is_dir():
        parser.error(f"Model directory not found: {model_dir}")

    os.environ["FIREREDVAD_MODEL_DIR"] = str(model_dir)
    os.environ["FIREREDVAD_USE_GPU"] = "true" if args.gpu else "false"
    app = create_app()
    client = TestClient(app)

    started = time.perf_counter()
    try:
        with audio_path.open("rb") as audio:
            response = client.post(
                "/analyze",
                files={"file": (audio_path.name, audio, "audio/wav")},
            )
    finally:
        client.close()
    elapsed = round(time.perf_counter() - started, 3)

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
    for event_name in ("singing", "speech", "music"):
        if event_name not in analysis or not isinstance(analysis[event_name].get("segments"), list):
            print(f"FAIL: missing normalized {event_name} segments")
            return 1

    print(json.dumps({
        "model": analysis.get("model"),
        "duration_seconds": analysis.get("duration_seconds"),
        "singing_total_duration_seconds": analysis["singing"].get("total_duration_seconds"),
        "singing_segment_count": len(analysis["singing"]["segments"]),
        "speech_segment_count": len(analysis["speech"]["segments"]),
        "music_segment_count": len(analysis["music"]["segments"]),
        "end_to_end_seconds": elapsed,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
