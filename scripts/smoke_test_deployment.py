#!/usr/bin/env python3
"""Deployment smoke test for the ARTIST DNA CPU API.

Uses only the Python standard library. It never prints bearer tokens or uploads
an audio file until the unauthenticated and invalid-token checks have passed.

Usage:
  ARTIST_DNA_API_TOKEN='...' python scripts/smoke_test_deployment.py \
    --base-url https://your-audio-host.example \
    --audio /path/to/a-real-test-track.mp3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    timeout: float = 20,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def multipart_audio(path: Path) -> tuple[str, bytes]:
    boundary = "----ArtistDNA" + uuid.uuid4().hex
    payload = path.read_bytes()
    filename = path.name.replace('"', "")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return boundary, body


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="HTTPS base URL, without /dna/analyze")
    parser.add_argument("--audio", required=True, type=Path, help="Real MP3/WAV/FLAC/OGG/M4A test track")
    parser.add_argument("--timeout", type=float, default=240, help="Per-request timeout in seconds (default: 240)")
    parser.add_argument("--max-seconds", type=float, default=180, help="Latency warning threshold (default: 180)")
    args = parser.parse_args()

    base = args.base_url.strip().rstrip("/")
    if not base.startswith("https://"):
        fail("base URL must use HTTPS")
    if any(char in base for char in "?#!") or "@" in base.split("://", 1)[1]:
        fail("base URL must not contain query, fragment, or embedded credentials")
    if not args.audio.is_file() or args.audio.stat().st_size <= 0:
        fail("audio file does not exist or is empty")
    if args.audio.suffix.lower() not in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}:
        fail("unsupported file extension")
    if args.audio.stat().st_size > 50 * 1024 * 1024:
        fail("audio file exceeds the API's 50 MiB limit")

    endpoint = base + "/dna/analyze"
    print("1/4 Health check…")
    try:
        status, raw = request(base + "/health", timeout=20)
    except Exception as exc:
        fail(f"health request failed ({type(exc).__name__})")
    if status != 200:
        fail(f"/health returned HTTP {status}")
    try:
        health: dict[str, Any] = json.loads(raw)
    except (ValueError, TypeError):
        fail("/health did not return JSON")
    if health.get("status") != "ok":
        fail("/health status is not ok")
    if health.get("ffmpeg_available") is not True:
        fail("FFmpeg is unavailable on the host")
    print("PASS: HTTPS health and FFmpeg")

    # Auth checks use a tiny invalid body; the server must reject it before parsing audio.
    boundary = "----ArtistDNAAuthCheck"
    tiny_body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="audio"; filename="probe.mp3"\r\n'
        "Content-Type: application/octet-stream\r\n\r\nx\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    content_type = f"multipart/form-data; boundary={boundary}"
    print("2/4 Authentication checks…")
    for label, headers, accepted in (
        ("missing token", {"Content-Type": content_type}, {401, 503}),
        ("invalid token", {"Content-Type": content_type, "Authorization": "Bearer invalid-smoke-test-token"}, {401}),
    ):
        try:
            status, _ = request(endpoint, method="POST", headers=headers, data=tiny_body, timeout=20)
        except Exception as exc:
            fail(f"{label} request failed ({type(exc).__name__})")
        if status not in accepted:
            fail(f"{label} should be rejected before inference; got HTTP {status}")
        print(f"PASS: {label} rejected (HTTP {status})")

    token = os.environ.get("ARTIST_DNA_API_TOKEN", "").strip()
    if not token:
        fail("set ARTIST_DNA_API_TOKEN in the environment; it is never read from CLI args")
    print("3/4 Real audio inference…")
    boundary, body = multipart_audio(args.audio)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Accept": "application/json",
    }
    started = time.monotonic()
    try:
        status, raw = request(
            endpoint,
            method="POST",
            headers=headers,
            data=body,
            timeout=args.timeout,
        )
    except Exception as exc:
        fail(f"authenticated inference failed ({type(exc).__name__}); inspect server logs without exposing secrets")
    elapsed = time.monotonic() - started
    if status != 200:
        # Do not dump response bodies: they can contain infrastructure details.
        fail(f"authenticated inference returned HTTP {status} after {elapsed:.1f}s")
    try:
        result: dict[str, Any] = json.loads(raw)
    except (ValueError, TypeError):
        fail("inference did not return valid JSON")
    dna = result.get("dna")
    if result.get("ok") is not True or not isinstance(dna, dict) or not dna:
        fail("response contract mismatch: expected {ok: true, dna: non-empty object}")
    print(f"PASS: real inference returned a valid DNA object in {elapsed:.1f}s")

    print("4/4 Resource/contract summary…")
    if elapsed > args.max_seconds:
        print(f"WARNING: latency {elapsed:.1f}s exceeds target {args.max_seconds:.1f}s")
    for key in ("genre_analysis", "audio_tag_analysis", "vocal_analysis"):
        if key not in dna:
            print(f"WARNING: expected analysis field is missing: {key}")
    print("PASS: smoke test complete; inspect warnings and validate genre quality before connecting DNA_URL.")
    print("No secrets or analysis payloads were printed.")


if __name__ == "__main__":
    main()
