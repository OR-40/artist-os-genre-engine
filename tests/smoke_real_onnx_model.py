"""Real private-artifact smoke test for the AST ONNX adapter.

Requires HF_TOKEN with read access to OR-40/artist-os-ast-onnx.
Uses deterministic synthetic audio; it validates model loading/inference, not genre accuracy.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from src.onnx_genre_classifier import LocalONNXGenreClassifier


def main() -> int:
    if not os.getenv("HF_TOKEN", "").strip():
        print("HF_TOKEN absent: real-model smoke test cannot run.", file=sys.stderr)
        return 2

    # Deterministic 10-second test signal; no user audio or copyrighted track is needed.
    sample_rate = 16_000
    time = np.arange(sample_rate * 10, dtype=np.float32) / sample_rate
    audio = (
        0.2 * np.sin(2 * np.pi * 220.0 * time)
        + 0.05 * np.sin(2 * np.pi * 440.0 * time)
    ).astype(np.float32)

    classifier = LocalONNXGenreClassifier()
    predictions = classifier.predict(audio, sample_rate=sample_rate, top_k=5)

    if not predictions:
        raise AssertionError("ONNX AST model returned no predictions.")
    if not all(
        isinstance(item.get("label"), str)
        and item["label"]
        and np.isfinite(float(item.get("score", float("nan"))))
        for item in predictions
    ):
        raise AssertionError("ONNX AST model returned malformed predictions.")

    print("REAL AST ONNX MODEL: LOAD + CPU INFERENCE OK")
    for item in predictions:
        print(f'{item["label"]}: {float(item["score"]):.4f}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
