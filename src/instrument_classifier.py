"""Experimental local ONNX instrument classifier; no network inference service.

Model candidate: onnx-community/Musical-Instrument-Classification-ONNX.
Inference is opt-in via ARTIST_DNA_INSTRUMENTS_ENABLED=true. The first enabled
analysis downloads/caches the model files from Hugging Face; inference itself
runs locally with ONNX Runtime on CPU.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "onnx-community/Musical-Instrument-Classification-ONNX"
MODEL_FILE = "onnx/model_q4.onnx"
TARGET_SAMPLE_RATE = 16000
CLIP_SECONDS = 3
CLIP_SAMPLES = TARGET_SAMPLE_RATE * CLIP_SECONDS
DEFAULT_MAX_SEGMENTS = 6


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32).reshape(-1)
    from math import gcd
    from scipy.signal import resample_poly

    divisor = gcd(source_rate, target_rate)
    return resample_poly(
        np.asarray(audio, dtype=np.float32).reshape(-1),
        target_rate // divisor,
        source_rate // divisor,
    ).astype(np.float32, copy=False)


def _sample_starts(audio_length: int, clip_samples: int, max_segments: int) -> list[int]:
    if audio_length <= 0 or clip_samples <= 0 or max_segments <= 0:
        return []
    last_start = max(0, audio_length - clip_samples)
    count = min(max_segments, max(1, int(np.ceil(audio_length / clip_samples))))
    if count == 1:
        return [0]
    return sorted({int(round(value)) for value in np.linspace(0, last_start, count)})


class LocalONNXInstrumentClassifier:
    """Local CPU inference over a small set of evenly spaced 3-second clips."""

    def __init__(
        self,
        enabled: bool | None = None,
        model_id: str = MODEL_ID,
        max_segments: int = DEFAULT_MAX_SEGMENTS,
        session: Any | None = None,
        feature_extractor: Any | None = None,
        labels: dict[int, str] | None = None,
    ) -> None:
        self.enabled = (
            os.getenv("ARTIST_DNA_INSTRUMENTS_ENABLED", "false").strip().lower() == "true"
            if enabled is None else enabled
        )
        if max_segments < 1:
            raise ValueError("max_segments must be at least 1")
        self.model_id = model_id
        self.max_segments = max_segments
        self._session = session
        self._feature_extractor = feature_extractor
        self._labels = labels
        if session is not None and feature_extractor is not None and labels is not None:
            self.enabled = True

    def _load(self) -> None:
        if self._session is not None and self._feature_extractor is not None and self._labels is not None:
            return
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import AutoConfig, AutoFeatureExtractor
        except ImportError as exc:
            raise RuntimeError(
                "Le classifieur d'instruments nécessite onnxruntime, huggingface_hub "
                "et transformers (requirements-dna.txt)."
            ) from exc

        model_path = hf_hub_download(repo_id=self.model_id, filename=MODEL_FILE)
        config = AutoConfig.from_pretrained(self.model_id)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
        self._labels = {
            int(index): str(label)
            for index, label in config.id2label.items()
        }
        self._session = ort.InferenceSession(
            str(Path(model_path)),
            providers=["CPUExecutionProvider"],
        )
        if not self._session.get_inputs():
            raise RuntimeError("Le modèle ONNX ne déclare aucune entrée.")
        self._validate_labels()

    def _validate_labels(self) -> None:
        if not self._labels or len(self._labels) < 2:
            raise RuntimeError("La configuration du modèle ne fournit pas les étiquettes d'instruments.")
        expected = set(range(len(self._labels)))
        if set(self._labels) != expected:
            raise RuntimeError("Les indices d'étiquettes du modèle ne sont pas contigus.")

    def _predict_clip(self, clip: np.ndarray) -> np.ndarray:
        assert self._session is not None
        assert self._feature_extractor is not None
        assert self._labels is not None
        features = self._feature_extractor(
            clip,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="np",
        )
        feeds: dict[str, np.ndarray] = {}
        for model_input in self._session.get_inputs():
            if model_input.name in features:
                feeds[model_input.name] = np.asarray(features[model_input.name])
            elif model_input.name == "attention_mask":
                feeds[model_input.name] = np.ones((1, len(clip)), dtype=np.int64)
            else:
                raise RuntimeError(f"Entrée ONNX non prise en charge : {model_input.name}")
        raw_outputs = self._session.run(None, feeds)
        if not raw_outputs:
            raise RuntimeError("Le modèle ONNX n'a retourné aucune sortie.")
        logits = np.asarray(raw_outputs[0], dtype=np.float64).reshape(-1)
        if len(logits) != len(self._labels) or not np.all(np.isfinite(logits)):
            raise RuntimeError("Sortie ONNX incompatible avec les étiquettes du modèle.")
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted)
        total = probabilities.sum()
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError("Scores du modèle ONNX invalides.")
        return probabilities / total

    def analyze(self, audio: np.ndarray, sample_rate: int, top_k: int = 8) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            raise ValueError("Audio vide ou fréquence d'échantillonnage invalide.")
        self._load()
        assert self._labels is not None
        prepared = _resample(audio, sample_rate, TARGET_SAMPLE_RATE)
        starts = _sample_starts(len(prepared), CLIP_SAMPLES, self.max_segments)
        if not starts:
            raise ValueError("Aucun extrait exploitable pour la classification d'instruments.")

        probabilities_by_clip = []
        top1_by_clip: list[int] = []
        for start in starts:
            clip = prepared[start:start + CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES:
                clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            probabilities = self._predict_clip(clip)
            probabilities_by_clip.append(probabilities)
            top1_by_clip.append(int(np.argmax(probabilities)))

        mean_scores = np.mean(np.stack(probabilities_by_clip, axis=0), axis=0)
        limit = min(max(0, int(top_k)), len(self._labels))
        ranked_ids = np.argsort(mean_scores)[::-1][:limit]
        return [
            {
                "name": self._labels[int(index)].replace("_", " ").strip(),
                "confidence": round(float(mean_scores[index]), 4),
                "top1_windows": top1_by_clip.count(int(index)),
                "windows_analyzed": len(starts),
                "role": "",
            }
            for index in ranked_ids
        ]
