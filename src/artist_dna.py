"""ARTIST DNA orchestration layer.

Combines independent genre-classifier outputs with FireRedVAD vocal-event
analysis. Model loading is lazy; the orchestration is dependency-injectable for
contract tests. This module does not claim a validated genre or instrument truth.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from src.instrument_classifier import LocalONNXInstrumentClassifier


logger = logging.getLogger(__name__)


DEFAULT_GENRE_MODELS = (
    ("genre_baseline", os.getenv("ARTIST_DNA_MODEL_BASELINE", "msaligs/ast_fine_tuned_music_genre_10")),
)
DEFAULT_AST_TAGGER_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    from math import gcd
    divisor = gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


def _normal_label(label: str) -> str:
    return " ".join(label.casefold().replace("_", " ").replace("-", " ").split())


class HuggingFaceGenreClassifier:
    # Model-card order for msaligs/ast_fine_tuned_music_genre_10.
    # The checkpoint config emits generic LABEL_n names, so map them explicitly.
    CANDIDATE_MODEL_ID = "msaligs/ast_fine_tuned_music_genre_10"
    CANDIDATE_LABELS = (
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock",
    )
    """Lazy Hugging Face audio-classification adapter for one candidate model."""

    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self._pipeline: Any = None
        self._sample_rate: int | None = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Le moteur de genres nécessite les dépendances de requirements-dna.txt."
            ) from exc
        self._pipeline = pipeline(
            "audio-classification",
            model=self.model_id,
            device=-1,
        )
        feature_extractor = getattr(self._pipeline, "feature_extractor", None)
        self._sample_rate = int(getattr(feature_extractor, "sampling_rate", 16000))

    def predict(self, audio: np.ndarray, sample_rate: int, top_k: int = 5) -> list[dict[str, Any]]:
        self._load()
        assert self._sample_rate is not None
        prepared = _resample(audio, sample_rate, self._sample_rate)
        raw = self._pipeline(
            {"raw": prepared, "sampling_rate": self._sample_rate},
            top_k=top_k,
        )
        if isinstance(raw, dict):
            raw = [raw]
        result = []
        for item in raw or []:
            label = str(item.get("label", "")).strip()
            if self.model_id == self.CANDIDATE_MODEL_ID and label.startswith("LABEL_"):
                try:
                    label_index = int(label.removeprefix("LABEL_"))
                    if 0 <= label_index < len(self.CANDIDATE_LABELS):
                        label = self.CANDIDATE_LABELS[label_index]
                except ValueError:
                    pass
            score = float(item.get("score", 0.0))
            if label and np.isfinite(score):
                result.append({"label": label, "score": max(0.0, min(score, 1.0))})
        return result


class HuggingFaceAudioTagger:
    """Lazy AST AudioSet tagger; outputs audio-event tags, not genre truth."""

    def __init__(self, model_id: str = DEFAULT_AST_TAGGER_MODEL) -> None:
        self.model_id = model_id
        self._pipeline: Any = None
        self._sample_rate: int | None = None

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Le tagueur AST nécessite les dépendances de requirements-dna.txt."
            ) from exc
        self._pipeline = pipeline("audio-classification", model=self.model_id, device=-1)
        feature_extractor = getattr(self._pipeline, "feature_extractor", None)
        self._sample_rate = int(getattr(feature_extractor, "sampling_rate", 16000))

    def predict(self, audio: np.ndarray, sample_rate: int, top_k: int = 10) -> list[dict[str, Any]]:
        self._load()
        assert self._sample_rate is not None
        prepared = _resample(audio, sample_rate, self._sample_rate)
        raw = self._pipeline(
            {"raw": prepared, "sampling_rate": self._sample_rate},
            top_k=top_k,
        )
        if isinstance(raw, dict):
            raw = [raw]
        result = []
        for item in raw or []:
            label = str(item.get("label", "")).strip()
            score = float(item.get("score", 0.0))
            if label and np.isfinite(score):
                result.append({"label": label, "score": max(0.0, min(score, 1.0))})
        return result


class ArtistDNAEngine:
    """Orchestrates genre classification, AST audio tagging and FireRedVAD."""

    def __init__(
        self,
        vocal_detector: Any,
        classifiers: list[Any] | None = None,
        window_seconds: int = 10,
        instrument_analyzer: Any | None = None,
        audio_tagger: Any | None = None,
    ) -> None:
        if window_seconds < 5:
            raise ValueError("window_seconds must be at least 5")
        self.vocal_detector = vocal_detector
        if classifiers is None:
            model_specs = (
                ("genre_baseline", os.getenv("ARTIST_DNA_MODEL_BASELINE", DEFAULT_GENRE_MODELS[0][1])),
            )
            classifiers = [HuggingFaceGenreClassifier(name, model_id) for name, model_id in model_specs]
        self.classifiers = classifiers
        self.window_seconds = window_seconds
        self.instrument_analyzer = instrument_analyzer or LocalONNXInstrumentClassifier()
        self.audio_tagger = audio_tagger or HuggingFaceAudioTagger(
            model_id=os.getenv("ARTIST_DNA_MODEL_AST_TAGGER", DEFAULT_AST_TAGGER_MODEL)
        )

    def _genre_windows(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, np.ndarray]]:
        """Ten evenly spaced 10-second windows for the tested genre-candidate protocol."""
        window_size = self.window_seconds * sample_rate
        if len(audio) == 0:
            raise ValueError("Fichier audio vide.")
        if len(audio) <= 3 * window_size:
            return [(0.0, audio)]
        last_start = len(audio) - window_size
        count = min(10, max(1, len(audio) // window_size))
        starts = sorted({int(round(i * last_start / (count - 1))) for i in range(count)}) if count > 1 else [0]
        return [(start / sample_rate, audio[start:start + window_size]) for start in starts]

    def _windows(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, np.ndarray]]:
        window_size = self.window_seconds * sample_rate
        if len(audio) == 0:
            raise ValueError("Fichier audio vide.")
        # For short tracks, keep the complete track as one sample. For longer
        # tracks, use three 10-second snapshots: beginning, middle, and end.
        if len(audio) <= 3 * window_size:
            return [(0.0, audio)]
        last_start = len(audio) - window_size
        starts = sorted({0, int(round(last_start / 2)), last_start})
        return [
            (start / sample_rate, audio[start:start + window_size])
            for start in starts
        ]

    def analyze_file(self, audio_path: str | Path) -> dict[str, Any]:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Fichier audio introuvable : {path}")
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        if sample_rate <= 0 or len(audio) == 0:
            raise ValueError("Fichier audio vide ou invalide.")
        mono = audio.mean(axis=1)
        duration = len(mono) / sample_rate

        per_model: dict[str, dict[str, Any]] = {}
        model_label_scores: dict[str, dict[str, list[float]]] = {}
        representative_windows = self._windows(mono, sample_rate)
        genre_windows = self._genre_windows(mono, sample_rate)
        # Keep auxiliary AST tags and instruments on three representative windows
        # to bound CPU latency; genre inference uses the tested 10-window protocol.
        instrument_audio = np.concatenate([window for _, window in representative_windows])
        for classifier in self.classifiers:
            window_results = []
            all_scores: dict[str, list[float]] = defaultdict(list)
            for start, window in genre_windows:
                predictions = classifier.predict(window, sample_rate, top_k=10)
                window_results.append({
                    "start_seconds": round(start, 3),
                    "predictions": predictions,
                })
                for prediction in predictions:
                    label = prediction["label"]
                    all_scores[label].append(float(prediction["score"]))
            model_label_scores[classifier.name] = {
                _normal_label(label): scores for label, scores in all_scores.items()
            }
            ranked = sorted(
                (
                    {"label": label, "mean_score": round(sum(scores) / len(scores), 4),
                     "windows_present": len(scores)}
                    for label, scores in all_scores.items()
                ),
                key=lambda item: item["mean_score"],
                reverse=True,
            )
            per_model[classifier.name] = {
                "model_id": getattr(classifier, "model_id", None),
                "window_count": len(window_results),
                "aggregated_predictions": ranked[:10],
                "windows": window_results,
            }

        # AST AudioSet is an audio-event tagger, not a genre classifier. Keep its
        # evidence separate so tags such as "Music" or "Psychedelic rock" are not
        # incorrectly mixed into the genre consensus.
        ast_window_results = []
        ast_scores: dict[str, list[float]] = defaultdict(list)
        ast_top1: dict[str, int] = defaultdict(int)
        ast_error: str | None = None
        try:
            for start, window in representative_windows:
                predictions = self.audio_tagger.predict(window, sample_rate, top_k=10)
                ast_window_results.append({
                    "start_seconds": round(start, 3),
                    "predictions": predictions,
                })
                if predictions:
                    ast_top1[str(predictions[0]["label"])] += 1
                for prediction in predictions:
                    ast_scores[str(prediction["label"])].append(float(prediction["score"]))
            ast_ranked = sorted(
                (
                    {
                        "label": label,
                        "mean_score": round(sum(scores) / len(scores), 4),
                        "windows_present": len(scores),
                        "top1_windows": ast_top1.get(label, 0),
                    }
                    for label, scores in ast_scores.items()
                ),
                key=lambda item: (item["mean_score"], item["top1_windows"]),
                reverse=True,
            )[:15]
            ast_status = "ok"
        except Exception as exc:
            ast_error = type(exc).__name__
            ast_ranked = []
            ast_status = "unavailable"
            logger.warning("AST audio tagging unavailable (%s).", ast_error, exc_info=True)

        # Consensus counts distinct models, not repeated windows from the same model.
        # Labels are matched only after conservative text normalization.
        label_by_model: dict[str, list[float]] = defaultdict(list)
        for model_name, scores_by_label in model_label_scores.items():
            for label, scores in scores_by_label.items():
                if scores:
                    label_by_model[label].append(sum(scores) / len(scores))
        consensus = sorted(
            (
                {
                    "label": label,
                    "mean_score_across_models": round(sum(scores) / len(scores), 4),
                    "models_agreeing": len(scores),
                    "model_count": len(self.classifiers),
                }
                for label, scores in label_by_model.items()
            ),
            key=lambda item: (item["models_agreeing"], item["mean_score_across_models"]),
            reverse=True,
        )

        vocal = self.vocal_detector.analyze_file(path)
        singing = vocal.get("singing", {})
        instrument_enabled = bool(getattr(self.instrument_analyzer, "enabled", False))
        instruments: list[dict[str, Any]] = []
        instrument_error: str | None = None
        if instrument_enabled:
            try:
                # The local ONNX classifier receives the selected representative windows.
                if hasattr(self.instrument_analyzer, "analyze_file"):
                    instruments = self.instrument_analyzer.analyze_file(path)
                else:
                    # Keep dependency injection compatible with lightweight tests.
                    instruments = self.instrument_analyzer.analyze(
                        instrument_audio, sample_rate, top_k=8
                    )
            except Exception as exc:
                # Instrument detection is optional: an outage must not discard
                # the genre/vocal analysis already completed.
                instrument_error = type(exc).__name__
                logger.warning(
                    "Instrument analysis unavailable (%s); returning no instrument predictions.",
                    instrument_error,
                    exc_info=True,
                )
                instruments = []

        instrument_analysis = {
            "status": (
                "not_enabled" if not instrument_enabled
                else "unavailable" if instrument_error
                else "experimental"
            ),
            "model_id": getattr(self.instrument_analyzer, "model_id", None),
            "predictions": instruments,
            "note": (
                "Prédictions du classifieur ONNX local, exécuté sur CPU sur des extraits représentatifs. "
                "Les scores restent expérimentaux et doivent être vérifiés sur des morceaux complets."
            ),
        }
        if not instrument_enabled:
            instrument_analysis["reason"] = (
                "ARTIST_DNA_INSTRUMENTS_ENABLED=false : l'analyse instrumentale ONNX est désactivée."
            )
        elif instrument_error:
            instrument_analysis["reason"] = (
                "Le classifieur ONNX n'a pas répondu correctement ; analyse instrumentale ignorée."
            )

        singing_seconds = float(singing.get("total_duration_seconds", 0) or 0)
        voice_summary = (
            f"Chant détecté sur environ {singing_seconds:.1f} s ; le timbre et le genre vocal ne sont pas évalués."
            if singing_seconds > 0
            else "Aucun segment chanté détecté ; la présence d'une voix n'est pas confirmée."
        )
        consensus_genres = [item["label"] for item in consensus[:3]]
        return {
            "engine": "ARTIST DNA",
            "engine_version": "0.1.0",
            "duration_seconds": round(duration, 3),
            "analysis_sampling": {
                "strategy": "genre: up to 10 evenly spaced 10-second windows; auxiliary AST/instruments: up to 3 representative windows",
                "selected_audio_seconds": round(sum(len(window) for _, window in genre_windows) / sample_rate, 3),
                "selected_windows": [
                    {"start_seconds": round(start, 3), "duration_seconds": round(len(window) / sample_rate, 3)}
                    for start, window in genre_windows
                ],
                "auxiliary_windows": [
                    {"start_seconds": round(start, 3), "duration_seconds": round(len(window) / sample_rate, 3)}
                    for start, window in representative_windows
                ],
            },
            "audio": {"channels": int(audio.shape[1]), "sample_rate": int(sample_rate)},
            # Compatibility fields consumed by ARTIST OS's existing Pro DNA catalog.
            # Detailed evidence remains available in the structured analysis below.
            "genres": consensus_genres,
            "voice": voice_summary,
            "instrumentation": instruments,
            "genre_analysis": {
                "aggregation_note": "Scores are model outputs, not calibrated probabilities. Consensus counts distinct genre classifiers sharing an exactly normalized label; AST audio tags are reported separately.",
                "models": per_model,
                "label_consensus": consensus[:10],
            },
            "audio_tag_analysis": {
                "status": ast_status,
                "model_id": getattr(self.audio_tagger, "model_id", DEFAULT_AST_TAGGER_MODEL),
                "task": "AudioSet event/audio tagging; not a dedicated genre classifier",
                "predictions": ast_ranked,
                "windows": ast_window_results,
                **({"reason": "Le tagueur AST a échoué ; l'analyse de genre et FireRedVAD restent disponibles."} if ast_error else {}),
            },
            "vocal_analysis": {
                "model": vocal.get("model", "FireRedVAD"),
                "singing": singing,
                "speech": vocal.get("speech", {}),
                "music": vocal.get("music", {}),
            },
            "artistic_signature": {
                "status": "not_generated",
                "reason": "A descriptive artistic signature requires validated musical evidence and must not be inferred from genre scores alone.",
            },
            "instrument_analysis": instrument_analysis,
        }
