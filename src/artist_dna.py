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
from src.artistic_report import build_artistic_report


logger = logging.getLogger(__name__)


# The baseline is the only default model. The AST candidate lacks a
# Transformers config.json in its published repository and is opt-in only after
# a dedicated loader has been validated.
DEFAULT_GENRE_MODELS = (
    ("genre_baseline", "dima806/music_genres_classification"),
)


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
            score = float(item.get("score", 0.0))
            if label and np.isfinite(score):
                result.append({"label": label, "score": max(0.0, min(score, 1.0))})
        return result


class ArtistDNAEngine:
    """Orchestrates two genre candidates and a vocal-event detector."""

    def __init__(
        self,
        vocal_detector: Any,
        classifiers: list[Any] | None = None,
        window_seconds: int = 30,
        instrument_analyzer: Any | None = None,
    ) -> None:
        if window_seconds < 10:
            raise ValueError("window_seconds must be at least 10")
        self.vocal_detector = vocal_detector
        if classifiers is None:
            model_specs = [
                ("genre_baseline", os.getenv("ARTIST_DNA_MODEL_BASELINE", DEFAULT_GENRE_MODELS[0][1])),
            ]
            # AST is deliberately disabled by default: the published checkpoint
            # does not load through Transformers' generic audio-classification
            # pipeline. Do not enable it until a dedicated loader is tested.
            if os.getenv("ARTIST_DNA_ENABLE_AST", "false").strip().lower() == "true":
                model_specs.append((
                    "genre_ast",
                    os.getenv("ARTIST_DNA_MODEL_AST", "neerajs7/AST-audio-classifier"),
                ))
            classifiers = [HuggingFaceGenreClassifier(name, model_id) for name, model_id in model_specs]
        self.classifiers = classifiers
        self.window_seconds = window_seconds
        self.instrument_analyzer = instrument_analyzer or LocalONNXInstrumentClassifier()

    def _windows(self, audio: np.ndarray, sample_rate: int) -> list[tuple[float, np.ndarray]]:
        """Sample 30-second excerpts across the complete track, not just three snapshots."""
        window_size = self.window_seconds * sample_rate
        if len(audio) == 0:
            raise ValueError("Fichier audio vide.")
        if len(audio) <= window_size:
            return [(0.0, audio)]
        last_start = len(audio) - window_size
        # Up to eight evenly distributed windows cover the beginning, middle,
        # transitions and ending. The classifier was trained on 30-second clips;
        # ten-second snapshots are a mismatched input for this baseline.
        count = min(8, max(2, int(np.ceil(len(audio) / window_size))))
        starts = sorted({int(round(value)) for value in np.linspace(0, last_start, count)})
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
        # The instrument candidate receives the same selected audio as genres.
        instrument_audio = np.concatenate([window for _, window in representative_windows])
        for classifier in self.classifiers:
            window_results = []
            all_scores: dict[str, list[float]] = defaultdict(list)
            for start, window in representative_windows:
                predictions = classifier.predict(window, sample_rate, top_k=10)
                window_results.append({
                    "start_seconds": round(start, 3),
                    "predictions": predictions,
                    "top1": predictions[0]["label"] if predictions else None,
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

        # The ONNX model's published training is single-instrument clips, not
        # full polyphonic mixes. Keep raw candidates for diagnostics, but only
        # expose an instrument as a compatibility result when both its score and
        # its temporal recurrence pass conservative thresholds.
        confirmed_instruments = [
            item for item in instruments
            if float(item.get("confidence", 0.0) or 0.0) >= 0.35
            and int(item.get("top1_windows", 0) or 0) >= max(
                2, int(np.ceil(int(item.get("windows_analyzed", 0) or 0) * 0.5))
            )
        ]
        instrument_analysis = {
            "status": (
                "not_enabled" if not instrument_enabled
                else "unavailable" if instrument_error
                else "experimental_mix_generalization_unvalidated"
            ),
            "model_id": getattr(self.instrument_analyzer, "model_id", None),
            "predictions": instruments,
            "confirmed_predictions": confirmed_instruments,
            "note": (
                "Le modèle est entraîné sur des extraits centrés sur un seul instrument, "
                "pas sur des mixages complets. Les candidats bruts ne sont pas des instruments "
                "confirmés dans le morceau. Seuls les résultats récurrents avec score >= 0,35 "
                "sont exposés dans le champ instrumentation."
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
        # A genre is promoted to the compatibility output only when it wins
        # repeatedly across independent time windows. One anomalous intro/outro
        # window must not define the entire song's genre.
        vote_counts: dict[str, int] = defaultdict(int)
        vote_score_totals: dict[str, list[float]] = defaultdict(list)
        vote_windows = 0
        for model_data in per_model.values():
            for window_result in model_data["windows"]:
                predictions = window_result["predictions"]
                if not predictions:
                    continue
                vote_windows += 1
                vote_counts[_normal_label(predictions[0]["label"])] += 1
                vote_score_totals[_normal_label(predictions[0]["label"])].append(
                    float(predictions[0]["score"])
                )
        required_votes = max(2, int(np.ceil(vote_windows * 0.20))) if vote_windows > 1 else 1
        vote_ranking = sorted(
            (
                {
                    "label": label,
                    "top1_windows": count,
                    "windows_total": vote_windows,
                    "mean_top1_score": round(float(np.mean(vote_score_totals[label])), 4),
                }
                for label, count in vote_counts.items()
            ),
            key=lambda item: (item["top1_windows"], item["mean_top1_score"]),
            reverse=True,
        )
        stable_genres = [item["label"] for item in vote_ranking if item["top1_windows"] >= required_votes]
        if not stable_genres and vote_ranking:
            stable_genres = [vote_ranking[0]["label"]]
        consensus_genres = stable_genres[:3]
        genre_decision_status = (
            "repeated_temporal_evidence" if vote_ranking and vote_ranking[0]["top1_windows"] >= required_votes
            else "low_temporal_evidence"
        )
        return {
            "engine": "ARTIST DNA",
            "engine_version": "0.1.0",
            "duration_seconds": round(duration, 3),
            "analysis_sampling": {
                "strategy": "full_track_if_at_most_30_seconds; otherwise up to eight evenly spaced 30-second windows covering the track",
                "selected_audio_seconds": round(sum(len(window) for _, window in representative_windows) / sample_rate, 3),
                "selected_windows": [
                    {"start_seconds": round(start, 3), "duration_seconds": round(len(window) / sample_rate, 3)}
                    for start, window in representative_windows
                ],
            },
            "audio": {"channels": int(audio.shape[1]), "sample_rate": int(sample_rate)},
            # Compatibility fields consumed by ARTIST OS's existing Pro DNA catalog.
            # Detailed evidence remains available in the structured analysis below.
            "genres": consensus_genres,
            "voice": voice_summary,
            "instrumentation": confirmed_instruments,
            "genre_analysis": {
                "aggregation_note": "Scores are model outputs, not calibrated probabilities. Genre output is based on repeated top-1 votes across full-track 30-second windows; isolated window predictions are retained as evidence but do not alone determine the track genre.",
                "models": per_model,
                "label_consensus": consensus[:10],
                "window_top1_votes": vote_ranking,
                "decision_status": genre_decision_status,
                "minimum_repeated_window_votes": required_votes,
            },
            "vocal_analysis": {
                "model": vocal.get("model", "FireRedVAD"),
                "singing": singing,
                "speech": vocal.get("speech", {}),
                "music": vocal.get("music", {}),
            },
            "artistic_signature": {
                "status": "not_generated",
                "reason": "La signature artistique attend encore un modèle de sous-genres, de timbre vocal et d'instrumentation polyphonique validé. Aucun texte artistique ne doit être fabriqué à partir de prédictions instables.",
            },
            "artistic_analysis": build_artistic_report(
                consensus_genres,
                genre_decision_status,
                singing,
                confirmed_instruments,
                duration,
            ),
            "instrument_analysis": instrument_analysis,
        }
