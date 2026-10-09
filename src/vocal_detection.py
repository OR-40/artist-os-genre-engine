"""FireRedVAD adapter for vocal-segment detection.

The FireRedVAD package is an optional runtime dependency. This module keeps
interval normalization independently testable without downloading model weights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, order=True)
class TimeSegment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a mapping or a FireRedVAD result object."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _event_intervals(result: Any, event_name: str) -> Iterable[Any]:
    event_map = _field(result, "event2timestamps", {})
    if not isinstance(event_map, Mapping):
        raise ValueError("FireRedVAD result has no valid event2timestamps mapping")
    intervals = event_map.get(event_name, [])
    if intervals is None:
        return []
    if not isinstance(intervals, (list, tuple)):
        raise ValueError(f"Invalid timestamp list for event: {event_name}")
    return intervals


def normalize_segments(raw_intervals: Iterable[Any], duration_seconds: float) -> list[TimeSegment]:
    """Validate, clamp and merge overlapping timestamp intervals."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    segments: list[TimeSegment] = []
    for interval in raw_intervals:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"Invalid timestamp interval: {interval!r}")
        try:
            start, end = float(interval[0]), float(interval[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric timestamp interval: {interval!r}") from exc
        if start != start or end != end or start in (float("inf"), float("-inf")) or end in (float("inf"), float("-inf")):
            raise ValueError(f"Non-finite timestamp interval: {interval!r}")
        start = max(0.0, min(start, duration_seconds))
        end = max(0.0, min(end, duration_seconds))
        if end <= start:
            continue
        segments.append(TimeSegment(round(start, 3), round(end, 3)))

    segments.sort()
    merged: list[TimeSegment] = []
    for segment in segments:
        if merged and segment.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = TimeSegment(previous.start, max(previous.end, segment.end))
        else:
            merged.append(segment)
    return merged


def summarize_event(result: Any, event_name: str, duration_seconds: float) -> dict[str, Any]:
    """Convert one FireRedVAD event type into a stable JSON-compatible shape."""
    intervals = normalize_segments(_event_intervals(result, event_name), duration_seconds)
    total = round(sum(segment.duration for segment in intervals), 3)
    return {
        "segments": [asdict(segment) | {"duration": segment.duration} for segment in intervals],
        "total_duration_seconds": total,
        "ratio": round(min(total / duration_seconds, 1.0), 4),
    }


def normalize_result(result: Any, duration_seconds: float) -> dict[str, Any]:
    """Normalize the singing, speech and music events returned by FireRedVAD."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    return {
        "model": "FireRedVAD",
        "duration_seconds": round(float(duration_seconds), 3),
        "singing": summarize_event(result, "singing", duration_seconds),
        "speech": summarize_event(result, "speech", duration_seconds),
        "music": summarize_event(result, "music", duration_seconds),
    }


class FireRedVADDetector:
    """Lazy-loading FireRedVAD wrapper; model weights are loaded only on first use."""

    def __init__(self, model_dir: str | Path, use_gpu: bool = False) -> None:
        self.model_dir = Path(model_dir)
        self.use_gpu = use_gpu
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"FireRedVAD model directory not found: {self.model_dir}")
        try:
            from fireredvad import FireRedAed, FireRedAedConfig
        except ImportError as exc:
            raise RuntimeError(
                "FireRedVAD is not installed. Install requirements-vocal.txt first."
            ) from exc

        config = FireRedAedConfig(
            use_gpu=self.use_gpu,
            smooth_window_size=5,
            speech_threshold=0.4,
            singing_threshold=0.5,
            music_threshold=0.5,
            min_event_frame=20,
            max_event_frame=2000,
            min_silence_frame=20,
            merge_silence_frame=0,
            extend_speech_frame=0,
            chunk_max_frame=30000,
        )
        self._model = FireRedAed.from_pretrained(str(self.model_dir), config)
        return self._model

    def analyze_file(self, audio_path: str | Path) -> dict[str, Any]:
        """Analyze a local audio file and return normalized vocal/music timestamps."""
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required to read audio duration.") from exc

        duration = float(sf.info(str(path)).duration)
        if duration <= 0:
            raise ValueError("Audio file has no positive duration")

        model = self._load()
        raw_result, _probabilities = model.detect(str(path))
        return normalize_result(raw_result, duration)
