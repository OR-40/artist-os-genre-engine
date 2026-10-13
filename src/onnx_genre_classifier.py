"""Local ONNX genre classifier for the exported AST checkpoint.

Set ARTIST_DNA_ONNX_GENRE_MODEL_PATH to a directory containing model.onnx,
config.json, preprocessor_config.json and genre_labels.json. CPU-only by default.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class LocalONNXGenreClassifier:
    """Run the exported AST genre model directly with ONNX Runtime."""

    def __init__(self, name: str = "genre_candidate_onnx", model_dir: str | Path | None = None) -> None:
        self.name = name
        self.model_dir = Path(model_dir or os.getenv("ARTIST_DNA_ONNX_GENRE_MODEL_PATH", "")).expanduser()
        self.model_id = "msaligs/ast_fine_tuned_music_genre_10"
        self._session: Any = None
        self._extractor: Any = None
        self._sample_rate: int | None = None
        self._labels: dict[int, str] = {}

    def _load(self) -> None:
        if self._session is not None:
            return
        if not str(self.model_dir) or not self.model_dir.is_dir():
            raise RuntimeError(
                "ARTIST_DNA_ONNX_GENRE_MODEL_PATH doit pointer vers le dossier local du modèle AST ONNX."
            )
        onnx_path = self.model_dir / "model.onnx"
        extractor_path = self.model_dir / "preprocessor_config.json"
        labels_path = self.model_dir / "genre_labels.json"
        for required in (onnx_path, extractor_path, labels_path):
            if not required.is_file():
                raise FileNotFoundError(f"Fichier requis pour AST ONNX introuvable : {required}")

        try:
            import onnxruntime as ort
            from transformers import ASTFeatureExtractor
        except ImportError as exc:
            raise RuntimeError(
                "Le classifieur AST ONNX nécessite onnxruntime et transformers."
            ) from exc

        with labels_path.open("r", encoding="utf-8") as f:
            raw_labels = json.load(f)
        self._labels = {int(key): str(value) for key, value in raw_labels.items()}
        self._extractor = ASTFeatureExtractor.from_pretrained(str(self.model_dir))
        self._sample_rate = int(self._extractor.sampling_rate)
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != "input_values":
            raise RuntimeError(
                f"Entrée ONNX AST inattendue : {[item.name for item in inputs]}"
            )
        output_shape = self._session.get_outputs()[0].shape
        if isinstance(output_shape[-1], int) and output_shape[-1] != len(self._labels):
            raise RuntimeError(
                f"Le modèle ONNX renvoie {output_shape[-1]} classes mais "
                f"{len(self._labels)} labels sont configurés."
            )

    def predict(self, audio: np.ndarray, sample_rate: int, top_k: int = 5) -> list[dict[str, Any]]:
        self._load()
        assert self._extractor is not None and self._sample_rate is not None
        values = np.asarray(audio, dtype=np.float32).reshape(-1)
        if values.size == 0:
            raise ValueError("Extrait audio vide.")
        if sample_rate != self._sample_rate:
            from scipy.signal import resample_poly
            from math import gcd
            divisor = gcd(int(sample_rate), self._sample_rate)
            values = resample_poly(
                values,
                self._sample_rate // divisor,
                int(sample_rate) // divisor,
            ).astype(np.float32)

        features = self._extractor(
            values,
            sampling_rate=self._sample_rate,
            return_tensors="np",
        )
        input_name = self._session.get_inputs()[0].name
        model_input = np.asarray(features["input_values"], dtype=np.float32)
        logits = np.asarray(self._session.run(None, {input_name: model_input})[0])
        logits = logits.reshape(-1)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= probs.sum()
        limit = max(1, min(int(top_k), len(probs)))
        ranked = np.argsort(probs)[::-1][:limit]
        return [
            {
                "label": self._labels.get(int(index), f"LABEL_{int(index)}"),
                "score": float(probs[index]),
            }
            for index in ranked
        ]
