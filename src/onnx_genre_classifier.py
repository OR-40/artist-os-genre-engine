"""CPU ONNX genre classifier with optional private Hugging Face artifact download.

Set ARTIST_DNA_ONNX_GENRE_MODEL_PATH to use an existing local model directory.
Otherwise, download the four model files from ARTIST_DNA_ONNX_GENRE_REPO_ID
(default: OR-40/artist-os-ast-onnx) using the server-side HF_TOKEN secret.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


_REQUIRED_FILES = (
    "model.onnx",
    "config.json",
    "preprocessor_config.json",
    "genre_labels.json",
)
_DEFAULT_REPO_ID = "OR-40/artist-os-ast-onnx"


class LocalONNXGenreClassifier:
    """Run the exported AST genre model directly with ONNX Runtime on CPU."""

    def __init__(
        self,
        name: str = "genre_candidate_onnx",
        model_dir: str | Path | None = None,
    ) -> None:
        self.name = name
        configured_dir = str(
            model_dir or os.getenv("ARTIST_DNA_ONNX_GENRE_MODEL_PATH", "")
        ).strip()
        self.model_dir = Path(configured_dir).expanduser() if configured_dir else None
        self.model_id = os.getenv("ARTIST_DNA_ONNX_GENRE_REPO_ID", _DEFAULT_REPO_ID)
        self._session: Any = None
        self._extractor: Any = None
        self._sample_rate: int | None = None
        self._labels: dict[int, str] = {}

    def _resolve_model_dir(self) -> Path:
        # Explicit local path wins and must be valid; do not silently download instead.
        if self.model_dir is not None:
            if not self.model_dir.is_dir():
                raise RuntimeError(
                    "ARTIST_DNA_ONNX_GENRE_MODEL_PATH ne pointe pas vers un dossier existant."
                )
            return self.model_dir

        token = os.getenv("HF_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "Le modèle AST ONNX privé nécessite le secret HF_TOKEN côté serveur, "
                "ou un dossier local défini par ARTIST_DNA_ONNX_GENRE_MODEL_PATH."
            )

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Le téléchargement du modèle privé nécessite huggingface_hub."
            ) from exc

        try:
            snapshot_path = snapshot_download(
                repo_id=self.model_id,
                repo_type="model",
                token=token,
                allow_patterns=list(_REQUIRED_FILES),
            )
        except Exception as exc:
            # Do not echo the token or request headers into logs/errors.
            raise RuntimeError(
                f"Impossible de télécharger l'artefact AST ONNX depuis {self.model_id}. "
                "Vérifie HF_TOKEN et les droits de lecture sur le dépôt privé."
            ) from exc

        return Path(snapshot_path)

    def _load(self) -> None:
        if self._session is not None:
            return

        self.model_dir = self._resolve_model_dir()
        for filename in _REQUIRED_FILES:
            path = self.model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"Fichier requis pour AST ONNX introuvable : {path.name}"
                )

        try:
            import onnxruntime as ort
            from transformers import ASTFeatureExtractor
        except ImportError as exc:
            raise RuntimeError(
                "Le classifieur AST ONNX nécessite onnxruntime et transformers."
            ) from exc

        with (self.model_dir / "genre_labels.json").open("r", encoding="utf-8") as f:
            raw_labels = json.load(f)
        self._labels = {int(key): str(value) for key, value in raw_labels.items()}
        self._extractor = ASTFeatureExtractor.from_pretrained(str(self.model_dir))
        self._sample_rate = int(self._extractor.sampling_rate)
        self._session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"),
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

    def predict(
        self, audio: np.ndarray, sample_rate: int, top_k: int = 5
    ) -> list[dict[str, Any]]:
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
