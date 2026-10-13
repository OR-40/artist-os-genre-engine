import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.onnx_genre_classifier import LocalONNXGenreClassifier


class FakeExtractor:
    sampling_rate = 16000

    def __call__(self, audio, sampling_rate, return_tensors):
        assert sampling_rate == 16000
        assert return_tensors == "np"
        return {"input_values": np.zeros((1, 1024, 128), dtype=np.float32)}


class FakeInput:
    name = "input_values"


class FakeOutput:
    shape = ["batch_size", 10]


class FakeSession:
    def get_inputs(self):
        return [FakeInput()]

    def get_outputs(self):
        return [FakeOutput()]

    def run(self, output_names, feeds):
        assert list(feeds) == ["input_values"]
        assert feeds["input_values"].shape == (1, 1024, 128)
        return [np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 3]], dtype=np.float32)]


class ONNXGenreClassifierTests(unittest.TestCase):
    def test_predict_maps_label_and_returns_ranked_scores(self):
        classifier = LocalONNXGenreClassifier()
        classifier._session = FakeSession()
        classifier._extractor = FakeExtractor()
        classifier._sample_rate = 16000
        classifier._labels = {
            0: "blues", 1: "classical", 2: "country", 3: "disco",
            4: "hiphop", 5: "jazz", 6: "metal", 7: "pop",
            8: "reggae", 9: "rock",
        }

        predictions = classifier.predict(
            np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            top_k=3,
        )

        self.assertEqual(len(predictions), 3)
        self.assertEqual(predictions[0]["label"], "rock")
        self.assertGreater(predictions[0]["score"], predictions[1]["score"])
        self.assertLess(sum(item["score"] for item in predictions), 1.0)

    def test_empty_audio_is_rejected(self):
        classifier = LocalONNXGenreClassifier()
        classifier._session = FakeSession()
        classifier._extractor = FakeExtractor()
        classifier._sample_rate = 16000
        classifier._labels = {9: "rock"}
        with self.assertRaisesRegex(ValueError, "vide"):
            classifier.predict(np.array([], dtype=np.float32), 16000)

    def test_private_repo_download_uses_server_token_and_only_model_files(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch.dict(
                os.environ,
                {
                    "HF_TOKEN": "test-secret-token",
                    "ARTIST_DNA_ONNX_GENRE_REPO_ID": "OR-40/artist-os-ast-onnx",
                },
                clear=False,
            ):
                with patch(
                    "huggingface_hub.snapshot_download",
                    return_value=cache_dir,
                ) as download:
                    classifier = LocalONNXGenreClassifier()
                    resolved = classifier._resolve_model_dir()

            self.assertEqual(resolved, Path(cache_dir))
            kwargs = download.call_args.kwargs
            self.assertEqual(kwargs["repo_id"], "OR-40/artist-os-ast-onnx")
            self.assertEqual(kwargs["token"], "test-secret-token")
            self.assertEqual(
                set(kwargs["allow_patterns"]),
                {
                    "model.onnx",
                    "config.json",
                    "preprocessor_config.json",
                    "genre_labels.json",
                },
            )

    def test_private_repo_download_requires_token(self):
        with patch.dict(os.environ, {}, clear=True):
            classifier = LocalONNXGenreClassifier()
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                classifier._resolve_model_dir()


if __name__ == "__main__":
    unittest.main()
