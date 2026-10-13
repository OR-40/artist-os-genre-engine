import unittest

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
        self.assertAlmostEqual(sum(item["score"] for item in predictions) < 1.0, True)

    def test_empty_audio_is_rejected(self):
        classifier = LocalONNXGenreClassifier()
        classifier._session = FakeSession()
        classifier._extractor = FakeExtractor()
        classifier._sample_rate = 16000
        classifier._labels = {9: "rock"}
        with self.assertRaisesRegex(ValueError, "vide"):
            classifier.predict(np.array([], dtype=np.float32), 16000)


if __name__ == "__main__":
    unittest.main()
