import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.artist_dna import ArtistDNAEngine


class FakeClassifier:
    def __init__(self, name, model_id, labels):
        self.name = name
        self.model_id = model_id
        self.labels = labels
        self.calls = 0

    def predict(self, audio, sample_rate, top_k=5):
        self.calls += 1
        return [{"label": label, "score": score} for label, score in self.labels]


class FakeInstrumentAnalyzer:
    url = "https://example.invalid/analyze"

    def analyze(self, audio, sample_rate, top_k=8):
        return [{"name": "electric guitar", "confidence": 0.91, "role": ""}]


class FakeVocalDetector:
    def analyze_file(self, path):
        return {
            "model": "FireRedVAD",
            "duration_seconds": 12.0,
            "singing": {"segments": [{"start": 2.0, "end": 5.0, "duration": 3.0}],
                        "total_duration_seconds": 3.0, "ratio": 0.25},
            "speech": {"segments": [], "total_duration_seconds": 0.0, "ratio": 0.0},
            "music": {"segments": [], "total_duration_seconds": 0.0, "ratio": 0.0},
        }


class ArtistDNAEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio_path = Path(self.tmp.name) / "sample.wav"
        sf.write(self.audio_path, np.zeros(12 * 16000, dtype=np.float32), 16000)
        self.a = FakeClassifier("candidate_a", "fake/a", [("rock", 0.8), ("metal", 0.2)])
        self.b = FakeClassifier("candidate_b", "fake/b", [("rock", 0.7), ("pop", 0.3)])
        self.engine = ArtistDNAEngine(
            FakeVocalDetector(),
            [self.a, self.b],
            window_seconds=10,
            instrument_analyzer=FakeInstrumentAnalyzer(),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_combines_two_genre_candidates_and_vocal_detection(self):
        result = self.engine.analyze_file(self.audio_path)
        self.assertEqual(result["engine"], "ARTIST DNA")
        self.assertEqual(set(result["genre_analysis"]["models"]), {"candidate_a", "candidate_b"})
        self.assertEqual(result["vocal_analysis"]["model"], "FireRedVAD")
        self.assertEqual(result["vocal_analysis"]["singing"]["total_duration_seconds"], 3.0)
        self.assertEqual(result["genre_analysis"]["label_consensus"][0]["label"], "rock")
        self.assertEqual(result["genre_analysis"]["label_consensus"][0]["models_agreeing"], 2)
        self.assertEqual(result["genre_analysis"]["label_consensus"][0]["model_count"], 2)
        self.assertEqual(self.a.calls, 1)
        self.assertEqual(self.b.calls, 1)

    def test_marks_signature_as_not_generated_instead_of_inventing(self):
        result = self.engine.analyze_file(self.audio_path)
        self.assertEqual(result["artistic_signature"]["status"], "not_generated")

    def test_instrument_service_predictions_are_included(self):
        result = self.engine.analyze_file(self.audio_path)
        self.assertEqual(result["instrument_analysis"]["status"], "configured")
        self.assertEqual(result["instrument_analysis"]["predictions"][0]["name"], "electric guitar")

    def test_missing_file_fails_clearly(self):
        with self.assertRaises(FileNotFoundError):
            self.engine.analyze_file(Path(self.tmp.name) / "missing.wav")


if __name__ == "__main__":
    unittest.main()
